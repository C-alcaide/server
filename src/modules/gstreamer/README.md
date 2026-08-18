# GStreamer module

Plays a GStreamer pipeline into a channel, and sends a channel out through one. It is
**off by default** — nothing about a build without `ENABLE_GSTREAMER` changes.

```
PLAY 1-10 [GSTREAMER] "filesrc location=clip.mp4 ! decodebin"
PLAY 1-10 [GSTREAMER] "srtsrc uri=srt://host:9010 ! tsdemux ! h264parse ! avdec_h264"
ADD  1 GSTREAMER "videoconvert ! x264enc ! mpegtsmux ! srtsink uri=srt://:9020?mode=listener"
```

## Why this can exist now

GStreamer ships its own FFmpeg. Until CasparCG moved to FFmpeg 8 the two collided: GStreamer
1.28's `avcodec-61`, `avutil-59`, `avformat-61`, `avfilter-10`, `swscale-8` and `swresample-5`
are the same base names CasparCG shipped at FFmpeg 7, and Windows keys loaded modules by base
name per process — so one of the two lost. At 8.x every soname differs (62/60/62/11/9/6) and
both stacks coexist: measured with all six of each resident in one process, and with
`avdec_h264` — that is `gstlibav`, GStreamer's own FFmpeg — decoding video inside
`casparcg.exe` while ours served the rest of the server.

## Building

```
cmake -DENABLE_GSTREAMER=ON -DGSTREAMER_ROOT=<install root> …
```

`GSTREAMER_ROOT` is the directory holding `bin/`, `lib/`, `libexec/` and `include/`. It
defaults to `%GSTREAMER_1_0_ROOT_MSVC_X86_64%`, which the Windows installer sets only for an
all-users install — a per-user install has to name it.

Nothing from GStreamer is copied next to `casparcg.exe`. Its libraries are **delay-loaded**
and opened by explicit path at first use, so a server built with the module still starts on a
machine that has no GStreamer; only a `PLAY` that asks for it fails.

## Configuring

```xml
<gstreamer>
    <path>C:\gstreamer\1.28.6</path>   <!-- installation root; needed when the installer set no variable -->
    <auto-load>false</auto-load>       <!-- load at startup so a wrong path is a startup diagnostic -->
</gstreamer>
```

and, inside a channel's `<consumers>`:

```xml
<gstreamer>
    <pipeline>videoconvert ! x264enc ! mpegtsmux ! srtsink uri=srt://:9020?mode=listener</pipeline>
</gstreamer>
```

`GST_PLUGIN_SYSTEM_PATH_1_0`, `GST_PLUGIN_SCANNER_1_0` and a **private** `GST_REGISTRY` are set
from that path before `gst_init`, and GStreamer's `bin` goes on this process's DLL search path
via `SetDllDirectory` — never `PATH`, which would put its FFmpeg on the search path of every
process on the machine. The two directories are compared at startup and any shared base name is
logged; the set is empty at FFmpeg 8 and was six at FFmpeg 7.

## The pipeline description

The description is **everything except the sink** (producer) or **the source** (consumer); the
module supplies the other end:

| | appended / prepended |
| :--- | :--- |
| producer, host path | `! videoconvert ! videorate ! appsink name=caspar_video` |
| producer, `GPU` | `! d3d11upload ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=BGRA ! appsink name=caspar_video` |
| consumer | `appsrc name=caspar_video is-live=true format=time !` |

Name `caspar_video` yourself and the description is taken as written, which is how a second
chain reaches `caspar_audio`:

```
PLAY 1-10 [GSTREAMER] "filesrc location=clip.mp4 ! decodebin name=d
                       d. ! queue ! videoconvert ! videorate ! appsink name=caspar_video
                       d. ! queue ! audioconvert ! audioresample ! appsink name=caspar_audio"
```

It is parsed with `gst_parse_launch`, so delayed links work and `filesrc ! decodebin` links up
when the pad appears. A description that comes back **incomplete** — an element that could not
be created, a link the parser had to drop — is refused rather than run, because a pipeline
missing a link runs, reaches EOS and delivers nothing.

## Transport and queries

```
CALL 1-10 PAUSE | RESUME | SEEK <frame> | LENGTH | POSITION
GST INFO
GST LIST <substring>
```

`SEEK` is flushing and frame-accurate. `LENGTH` answers 0 on a live source rather than
inventing a duration. `LOOP`, `IN` and `OUT` are deliberately absent: they belong to a producer
that owns its reader, and this one owns a pipeline whose source may be a socket.

## The GPU path

`PLAY … "<pipeline>" GPU` keeps the picture in video memory from the decoder to the mixer: a
ring of three shared textures on GStreamer's D3D11 device, a `CopySubresourceRegion` into the
next one, an event query so the copy retires before the mixer reads it, and
`frame_factory::import_shared_texture` on the other side. Both mixers import — OpenGL through
`WGL_NV_DX_interop2`, Vulkan through `VK_KHR_external_memory_win32`.

**It is not zero-copy**, and calling it that would be wrong: it trades a download and an upload
for one GPU-to-GPU copy on the same adapter. It is opt-in per `PLAY` because the tail changes
what the pipeline must negotiate, and a source that cannot reach D3D11 memory would pay an
upload for nothing — the sink offers both memory types, GPU first, so such a pipeline still
links and runs on the host path.

It falls back to host memory, once and with a logged reason, when the sample is not D3D11
memory, when GStreamer's device is on a different adapter (`d3d11h264dec` versus
`d3d11h264device1dec` decides that), or when the mixer will not take the texture.

Windows only. Everything else in the module is portable in principle, though only Windows has
been built and measured.

## What it reports

`INFO 1-10` carries `received`, `dropped`, `starved`, `queue`, `queue-peak`, `gpu-frames`,
`restarts`, `underruns`, `audio`, `eos`, `position` and `length`.

`starved` is the one worth knowing: it counts ticks that found the queue empty and repeated the
last picture. Frame counts cannot tell a healthy producer from a frozen one — a source that
stops leaves `received` sitting still while the channel keeps ticking — and a source merely
*slower* than the channel does not starve it, because the appended `videorate` duplicates up to
the channel rate.

## Failure handling

A pipeline that errors is rebuilt, backing off 500 ms → 1 s → 2 s → 4 s to a 5 s ceiling, rather
than stopping the producer for good. For a file that would be survivable; for a live source it
means a layer dies with its sender and never comes back.

## Limits

* **No YUV planes.** The GPU path converts to BGRA rather than handing the mixer NV12: there is
  no semi-planar pixel format in `core::pixel_format`, and adding one means shader work in both
  backends.
* **Audio on the GPU path is carried, not converted** — a 16-channel bus, the source's channels
  mapped into the first of them.
* **No throughput or latency measurement.** Frames are counted; rates are not.
* **Tested on Windows only**, against GStreamer 1.28.6 MSVC x86_64.
