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
| producer, `GPU` | `! d3d11upload ! video/x-raw(memory:D3D11Memory),format=NV12 ! appsink name=caspar_video` — **not built yet**; `GPU` currently gets the host tail |
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

## Semi-planar to the mixer, rather than converted on the way

This tree has `core::pixel_format::nv12` and both shaders carry a case for it, so a hardware
decoder's NV12 travels to the mixer **as two planes** — Y as one component, interleaved CbCr
as two at half resolution — and the mixer's own `ycbcra_to_rgba` does the conversion.

That is not only cheaper than letting `videoconvert` produce BGRA. It is the *mixer's*
conversion, so the matrix, the range and the chroma siting come off the caps and reach the
shader, instead of being consumed by a converter whose choices then have to agree with the
mixer's by luck. It also means 10-bit survives: `d3d11h265dec` produces P010, which is carried
at `bit16` with `precision_factor` 1.0, where BGRA8 would truncate it.

`NV12` and `P010_10LE` are therefore in the accepted caps. They were removed from the upstream
module for a good reason — upstream has no semi-planar format, `pixel_format_desc` returned
`invalid`, and a hardware-decoded file played as 164 black frames the producer counted as
*received*. The guard that catches that is still in `make_frame`, and the battery's
`nv12-host` case asserts both halves: that NV12 was negotiated **and** that the picture matches
the FFmpeg producer. Neither alone is worth anything — a picture check passes a `videoconvert`
that silently did the work, and a format check passes the black-frame defect exactly.

`INFO` reports the negotiated format for this reason, including its memory type.

## The GPU path

**Not built yet in this tree.** Upstream's version is `frame_factory::import_shared_texture`,
which CasparVP deliberately does not have; the replacement extracts the two NV12 planes on
GStreamer's own D3D11 device and hands them to `vulkan::d3d11_import_bridge::copy_planes` or
`ogl::dx_interop::copy_to_pooled`, which is what the ffmpeg producer's GPU-direct decode
already does.

Until it lands, `PLAY … GPU` **refuses in the log and uses host memory**. That is deliberate:
a GPU path that silently falls back is indistinguishable from one that is working, and the
producer counts frames either way.

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

* **No GPU path yet** — see above. `GPU` is accepted, refused with a reason, and served from
  host memory.
* **P010 through the host path is untested.** It is in the accepted caps and `av_util` maps it,
  but nothing has measured a 10-bit clip end to end; the battery's fixture is 8-bit.
* **No throughput or latency measurement.** Frames are counted; rates are not.
* **Tested on Windows only**, against GStreamer 1.28.6 MSVC x86_64.

## One thing to know about this fork's DLLs

CasparVP runs a MinGW-built FFmpeg so NVENC works on the Pascal card, which puts four runtime
DLLs in `build/shell` whose base names also exist in GStreamer's `bin` — `libcrypto-3-x64`,
`libgcc_s_seh-1`, `libstdc++-6`, `libwinpthread-1`. Windows keys modules by base name per
process, so ours win.

Measured rather than assumed: three of the four are reached by **none** of GStreamer's 271
plugins. `libcrypto` is reached by nine — SRT, HLS, DASH, RTMP and the WebRTC/ICE stack — and
all nine load and reach PAUSED against ours (OpenSSL 3.6.1) as they do against GStreamer's
(3.5.0). `CasparCG-TestRunner`'s `gst-dll-probe` is the standing check, and it verifies the two
arms genuinely differed rather than reporting agreement it never tested.
