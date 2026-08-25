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
| producer, `GPU` | `! d3d11upload ! d3d11convert ! video/x-raw(memory:D3D11Memory),format=NV12 ! appsink name=caspar_video` |
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

## Where the picture goes, and where the conversion happens

The order is the point, so here it is once rather than four times in prose. The two routes
reach the same mixer; what differs is whether the picture visits host memory, and **neither
route converts colour** — that is the mixer's job, at the end, with the channel's colour
management.

```mermaid
flowchart TD
    SRC["source<br/><i>filesrc · srtsrc · udpsrc</i>"] --> DEC{"decoder"}

    DEC -->|"software<br/>avdec_h264"| HOSTBUF["system memory<br/><i>I420 · NV12 · P010</i>"]
    DEC -->|"hardware<br/>d3d11h264dec · d3d11h265dec"| D3D["D3D11 memory<br/><i>NV12 · P010</i><br/>decoder pool"]

    HOSTBUF --> APPSINK["appsink<br/>caspar_video"]
    D3D -->|"PLAY … GPU"| BRIDGE
    D3D -->|"d3d11download<br/>(host route)"| HOSTBUF

    APPSINK --> MAKE["make_frame<br/><i>row copy per plane</i>"]
    MAKE --> UPLOAD["host upload"]

    subgraph BRIDGE ["gst_gpu_bridge — GStreamer's D3D11 device"]
        direction TB
        STAGE["CopySubresourceRegion<br/><i>only if the decoder pool is not<br/>BIND_SHADER_RESOURCE</i>"]
        EXTRACT["plane extraction draw<br/><i>Load() not Sample() — the surface<br/>is padded to the macroblock grid</i>"]
        PLANES["R8 + R8G8 &nbsp;·&nbsp; R16 + R16G16"]
        STAGE --> EXTRACT --> PLANES
    end

    PLANES -->|"OpenGL<br/>WGL_NV_DX_interop2"| POOLED["pooled mixer textures"]
    PLANES -->|"Vulkan<br/>VK_KHR_external_memory_win32"| POOLED
    UPLOAD --> POOLED

    POOLED --> SHADER["mixer shader<br/><b>ycbcra_to_rgba</b><br/><i>this is the ONLY colour conversion</i>"]
    SHADER --> OUT(["channel"])
```

Two things the picture makes obvious that the prose kept having to repeat:

* **The conversion happens once, at the end, in the mixer.** No `videoconvert` and no
  `d3d11convert` colour work sits on either route, which is why the two agree byte for byte
  rather than by luck — and why the caps' matrix, range and chroma siting reach the shader
  instead of being consumed on the way.
* **The GPU route is not zero-copy and does not pretend to be.** It is one extraction pass plus
  one mixer-side copy, against a download and an upload. The staging copy is a third pass and
  only appears when the decoder's own pool is not shader-readable, which is the usual case.

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

`PLAY … "<pipeline>" GPU` keeps the picture in video memory from the decoder to the mixer. A
D3D11 pass on **GStreamer's own device** extracts the two NV12 planes into textures the module
owns, and those go to `vulkan::d3d11_import_bridge::copy_planes` or two
`ogl::dx_interop::copy_to_pooled` calls — the same two routes the ffmpeg producer's GPU-direct
decode uses. Upstream's `frame_factory::import_shared_texture` is deliberately absent from this
tree, so none of its mixer-facing half is reused.

**It is not zero-copy**, and calling it that would be wrong. The decoded surface belongs to the
decoder's pool and is recycled, so it cannot simply be handed over: one extraction pass writes
the planes, and the mixer copies those into pooled textures. Two full-frame passes at 1.5
bytes per pixel, against three at 4 for a BGRA route.

**A decoder surface is usually not shader-readable, and that is the trap here.**
`d3d11h264dec` allocates its output pool with `D3D11_BIND_DECODER` and no
`BIND_SHADER_RESOURCE`, so it can have no shader resource view at all — GStreamer publishes
none and `CreateShaderResourceView1` fails. The bridge checks the bind flags once and, when
needed, adds a `CopySubresourceRegion` into a shader-readable texture of its own before
extracting. That is a third full-frame pass on such decoders, and still no host round trip.

It hid at first because the earliest test fed `decodebin` a **lossless** clip, which decodes in
software, so `d3d11convert` had real work to do and produced a shader-readable texture as a
side effect. Point the same route at a hardware decoder and the converter becomes passthrough,
the decoder's own surface arrives, and nothing can sample it. *It worked until we made it
faster* is the shape to recognise.

`d3d11convert` stays in the appended tail for the same reason it is harmless: at matching caps
it passes buffers through, and when they do not match it is the difference between a pipeline
that links and one that refuses — `d3d11upload` changes the memory type, never the format.

It falls back to host memory, once and with a logged reason, when the sample is not D3D11
memory, when the surface format is not NV12/P010/P016, when GStreamer's device is on a
different adapter (`d3d11h264dec` versus `d3d11h264device1dec` decides that), or when the
mixer will not take the textures.

Windows only. Measured on both mixers: 64/64 (OpenGL) and 68/68 (Vulkan) frames on the GPU
path, byte-identical to the host path.

## How much it carries, and why there is no CUDA route

Measured 2026-08-24 on this box (RTX-class adapter 0, driver 582.53), simultaneous 1080p50
H.264 streams, each its own layer, each `d3d11h264dec` with `GPU`:

| streams | worst delivered | starved ticks | not on the GPU path |
| ---: | ---: | ---: | ---: |
| 1 | 1.02x | 0 | 0 |
| 2 | 1.05x | 0 | 0 |
| 4 | 1.10x | 0 | 0 |
| 6 | 1.15x | 0 | 0 |
| 8 | 1.20x | 0 | 0 |

Eight with no pressure at all and the queue never deeper than 2 of 4. (Above 1.00x because a
`filesrc` is not real-time paced — the decoder simply runs ahead of the channel. On the
real-time SRT source `ingest-ab` uses, both producers sit at 1.00x.)

**That is why there is no CUDA ingest route**, and the absence is a decision rather than an
omission. `nvh264dec`, `cudaupload` and `gst_cuda_context_new_wrapped` all work here, and
`CudaVkTexture` exists in this tree, so it could be built. It would add an NVIDIA-only second
path, a cross-context device-pointer failure mode that takes the process down rather than
returning an error, and a second thing to keep correct — to relieve a bottleneck that the
measurement says is not there. The D3D11 route is vendor-neutral and already carries eight
streams.

If that changes — 4K, many more streams, a box where this saturates — the measurement to
re-run is above, and the justification would be a row where `starved` is not zero.

### The consumer's readback, now measured

The consumer declares `needs_cpu_frame_data()`, so the channel reads the composited frame back
for it — about 415 MB/s at 1080p50. That cost was unmeasured until 2026-08-24; it is not now.

`cli.py gst-consumer-cost` compares the channel with and without a GStreamer consumer whose
pipeline is a bare `fakesink` — a fakesink and not an encoder, so the number is the readback
rather than the encoder. Two arms, interleaved round by round. Three runs at 1080p50 with one
GPU-direct producer:

| | late frames | server CPU |
| :--- | :--- | :--- |
| no GStreamer consumer | 0 / 0 / 0 | 1.85 cores |
| with one | 4 / 3 / 4 | 1.87 cores |

**The consumer arm was worse in every round of every run** — 0.20–0.27% of frames late, and
+0.02 cores. Small, consistent and load-dependent.

So: the readback does cost something, and it is not nothing. `cuda_vk_uploader` in the ffmpeg
consumer is the template for removing it. Whether a quarter of a percent of frames justifies a
second GPU path depends on how close to the edge the deployment runs, which is a judgement for
whoever runs it rather than one this measurement can make.

The verdict is a **sign test** across interleaved rounds, not a threshold, and that matters: an
earlier absolute late-frame budget gave opposite answers on two runs an hour apart, because
baseline server CPU on this box was 0.86 cores in one and 1.86 in the next. Asking "was the
consumer arm worse in every round" survives that; a fixed number decides on machine load.

Not covered: several consumers (the module allows one per channel), 4K, or a machine already
under other load.

## What is verified, and by what

Every row is exercised by `CasparCG-TestRunner`'s `gstreamer` battery on **both** mixers.

| element / feature | verified |
| :--- | :--- |
| `d3d11h264dec` | NV12 straight to the mixer, byte-identical to the FFmpeg producer |
| `d3d11h265dec` | `P010_10LE` GPU-direct, levels within 1 LSB of the 8-bit reference |
| `d3d11av1dec` | AV1 over the same bridge, no code of its own |
| `d3d11upload` / `d3d11convert` | software frames uploaded and converted on the GPU (`GPU BGRA`) |
| `srtsrc` caller / listener | real ingest; and a listener with no caller does not hang the server |
| SRT recovery | sender lost and returned, picture byte-identical after |
| `srtsrc` `stats` | RTT, bandwidth and loss into `INFO` and OSC |
| `cccombiner` / `capssetter` | CEA-708 carried through the channel, and repeats suppressed |
| `cudadownload` / `nvh264enc` | the channel's texture handed to an encoder as CUDA memory |
| third-party plugins | a `<plugin-path>` directory loaded and its elements usable |

**Network sources and sinks are GPU-accelerated on the same terms as files.** The transport
is never the accelerated part; the decode, the encode and staying in video memory are.
Measured 2026-08-25: `srtsrc … ! d3d11h264dec` with `GPU` gives 249/249 frames GPU-direct, and
`nvh264enc ! mpegtsmux ! srtsink` with `GPU` sends 298/298 with no readback. The equivalent
FFmpeg paths do the same — `STREAM` and `FILE` are one consumer with a flag, and its GPU gate
never inspects it.

The catch is naming a hardware decoder: `avdec_h264` works and is software, `d3d11h264dec` is
the one that stays on the GPU. `gpu-frames` is how you tell them apart, and the battery's own
`srt-ingest` case uses the software one deliberately — which is why this combination had never
been run here until it was measured directly.

**`fallbacksrc` is the one with a caveat.** It takes over from a dead primary correctly and
shows the fallback file — then, once the dead primary's socket errors, it keeps producing
frames that carry no picture. `restart-timeout`, `retry-timeout` and `restart-on-eos` do not
change it. Nothing in CasparCG is involved: the frames arrive and are built correctly. Its
battery case reports rather than gates for that reason.

## What it reports

`INFO 1-10` carries `received`, `dropped`, `starved`, `queue`, `queue-peak`, `gpu-frames`,
`restarts`, `underruns`, `audio`, `eos`, `position`, `length` and `format` — the last being
what the sink actually negotiated, which is the difference between "the pipeline names
`d3d11h264dec`" and "the mixer got NV12".

`INFO 1` carries the **consumer's** counters: `sent`, `dropped`, `captions`,
`egress-frames`, `captions-queued`, `captions-dropped`, `cc-triplets-in`,
`cc-triplets-out` and `cc-suppressed`.

**`egress-frames` and not `gpu-frames`**, and the name is load-bearing. The producer publishes
`gstreamer/gpu-frames` for frames its decoder put on the GPU, one `INFO` response carries both
halves of the channel, and while they shared a leaf name anything reading by tag got whichever
came first. A host consumer's constant 0 hid a producer running 199/199 and was diagnosed as a
GPU bridge defect that did not exist.

**`received` counting up while the picture is black** means samples are arriving that cannot be
made into frames. The producer now logs that once, naming the negotiated format, because the
silent version of it has cost two investigations here.

The diag window carries the same story for someone watching rather than polling. The
**producer** plots `frame-time`, `tick-time` and `buffer` — the queue depth normalised to its
own limit, which falls *before* the producer starves, so a sender that is slipping shows up
while the picture is still perfect — and tags `dropped-frame`, `audio-underrun`, `starved` and
`restart`. The **consumer** plots `frame-time` and `input` (how full `appsrc`'s queue is
against its `max-bytes`) and tags `dropped-frame`.

Every one of those plots a quantity `INFO` also reports, which is deliberate: the numbers are
gated by the battery, and the graph is a second view of the same values rather than a separate
source of truth. Note that diagnostics graphs are **drawn and nothing else** — they are not
published over OSC or in monitor state — so nothing can assert the window itself.

`starved` is the one worth knowing: it counts ticks that found the queue empty and repeated the
last picture. Frame counts cannot tell a healthy producer from a frozen one — a source that
stops leaves `received` sitting still while the channel keeps ticking — and a source merely
*slower* than the channel does not starve it, because the appended `videorate` duplicates up to
the channel rate.

## Closed captions — and a caveat that matters

A source's CEA-608/708 captions reach the mixer, survive compositing and are re-emitted by the
consumer. `INFO` reports `captions-in` on the producer and `captions` on the consumer, and the
pair is the point: the picture is identical whether captions travel or vanish.

**CEA-708 is re-paced, not copied.** A repeated frame must not re-issue its captions: the
channel repeats its last picture whenever the producer starves, and CEA-708 is a *command*
stream, so a doubled `RollUp` or `SetPenLocation` changes what the viewer sees. `cc_data` goes
through a queue keyed on the frame's identity, and comes out at the per-frame budget the
standard defines — 12 triplets at 50p, 24 at 25p, 25 at 24p, taken from FFmpeg's
`libavutil/ccfifo.c` rather than derived here.

Measured with a source delayed to ~33 fps in a 50p channel: **177 repeated frames, 2100 of
5376 arriving triplets withheld as duplicates**, where every one of them used to go out.
`INFO 1` reports `cc-triplets-in`, `cc-triplets-out` and `cc-suppressed`; those three are the
only counters that can tell a de-duplicating channel from a copying one, because both attach
exactly one caption meta per frame.

That fix needed a second one: **`core::mixer` was copying the frame's metadata every tick**, so
the identity the queue keys on was new on every frame and 177 repeats produced zero
suppressions. `const_frame::metadata_ptr()` shares the stored pointer instead — one allocation
and one copy less per frame, as well.

**608 and 708-in-CDP are still passed through unpaced**, and that is stated rather than left to
be discovered: 608 is two bytes per field per frame and a CDP carries its own framing and
sequence counter, so re-packing either means rewriting a header rather than moving triplets.

**And it is interim.** Upstream `CasparCG/server#1637` implements a fuller design -- an
extensible side-data system named after FFmpeg's `AVFrameSideDataType`, a `side_data_mixer`,
per-layer float priority to choose the caption source, and a rate- and interlace-aware
`a53_cc_queue`. It is close to merge and touches the same files this does, so the plan should
be to adopt it rather than develop this further. See `src/core/frame/frame_metadata.h`.

## What the pipeline's own elements report

A whole class of GStreamer element does its work by **posting messages** or by publishing a
`stats` property, and touches the video not at all. Before this the module could carry such an
element and tell you nothing about it. Both routes now land in monitor state under
`gstreamer/element/<element>/<field>`, so they reach `INFO` and OSC.

**Generic on purpose.** There is no per-element code and no allow-list: an element nobody
anticipated still reports, and supporting a new one is a pipeline change rather than a server
change.

```
PLAY 1-10 [GSTREAMER] "... ! ebur128level ! level ! audioconvert ! appsink name=caspar_audio"
```

    gstreamer/element/ebur128-level/momentary-loudness   -6.01
    gstreamer/element/ebur128-level/shortterm-loudness   -6.01
    gstreamer/element/ebur128-level/global-loudness      -6.05
    gstreamer/element/ebur128-level/true-peak             0.50

Verified against a known input: a 1 kHz sine at 0.5 amplitude reads true-peak 0.50 and
-6.05 LUFS, where 0.5 amplitude is -6.02 dBFS. **This is EBU R128 loudness in CasparCG, which
it has never had.**

The `stats` half covers the transports:

```
PLAY 1-10 [GSTREAMER] "srtsrc uri=srt://host:9010 ! tsdemux ! h264parse ! avdec_h264"
```

    gstreamer/element/srtsrc0/rtt-ms                  0.03
    gstreamer/element/srtsrc0/bandwidth-mbps          0.54
    gstreamer/element/srtsrc0/packets-received-lost   0
    gstreamer/element/srtsrc0/packets-retransmitted   0
    gstreamer/element/srtsrc0/negotiated-latency-ms   125

so "the picture is breaking up" stops being an opinion. `rtspsrc`, `rtpbin` and the RIST
elements have the same property and come along without another line of code.

Two limits worth knowing. Only **scalar** fields are taken, and an array is reduced to its
first value plus a `-count` — monitor state is a flat key/value space and a per-channel fan-out
would be unbounded on a 16-channel bus. And this carries **measurements, not per-frame
metadata**: closed captions and analytics results ride on the buffer rather than the bus, and
`core::const_frame` has nowhere to put them yet.

## Failure handling

A pipeline that errors is rebuilt, backing off 500 ms → 1 s → 2 s → 4 s to a 5 s ceiling, rather
than stopping the producer for good. For a file that would be survivable; for a live source it
means a layer dies with its sender and never comes back.

## Limits

* **The GPU path is Windows-only and opt-in per PLAY.** It is not automatic because the tail
  changes what the pipeline must negotiate, and a source that cannot reach D3D11 memory would
  pay an upload for nothing.
* **Audio on the GPU path is carried, not converted** — a 16-channel bus with the source's
  channels mapped into the first of them.
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
