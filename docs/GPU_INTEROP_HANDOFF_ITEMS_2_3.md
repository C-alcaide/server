# The interop plan, items 2 and 3: what was measured

> **Audited 2026-08-16. Items 1 and 2 are done and item 3 is closed — this is a record, not
> a work list.** The title said "Continuing…" long after there was nothing to continue.
>
> **It is kept, unlike `OCIO_HANDOFF_2026-08-11.md`, which was pruned the same day.** That
> one was closed *and* duplicated: every finding already lived in `CHANGELOG.md` or the
> study. This one is closed and **load-bearing** — `GPU_INTEROP_PLAN.md` §Why explicitly
> delegates to it for item 2's current numbers and item 3's reasoning, and the item-1
> figures (41 % H.264, 38 % HEVC and VP9) appear in no other file. Deleting it would delete
> the evidence.
>
> Fold it into the plan and prune it only together, and only after checking those numbers
> land somewhere.

Companion to `GPU_INTEROP_PLAN.md`, which holds the design and the reasoning.
This is the delta — what changed while it was being carried out, what item 1 taught
that applied to what was left, and the baselines that were measured against.

Read the plan first. Do not re-derive it from here.

## State

**Item 1 is done.** GPU-direct hardware decode works on the Vulkan mixer
(`489b02fbc`). It measured 41 % on H.264, 38 % on HEVC and VP9, byte-identical to
OpenGL's GPU-direct output on all seven clips, with the OpenGL path proved
untouched by hashing captures against HEAD.

**Item 2 is done, route 2a, both callers.** ISF (`b1f591025`) and the Spout
producer (`9193ea300`) now render and receive straight into Vulkan memory. The
figures are in the table below and in those commit messages. Route 2b, via CUDA,
was not needed.

**Item 3 is measured and not worth doing.** See the section below. The plan
guessed item 2 might delete most of it; it deleted effectively all of it.

Two things found on the way that are worth knowing before reading anything else
here:

- **`kGlHandleType` was `GL_DEVICE_LUID_EXT`** (`edf668551`). The constant every
  VK → GL import passed to the driver was `0x9462`, which is not a handle type;
  `GL_EXT_memory_object` defines `0x9587` on Windows and `0x9586` on Linux. No
  call site checked `glGetError`, so every import had been failing silently —
  previz's channel textures and the `vulkan_output` consumer's shared slots
  both. This is what "the import half already exists in
  `previz_texture_bridge`" actually meant, and it is why the item 2a probe
  failed at first with previz's own settings.
- **The mixers disagree about the byte order of a texture-backed `bgra` frame,**
  and each is self-consistent. OpenGL wants RGBA bytes in the texture, because
  its own CPU upload path reorders through `GL_BGRA` and the two paths must
  agree on the result. Vulkan wants BGRA bytes, because its upload is a verbatim
  copy. Get this wrong and red and blue exchange, silently, on the fast path
  only. ISF writes BGRA and stays labelled `bgra`; Spout cannot swizzle what
  the SDK writes, so it labels the frame `rgba` instead and the mixer's
  swizzle stays out of the way.

**The separate GPU-direct seek defect is fixed** (`1423934ec`, `b8ca62ff7`,
`08deff786`), so the caution in the ground rules about not benchmarking a looping
clip no longer has a stalling producer behind it. `GPUDIRECT_SEEK_STALL.md` has
the three defects it turned out to be.

**What is left of this plan.** Nothing in items 1–3. Two threads it opened and
never closed, both now measured — see the two sections at the end of this
document. 10-bit turned out to be already implemented, and the one thing wrong
with it was on the software side rather than in GPU-direct; it is fixed. CEF is a
worse idea than the plan assumed, and is not worth starting.

## 10-bit GPU-direct: it already worked, and what differed was the other path

Nothing gated it. `d3d11_gl_bridge::setup_planes` has always mapped
`DXGI_FORMAT_P010`/`P016` to `R16_UNORM` / `R16G16_UNORM` and declared the planes
`bit16`, which is correct because P010's significant bits are high-aligned and so
already normalise — `precision_factor` 64 is for *low*-aligned 10-bit, which is
what the software path carries. There is no 10-bit condition anywhere in the
decline logic either. It was untested, not unimplemented.

Tested (`gpudirect/cpu_matrix.py` with the new `L_hevc_10_prog`, four layers):

| | software | GPU-direct | |
|---|---|---|---|
| OpenGL | 2.61 cores | 1.21 | **−53.7 %** |
| Vulkan | 2.56 | 1.15 | **−55.0 %** |

(Those are the figures as first measured. The fix below made the software column
cheaper, so the saving is now 42 % / 51 % against a better baseline — the
GPU-direct column did not move.)

A larger saving than 8-bit's 41 % / 38 %, which stands to reason: the host
transfer it removes is twice the size. H.264 High10 is not hardware-decoded on
this GPU and falls back to software cleanly on both mixers, which is the right
behaviour and worth keeping in the matrix as the negative case.

**The ±1 difference against the software path is fixed** (`f171f56ea`), and it
was never GPU-direct's defect. It reported 69.637010 dB, ±1 code value on 2 % of
pixels, identically on both mixers. GPU-direct is byte-identical to the software
path as soon as both hand the mixer the same layout; the residual was a precision
loss on the *software* side.

`yuv420p10le` arrives as three planes of codes 0..1023 in 16-bit words, so it is
declared `bit10` and the shader multiplies by a precision factor of 64. The data
occupies only the bottom 1/64 of the texture's UNORM range, and chroma is
upsampled by the texture unit *before* that multiply — so whatever the filter
rounds away is amplified 64-fold. P010 carries the same codes high-aligned and
needs no multiply. 8-bit was byte-identical all along precisely because there the
software path already delivers `nv12`: 10-bit was the only case comparing two
different routes through the mixer.

Progressive 10-bit 4:2:0 now requires `p010le` at the filter sink. The picture
matrix is 7/7 identical on both mixers, the two mixers agree at `inf` dB on every
clip, and the software path got *cheaper* — 2.61 → 2.20 cores on OpenGL, 2.56 →
2.32 on Vulkan — because these clips decode in hardware even with GPU-direct off,
so converting the already-P010 surface to `yuv420p10le` every frame was the cost
of the less accurate layout.

**Progressive only, and deliberately.** With `bwdif` in the graph, requiring
`p010le` changes the chroma layout the deinterlacer works on, and the
deinterlaced picture moves by 53.4 dB with differences up to 83 code values —
two orders of magnitude more than the ±1 being fixed. Interlaced 4:2:0 chroma is
delicate; `bwdif` keeps the planar layout it has always had, and interlaced
10-bit is verified byte-identical to the pre-change build. GPU-direct declines
interlaced content anyway, so the fast path loses nothing.

Two dead ends on the way there, both plausible and both wrong. The decoders were
suspected of disagreeing — they are bit-exact, confirmed by decoding one frame
both ways outside the server. And the error was expected to follow chroma-sample
parity — it does not, because chroma texel centres land *between* output pixels,
so every pixel is interpolated and there is no parity signature.

**`m_int_10` is the regression check for this**, and it is not in the decode
matrix. It is the only 10-bit 4:2:0 *interlaced* clip in `media/`, so it is the
one thing the progressive-only condition exists to protect; the matrix's
interlaced clip is 8-bit and its other 10-bit clips are progressive. It must keep
arriving as `yuv420p10le -> pixel_format 5 (3 planes)`. If it ever arrives as
`p010le`, the condition has been loosened and the deinterlaced picture has moved.
There is no harness for it: it was checked by stashing the change, rebuilding,
capturing, restoring and comparing — worth repeating rather than trusting, because
nothing else will catch it.

## HTML/CEF accelerated paint: measured, and the premise is wrong

The plan's table says HTML is CPU `OnPaint` on both mixers, cost "not measured",
and item 1 notes its D3D11 → Vulkan bridge would later unlock accelerated paint.
The mechanism is indeed there — CEF 142's `OnAcceleratedPaint` hands over a
shared D3D11 texture handle, and item 1 built the import for exactly that. The
mechanism is not the obstacle. The configuration is.

`OnAcceleratedPaint` is only reached when `CefWindowInfo::shared_texture_enabled`
is set, which needs CEF's GPU compositor. CasparCG launches CEF with
`--disable-gpu --disable-gpu-compositing` by default —
`configuration.html.enable-gpu` is false, with a comment that a single 1080p
producer cannot run smoothly otherwise. So the first question is not what the
bridge saves but what turning GPU compositing on costs. Measured
(`vkdispatch/html_gpu.py`, animated 1080p page):

| mixer | CEF GPU | 1 layer | 4 layers | per extra layer |
|---|---|---|---|---|
| OpenGL | off | 1.41 cores | 2.24 | +0.274 |
| OpenGL | on | 1.54 | 2.66 | +0.372 |
| Vulkan | off | 1.38 | 2.15 | +0.254 |
| Vulkan | on | 1.54 | 2.81 | **+0.422** |

Enabling it costs **more**, by +0.10 cores per layer on OpenGL and +0.17 on
Vulkan — 66 % more per layer on the mixer that would benefit. `inf` dB either
way, so it renders the same page; this is overhead, not a different picture.

That is the bar accelerated paint has to clear before it wins anything, and what
it removes is one host-to-host BGRA memcpy per frame per layer. For scale, ISF's
full GPU → host → GPU round trip measured +0.230 cores per layer and a memcpy is
a fraction of that. **On these numbers accelerated paint plausibly loses**, and
nothing should be built until that is settled.

Two constraints for whoever does take it on, both from the CEF header rather than
from reasoning: the shared texture is "released to the underlying pool for reuse
when the callback returns from client code", so it must be copied into a pooled
texture *inside* `OnAcceleratedPaint` — the same shape as
`d3d11_import_bridge::copy_planes`, and for the same reason. And it is
"instantiated without a keyed mutex", so synchronisation is entirely the caller's
problem.

One trap already paid for: CEF paints on damage, so a **static** page reaches
`OnPaint` about twice and then never again. The first version of this measurement
used one and reported +0.003 cores per layer, a confident-looking figure for four
idle producers. The producer also resolves a bare name against the *template*
folder without the extension — pass a filename and CEF tries it as a hostname,
logs `ERR_NAME_NOT_RESOLVED`, and renders nothing. `html_gpu.py` now fails loudly
on both.

## What item 1 changed under item 2's feet

- `d3d11_import_bridge` (`accelerator/vulkan/util/`) is new. It is **not**
  reusable here — it hardcodes `eD3D11Texture` and runs D3D11 → Vulkan, where
  item 2 needs Vulkan → GL. It is still worth reading for the
  `vkGetMemoryWin32HandleProperties` mechanics and the handle lifetime.
- `create_exportable_texture` **still lacks `eColorAttachment`**
  (`eTransferDst | eSampled | eTransferSrc`). Item 2a's one-line prerequisite is
  outstanding.
- `frame_factory` gained `gpu_device_backend()` so a handle's API is stated
  rather than assumed. Use it rather than inferring the backend from a cast.

Since the plan was written, ISF and Spout have both changed:

- Spout's producer now has a working zero-copy receive **on OpenGL**
  (`9e7f0985a`), and `CASPAR_SPOUT_FORCE_READBACK` exists to run the readback
  path on a mixer that would otherwise take the fast one. Item 2 extends that
  branch to Vulkan rather than writing it from scratch.
- ISF had a crash fixed (`bc357e7d1` — an SFML context left active on the channel
  thread but destroyed on the producer destroyer pool) and its readback made
  cheaper (`bc47dd5e4`). The readback work is what established that the remaining
  gap is the transfer, not CPU arithmetic.

## Three lessons from item 1 that apply directly

**1. A capability query is not a feasibility test.** The plan recorded
`eOpaqueWin32` as importable for NV12 on the strength of
`getImageFormatProperties2`. Asked to import a real shared D3D11 texture it fails
outright. For item 2, prove an actual export-import-render round trip on a real
texture before building on it — not that the extension string is present.

**2. Prove the risky primitive before writing production code.** Item 1 was told
to prove per-plane views first, and that instruction is what surfaced a D3D11 and
Vulkan disagreement about multi-planar layout before any of it was built on. For
item 2 the equivalent primitive is: **allocate an exportable Vulkan image with
`eColorAttachment`, import it into a GL context, render into it from GL, and read
it back through Vulkan.** If that round trip is not byte-exact, take route 2b
(via CUDA) and stop trying to make 2a work.

**3. The synchronisation can eat a third of the gain.** Item 1's first
implementation used a `D3D11_QUERY_EVENT` spin: 1.44 cores against the fence's
1.16, on a path whose whole saving was about 0.8. The plan says to start item 2
with `glFinish()` and leave `GL_EXT_semaphore` for later — that is still the
right order, but **measure what `glFinish` costs** rather than assuming it is
free. If it is large, that measurement is the argument for the semaphore, and it
belongs in the commit message either way.

## What item 2 achieved

Both harnesses now run the slow path as an explicit column rather than trusting
that it still works: `CASPAR_ISF_FORCE_READBACK` and `CASPAR_SPOUT_FORCE_READBACK`.

**ISF** — `vkdispatch/isf_matrix.py`, `isftest.fs`, NDI consumer attached:

| | 1 layer | 4 layers | per extra layer |
|---|---|---|---|
| OpenGL | 1.18 cores | 1.21 | +0.008 |
| Vulkan, zero-copy | 1.16 | 1.22 | **+0.019** |
| Vulkan, readback (before) | 1.35 | 2.08 | +0.242 |

+0.242 → +0.019, against OpenGL's +0.008. The readback row reproduces the
1.37 / 2.06 / +0.230 this document used to record, which is how the new harness
was checked rather than merely believed. `inf` dB against OpenGL on both paths.

**Spout** — `vkdispatch/spout_loop.py`, still source, OpenGL sender throughout
so only the receiver varies:

| receiver | path | CPU |
|---|---|---|
| OpenGL | zero-copy | 4.44 cores |
| OpenGL | readback | 6.44 |
| Vulkan | zero-copy | **4.39** |
| Vulkan | readback (before) | 6.33 |

19.226120 dB on all four, the figure this document already recorded.

**Correction to the old baselines.** The readback receiver is recorded above as
7.02 cores and measures 6.4 on the current build. The gap and its ordering hold;
the absolute 7.02 does not, and should not be quoted as current. Also, the Spout
*consumer* reports no GPU texture path in every one of these runs, so the sender
side is on its own readback throughout — identical across the comparison, but it
means none of the sender figures measure anything that improved.

## Item 3: measured, and the recommendation is to drop it

Each ISF producer creates its own SFML context, so four layers mean four contexts
and four `setActive` pairs per frame. Item 3 was to share one, guarded by the
mutex that already serialises rendering.

It was left last because item 2 might delete most of it. It did. What is left to
win is now bounded by measurement rather than estimated — `isf_matrix.py 1,8`:

| mixer | 1 layer | 8 layers | per extra layer |
|---|---|---|---|
| OpenGL | 1.17 cores | 1.27 | +0.014 |
| Vulkan | 1.16 | 1.27 | +0.016 |

The OpenGL path creates no SFML context at all, so the difference between the two
rows is an upper bound on what per-producer contexts cost: **about 0.002 cores
per layer**, roughly 0.08 ms per layer per frame, and 0.016 cores across eight
layers. Before item 2 the Vulkan row was +0.230 and a context per producer was a
plausible contributor; the readback was hiding it, and with the readback gone
there is nothing there.

Against that, sharing one context across producers puts back exactly the question
`bc357e7d1` was a crash about — who owns the context, and on which thread is it
destroyed — and ISF already has to hand its context back after every render for
that reason. **Recommendation: close item 3 as not worth doing**, and reopen it
only if a profile of many simultaneous ISF layers points here specifically.

If it is done anyway, re-run the three play-and-clear cycles as well as the CPU
figures.

## Ground rules

Each of these is a mistake already made in this work, not general advice.

- **Compare on still sources.** Three separate measurements here were wrong
  because two servers sat on different frames of moving content and the
  difference was read as a defect.
  - Pausing and seeking to a fixed frame is the other way to get one, and it
    beats encoding a still clip when the content has to be real footage.
    Results can vary by codec, so check that the two ends actually landed on the
    same frame rather than assuming a seek is exact.
  - And check that a "still" source is still. `gradients` animates by default,
    which produced a comparison of two different frames here and a set of
    numbers that meant nothing until the source was verified flat first.
- **Read the log to decide which path ran**, never infer from timings. A silent
  fallback and a path with no benefit are indistinguishable in a CPU figure.
  Both Spout and GPU-direct log which path they took for exactly this reason.
- **Do not benchmark a looping clip.** A stalled producer costs less than a
  working one, so the stall reports as a saving — this produced plausible,
  documented-looking numbers once already. Use a clip longer than the run.
- **Instrument before theorising.** Four consecutive explanations in the seek
  stall were plausible, consistent with the code, and wrong.
- **Keep the fallback working and verified.** Every one of these adds a fast path
  beside a slow one, and the slow one runs whenever a driver, a GPU or a
  configuration says no.

## Build and environment

```
cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\
        VC\Auxiliary\Build\vcvars64.bat\" -vcvars_ver=14.50 &&
        cmake --build d:\Github\CasparVP\build --target casparcg"
```

**BuildTools**, with the `-vcvars_ver=14.50` pin — the Community install has a
different toolset and mixing them fails confusingly. Before every link, kill
leftover servers or it fails with `LNK1168`; CEF spawns the executable as helper
subprocesses that hold the file:

```
Get-Process | Where-Object {$_.Path -like "*CasparVP*casparcg.exe"} |
    Stop-Process -Force
```

A consumer must be attached (`ADD 1 NDI NAME "x"`) or the channel skips
compositing and nothing is exercised. Configs in `build/shell/`:
`smoke_amcp.config` (OpenGL, AMCP 5260), `smoke_amcp_vk.config` (Vulkan); copy
one and change the port for a second instance, which the Spout loopback needs.

Two AMCP details that cost time here:

- **`VF`, not `FILTER`, for a video filter.** `FILTER` sets the video *and* audio
  graphs, so a video-only filter reaches the audio graph and it fails to build
  with "Media type mismatch" — the producer then plays nothing and the reason is
  four lines further up the log than the failure.
- **`[HTML]` resolves a bare name against the *template* folder, without the
  extension.** Pass a filename and CEF treats it as a hostname, logs
  `ERR_NAME_NOT_RESOLVED`, and renders an empty page that still composites
  happily.

Harnesses: `d:/Github/CasparCG-TestRunner/vkdispatch/` (`isf_matrix.py`,
`spout_loop.py`, `spout_matrix.py`) and `gpudirect/` (the decode matrix). One item
per commit, before-and-after figures in the message, stage files by name — another
session may be in this repository.

The Spout loopback needs two instances, so `smoke_amcp_b.config` and
`smoke_amcp_vk_b.config` (the same configs on AMCP 5261) must exist in
`build/shell`. They are not in the repository — `build/` is ignored.
