# Continuing the interop plan: items 2 and 3

Companion to `GPU_INTEROP_PLAN.md`, which holds the design and the reasoning.
This is the delta — what has changed since it was written, what item 1 taught
that applies to what is left, and the baselines to measure against.

Read the plan first. Do not re-derive it from here.

## State

**Item 1 is done.** GPU-direct hardware decode works on the Vulkan mixer
(`489b02fbc`). It measured 41 % on H.264, 38 % on HEVC and VP9, byte-identical to
OpenGL's GPU-direct output on all seven clips, with the OpenGL path proved
untouched by hashing captures against HEAD.

**Items 2 and 3 are not started.** Item 2 is GL → Vulkan, serving both ISF and
the Spout producer. Item 3 is a shared GL context across ISF producers.

**One known defect is open and separate**: GPU-direct decode does not resume
after a seek, which is why looping stalls. It has its own hand-off,
`GPUDIRECT_SEEK_STALL.md`. It does not block items 2 or 3, and the two should not
be mixed into one change.

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

## Baselines to beat, and to not break

Measured on the reference rig, on the current build.

**ISF** (`isftest.fs`, deterministic, so the mixers must stay byte-identical):

| | 1 layer | 4 layers | per extra layer |
|---|---|---|---|
| OpenGL | 1.19 cores | 1.19 | +0.000 |
| Vulkan | 1.37 | 2.06 | **+0.230** |

That +0.230 — about 9.2 ms per layer per frame — is the target. OpenGL's +0.000
is what success looks like. Picture is `inf` dB between mixers today and must
stay so.

**Spout** (loopback, two instances, still source):

| | |
|---|---|
| OpenGL receiver, readback | 7.02 cores |
| OpenGL receiver, zero-copy | 4.43 |
| picture, all four sender/receiver combinations | 19.226120 dB |

Vulkan receivers still pay the 7.02. All four combinations currently agree to six
decimals; that must survive.

## Item 3, and why it is last

Each ISF producer creates its own SFML context, so four layers mean four contexts
and four `setActive` pairs per frame. Sharing one, guarded by the mutex that
already serialises rendering, removes that.

It is deliberately last because item 2 may delete most of it — if ISF stops
rendering on its own GL context under Vulkan, there is less to share. Doing it
first risks work that item 2 throws away.

When it is done, re-run the three play-and-clear cycles as well as the CPU
figures. The context-lifetime crash fixed in `bc357e7d1` lived in exactly this
area, and sharing a context between producers puts the same question — who owns
it, and on which thread is it destroyed — straight back on the table.

## Ground rules

Each of these is a mistake already made in this work, not general advice.

- **Compare on still sources.** Three separate measurements here were wrong
  because two servers sat on different frames of moving content and the
  difference was read as a defect.
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

Harnesses: `d:/Github/CasparCG-TestRunner/vkdispatch/` (ISF, Spout loopback) and
`gpudirect/` (the decode matrix). One item per commit, before-and-after figures
in the message, stage files by name — another session may be in this repository.
