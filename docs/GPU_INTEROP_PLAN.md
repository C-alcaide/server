# Closing the Vulkan interop gaps

## Why

Four producers already hand the Vulkan mixer a texture without a host round trip
— HAP by uploading compressed data, and CUDA ProRes, CUDA NotchLC, OFX and
remotewall through `CudaVkTexture`. Three do not, and each pays a full
GPU→host→GPU round trip per frame to reach the same place:

| path | on OpenGL | on Vulkan | measured cost of the gap |
|---|---|---|---|
| ~~GPU-direct decode~~ | works | **works** (item 1, `489b02fbc`) | was 43 % more CPU at four layers |
| ~~ISF~~ | zero-copy texture | **zero-copy texture** (item 2a, `b1f591025`) | was +0.230 cores per layer; now +0.019 |
| ~~Spout producer~~ | zero-copy texture | **zero-copy texture** (item 2a, `9193ea300`) | was 6.33 cores on Vulkan; now 4.39 |
| HTML/CEF | CPU `OnPaint` | CPU `OnPaint` | not measured |

**Items 1 and 2 are done and item 3 is closed as not worth doing.** The current
figures, the corrections to the baselines below, and the reasoning for closing
item 3 are in `GPU_INTEROP_HANDOFF_ITEMS_2_3.md`; this document is kept for the
design and the mechanism survey. HTML/CEF is the only line left, and it wants
item 1's D3D11 → Vulkan bridge rather than anything here.

The first line is the one that should drive the ordering. GPU-direct decode is
not merely slower on Vulkan, it is **unavailable** — the producer logs *"mixer
exposes no OpenGL device (Vulkan backend, or GPU affinity moved it)"* and falls
back to software decoding entirely. Anyone who chooses the Vulkan mixer today
silently gives up hardware decode.

## What the transfers actually cost

Worth settling before choosing a mechanism, because "GL → CUDA → Vulkan" sounds
like two transfers and is not.

`cudaGraphicsGLRegisterImage` and `cudaImportExternalMemory` both *alias* memory
into another API's view. Neither copies. A GL→CUDA→VK route therefore performs
one device-local copy — from the GL-backed array into the VK-backed one — and
nothing crosses the bus:

| route | copies | cost at 1080p |
|---|---|---|
| today | GPU→host, then host→GPU | ≈3.1 ms + ≈0.9 ms, from the measured 2.7 GB/s readback and 9.1 GB/s upload |
| GL→CUDA→VK | one, device-local | ≈0.02 ms at ~500 GB/s |
| GL→VK direct | none — the producer renders into Vulkan memory | — |

The CUDA hop costs about half a percent of what it removes. It is not the reason
to prefer one mechanism over the other; correctness, reuse and portability are.

## Mechanisms available

| bridge | exists? | used by | notes |
|---|---|---|---|
| CUDA ↔ VK | yes, `CudaVkTexture` | ProRes, NotchLC, OFX, remotewall | aliasing; timeline semaphores already exported from `texture_wrapper.h` |
| GL → CUDA | yes, `cuda_gl_uploader` | ffmpeg NVENC recording | `cudaGraphicsGLRegisterImage` |
| VK → GL | yes, `gl_export_bridge` | ISF, Spout, previz | `GL_EXT_memory_object_win32` |
| **GL → VK** | **no** | — | wanted by ISF and Spout |
| D3D11 → VK | yes, `d3d11_import_bridge` | GPU-direct decode | `VK_KHR_external_memory_win32` (item 1) |
| D3D11 → GL | yes, in `av_producer` | GPU-direct decode | `WGL_NV_DX_interop2` |

Only two are missing, and each serves more than one caller.

**Both have since been built.** D3D11 → VK is `d3d11_import_bridge` (item 1).
GL → VK turned out not to be the right shape at all: rather than moving a GL
render into Vulkan, Vulkan allocates and exports and GL renders into that, so
what exists is a VK → GL memory export — `gl_export_bridge`, which also replaced
`previz_texture_bridge`'s copy of the same code. The "VK → GL: yes" row above was
also wrong when written: the mechanism was there but the handle-type constant was
not a handle type, so no import had ever succeeded (`edf668551`).

---

## Item 1 — D3D11 → Vulkan

**Unlocks:** GPU-direct decode on the Vulkan mixer, and later CEF accelerated
paint. Largest payoff of the three.

A hardware-decoded frame arrives as a D3D11 texture. `av_producer` already
bridges that to OpenGL with `WGL_NV_DX_interop2`; Vulkan needs the equivalent
through `VK_KHR_external_memory_win32`, importing the D3D11 shared handle as a
`VkImage`.

1. Give the decoder's texture pool a shared handle. FFmpeg's `d3d11va` frames
   come from a pool the producer does not own, so either request
   `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` on the pool the producer creates, or
   keep the existing per-frame copy into a texture that does have one. Start
   with the copy: it is what the OpenGL path already does and it removes one
   variable.
2. Import as `VkImage` via `vk::ImportMemoryWin32HandleInfoKHR`, sharing the
   pattern in `vulkan_output/util/shared_texture_pool.cpp`.
3. Hand it over as NV12 planes, not converted RGB. The OpenGL path already does
   this — *"NV12 planes handed to the mixer, which performs the colour
   conversion (no CPU frame, no VideoProcessor)"* — and it is why GPU-direct is
   byte-identical to software there. Keep that property.
4. Relax the decline in `av_producer` so a Vulkan mixer is eligible.

**Feasibility: checked, both unknowns cleared.** Queried on the reference GPU
before any code was written, which is the only reason this item is not
speculative:

| | result |
|---|---|
| NV12 `eG8B8R82Plane420Unorm` sampled | **yes**, with midpoint chroma |
| P010 `eG10X6B10X6R10X62Plane420Unorm3Pack16` sampled | **yes** — 10-bit is open later |
| import NV12 as `eD3D11Texture` | **importable**, and works in practice |
| import NV12 as `eD3D11TextureKmt` | **importable**, and works in practice |
| ~~import NV12 as `eOpaqueWin32`~~ | query says importable, **but a real D3D11 NV12 texture does not import through it** |

The last row is a correction to this table, and the distinction matters:
`getImageFormatProperties2` answers *"could this format be imported through this
handle type"*, not *"will this resource import"*. Asked to import an actual
shared D3D11 NV12 texture, `eOpaqueWin32` fails --
`vkGetMemoryWin32HandleProperties` returns `ERROR_INITIALIZATION_FAILED` and
`vkAllocateMemory` returns `ERROR_OUT_OF_DEVICE_MEMORY`. Only the two
D3D11-specific handle types work. A capability query is not a feasibility test.

The Vulkan mixer already accepts what this produces: `core::pixel_format::nv12`
exists and the Vulkan fragment shader has the case for it, sampling PLANE0 as Y
and PLANE1 as interleaved CbCr. So the hand-over is the same shape the OpenGL
path already builds -- a two-plane desc, Y at full resolution and CbCr at half --
and no shader work is needed.

**Shape settled by measurement: two separate single-plane imports, not one
multi-planar import.**

Plane views were provable -- `ePlane0` and `ePlane1` views on one imported
multi-planar `VkImage` both create successfully. That was not the blocker. The
blocker is a layout disagreement found immediately after, isolated with a
four-case control under the validation layer:

| case | isolates | result |
|---|---|---|
| Y plane, D3D11 writes, VK reads `PLANE_0` | luma across the boundary | byte-identical |
| pure-Vulkan NV12, VK writes and reads `PLANE_1` | is Vulkan's plane handling sane? | correct |
| imported NV12, VK writes and reads `PLANE_1` | is Vulkan self-consistent on imported memory? | correct |
| imported NV12, **D3D11 writes**, VK reads `PLANE_1` | do the two APIs agree? | **844800 / 1036800 bytes wrong** |
| two separate textures, R8 + R8G8, D3D11 writes, VK reads | the alternative shape | **byte-identical** |

Rows two and three passing while row four fails is the decisive pair: Vulkan's
plane machinery is correct and Vulkan is self-consistent on the imported
allocation, so what disagrees is D3D11 and the Vulkan driver about where plane 1
sits inside a shared NV12 allocation. Luma agrees; chroma does not. The
corruption is a tiling or swizzle disagreement rather than an offset -- a base
offset would not leave 18.5 % of bytes coincidentally matching -- so there is no
arithmetic fix. Validation was clean throughout, and the alternative copy extent
is rejected by `VUID-vkCmdCopyImage-srcOffset-00144`, so no untried spelling
remains. The D3D11 side was cleared separately: a staging readback of the shared
texture after `CopySubresourceRegion` is byte-exact in both planes.

**This makes the item smaller, not larger, and the result more portable.**
`d3d11_gl_bridge::setup_planes` already creates exactly the two textures the
working case imports -- `y_texture_` R8_UNORM at full resolution and
`uv_texture_` R8G8_UNORM at half -- filled by a deliberately arithmetic-free
`Load()` pass-through shader. The Vulkan bridge reuses that extraction verbatim
and differs only in adding `D3D11_RESOURCE_MISC_SHARED |
D3D11_RESOURCE_MISC_SHARED_NTHANDLE` to those two textures and importing each
handle as an ordinary single-plane `VkImage`, where the OpenGL path calls
`wglDXRegisterObjectNV`. The two-plane hand-over is unchanged, so
byte-identical-to-software is preserved by construction. It also removes any
dependence on the two APIs agreeing about multi-planar layout, which is a
better property to have than a fallback.

**Remaining risk.** Frame lifetime: the decoder pool is ~20 frames deep and
holding imports too long stalls it, already noted for the OpenGL path. Also
unmeasured: what the cross-API synchronisation costs per frame in the server (a
`D3D11_QUERY_EVENT` wait was correct in the probes but its cost is unknown), and
whether P010/P016 behave the same -- moot for layout now the shape is two
single-plane imports, but untested.

**Verification.** The gpu-direct matrix already exists: four codecs, software vs
GPU-direct, CPU and tick, plus a picture comparison against the software path on
a **still** source. On OpenGL that comparison is byte-identical; Vulkan must
match its own software path to the same standard before this is considered done.

---

## Item 2 — GL → Vulkan

**Done, route 2a, both callers** (`b1f591025`, `9193ea300`). 2b was not needed.
Four things this section got wrong, kept because each was a real mistake:

1. The import half did not "already exist" in a working state — the handle-type
   constant was `GL_DEVICE_LUID_EXT` and no import had ever succeeded
   (`edf668551`). A mechanism nobody checks the error code of is not a mechanism.
2. `eOptimal` tiling is mandatory. GL rejects Vulkan `eLinear` memory as "memory
   object too small", so previz's Pascal `eLinear` workaround cannot use this path.
3. `eGeneral` was not needed. The mixer already handles exportable images written
   by another API — that is what the CUDA producers do — and no explicit layout
   transition was added.
4. `glFinish` was the right call, and now for a measured reason: 0.053 ms per
   frame against the ≈9.2 ms per layer it replaces. `GL_EXT_semaphore` stays
   deferred on evidence.

**Unlocks:** ISF, and the Spout producer's zero-copy receive on Vulkan.

Two routes. Prefer the direct one.

**2a, direct (preferred).** Vulkan allocates an exportable image, GL imports its
memory as a texture, and the producer renders into it. Zero copies.

- `create_exportable_texture` needs `eColorAttachment` added to its usage flags;
  it currently has `eTransferDst | eSampled | eTransferSrc`, which is enough to
  sample but not to be rendered into.
- The GL import half already exists in `previz_texture_bridge.cpp`
  (`glCreateMemoryObjectsEXT`, `glImportMemoryWin32HandleEXT`,
  `glTextureStorageMem2DEXT`) and should be lifted into something both that
  bridge and this can share, rather than copied.
- Synchronisation: `glFinish()` after rendering, then publish. Crude but
  correct, and exactly what the HAP producer does today. `GL_EXT_semaphore` is
  the better answer later; do not start there.
- Image layout: GL writes outside Vulkan's knowledge, so the image wants
  `eGeneral` rather than a transition Vulkan believes it controls.

**2b, via CUDA (fallback).** `cudaGraphicsGLRegisterImage` on the rendered
texture, `CudaVkTexture` on the Vulkan side, one device-local copy between them.
Both halves are proven in this tree. Worth taking if 2a hits a driver problem —
one device-local copy is not worth agonising over.

**Callers.**
- ISF: `render_readback` becomes render-into-the-imported-texture; the frame is
  then texture-backed exactly as the OpenGL path builds it.
- Spout producer: the `zero_copy` branch stops being conditional on
  `ogl_device_` and takes a Vulkan texture instead. `CASPAR_SPOUT_FORCE_READBACK`
  already exists to compare the two paths on one mixer.

**Verification.** Both already have harnesses. ISF: `isftest.fs` is
deterministic, so the two mixers must stay byte-identical, and the per-layer cost
should approach OpenGL's +0.000 cores. Spout: the loopback rig, four
sender/receiver combinations, currently 19.226120 dB on all four.

---

## Item 3 — Share one GL context across ISF producers

**Closed as not worth doing.** With item 2 in, per-producer contexts cost about
0.002 cores per layer — Vulkan's +0.016 against OpenGL's +0.014 at eight layers,
where OpenGL creates no such context at all. The readback was hiding the fact
that there was nothing there. Reasoning and figures in
`GPU_INTEROP_HANDOFF_ITEMS_2_3.md`.


**Unlocks:** less context switching when several ISF layers run at once.

Each ISF producer creates its own SFML context today, so four layers mean four
contexts and four `setActive` pairs per frame. One shared context, guarded by the
mutex that already serialises rendering, removes that.

Small and low risk. Do it **after** item 2 — if ISF stops needing its own GL
context on Vulkan, part of this evaporates, and doing it first risks work that
item 2 deletes.

**Verification.** The ISF measurement at one and four layers; look for the gap
between them to narrow. Also re-run the three play-and-clear cycles: the context
lifetime bug fixed in `bc357e7d1` lived in exactly this area, and sharing a
context across producers puts the same question back on the table.

---

## Ordering

1. ~~**Item 1**, D3D11 → Vulkan.~~ Done, `489b02fbc`.
2. ~~**Item 2a**, GL → Vulkan direct, serving ISF and Spout together.~~ Done,
   `b1f591025` and `9193ea300`. 2b never became necessary.
3. ~~**Item 3**, shared ISF context.~~ Closed: item 2 removed the reason for it,
   which is why it was scheduled last.

The ordering held up. Putting item 3 last is what turned it from work into a
measurement.

## Ground rules

- One item per commit, with the before/after measurement in the message. Every
  number in this document came from a harness that still exists and can be
  re-run.
- Compare on **still** sources. Three separate measurements in this area were
  wrong because two servers were on different frames of moving content and the
  difference was read as a defect.
- Read the log, not the timings, to decide which path ran. A silent fallback and
  a path with no benefit look identical in a CPU figure, and both Spout and
  GPU-direct log which path they took precisely because of this.
- Keep the fallback working and verified. Every one of these adds a faster path
  beside a slower one; the slower one is what runs when a driver, a GPU or a
  configuration says no.
