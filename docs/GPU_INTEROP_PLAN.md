# Closing the Vulkan interop gaps

## Why

Four producers already hand the Vulkan mixer a texture without a host round trip
— HAP by uploading compressed data, and CUDA ProRes, CUDA NotchLC, OFX and
remotewall through `CudaVkTexture`. Three do not, and each pays a full
GPU→host→GPU round trip per frame to reach the same place:

| path | on OpenGL | on Vulkan | measured cost of the gap |
|---|---|---|---|
| GPU-direct decode | works | **declines outright** | 43 % more CPU on H.264/HEVC/VP9 at four layers |
| ISF | zero-copy texture | readback | +0.230 cores per layer, ≈9.2 ms per layer per frame |
| Spout producer | zero-copy texture | readback | 7.02 → 4.43 cores measured on OpenGL; Vulkan still pays the 7.02 |
| HTML/CEF | CPU `OnPaint` | CPU `OnPaint` | not measured |

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
| VK → GL | yes, `previz_texture_bridge` | previz | `GL_EXT_memory_object_win32`, the import half is reusable |
| **GL → VK** | **no** | — | wanted by ISF and Spout |
| **D3D11 → VK** | **no** | — | wanted by GPU-direct decode and HTML |
| D3D11 → GL | yes, in `av_producer` | GPU-direct decode | `WGL_NV_DX_interop2` |

Only two are missing, and each serves more than one caller.

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

**Risks.** NV12 as two Vulkan planes needs `eG8B8R82Plane420Unorm` or two
single-plane images; check `formatProperties` before assuming. Keyed-frame
lifetime matters — the decoder pool is ~20 frames deep and holding imports too
long stalls it, which is already noted for the OpenGL path.

**Verification.** The gpu-direct matrix already exists: four codecs, software vs
GPU-direct, CPU and tick, plus a picture comparison against the software path on
a **still** source. On OpenGL that comparison is byte-identical; Vulkan must
match its own software path to the same standard before this is considered done.

---

## Item 2 — GL → Vulkan

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

1. **Item 1**, D3D11 → Vulkan. It is the only one that turns something off into
   something on, and 43 % on every hardware-decoded clip is the largest number
   on the page.
2. **Item 2a**, GL → Vulkan direct, serving ISF and Spout together. Fall back to
   2b if the driver argues.
3. **Item 3**, shared ISF context, once item 2 has settled what ISF still needs.

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
