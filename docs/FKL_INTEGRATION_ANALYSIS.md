# Fused Kernel Library (FKL) — Integration Analysis for CasparCG

**Date:** 2025-05-10  
**Branch:** CasparVP  
**Status:** Research / Future consideration

---

## 1. Overview of the Libraries

### Fused Kernel Library (FKL)

- **Repository:** https://github.com/Libraries-Openly-Fused/FusedKernelLibrary
- **Paper:** https://arxiv.org/abs/2508.07071
- **License:** Apache-2.0
- **Language:** C++20 (LTS branch: C++17), CUDA
- **Maturity:** Beta-0.1.14-LTS (self-described as "<1% feature complete")

FKL is a compile-time CUDA kernel fusion framework. It defines `InstantiableOperation`
structs that encapsulate both the GPU code (as static members) and runtime parameters
(as instance data). When composed via `executeOperations<>()`, the C++ template
machinery collapses the entire operation chain into a **single CUDA kernel** at compile
time — one global memory read, one global memory write, all intermediates kept in
registers/SRAM.

Four fusion techniques are implemented:

| Technique | Description | Relevant scenario |
|---|---|---|
| **Vertical Fusion** | Chain N element-wise ops into one kernel | color convert → normalize → repack |
| **Horizontal Fusion** | Process M independent data planes in one launch via `blockIdx.z` | batch crop/resize of N detections |
| **Backwards Vertical Fusion** | Only read source pixels needed by downstream ops (e.g., crop region from full frame before color convert) | decode only the ROI, not the full 4K/8K frame |
| **Divergent Horizontal Fusion** | Run structurally different kernels in one launch, each on its own z-plane set | simultaneous encode + preview conversion |

### cvGPUSpeedup

- **Repository:** https://github.com/Libraries-Openly-Fused/cvGPUSpeedup
- **License:** Apache-2.0

A wrapper around FKL that mirrors the OpenCV-CUDA API (`cv::cuda::resize`, `cv::cuda::split`, etc.)
but returns deferred operation descriptors instead of launching kernels immediately. The user
calls `cvGS::executeOperations(stream, op1, op2, ..., opN)` to fuse and execute them all at once.

**Not directly useful** — CasparCG does not use OpenCV-CUDA. Only the raw FKL library is relevant.

---

## 2. CasparCG CUDA Pipeline Inventory

### 2.1 ProRes Decode (`src/modules/cuda_prores/cuda/`)

```
Host upload (H→D async)
  → k_prores_entropy_decode          [compute-bound, per-slice bitstream parsing]
  → k_prores_idct_dequant            [compute-bound, 8×8 DCT per Y/Cb/Cr plane]
  → k_ycbcr422p10_to_bgra16          [memory-bound, per-pixel color matrix + clamp]
  → cudaMemcpy2DToArrayAsync          [memory-bound, copy to GL cudaArray]
```

Kernel launch count per progressive frame: **4** (entropy, IDCT×1 fused, YCbCr→BGRA, memcpy).  
Kernel launch count per interlaced frame: **8** (×2 fields).

### 2.2 ProRes Encode (`src/modules/cuda_prores/cuda/`)

```
d_bgra (mixer output)
  → k_bgra_to_v210  OR  k_bgra8_to_field422p10   [memory-bound, pixel packing]
  → k_v210_unpack  OR  k_v210_unpack_field        [memory-bound, pixel unpacking]
  → k_dct_quantise                                 [compute-bound, forward DCT + quant]
  → k_prores_entropy_encode                        [compute-bound, bitstream packing]
```

For the **422 progressive** path, `k_bgra_to_v210` + `k_v210_unpack` is a round-trip
through V210 format that exists because the original DeckLink capture pipeline delivered V210.
The `k_bgra8_to_field422p10` bypass kernel already fuses BGRA→YUV conversion with field
extraction for the interlaced BGRA path, avoiding the V210 intermediate.

### 2.3 NotchLC Decode (`src/modules/cuda_notchlc/cuda/`)

```
Host upload (H→D async, LZ4-compressed payload)
  → nvcomp LZ4 decompress             [compute-bound, GPU LZ4]
  → k_notchlc_decode_y / _uv / _a     [compute-bound, entropy + dequant]
  → k_notch_ycocg_to_bgra16           [memory-bound, per-pixel YCoCg→RGB matrix]
  → cudaMemcpy2DToArrayAsync           [memory-bound, copy to GL cudaArray]
```

### 2.4 Vulkan Output (`src/modules/vulkan_output/shaders/`)

```
color_convert.comp (GLSL compute shader):
  sRGB EOTF → 3×3 gamut matrix → output OETF (sRGB/PQ/HLG/gamma)
```

This runs on the Vulkan side and is not a CUDA kernel, so FKL does not apply here.

### 2.5 CUDA↔GL Interop (`cuda_gl_texture.h`, `cuda_peer_transfer.cpp`)

All decode paths write to a `cudaArray_t` obtained from `cudaGraphicsGLRegisterImage`.
This is a `cudaMemcpy2DToArrayAsync` call — a DMA transfer, not a compute kernel.
It cannot be fused with preceding compute kernels via FKL.

---

## 3. Fusion Opportunities

### 3.1 ProRes Decode: `k_ycbcr422p10_to_bgra16` + output write

**Current state:**  
`k_ycbcr422p10_to_bgra16` reads three separate planes (`d_y`, `d_cb`, `d_cr`), applies a
3×3 color matrix with clamping, and writes interleaved `uint16_t[4]` BGRA to `d_bgra16`.
A separate `cudaMemcpy2DToArrayAsync` then copies `d_bgra16` → `d_gl_array`.

**Potential fusion:**  
If the output target were a plain `Ptr2D<ushort4>` instead of a `cudaArray_t`, we could
fuse the color conversion write with the output write, eliminating the `d_bgra16`
intermediate buffer entirely (~33 MB for 4K, ~132 MB for 8K).

**Blocker:** The GL interop target is a `cudaArray_t` with opaque layout. Writing to it
requires `cudaMemcpy2DToArray` or surface objects — FKL currently has no surface-write
operation. A custom `SurfaceWrite` FKL operation would need to be implemented.

**Estimated gain:** Modest. The memcpy is already a pure DMA transfer running at
PCIe/NVLink bandwidth. Fusing would save one kernel launch overhead (~5–10 µs) and one
full-frame read+write cycle (~0.5 ms at 4K on a mid-range GPU). For 8K or 12K this
becomes more meaningful.

### 3.2 ProRes Encode: `k_bgra_to_v210` + `k_v210_unpack` elimination

**Current state (progressive 422 from mixer BGRA):**
```
BGRA8 → [k_bgra_to_v210] → V210 → [k_v210_unpack] → YCbCr422 planar
```

This is a pointless round-trip through V210 when the source is already BGRA8 on the GPU.
A single fused kernel `BGRA8 → YCbCr422P10` would:
- Eliminate the V210 intermediate buffer
- Halve the global memory traffic
- Remove one kernel launch

**This does not require FKL** — it's a straightforward custom kernel, similar to the
existing `k_bgra8_to_field422p10` for interlaced. This should be done regardless of FKL.

### 3.3 ProRes Encode: `BGRA→YCbCr` + `DCT+Quantise` fusion

**Current state:**
```
[k_bgra8_to_yuv422p10] → d_y, d_cb, d_cr → [k_dct_quantise] → d_coeffs
```

**Potential fusion:**  
Backwards Vertical Fusion could avoid writing the YCbCr planes to DRAM. Each DCT block
thread would pull the 8×8 source pixels via the fused color conversion path, keeping
intermediates in registers.

**Complexity:** The DCT kernel operates on 8×8 blocks using shared memory for the
butterfly passes. FKL's fusion model works best for per-pixel operations; integrating
a shared-memory reduction pattern (DCT) would require a custom `MidOp` or `UnaryOp`
with `__shared__` usage that FKL may not currently support.

**Estimated gain:** Eliminates 3 plane writes + 3 plane reads (~50 MB for 1080p,
~200 MB for 4K). Likely 0.3–1.0 ms at 4K.

### 3.4 NotchLC Decode: `k_notch_ycocg_to_bgra16` + output

Same analysis as §3.1. The color conversion kernel is element-wise and memory-bound.
Fusion with a hypothetical surface write would save one intermediate buffer and one
read+write pass.

### 3.5 Batch/Multi-Output: Horizontal Fusion

If CasparCG ever needs to produce multiple output formats from the same source frame
(e.g., 4K HDR + 1080p SDR preview + proxy), Divergent Horizontal Fusion could run all
conversions in a single kernel launch, sharing the source frame read.

Currently each output is handled by a separate consumer with its own pipeline, so this
would require architectural changes beyond just kernel fusion.

---

## 4. Integration Requirements

### 4.1 Build System

FKL requires:
- CUDA SDK 11.8, 12.1–12.3, or 12.8+ (12.4–12.6 have an nvcc template deduction bug)
- C++17 minimum (LTS branch), C++20 for main branch
- CMake 3.24+

CasparCG CasparVP branch currently uses:
- CUDA 12.x (compatible)
- C++17 (compatible with FKL LTS branch)
- CMake (compatible)

FKL is header-only — integration is adding include paths and compiling affected `.cu`
files with the FKL headers visible.

### 4.2 Code Changes Required

Each kernel to be fused must be refactored into FKL's operation model:

```cpp
// Current CasparCG kernel style:
__global__ void k_ycbcr422p10_to_bgra16(
    const int16_t* d_y, const int16_t* d_cb, const int16_t* d_cr,
    uint16_t* d_bgra16, int width, int height, int color_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    // ... color matrix math, write to d_bgra16 ...
}

// FKL operation style (conceptual):
struct YCbCr422ToBGRA16 {
    using InputType  = fk::Tuple<int16_t, int16_t, int16_t>; // Y, Cb, Cr
    using OutputType = ushort4;                                // BGRA16
    struct Params { int color_matrix; };

    __device__ static OutputType exec(InputType in, Params p) {
        // ... same color matrix math ...
    }

    static auto build(int color_matrix) {
        return UnaryOp<YCbCr422ToBGRA16>::build(Params{color_matrix});
    }
};
```

The thread indexing, grid launch, and memory access patterns are abstracted away by FKL —
this is both the benefit (composability) and the cost (loss of fine-grained control over
shared memory, warp-level primitives, etc.).

### 4.3 Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| FKL API instability (main branch) | Medium | Pin to LTS branch + specific tag |
| nvcc template instantiation time increase | Low-Medium | Isolate fused kernels in dedicated `.cu` TUs |
| Debugging difficulty (deep template nesting) | Medium | Keep non-fused fallback kernels; `#ifdef` switch |
| FKL does not support shared-memory ops (DCT) | High | Only fuse element-wise stages; keep DCT as-is |
| Surface write to `cudaArray_t` not in FKL | High | Implement custom `SurfaceWrite` operation or upstream PR |

---

## 5. Recommendations

### Do Now (no FKL needed)

1. **Fuse `k_bgra_to_v210` + `k_v210_unpack` into `k_bgra8_to_yuv422p10`** for the
   progressive 422 encode path. This is pure waste today — a V210 round-trip that can
   be replaced with a single direct conversion kernel, identical in structure to the
   already-existing `k_bgra8_to_field422p10`.

### Revisit When FKL Matures

2. **Watch for `SurfaceWrite` or `cudaArray` output support** in FKL. Once available,
   fusing `ycbcr_to_bgra16` with the GL texture write becomes viable and would eliminate
   the largest intermediate buffer in the decode path.

3. **Watch for shared-memory operation support.** If FKL adds a `BlockOp` pattern that
   supports `__shared__` memory and block-level synchronization, fusing color conversion
   with the DCT stage becomes possible.

4. **Consider FKL for any new GPU pipeline** (e.g., GPU-based scaler, LUT application,
   360° reprojection) where the operation chain is purely element-wise from the start.
   Building new code on FKL from scratch is far cheaper than retrofitting existing kernels.

### Do Not Do

5. **Do not retrofit existing working CUDA kernels into FKL operations** at this time.
   The integration cost exceeds the performance benefit for the current pipeline, and
   FKL's API is still evolving.

---

## 6. References

- FKL source: https://github.com/Libraries-Openly-Fused/FusedKernelLibrary
- cvGPUSpeedup (OpenCV wrapper): https://github.com/Libraries-Openly-Fused/cvGPUSpeedup
- Paper: O. Amoros et al., "The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU Libraries," arXiv:2508.07071, 2025.
- GTC 2025 poster: https://www.nvidia.com/gtc/posters/?search=P73324
- FKL playground (Colab): https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl
