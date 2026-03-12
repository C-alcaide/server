// cuda_bgra_to_v210.cuh
// Convert BGRA8 (CasparCG mixer output, full-range) to V210 packed YCbCr 4:2:2 10-bit.
//
// Color space: BT.709 full-range RGB → limited-range studio-swing YCbCr
//   Y:  [64 .. 940]  (10-bit, maps to [0..255] luma)
//   Cb: [64 .. 960]  (10-bit, mid-point 512)
//   Cr: [64 .. 960]  (10-bit, mid-point 512)
//
// V210 word layout (each group = 6 horizontal pixels = 4 × uint32_t = 16 bytes):
//   Word 0: bits[9:0]=Cb0, bits[19:10]=Y0,  bits[29:20]=Cr0
//   Word 1: bits[9:0]=Y1,  bits[19:10]=Cb1, bits[29:20]=Y2
//   Word 2: bits[9:0]=Cr1, bits[19:10]=Y3,  bits[29:20]=Cb2
//   Word 3: bits[9:0]=Y4,  bits[19:10]=Cr2, bits[29:20]=Y5
//
// Thread grid: 1 thread per 6-pixel group.
//   gridDim.x  = ceil(width  / 6)
//   gridDim.y  = height
//   blockDim.x = 32 (or tuned)
//   blockDim.y = 1
//
// Input:  d_bgra  — BGRA8 packed, row-major, stride = width * 4 bytes
//                   (CasparCG produces BGRA8 output from its OpenGL compositor)
// Output: d_v210  — V210 packed, row stride = ceil(width/6) * 16 bytes
//                   (matches DeckLink bmdFormat10BitYUV row stride convention)
//
// Reference: SMPTE ST 274 (V210), Apple ProRes White Paper,
//            BT.709 colour matrix coefficients.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// BT.709 full-range RGB [0..255] → 10-bit limited-range YCbCr
//
// Derived from:
//   KR = 0.2126,  KG = 0.7152,  KB = 0.0722  (BT.709)
//
// Using integer fixed-point (×2^17 = 131072) to avoid float on SIMD paths:
//
//   Y10  = 64  + (( 95787*R + 322497*G + 32557*B) >> 17)   [64..940]
//   Cb10 = 512 + ((-52785*R - 177620*G + 230389*B) >> 17)  [64..960]
//   Cr10 = 512 + (( 230389*R - 209206*G - 21123*B) >> 17)  [64..960]
//
// Clamping to valid ranges is applied after the matrix multiply.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void bgra8_to_ycbcr10(uint8_t R, uint8_t G, uint8_t B,
                       int &Y, int &Cb, int &Cr)
{
    int r = (int)R, g = (int)G, b = (int)B;

    Y  = 64  + (( 95787 * r + 322497 * g +  32557 * b) >> 17);
    Cb = 512 + ((-52785 * r - 177620 * g + 230389 * b) >> 17);
    Cr = 512 + (( 230389 * r - 209206 * g -  21123 * b) >> 17);

    // Clamp to valid ranges (should be in range for valid input, but be safe)
    Y  = Y  < 64 ? 64 : (Y  > 940 ? 940 : Y);
    Cb = Cb < 64 ? 64 : (Cb > 960 ? 960 : Cb);
    Cr = Cr < 64 ? 64 : (Cr > 960 ? 960 : Cr);
}

// ---------------------------------------------------------------------------
// Main conversion kernel
// ---------------------------------------------------------------------------
__global__ void k_bgra_to_v210(
    const uint8_t * __restrict__ d_bgra,  // input: BGRA8, row-major, stride=width*4
    uint32_t       * __restrict__ d_v210, // output: V210 words, stride = ceil(width/6)*4 uint32s
    int width,
    int height)
{
    // Each thread processes 6 horizontal pixels (one V210 group = 4 uint32 words)
    const int group_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int row     = blockIdx.y;

    if (row >= height) return;

    // Number of V210 groups per row (rounded up; padded pixels are black)
    const int groups_per_row = (width + 5) / 6;
    if (group_x >= groups_per_row) return;

    // Pointer to the first BGRA pixel in this group
    const int px_base = group_x * 6;
    const uint8_t *row_bgra = d_bgra + (size_t)row * width * 4;

    // Load 6 pixels (handle right edge with clamping)
    auto load_pixel = [&](int px_offset, uint8_t &B, uint8_t &G, uint8_t &R) {
        const int px = px_base + px_offset;
        if (px < width) {
            const uint8_t *p = row_bgra + px * 4;
            B = p[0]; G = p[1]; R = p[2]; // BGRA layout — alpha ignored
        } else {
            B = 0; G = 0; R = 0; // black padding
        }
    };

    uint8_t B0,G0,R0,  B1,G1,R1,  B2,G2,R2,
            B3,G3,R3,  B4,G4,R4,  B5,G5,R5;
    load_pixel(0, B0,G0,R0);
    load_pixel(1, B1,G1,R1);
    load_pixel(2, B2,G2,R2);
    load_pixel(3, B3,G3,R3);
    load_pixel(4, B4,G4,R4);
    load_pixel(5, B5,G5,R5);

    // Convert each pixel to 10-bit YCbCr
    int Y0,Cb0_raw,Cr0_raw;
    int Y1,Cb1_raw,Cr1_raw;
    int Y2,Cb2_raw,Cr2_raw;
    int Y3,Cb3_raw,Cr3_raw;
    int Y4,Cb4_raw,Cr4_raw;
    int Y5,Cb5_raw,Cr5_raw;
    bgra8_to_ycbcr10(R0,G0,B0, Y0,Cb0_raw,Cr0_raw);
    bgra8_to_ycbcr10(R1,G1,B1, Y1,Cb1_raw,Cr1_raw);
    bgra8_to_ycbcr10(R2,G2,B2, Y2,Cb2_raw,Cr2_raw);
    bgra8_to_ycbcr10(R3,G3,B3, Y3,Cb3_raw,Cr3_raw);
    bgra8_to_ycbcr10(R4,G4,B4, Y4,Cb4_raw,Cr4_raw);
    bgra8_to_ycbcr10(R5,G5,B5, Y5,Cb5_raw,Cr5_raw);

    // Horizontally subsample Cb and Cr (4:2:2): average each pair
    int Cb0 = (Cb0_raw + Cb1_raw + 1) >> 1;
    int Cr0 = (Cr0_raw + Cr1_raw + 1) >> 1;
    int Cb1 = (Cb2_raw + Cb3_raw + 1) >> 1;
    int Cr1 = (Cr2_raw + Cr3_raw + 1) >> 1;
    int Cb2 = (Cb4_raw + Cb5_raw + 1) >> 1;
    int Cr2 = (Cr4_raw + Cr5_raw + 1) >> 1;

    // Pack into V210 words (10 bits each, zero-padded bits 31:30)
    // Word 0: [9:0]=Cb0, [19:10]=Y0, [29:20]=Cr0
    // Word 1: [9:0]=Y1,  [19:10]=Cb1,[29:20]=Y2
    // Word 2: [9:0]=Cr1, [19:10]=Y3, [29:20]=Cb2
    // Word 3: [9:0]=Y4,  [19:10]=Cr2,[29:20]=Y5
    uint32_t w0 = (uint32_t)Cb0        | ((uint32_t)Y0  << 10) | ((uint32_t)Cr0 << 20);
    uint32_t w1 = (uint32_t)Y1         | ((uint32_t)Cb1 << 10) | ((uint32_t)Y2  << 20);
    uint32_t w2 = (uint32_t)Cr1        | ((uint32_t)Y3  << 10) | ((uint32_t)Cb2 << 20);
    uint32_t w3 = (uint32_t)Y4         | ((uint32_t)Cr2 << 10) | ((uint32_t)Y5  << 20);

    // Write to output (4 words per group)
    uint32_t *out = d_v210 + (size_t)row * groups_per_row * 4 + group_x * 4;
    out[0] = w0;
    out[1] = w1;
    out[2] = w2;
    out[3] = w3;
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
// Returns cudaSuccess or an error code.
// d_bgra   : BGRA8, full-range, width*height*4 bytes on device
// d_v210   : pre-allocated V210 output, ceil(width/6)*16*height bytes on device
// stream   : CUDA stream
static inline cudaError_t launch_bgra_to_v210(
    const uint8_t *d_bgra,
    uint32_t      *d_v210,
    int            width,
    int            height,
    cudaStream_t   stream)
{
    const int groups_per_row = (width + 5) / 6;
    dim3 block(32, 1);
    dim3 grid((groups_per_row + 31) / 32, height);
    k_bgra_to_v210<<<grid, block, 0, stream>>>(d_bgra, d_v210, width, height);
    return cudaGetLastError();
}
