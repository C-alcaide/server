/*
 * CUDA kernel compilation unit for CUDA-VK direct decklink strategy.
 * This file is compiled by nvcc and exposes C-linkage launcher functions
 * so the strategy (.cpp) can call them without requiring nvcc.
 */
#include "cuda_vk_v210.cuh"

// C-linkage wrappers callable from .cpp code
extern "C" {

cudaError_t cuda_vk_launch_surface_to_v210(
    cudaSurfaceObject_t surf,
    uint32_t*           d_v210,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h,
    int is_16bit,
    int use_bt2020,
    cudaStream_t stream)
{
    return launch_vk_surface_to_v210(
        surf, d_v210, src_x, src_y, dst_w, dst_h, src_w, src_h,
        is_16bit != 0, use_bt2020 != 0, stream);
}

cudaError_t cuda_vk_launch_surface_to_bgra8(
    cudaSurfaceObject_t surf,
    uint8_t*            d_bgra,
    int src_x, int src_y,
    int dst_w, int dst_h,
    int src_w, int src_h,
    cudaStream_t stream)
{
    return launch_vk_surface_to_bgra8(
        surf, d_bgra, src_x, src_y, dst_w, dst_h, src_w, src_h, stream);
}

} // extern "C"
