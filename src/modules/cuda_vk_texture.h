/*
 * Copyright (c) 2025 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 *
 * CUDA-Vulkan interop texture for zero-copy GPU decode into VK textures.
 * Replaces CudaGLTexture when the Vulkan accelerator backend is active.
 *
 * Usage:
 *   // On VK device thread:
 *   auto vk_tex = vk_device->create_exportable_texture(w, h, 4, bit_depth::bit16);
 *
 *   // On CUDA thread:
 *   CudaVkTexture cvt(vk_tex, vk_device->getVkDevice());
 *   cudaArray_t arr = cvt.map();
 *   cudaMemcpy2DToArrayAsync(arr, 0, 0, d_bgra16, pitch, w*8, h,
 *                            cudaMemcpyDeviceToDevice, stream);
 *   cvt.unmap();
 *   cudaStreamSynchronize(stream);
 */
#pragma once

#ifdef ENABLE_VULKAN

#include <accelerator/vulkan/util/texture.h>
#include <accelerator/vulkan/util/texture_wrapper.h>

#include <common/log.h>

#include <cuda_runtime.h>

#include <vulkan/vulkan.hpp>

#ifdef WIN32
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#endif

#include <memory>
#include <stdexcept>
#include <string>

namespace caspar {

inline void cuda_vk_check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::string msg = std::string(what) + ": " + cudaGetErrorString(e);
        CASPAR_LOG(error) << L"[cuda_vk_texture] " << msg.c_str();
        throw std::runtime_error(msg);
    }
}

class CudaVkTexture
{
  public:
    /// Construct from a VK texture that was created with create_exportable_texture().
    /// vk_device is needed to call vkGetMemoryWin32HandleKHR.
    explicit CudaVkTexture(std::shared_ptr<accelerator::vulkan::texture> vk_tex, VkDevice vk_device)
        : vk_tex_(std::move(vk_tex))
        , ext_mem_(nullptr)
        , mipmap_(nullptr)
        , array_(nullptr)
    {
        // 1. Get the Win32 handle for the VK memory
        VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
        getHandleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        getHandleInfo.memory     = vk_tex_->memory();
        getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        auto vkGetMemoryWin32HandleKHR_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(vk_device, "vkGetMemoryWin32HandleKHR"));
        if (!vkGetMemoryWin32HandleKHR_)
            throw std::runtime_error("vkGetMemoryWin32HandleKHR not available");

        HANDLE handle = nullptr;
        if (vkGetMemoryWin32HandleKHR_(vk_device, &getHandleInfo, &handle) != VK_SUCCESS)
            throw std::runtime_error("vkGetMemoryWin32HandleKHR failed");

        // 2. Import external memory into CUDA
        cudaExternalMemoryHandleDesc extMemDesc{};
        extMemDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        extMemDesc.handle.win32.handle = handle;
        extMemDesc.size                = vk_tex_->alloc_size();
        extMemDesc.flags               = 0;

        cuda_vk_check(cudaImportExternalMemory(&ext_mem_, &extMemDesc), "cudaImportExternalMemory");

        // Close the Win32 handle — CUDA has imported the memory and no longer
        // needs the handle.  Per the Vulkan spec, each call to
        // vkGetMemoryWin32HandleKHR returns a new handle the caller must close.
        CloseHandle(handle);

        // 3. Map as mipmapped array
        cudaExternalMemoryMipmappedArrayDesc mipmapDesc{};
        mipmapDesc.offset = 0;
        mipmapDesc.formatDesc =
            make_channel_desc(vk_tex_->stride(), vk_tex_->depth() == common::bit_depth::bit16);
        mipmapDesc.extent.width  = static_cast<unsigned int>(vk_tex_->width());
        mipmapDesc.extent.height = static_cast<unsigned int>(vk_tex_->height());
        mipmapDesc.extent.depth  = 0; // 2D
        mipmapDesc.numLevels     = 1;
        mipmapDesc.flags         = cudaArrayDefault;

        cuda_vk_check(cudaExternalMemoryGetMappedMipmappedArray(&mipmap_, ext_mem_, &mipmapDesc),
                      "cudaExternalMemoryGetMappedMipmappedArray");

        // 4. Get level 0 array
        cuda_vk_check(cudaGetMipmappedArrayLevel(&array_, mipmap_, 0), "cudaGetMipmappedArrayLevel");
    }

    ~CudaVkTexture()
    {
        if (mipmap_)
            cudaFreeMipmappedArray(mipmap_);
        if (ext_mem_)
            cudaDestroyExternalMemory(ext_mem_);
    }

    CudaVkTexture(const CudaVkTexture&)            = delete;
    CudaVkTexture& operator=(const CudaVkTexture&) = delete;

    CudaVkTexture(CudaVkTexture&& o) noexcept
        : vk_tex_(std::move(o.vk_tex_))
        , ext_mem_(o.ext_mem_)
        , mipmap_(o.mipmap_)
        , array_(o.array_)
    {
        o.ext_mem_ = nullptr;
        o.mipmap_  = nullptr;
        o.array_   = nullptr;
    }

    /// Get the CUDA array for writing. No map/unmap needed — the array is
    /// persistently mapped after construction (unlike the GL interop path).
    cudaArray_t array() const { return array_; }

    /// Return the underlying VK texture wrapped as core::texture for frame creation.
    std::shared_ptr<accelerator::vulkan::texture> vk_texture() const { return vk_tex_; }

    /// Check if the underlying VK texture has no external references.
    /// True when only the CudaVkTexture itself holds the shared_ptr (use_count == 1),
    /// meaning no draw_frame / texture_wrapper is still reading from it.
    bool is_free() const { return vk_tex_.use_count() == 1; }

    /// Return a core::texture wrapper for embedding in const_frame.
    /// The returned wrapper supports on-demand GPU→CPU readback via read_pixels().
    std::shared_ptr<core::texture> core_texture() const;

  private:
    static cudaChannelFormatDesc make_channel_desc(int stride, bool is_16bit)
    {
        int bits = is_16bit ? 16 : 8;
        switch (stride) {
            case 1:
                return cudaCreateChannelDesc(bits, 0, 0, 0, is_16bit ? cudaChannelFormatKindUnsigned
                                                                     : cudaChannelFormatKindUnsigned);
            case 2:
                return cudaCreateChannelDesc(bits, bits, 0, 0, cudaChannelFormatKindUnsigned);
            case 3:
                return cudaCreateChannelDesc(bits, bits, bits, 0, cudaChannelFormatKindUnsigned);
            case 4:
            default:
                return cudaCreateChannelDesc(bits, bits, bits, bits, cudaChannelFormatKindUnsigned);
        }
    }

    std::shared_ptr<accelerator::vulkan::texture> vk_tex_;
    cudaExternalMemory_t                          ext_mem_;
    cudaMipmappedArray_t                          mipmap_;
    cudaArray_t                                   array_;
};

/// A texture_wrapper subclass that supports on-demand GPU→CPU pixel readback
/// via cudaMemcpy2DFromArray from the persistent CUDA array.
/// Zero cost during normal playback — read_pixels() is only called by consumers
/// that explicitly need CPU data (e.g. PRINT RAW / write_frame_png).
class CudaVkReadableWrapper final : public accelerator::vulkan::texture_wrapper
{
  public:
    CudaVkReadableWrapper(std::shared_ptr<accelerator::vulkan::texture> vk_tex,
                          cudaArray_t cuda_array, int width, int height, int stride, bool is_16bit)
        : texture_wrapper(std::move(vk_tex))
        , cuda_array_(cuda_array)
        , width_(width)
        , height_(height)
        , stride_(stride)
        , is_16bit_(is_16bit)
    {
    }

    std::vector<std::uint8_t> read_pixels() const override
    {
        if (!cuda_array_ || width_ <= 0 || height_ <= 0)
            return {};

        const size_t bpp       = static_cast<size_t>(stride_) * (is_16bit_ ? 2 : 1);
        const size_t row_bytes = static_cast<size_t>(width_) * bpp;
        const size_t total     = row_bytes * static_cast<size_t>(height_);

        std::vector<std::uint8_t> buf(total);
        cudaError_t err = cudaMemcpy2DFromArray(
            buf.data(), row_bytes, cuda_array_, 0, 0, row_bytes, height_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            CASPAR_LOG(warning) << L"[CudaVkReadableWrapper] cudaMemcpy2DFromArray failed: "
                                << cudaGetErrorString(err);
            return {};
        }
        return buf;
    }

  private:
    cudaArray_t cuda_array_;
    int         width_;
    int         height_;
    int         stride_;
    bool        is_16bit_;
};

inline std::shared_ptr<core::texture> CudaVkTexture::core_texture() const
{
    return std::make_shared<CudaVkReadableWrapper>(
        vk_tex_, array_,
        vk_tex_->width(), vk_tex_->height(), vk_tex_->stride(),
        vk_tex_->depth() == common::bit_depth::bit16);
}

} // namespace caspar

#endif // ENABLE_VULKAN
