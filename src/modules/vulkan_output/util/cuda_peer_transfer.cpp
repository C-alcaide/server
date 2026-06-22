/*
 * Copyright (c) 2026 CasparCG Contributors
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
 */

#include "cuda_peer_transfer.h"

#ifdef CASPAR_CUDA_P2P_ENABLED

#include "platform_handles.h"

#include <common/except.h>
#include <common/log.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <cstring>

namespace caspar { namespace vulkan_output {

namespace {

void cuda_check(cudaError_t err, const char* context)
{
    if (err != cudaSuccess) {
        auto msg = std::string(context) + ": " + cudaGetErrorString(err);
        CASPAR_LOG(error) << L"[cuda_peer] " << msg.c_str();
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(msg));
    }
}

// Map Vulkan physical device to CUDA device index via UUID matching.
int cuda_device_from_vulkan(vk::PhysicalDevice physical)
{
    vk::PhysicalDeviceIDProperties id_props{};
    vk::PhysicalDeviceProperties2  props2{};
    props2.pNext = &id_props;
    physical.getProperties2(&props2);

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        // Compare UUID (16 bytes)
        if (std::memcmp(prop.uuid.bytes, id_props.deviceUUID, 16) == 0)
            return i;
    }

    // Fallback: try LUID match (Windows)
#ifdef _WIN32
    if (id_props.deviceLUIDValid) {
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, i);
            // prop.luid and prop.luidDeviceNodeMask available in CUDA 11.4+
            if (std::memcmp(prop.luid, id_props.deviceLUID, 8) == 0)
                return i;
        }
    }
#endif

    auto vk_props = physical.getProperties();
    CASPAR_LOG(warning) << L"[cuda_peer] UUID match failed for \""
                        << vk_props.deviceName.data() << L"\", falling back to ordinal.";
    return -1;
}

uint32_t find_memory_type(vk::PhysicalDevice physical, uint32_t type_bits, vk::MemoryPropertyFlags props)
{
    auto mem_props = physical.getMemoryProperties();
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No suitable memory type found"));
}

} // anonymous namespace

// ─── Static: availability check ─────────────────────────────────────────────

bool cuda_peer_transfer::is_available(vk::PhysicalDevice src, vk::PhysicalDevice dst)
{
    int src_cuda = cuda_device_from_vulkan(src);
    int dst_cuda = cuda_device_from_vulkan(dst);

    if (src_cuda < 0 || dst_cuda < 0 || src_cuda == dst_cuda)
        return false;

    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, src_cuda, dst_cuda);
    if (err != cudaSuccess)
        return false;

    // Even without direct peer access, cudaMemcpyPeer works (routed through host).
    // But we only report "available" if direct P2P is possible (otherwise host_staging is equivalent).
    return can_access != 0;
}

// ─── Construction / Destruction ─────────────────────────────────────────────

cuda_peer_transfer::cuda_peer_transfer(vk::Device         src_device,
                                       vk::PhysicalDevice src_physical,
                                       vk::Device         dst_device,
                                       vk::PhysicalDevice dst_physical,
                                       uint32_t           width,
                                       uint32_t           height,
                                       vk::Format         format)
    : src_device_(src_device)
    , src_physical_(src_physical)
    , dst_device_(dst_device)
    , dst_physical_(dst_physical)
    , width_(width)
    , height_(height)
    , format_(format)
{
    // Calculate image size based on format
    uint32_t bytes_per_pixel = 4; // BGRA8
    if (format == vk::Format::eR16G16B16A16Sfloat || format == vk::Format::eR16G16B16A16Unorm)
        bytes_per_pixel = 8;
    image_size_ = static_cast<vk::DeviceSize>(width_) * height_ * bytes_per_pixel;

    src_cuda_device_ = cuda_device_from_vulkan(src_physical_);
    dst_cuda_device_ = cuda_device_from_vulkan(dst_physical_);

    if (src_cuda_device_ < 0 || dst_cuda_device_ < 0)
        CASPAR_THROW_EXCEPTION(caspar_exception()
            << msg_info("Cannot map Vulkan devices to CUDA devices for P2P transfer"));

    setup_cuda_peer_access();
    create_exportable_buffers();

    // Create a CUDA stream on the source device for the peer copy
    cudaSetDevice(src_cuda_device_);
    cuda_check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreate");

    // Create Vulkan fences for synchronization
    src_fence_ = src_device_.createFence({});
    dst_fence_ = dst_device_.createFence({});

    CASPAR_LOG(info) << L"[cuda_peer] P2P transfer initialized: CUDA device "
                     << src_cuda_device_ << L" -> " << dst_cuda_device_
                     << (nvlink_ ? L" (NVLink)" : L" (PCIe)")
                     << L" " << width_ << L"x" << height_
                     << L" (" << (image_size_ / (1024 * 1024)) << L" MB/frame)";
}

cuda_peer_transfer::~cuda_peer_transfer()
{
    if (copy_stream_) {
        cudaSetDevice(src_cuda_device_);
        cudaStreamSynchronize(copy_stream_);
        cudaStreamDestroy(copy_stream_);
    }

    // Free CUDA imported memory
    if (src_cuda_ptr_) {
        cudaSetDevice(src_cuda_device_);
        cudaFree(src_cuda_ptr_);
    }
    if (dst_cuda_ptr_) {
        cudaSetDevice(dst_cuda_device_);
        cudaFree(dst_cuda_ptr_);
    }

    // Disable peer access
    if (peer_access_) {
        cudaSetDevice(src_cuda_device_);
        cudaDeviceDisablePeerAccess(dst_cuda_device_);
        cudaSetDevice(dst_cuda_device_);
        cudaDeviceDisablePeerAccess(src_cuda_device_);
    }

    // Destroy Vulkan resources
    if (src_fence_) src_device_.destroyFence(src_fence_);
    if (dst_fence_) dst_device_.destroyFence(dst_fence_);

    if (src_buffer_) src_device_.destroyBuffer(src_buffer_);
    if (src_memory_) src_device_.freeMemory(src_memory_);
    if (dst_buffer_) dst_device_.destroyBuffer(dst_buffer_);
    if (dst_memory_) dst_device_.freeMemory(dst_memory_);
    if (dst_landing_image_) dst_device_.destroyImage(dst_landing_image_);
    if (dst_image_memory_) dst_device_.freeMemory(dst_image_memory_);
}

// ─── Transfer execution ─────────────────────────────────────────────────────

vk::Image cuda_peer_transfer::transfer(vk::Image        src_image,
                                       vk::CommandBuffer src_cmd,
                                       vk::Queue         src_queue,
                                       vk::CommandBuffer dst_cmd,
                                       vk::Queue         dst_queue)
{
    // Phase 1: VkImage → src_buffer_ (on GPU A, Vulkan)
    src_cmd.reset();
    src_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    region.imageOffset       = vk::Offset3D{0, 0, 0};
    region.imageExtent       = vk::Extent3D{width_, height_, 1};

    src_cmd.copyImageToBuffer(src_image, vk::ImageLayout::eTransferSrcOptimal, src_buffer_, region);
    src_cmd.end();

    src_device_.resetFences(src_fence_);
    vk::SubmitInfo src_submit{};
    src_submit.commandBufferCount = 1;
    src_submit.pCommandBuffers    = &src_cmd;
    src_queue.submit(src_submit, src_fence_);
    (void)src_device_.waitForFences(src_fence_, VK_TRUE, UINT64_MAX);

    // Phase 2: cudaMemcpyPeerAsync (GPU A → GPU B via NVLink/PCIe DMA)
    cudaSetDevice(src_cuda_device_);
    cuda_check(cudaMemcpyPeerAsync(dst_cuda_ptr_, dst_cuda_device_,
                                   src_cuda_ptr_, src_cuda_device_,
                                   static_cast<size_t>(image_size_), copy_stream_),
               "cudaMemcpyPeerAsync");
    cuda_check(cudaStreamSynchronize(copy_stream_), "cudaStreamSynchronize");

    // Phase 3: dst_buffer_ → dst_landing_image_ (on GPU B, Vulkan)
    dst_cmd.reset();
    dst_cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Transition landing image: undefined → transfer dst
    vk::ImageMemoryBarrier bar_to_dst{};
    bar_to_dst.dstAccessMask    = vk::AccessFlagBits::eTransferWrite;
    bar_to_dst.oldLayout        = vk::ImageLayout::eUndefined;
    bar_to_dst.newLayout        = vk::ImageLayout::eTransferDstOptimal;
    bar_to_dst.image            = dst_landing_image_;
    bar_to_dst.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    dst_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_to_dst);

    dst_cmd.copyBufferToImage(dst_buffer_, dst_landing_image_,
                              vk::ImageLayout::eTransferDstOptimal, region);

    // Transition landing image: transfer dst → transfer src (ready for blit to swapchain)
    vk::ImageMemoryBarrier bar_to_src{};
    bar_to_src.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
    bar_to_src.dstAccessMask    = vk::AccessFlagBits::eTransferRead;
    bar_to_src.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
    bar_to_src.newLayout        = vk::ImageLayout::eTransferSrcOptimal;
    bar_to_src.image            = dst_landing_image_;
    bar_to_src.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    dst_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {}, nullptr, nullptr, bar_to_src);

    dst_cmd.end();

    dst_device_.resetFences(dst_fence_);
    vk::SubmitInfo dst_submit{};
    dst_submit.commandBufferCount = 1;
    dst_submit.pCommandBuffers    = &dst_cmd;
    dst_queue.submit(dst_submit, dst_fence_);
    (void)dst_device_.waitForFences(dst_fence_, VK_TRUE, UINT64_MAX);

    return dst_landing_image_;
}

// ─── Private: CUDA peer access ──────────────────────────────────────────────

void cuda_peer_transfer::setup_cuda_peer_access()
{
    int can_access_ab = 0, can_access_ba = 0;
    cudaDeviceCanAccessPeer(&can_access_ab, src_cuda_device_, dst_cuda_device_);
    cudaDeviceCanAccessPeer(&can_access_ba, dst_cuda_device_, src_cuda_device_);

    if (can_access_ab) {
        cudaSetDevice(src_cuda_device_);
        cudaError_t err = cudaDeviceEnablePeerAccess(dst_cuda_device_, 0);
        if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled)
            peer_access_ = true;
    }
    if (can_access_ba) {
        cudaSetDevice(dst_cuda_device_);
        cudaError_t err = cudaDeviceEnablePeerAccess(src_cuda_device_, 0);
        (void)err; // Best-effort reverse access
    }

    // Check if this is NVLink (high perf rank indicates direct interconnect)
    if (peer_access_) {
        int perf_rank = 0;
        cudaDeviceGetP2PAttribute(&perf_rank, cudaDevP2PAttrPerformanceRank,
                                  src_cuda_device_, dst_cuda_device_);
        // NVLink typically reports perf_rank > 0; PCIe reports 0
        nvlink_ = (perf_rank > 0);

        int native_atomic = 0;
        cudaDeviceGetP2PAttribute(&native_atomic, cudaDevP2PAttrNativeAtomicSupported,
                                  src_cuda_device_, dst_cuda_device_);
        if (native_atomic)
            nvlink_ = true; // Native atomics only available over NVLink
    }
}

// ─── Private: Create exportable Vulkan buffers ──────────────────────────────

void cuda_peer_transfer::create_exportable_buffers()
{
    // Both buffers need VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT (or _FD on Linux)
    // so CUDA can import them via cudaImportExternalMemory.

    auto handle_type = platform::kExternalMemoryHandleType;

    // Source buffer (device-local, exportable) on GPU A
    {
        vk::ExternalMemoryBufferCreateInfo ext_info{};
        ext_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits(handle_type);

        vk::BufferCreateInfo ci{};
        ci.pNext = &ext_info;
        ci.size  = image_size_;
        ci.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
        src_buffer_ = src_device_.createBuffer(ci);

        auto reqs = src_device_.getBufferMemoryRequirements(src_buffer_);

        vk::ExportMemoryAllocateInfo export_info{};
        export_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits(handle_type);

        vk::MemoryAllocateInfo alloc{};
        alloc.pNext           = &export_info;
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(src_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        src_memory_ = src_device_.allocateMemory(alloc);
        src_device_.bindBufferMemory(src_buffer_, src_memory_, 0);

        // Export to OS handle, then import into CUDA
        platform::native_handle_t src_handle = platform::kInvalidHandle;
#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR get_handle_info{};
        get_handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        get_handle_info.memory     = src_memory_;
        get_handle_info.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handle_type);
        auto fn_get_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(src_device_, "vkGetMemoryWin32HandleKHR"));
        if (!fn_get_handle)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryWin32HandleKHR not available"));
        fn_get_handle(src_device_, &get_handle_info, &src_handle);
#else
        VkMemoryGetFdInfoKHR get_fd_info{};
        get_fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        get_fd_info.memory     = src_memory_;
        get_fd_info.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handle_type);
        auto fn_get_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(src_device_, "vkGetMemoryFdKHR"));
        if (!fn_get_fd)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryFdKHR not available"));
        fn_get_fd(src_device_, &get_fd_info, &src_handle);
#endif

        // Import into CUDA
        cudaExternalMemoryHandleDesc cuda_desc{};
#ifdef _WIN32
        cuda_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        cuda_desc.handle.win32.handle = src_handle;
#else
        cuda_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
        cuda_desc.handle.fd = src_handle;
#endif
        cuda_desc.size = static_cast<unsigned long long>(reqs.size);

        cudaExternalMemory_t cuda_ext_mem = nullptr;
        cudaSetDevice(src_cuda_device_);
        cuda_check(cudaImportExternalMemory(&cuda_ext_mem, &cuda_desc), "cudaImportExternalMemory (src)");

        cudaExternalMemoryBufferDesc buf_desc{};
        buf_desc.offset = 0;
        buf_desc.size   = static_cast<unsigned long long>(image_size_);
        cuda_check(cudaExternalMemoryGetMappedBuffer(&src_cuda_ptr_, cuda_ext_mem, &buf_desc),
                   "cudaExternalMemoryGetMappedBuffer (src)");

        // Note: On Linux, fd is consumed by import (no close needed).
        // On Windows, we don't need to keep the handle — CUDA holds a reference.
#ifdef _WIN32
        platform::close_handle(src_handle);
#endif
    }

    // Destination buffer (device-local, exportable) on GPU B
    {
        vk::ExternalMemoryBufferCreateInfo ext_info{};
        ext_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits(handle_type);

        vk::BufferCreateInfo ci{};
        ci.pNext = &ext_info;
        ci.size  = image_size_;
        ci.usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        dst_buffer_ = dst_device_.createBuffer(ci);

        auto reqs = dst_device_.getBufferMemoryRequirements(dst_buffer_);

        vk::ExportMemoryAllocateInfo export_info{};
        export_info.handleTypes = vk::ExternalMemoryHandleTypeFlagBits(handle_type);

        vk::MemoryAllocateInfo alloc{};
        alloc.pNext           = &export_info;
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(dst_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        dst_memory_ = dst_device_.allocateMemory(alloc);
        dst_device_.bindBufferMemory(dst_buffer_, dst_memory_, 0);

        // Export and import into CUDA
        platform::native_handle_t dst_handle = platform::kInvalidHandle;
#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR get_handle_info2{};
        get_handle_info2.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        get_handle_info2.memory     = dst_memory_;
        get_handle_info2.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handle_type);
        auto fn_get_handle2 = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(dst_device_, "vkGetMemoryWin32HandleKHR"));
        if (!fn_get_handle2)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryWin32HandleKHR not available (dst)"));
        fn_get_handle2(dst_device_, &get_handle_info2, &dst_handle);
#else
        VkMemoryGetFdInfoKHR get_fd_info2{};
        get_fd_info2.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        get_fd_info2.memory     = dst_memory_;
        get_fd_info2.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handle_type);
        auto fn_get_fd2 = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(dst_device_, "vkGetMemoryFdKHR"));
        if (!fn_get_fd2)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryFdKHR not available (dst)"));
        fn_get_fd2(dst_device_, &get_fd_info2, &dst_handle);
#endif

        cudaExternalMemoryHandleDesc cuda_desc{};
#ifdef _WIN32
        cuda_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        cuda_desc.handle.win32.handle = dst_handle;
#else
        cuda_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
        cuda_desc.handle.fd = dst_handle;
#endif
        cuda_desc.size = static_cast<unsigned long long>(reqs.size);

        cudaExternalMemory_t cuda_ext_mem = nullptr;
        cudaSetDevice(dst_cuda_device_);
        cuda_check(cudaImportExternalMemory(&cuda_ext_mem, &cuda_desc), "cudaImportExternalMemory (dst)");

        cudaExternalMemoryBufferDesc buf_desc{};
        buf_desc.offset = 0;
        buf_desc.size   = static_cast<unsigned long long>(image_size_);
        cuda_check(cudaExternalMemoryGetMappedBuffer(&dst_cuda_ptr_, cuda_ext_mem, &buf_desc),
                   "cudaExternalMemoryGetMappedBuffer (dst)");

#ifdef _WIN32
        platform::close_handle(dst_handle);
#endif
    }

    // Destination landing image (receives data from dst_buffer_)
    {
        vk::ImageCreateInfo ci{};
        ci.imageType   = vk::ImageType::e2D;
        ci.format      = format_;
        ci.extent      = vk::Extent3D{width_, height_, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = 1;
        ci.samples     = vk::SampleCountFlagBits::e1;
        ci.tiling      = vk::ImageTiling::eOptimal;
        ci.usage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
        ci.sharingMode = vk::SharingMode::eExclusive;
        dst_landing_image_ = dst_device_.createImage(ci);

        auto reqs = dst_device_.getImageMemoryRequirements(dst_landing_image_);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = reqs.size;
        alloc.memoryTypeIndex = find_memory_type(dst_physical_, reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        dst_image_memory_ = dst_device_.allocateMemory(alloc);
        dst_device_.bindImageMemory(dst_landing_image_, dst_image_memory_, 0);
    }
}

}} // namespace caspar::vulkan_output

#endif // CASPAR_CUDA_P2P_ENABLED
