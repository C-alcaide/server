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

#include "previz_texture_bridge.h"

#include "../util/device.h"
#include "../../ogl/image/previz_scene.h"
#include "../../ogl/util/device.h"

#include <common/except.h>
#include <common/log.h>

#include <vulkan/vulkan.hpp>

#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <EGL/egl.h>
#endif

#include <algorithm>
#include <cstring>

namespace caspar { namespace accelerator { namespace vulkan {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult result_ = (call);                                                                                     \
        if (result_ != VK_SUCCESS) {                                                                                   \
            CASPAR_THROW_EXCEPTION(caspar_exception()                                                                  \
                                   << msg_info("VK call failed: " #call " = " + std::to_string(result_)));             \
        }                                                                                                              \
    } while (0)

// GL_EXT_memory_object function pointers
PFNGLCREATEMEMORYOBJECTSEXTPROC        glCreateMemoryObjectsEXT_       = nullptr;
PFNGLDELETEMEMORYOBJECTSEXTPROC        glDeleteMemoryObjectsEXT_       = nullptr;
PFNGLTEXTURESTORAGEMEM2DEXTPROC        glTextureStorageMem2DEXT_       = nullptr;

#ifdef _WIN32
PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC    glImportMemoryWin32HandleEXT_   = nullptr;
#else
using PFNGLIMPORTMEMORYFDEXTPROC = void (*)(GLuint, GLuint64, GLenum, GLint);
PFNGLIMPORTMEMORYFDEXTPROC             glImportMemoryFdEXT_            = nullptr;
#endif

std::once_flag gl_ext_flag;
bool           gl_ext_loaded = false;

void try_load_gl_extensions()
{
    std::call_once(gl_ext_flag, [] {
#ifdef _WIN32
        glCreateMemoryObjectsEXT_      = (PFNGLCREATEMEMORYOBJECTSEXTPROC)wglGetProcAddress("glCreateMemoryObjectsEXT");
        glDeleteMemoryObjectsEXT_      = (PFNGLDELETEMEMORYOBJECTSEXTPROC)wglGetProcAddress("glDeleteMemoryObjectsEXT");
        glImportMemoryWin32HandleEXT_  = (PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC)wglGetProcAddress("glImportMemoryWin32HandleEXT");
        glTextureStorageMem2DEXT_      = (PFNGLTEXTURESTORAGEMEM2DEXTPROC)wglGetProcAddress("glTextureStorageMem2DEXT");

        gl_ext_loaded = glCreateMemoryObjectsEXT_ && glImportMemoryWin32HandleEXT_ &&
                        glTextureStorageMem2DEXT_;
#else
        glCreateMemoryObjectsEXT_      = (PFNGLCREATEMEMORYOBJECTSEXTPROC)eglGetProcAddress("glCreateMemoryObjectsEXT");
        glDeleteMemoryObjectsEXT_      = (PFNGLDELETEMEMORYOBJECTSEXTPROC)eglGetProcAddress("glDeleteMemoryObjectsEXT");
        glImportMemoryFdEXT_           = (PFNGLIMPORTMEMORYFDEXTPROC)eglGetProcAddress("glImportMemoryFdEXT");
        glTextureStorageMem2DEXT_      = (PFNGLTEXTURESTORAGEMEM2DEXTPROC)eglGetProcAddress("glTextureStorageMem2DEXT");

        gl_ext_loaded = glCreateMemoryObjectsEXT_ && glImportMemoryFdEXT_ &&
                        glTextureStorageMem2DEXT_;
#endif

        if (!gl_ext_loaded) {
            CASPAR_LOG(warning) << L"[previz_bridge] GL_EXT_memory_object not available - "
                                   L"falling back to CPU readback for previz channel textures.";
        }
    });
}

} // anonymous namespace

// ── channel_slot ─────────────────────────────────────────────────────────────

struct previz_texture_bridge::channel_slot
{
    int width  = 0;
    int height = 0;
    bool use_16bit = false;

    // VK side
    VkImage        vk_image      = VK_NULL_HANDLE;
    VkDeviceMemory vk_memory     = VK_NULL_HANDLE;
    platform::native_handle_t memory_handle = platform::kInvalidHandle;
    VkDeviceSize   memory_size      = 0;

    // GL side (interop)
    GLuint gl_memory_object = 0;
    GLuint gl_texture       = 0;

    // CPU fallback
    std::vector<uint8_t> cpu_buffer;
    GLuint               cpu_gl_texture = 0;
    bool                 cpu_dirty      = false;

    bool ready = false; // true after first post_channel
};

// ── Constructor / Destructor ─────────────────────────────────────────────────

previz_texture_bridge::previz_texture_bridge(const spl::shared_ptr<device>&      vk_device,
                                             const std::shared_ptr<ogl::device>& ogl_device)
    : vk_device_(vk_device)
    , ogl_device_(ogl_device)
{
    auto vk_dev = static_cast<VkDevice>(vk_device_->getVkDevice());

    // Load VK extension function pointers
#ifdef _WIN32
    vkGetMemoryHandleKHR_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(vk_dev, "vkGetMemoryWin32HandleKHR"));
#else
    vkGetMemoryHandleKHR_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(vk_dev, "vkGetMemoryFdKHR"));
#endif

    // Detect Pascal GPU for LINEAR tiling workaround
    vk_device_->dispatch_sync([this] {
        VkPhysicalDeviceProperties vk_props;
        vkGetPhysicalDeviceProperties(
            static_cast<VkPhysicalDevice>(vk_device_->getVkPhysicalDevice()), &vk_props);
        if (vk_props.vendorID == 0x10DE) { // NVIDIA
            uint32_t major = VK_API_VERSION_MAJOR(vk_props.apiVersion);
            uint32_t minor = VK_API_VERSION_MINOR(vk_props.apiVersion);
            if (major == 1 && minor <= 1) {
                use_linear_tiling_ = true;
                CASPAR_LOG(info) << L"[previz_bridge] Pascal GPU detected - using LINEAR tiling.";
            }
        }
    });

    // Load GL extensions on the OGL thread
    ogl_device_->dispatch_sync([this] {
        try_load_gl_extensions();
        interop_available_ = gl_ext_loaded;
    });

    CASPAR_LOG(info) << L"[previz_bridge] Initialized ("
                     << (interop_available_ ? L"zero-copy interop" : L"CPU fallback") << L").";
}

previz_texture_bridge::~previz_texture_bridge()
{
    // Destroy GL resources on the OGL thread
    if (ogl_device_) {
        ogl_device_->dispatch_sync([this] {
            for (auto& [id, s] : slots_) {
                if (s.gl_texture)
                    glDeleteTextures(1, &s.gl_texture);
                if (s.gl_memory_object && glDeleteMemoryObjectsEXT_)
                    glDeleteMemoryObjectsEXT_(1, &s.gl_memory_object);
                if (s.cpu_gl_texture)
                    glDeleteTextures(1, &s.cpu_gl_texture);
            }
        });
    }

    // Destroy VK resources on the VK thread to avoid racing with in-flight work
    vk_device_->dispatch_sync([this] {
        auto vk_dev = static_cast<VkDevice>(vk_device_->getVkDevice());
        for (auto& [id, s] : slots_) {
            if (s.vk_image != VK_NULL_HANDLE)
                vkDestroyImage(vk_dev, s.vk_image, nullptr);
            if (s.vk_memory != VK_NULL_HANDLE)
                vkFreeMemory(vk_dev, s.vk_memory, nullptr);
            platform::close_handle(s.memory_handle);
        }
    });
}

// ── Slot management ──────────────────────────────────────────────────────────

void previz_texture_bridge::create_slot(channel_slot& s, int width, int height, bool use_16bit)
{
    s.width    = width;
    s.height   = height;
    s.use_16bit = use_16bit;

    auto vk_dev  = static_cast<VkDevice>(vk_device_->getVkDevice());
    auto vk_phys = static_cast<VkPhysicalDevice>(vk_device_->getVkPhysicalDevice());

    VkFormat format = use_16bit ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;

    // ── VK: Create exportable image ──────────────────────────────────────

    VkExternalMemoryImageCreateInfo ext_mem_img{};
    ext_mem_img.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_mem_img.handleTypes = platform::kExternalMemoryHandleType;

    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext         = &ext_mem_img;
    img_info.imageType     = VK_IMAGE_TYPE_2D;
    img_info.format        = format;
    img_info.extent        = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    img_info.mipLevels     = 1;
    img_info.arrayLayers   = 1;
    img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling        = use_linear_tiling_ ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT; // blit target from channel output
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(vk_dev, &img_info, nullptr, &s.vk_image));

    // ── VK: Allocate exportable memory ───────────────────────────────────

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(vk_dev, s.vk_image, &mem_reqs);
    s.memory_size = mem_reqs.size;

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = platform::kExternalMemoryHandleType;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk_phys, &mem_props);

    uint32_t mem_type_index = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type_index = i;
            break;
        }
    }
    if (mem_type_index == UINT32_MAX)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No suitable device-local memory type for previz bridge"));

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext           = &export_info;
    alloc_info.allocationSize  = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_index;

    VK_CHECK(vkAllocateMemory(vk_dev, &alloc_info, nullptr, &s.vk_memory));
    VK_CHECK(vkBindImageMemory(vk_dev, s.vk_image, s.vk_memory, 0));

    // Export memory handle
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR get_handle{};
    get_handle.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    get_handle.memory     = s.vk_memory;
    get_handle.handleType = platform::kExternalMemoryHandleType;

    if (!vkGetMemoryHandleKHR_)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryWin32HandleKHR not loaded"));
    VK_CHECK(vkGetMemoryHandleKHR_(vk_dev, &get_handle, &s.memory_handle));
#else
    VkMemoryGetFdInfoKHR get_handle{};
    get_handle.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_handle.memory     = s.vk_memory;
    get_handle.handleType = platform::kExternalMemoryHandleType;

    if (!vkGetMemoryHandleKHR_)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetMemoryFdKHR not loaded"));
    VK_CHECK(vkGetMemoryHandleKHR_(vk_dev, &get_handle, &s.memory_handle));
#endif

    // ── GL: Import memory + create texture ───────────────────────────────

    if (interop_available_) {
        glCreateMemoryObjectsEXT_(1, &s.gl_memory_object);
#ifdef _WIN32
        glImportMemoryWin32HandleEXT_(s.gl_memory_object,
                                      mem_reqs.size,
                                      platform::kGlHandleType,
                                      s.memory_handle);
#else
        // glImportMemoryFdEXT consumes the fd — duplicate it first since we
        // want to keep memory_handle valid for the slot's lifetime.
        int import_fd = dup(s.memory_handle);
        glImportMemoryFdEXT_(s.gl_memory_object,
                             mem_reqs.size,
                             platform::kGlHandleType,
                             import_fd);
        // fd is consumed by GL, no need to close import_fd
#endif

        glCreateTextures(GL_TEXTURE_2D, 1, &s.gl_texture);
        glTextureParameteri(s.gl_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(s.gl_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(s.gl_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(s.gl_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureStorageMem2DEXT_(s.gl_texture, 1,
                                  use_16bit ? GL_RGBA16 : GL_RGBA8,
                                  width, height, s.gl_memory_object, 0);
    } else {
        // CPU fallback: create a regular GL texture for upload
        int bpp = use_16bit ? 8 : 4;
        s.cpu_buffer.resize(width * height * bpp, 0);

        glCreateTextures(GL_TEXTURE_2D, 1, &s.cpu_gl_texture);
        glTextureParameteri(s.cpu_gl_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(s.cpu_gl_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(s.cpu_gl_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(s.cpu_gl_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureStorage2D(s.cpu_gl_texture, 1,
                           use_16bit ? GL_RGBA16 : GL_RGBA8,
                           width, height);
    }
}

void previz_texture_bridge::destroy_slot(channel_slot& s)
{
    // GL cleanup (must be on OGL thread — caller responsible)
    if (s.gl_texture)
        glDeleteTextures(1, &s.gl_texture);
    if (s.gl_memory_object && glDeleteMemoryObjectsEXT_)
        glDeleteMemoryObjectsEXT_(1, &s.gl_memory_object);
    if (s.cpu_gl_texture)
        glDeleteTextures(1, &s.cpu_gl_texture);

    // VK cleanup — dispatch to VK thread to avoid racing with in-flight work
    auto vk_dev = static_cast<VkDevice>(vk_device_->getVkDevice());
    auto vk_image  = s.vk_image;
    auto vk_memory = s.vk_memory;
    auto mem_handle = s.memory_handle;
    vk_device_->dispatch_sync([vk_dev, vk_image, vk_memory, mem_handle]() mutable {
        if (vk_image != VK_NULL_HANDLE)
            vkDestroyImage(vk_dev, vk_image, nullptr);
        if (vk_memory != VK_NULL_HANDLE)
            vkFreeMemory(vk_dev, vk_memory, nullptr);
        platform::close_handle(mem_handle);
    });

    s = {};
}

// ── post_channel (VK thread) ─────────────────────────────────────────────────

void previz_texture_bridge::post_channel(int    channel_id,
                                          VkImage source,
                                          VkImageLayout source_layout,
                                          int    width,
                                          int    height,
                                          bool   use_16bit)
{
    // ensure_slot may call dispatch_sync on the OGL thread, so we must NOT
    // hold mutex_ (the OGL thread might be in sync_to_store locking mutex_).
    channel_slot* slot_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slots_.find(channel_id);
        if (it != slots_.end() && it->second.width == width &&
            it->second.height == height && it->second.use_16bit == use_16bit) {
            slot_ptr = &it->second;
        }
    }

    if (!slot_ptr) {
        // Need to create/recreate — done outside mutex since it blocks on OGL thread
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = slots_.find(channel_id);
            if (it != slots_.end()) {
                ogl_device_->dispatch_sync([this, &it] { destroy_slot(it->second); });
                slots_.erase(it);
            }
        }

        // Create new slot (blocks on OGL thread)
        channel_slot new_slot;
        ogl_device_->dispatch_sync([this, &new_slot, width, height, use_16bit] {
            create_slot(new_slot, width, height, use_16bit);
        });

        std::lock_guard<std::mutex> lock(mutex_);
        slots_[channel_id] = std::move(new_slot);
        slot_ptr = &slots_[channel_id];
    }

    auto& s = *slot_ptr;

    if (interop_available_) {
        // Blit source → shared exportable image on the VK thread.
        // The caller is on a deferred future thread after VK compositing.
        // We dispatch the blit to the VK device thread for queue access.
        vk_device_->dispatch_sync([&] {
            auto vk_dev = static_cast<VkDevice>(vk_device_->getVkDevice());

            // Allocate a one-shot command buffer
            auto cmd_bufs = vk_device_->allocateCommandBuffers(1);
            VkCommandBuffer cmd = static_cast<VkCommandBuffer>(cmd_bufs[0]);

            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin);

            // Transition shared image: UNDEFINED → TRANSFER_DST
            VkImageMemoryBarrier barrier{};
            barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
            barrier.image                           = s.vk_image;
            barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel   = 0;
            barrier.subresourceRange.levelCount     = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount     = 1;
            barrier.srcAccessMask                   = 0;
            barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Transition source: current layout → TRANSFER_SRC
            VkImageMemoryBarrier src_barrier = barrier;
            src_barrier.image         = source;
            src_barrier.oldLayout     = source_layout;
            src_barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            src_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            src_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

            // Copy
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent         = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};

            vkCmdCopyImage(cmd,
                           source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           s.vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);

            // Transition shared image: TRANSFER_DST → GENERAL (for GL reading)
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Restore source layout so the VK mixer's cached texture remains valid
            VkImageMemoryBarrier restore_barrier = src_barrier;
            restore_barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            restore_barrier.newLayout     = source_layout;
            restore_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            restore_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &restore_barrier);

            vkEndCommandBuffer(cmd);

            // Submit WITHOUT semaphore — we use fence-only sync.
            // Binary semaphores can't be signaled twice without an intervening
            // wait, and we can't guarantee GL will consume every signal.
            VkSubmitInfo submit{};
            submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount   = 1;
            submit.pCommandBuffers      = &cmd;

            VkFence fence;
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            vkCreateFence(vk_dev, &fence_info, nullptr, &fence);

            // vk::SubmitInfo is layout-compatible with VkSubmitInfo
            vk_device_->submit(
                *reinterpret_cast<const vk::SubmitInfo*>(&submit),
                vk::Fence(fence));

            vkWaitForFences(vk_dev, 1, &fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(vk_dev, fence, nullptr);

            // Free the one-shot command buffer
            auto vk_pool = static_cast<VkCommandPool>(vk_device_->getCommandPool());
            vkFreeCommandBuffers(vk_dev, vk_pool, 1, &cmd);
        });
    } else {
        // CPU fallback: not yet implemented — previz will show black textures
        // for this channel until GPU interop is available.
        static bool warned = false;
        if (!warned) {
            CASPAR_LOG(warning) << L"[previz_bridge] CPU fallback path not implemented - "
                                   L"previz channel textures will be black.";
            warned = true;
        }
        s.cpu_dirty = true;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        s.ready = true;
    }
}

// ── get_gl_texture (OGL thread) ──────────────────────────────────────────────

previz_texture_bridge::gl_entry
previz_texture_bridge::get_gl_texture(int channel_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = slots_.find(channel_id);
    if (it == slots_.end() || !it->second.ready)
        return {};

    auto& s = it->second;

    if (interop_available_) {
        // No semaphore wait needed — post_channel() already fence-waited,
        // guaranteeing the VK write to shared memory is complete.
        return {s.gl_texture, s.width, s.height};
    }

    // CPU fallback: upload cpu_buffer to GL texture
    if (s.cpu_dirty && !s.cpu_buffer.empty()) {
        GLenum fmt  = GL_RGBA;
        GLenum type = s.use_16bit ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
        glTextureSubImage2D(s.cpu_gl_texture, 0, 0, 0, s.width, s.height, fmt, type, s.cpu_buffer.data());
        s.cpu_dirty = false;
    }

    return {s.cpu_gl_texture, s.width, s.height};
}

// ── sync_to_store ────────────────────────────────────────────────────────────

void previz_texture_bridge::sync_to_store(ogl::channel_texture_store& store)
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [channel_id, s] : slots_) {
        if (!s.ready)
            continue;

        GLuint tex = 0;
        if (interop_available_) {
            // No semaphore wait needed — post_channel() already fence-waited.
            tex = s.gl_texture;
        } else {
            if (s.cpu_dirty && !s.cpu_buffer.empty()) {
                GLenum fmt  = GL_RGBA;
                GLenum type = s.use_16bit ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
                glTextureSubImage2D(s.cpu_gl_texture, 0, 0, 0, s.width, s.height, fmt, type, s.cpu_buffer.data());
                s.cpu_dirty = false;
            }
            tex = s.cpu_gl_texture;
        }

        if (tex == 0)
            continue;

        store.update(channel_id, nullptr, tex, s.width, s.height);
    }
}

}}} // namespace caspar::accelerator::vulkan
