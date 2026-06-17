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

#include "shared_texture_pool.h"
#include "vulkan_device.h"
#include "platform_handles.h"

#include <accelerator/ogl/util/device.h>

#include <common/except.h>
#include <common/gl/gl_check.h>
#include <common/log.h>

#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <EGL/egl.h>
#include <unistd.h> // close()
#endif

#include <algorithm>

#include <mutex>

namespace caspar { namespace vulkan_output {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult result_ = (call);                                                                                     \
        if (result_ != VK_SUCCESS) {                                                                                   \
            CASPAR_THROW_EXCEPTION(caspar_exception()                                                                  \
                                   << msg_info("VK call failed: " #call " = " + std::to_string(result_)));             \
        }                                                                                                              \
    } while (0)

// GL_EXT_memory_object / GL_EXT_memory_object_{win32,fd} function pointers
PFNGLCREATEMEMORYOBJECTSEXTPROC        glCreateMemoryObjectsEXT_       = nullptr;
PFNGLDELETEMEMORYOBJECTSEXTPROC        glDeleteMemoryObjectsEXT_       = nullptr;
PFNGLTEXTURESTORAGEMEM2DEXTPROC        glTextureStorageMem2DEXT_       = nullptr;
#ifdef _WIN32
PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC    glImportMemoryWin32HandleEXT_   = nullptr;
#else
PFNGLIMPORTMEMORYFDEXTPROC             glImportMemoryFdEXT_            = nullptr;
#endif

// GL_EXT_semaphore / GL_EXT_semaphore_{win32,fd} function pointers
PFNGLGENSEMAPHORESEXTPROC              glGenSemaphoresEXT_             = nullptr;
PFNGLDELETESEMAPHORESEXTPROC           glDeleteSemaphoresEXT_          = nullptr;
PFNGLSIGNALSEMAPHOREEXTPROC            glSignalSemaphoreEXT_           = nullptr;
PFNGLWAITSEMAPHOREEXTPROC              glWaitSemaphoreEXT_             = nullptr;
#ifdef _WIN32
PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC glImportSemaphoreWin32HandleEXT_ = nullptr;
#else
PFNGLIMPORTSEMAPHOREFDEXTPROC          glImportSemaphoreFdEXT_         = nullptr;
#endif

std::once_flag gl_ext_flag;

void load_gl_extensions()
{
    std::call_once(gl_ext_flag, [] {
#ifdef _WIN32
        auto get_proc = [](const char* name) { return (void*)wglGetProcAddress(name); };
#else
        auto get_proc = [](const char* name) { return (void*)eglGetProcAddress(name); };
#endif

        glCreateMemoryObjectsEXT_      = (PFNGLCREATEMEMORYOBJECTSEXTPROC)get_proc("glCreateMemoryObjectsEXT");
        glDeleteMemoryObjectsEXT_      = (PFNGLDELETEMEMORYOBJECTSEXTPROC)get_proc("glDeleteMemoryObjectsEXT");
        glTextureStorageMem2DEXT_      = (PFNGLTEXTURESTORAGEMEM2DEXTPROC)get_proc("glTextureStorageMem2DEXT");
#ifdef _WIN32
        glImportMemoryWin32HandleEXT_  = (PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC)get_proc("glImportMemoryWin32HandleEXT");
#else
        glImportMemoryFdEXT_           = (PFNGLIMPORTMEMORYFDEXTPROC)get_proc("glImportMemoryFdEXT");
#endif

        glGenSemaphoresEXT_             = (PFNGLGENSEMAPHORESEXTPROC)get_proc("glGenSemaphoresEXT");
        glDeleteSemaphoresEXT_          = (PFNGLDELETESEMAPHORESEXTPROC)get_proc("glDeleteSemaphoresEXT");
        glSignalSemaphoreEXT_           = (PFNGLSIGNALSEMAPHOREEXTPROC)get_proc("glSignalSemaphoreEXT");
        glWaitSemaphoreEXT_             = (PFNGLWAITSEMAPHOREEXTPROC)get_proc("glWaitSemaphoreEXT");
#ifdef _WIN32
        glImportSemaphoreWin32HandleEXT_ = (PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC)get_proc("glImportSemaphoreWin32HandleEXT");
#else
        glImportSemaphoreFdEXT_         = (PFNGLIMPORTSEMAPHOREFDEXTPROC)get_proc("glImportSemaphoreFdEXT");
#endif

#ifdef _WIN32
        if (!glCreateMemoryObjectsEXT_ || !glImportMemoryWin32HandleEXT_ || !glTextureStorageMem2DEXT_) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("GL_EXT_memory_object_win32 not available on this GPU/driver"));
        }

        if (!glGenSemaphoresEXT_ || !glImportSemaphoreWin32HandleEXT_ || !glSignalSemaphoreEXT_) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("GL_EXT_semaphore_win32 not available on this GPU/driver"));
        }
#else
        if (!glCreateMemoryObjectsEXT_ || !glImportMemoryFdEXT_ || !glTextureStorageMem2DEXT_) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("GL_EXT_memory_object_fd not available on this GPU/driver"));
        }

        if (!glGenSemaphoresEXT_ || !glImportSemaphoreFdEXT_ || !glSignalSemaphoreEXT_) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("GL_EXT_semaphore_fd not available on this GPU/driver"));
        }
#endif
    });
}

} // namespace

shared_texture_pool::shared_texture_pool(std::shared_ptr<accelerator::ogl::device> ogl_device,
                                         vulkan_device&                            vk_device,
                                         uint32_t                                  width,
                                         uint32_t                                  height,
                                         bool                                      use_16bit)
    : ogl_device_(std::move(ogl_device))
    , vk_device_(vk_device)
    , width_(width)
    , height_(height)
    , use_16bit_(use_16bit)
{
    // Detect Pascal GPUs (GP10x, Vulkan API 1.1 era, vendor NVIDIA).
    // Pascal's GL driver has a tiling incompatibility with GL_EXT_memory_object:
    // VK_IMAGE_TILING_OPTIMAL textures imported via glTextureStorageMem2DEXT
    // produce visible pixel grid artifacts. LINEAR tiling avoids this.
    VkPhysicalDeviceProperties vk_props;
    vkGetPhysicalDeviceProperties(vk_device_.physical_device(), &vk_props);
    if (vk_props.vendorID == 0x10DE) { // NVIDIA
        uint32_t major = VK_API_VERSION_MAJOR(vk_props.apiVersion);
        uint32_t minor = VK_API_VERSION_MINOR(vk_props.apiVersion);
        // Pascal (GP10x) supports up to Vulkan 1.1. Turing+ supports 1.2+.
        if (major == 1 && minor <= 1) {
            use_linear_tiling_ = true;
            CASPAR_LOG(info) << L"[vulkan_output] Pascal GPU detected ("
                             << vk_props.deviceName
                             << L") - forcing LINEAR tiling for GL_EXT_memory_object compatibility.";
        }
    }

    if (use_linear_tiling_) {
        // Verify LINEAR tiling support
        VkFormat format = use_16bit_ ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        VkFormatProperties fmt_props;
        vkGetPhysicalDeviceFormatProperties(vk_device_.physical_device(), format, &fmt_props);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                              VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                              VK_FORMAT_FEATURE_BLIT_SRC_BIT;
        if ((fmt_props.linearTilingFeatures & required) != required) {
            CASPAR_LOG(warning) << L"[vulkan_output] LINEAR tiling not fully supported -"
                                << L" falling back to OPTIMAL (may cause artifacts).";
            use_linear_tiling_ = false;
        }
    }

    // Create shared textures on the OGL thread
    ogl_device_->dispatch_sync([this] {
        load_gl_extensions();

        for (int i = 0; i < BUFFER_COUNT; ++i) {
            create_slot(slots_[i]);
        }

        glGenFramebuffers(1, &read_fbo_);
        glGenFramebuffers(1, &draw_fbo_);
    });

    CASPAR_LOG(info) << L"[vulkan_output] Shared texture pool created: "
                     << width_ << L"x" << height_ << L" (" << BUFFER_COUNT << L" slots"
                     << (use_linear_tiling_ ? L", LINEAR tiling" : L"") << L")";
}

shared_texture_pool::shared_texture_pool(vulkan_device& vk_device,
                                         uint32_t       width,
                                         uint32_t       height,
                                         bool           use_16bit)
    : vk_device_(vk_device)
    , width_(width)
    , height_(height)
    , use_16bit_(use_16bit)
    , use_linear_tiling_(true) // Cross-GPU (affinity) path always uses LINEAR tiling.
                               // Pascal GPUs have GL_EXT_memory_object tiling incompatibilities,
                               // and affinity GL contexts are even more restrictive.
{
    // Verify LINEAR tiling support for our format+usage
    VkFormat format = use_16bit_ ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    VkFormatProperties fmt_props;
    vkGetPhysicalDeviceFormatProperties(vk_device_.physical_device(), format, &fmt_props);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                          VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                          VK_FORMAT_FEATURE_BLIT_SRC_BIT;
    if ((fmt_props.linearTilingFeatures & required) != required) {
        CASPAR_LOG(warning) << L"[vulkan_output] LINEAR tiling not fully supported for "
                            << (use_16bit_ ? L"RGBA16" : L"RGBA8")
                            << L" - falling back to OPTIMAL (may cause artifacts on Pascal GPUs)";
        use_linear_tiling_ = false;
    }

    // Affinity constructor: caller guarantees we're on a valid GL context thread
    load_gl_extensions();
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        create_slot(slots_[i]);
    }

    // Create FBOs for blit — always, not just 16-bit (see above).
    glGenFramebuffers(1, &read_fbo_);
    glGenFramebuffers(1, &draw_fbo_);

    CASPAR_LOG(info) << L"[vulkan_output] Shared texture pool created (affinity): "
                     << width_ << L"x" << height_ << L" (" << BUFFER_COUNT << L" slots"
                     << (use_linear_tiling_ ? L", LINEAR tiling" : L", OPTIMAL tiling") << L")";
}

shared_texture_pool::~shared_texture_pool()
{
    if (ogl_device_) {
        ogl_device_->dispatch_sync([this] {
            for (int i = 0; i < BUFFER_COUNT; ++i) {
                destroy_slot(slots_[i]);
            }
            if (read_fbo_) { glDeleteFramebuffers(1, &read_fbo_); read_fbo_ = 0; }
            if (draw_fbo_) { glDeleteFramebuffers(1, &draw_fbo_); draw_fbo_ = 0; }
        });
    } else {
        // Affinity path: no GL context is current on this thread.
        // Only destroy VK-side and Win32 handles. GL resources (texture, memory object,
        // semaphore, FBOs) will be freed when the affinity GL context is destroyed.
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            destroy_slot_vk_only(slots_[i]);
        }
    }
}

void shared_texture_pool::create_slot(slot& s)
{
    auto dev = vk_device_.device();

    // ─── Vulkan side: create exportable memory + image ───────────────────────

    // Create VkImage
    VkExternalMemoryImageCreateInfo ext_mem_img{};
    ext_mem_img.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_mem_img.handleTypes = platform::kExternalMemoryHandleType;

    VkImageCreateInfo img_info{};
    img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.pNext         = &ext_mem_img;
    img_info.imageType     = VK_IMAGE_TYPE_2D;
    img_info.format        = use_16bit_ ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    img_info.extent        = {width_, height_, 1};
    img_info.mipLevels     = 1;
    img_info.arrayLayers   = 1;
    img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling        = use_linear_tiling_ ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
    img_info.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // SAMPLED_BIT is needed for the color conversion compute shader (reads via imageView).
    // LINEAR tiling may not support it on all GPUs — check before adding.
    if (!use_linear_tiling_) {
        img_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    } else {
        VkFormat format = use_16bit_ ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        VkFormatProperties fmt_props;
        vkGetPhysicalDeviceFormatProperties(vk_device_.physical_device(), format, &fmt_props);
        if (fmt_props.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
            img_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
    }
    img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(dev, &img_info, nullptr, &s.vk_image));

    // Allocate exportable memory
    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(dev, s.vk_image, &mem_reqs);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = platform::kExternalMemoryHandleType;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk_device_.physical_device(), &mem_props);

    uint32_t mem_type_index = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type_index = i;
            break;
        }
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext           = &export_info;
    alloc_info.allocationSize  = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_index;

    VK_CHECK(vkAllocateMemory(dev, &alloc_info, nullptr, &s.vk_memory));
    VK_CHECK(vkBindImageMemory(dev, s.vk_image, s.vk_memory, 0));

    // Export memory handle
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR get_handle_info{};
    get_handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    get_handle_info.memory     = s.vk_memory;
    get_handle_info.handleType = platform::kExternalMemoryHandleType;

    auto vkGetMemoryWin32HandleKHR_ = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(dev, "vkGetMemoryWin32HandleKHR"));
    VK_CHECK(vkGetMemoryWin32HandleKHR_(dev, &get_handle_info, &s.memory_handle));
#else
    VkMemoryGetFdInfoKHR get_handle_info{};
    get_handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_handle_info.memory     = s.vk_memory;
    get_handle_info.handleType = platform::kExternalMemoryHandleType;

    auto vkGetMemoryFdKHR_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR"));
    VK_CHECK(vkGetMemoryFdKHR_(dev, &get_handle_info, &s.memory_handle));
#endif

    // Create VkImageView
    VkImageViewCreateInfo view_info{};
    view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image                           = s.vk_image;
    view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format                          = use_16bit_ ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel   = 0;
    view_info.subresourceRange.levelCount     = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount     = 1;

    VK_CHECK(vkCreateImageView(dev, &view_info, nullptr, &s.vk_image_view));

    // ─── Vulkan side: create exportable semaphore ────────────────────────────

    VkExportSemaphoreCreateInfo export_sem_info{};
    export_sem_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_sem_info.handleTypes = platform::kExternalSemaphoreHandleType;

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &export_sem_info;

    VK_CHECK(vkCreateSemaphore(dev, &sem_info, nullptr, &s.vk_semaphore));

    // Export semaphore handle
#ifdef _WIN32
    VkSemaphoreGetWin32HandleInfoKHR get_sem_handle{};
    get_sem_handle.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    get_sem_handle.semaphore  = s.vk_semaphore;
    get_sem_handle.handleType = platform::kExternalSemaphoreHandleType;

    auto vkGetSemaphoreWin32HandleKHR_ = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(dev, "vkGetSemaphoreWin32HandleKHR"));
    VK_CHECK(vkGetSemaphoreWin32HandleKHR_(dev, &get_sem_handle, &s.semaphore_handle));
#else
    VkSemaphoreGetFdInfoKHR get_sem_handle{};
    get_sem_handle.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    get_sem_handle.semaphore  = s.vk_semaphore;
    get_sem_handle.handleType = platform::kExternalSemaphoreHandleType;

    auto vkGetSemaphoreFdKHR_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR"));
    VK_CHECK(vkGetSemaphoreFdKHR_(dev, &get_sem_handle, &s.semaphore_handle));
#endif

    // ─── GL side: import memory + create texture ─────────────────────────────

    // Import VK memory into GL
    glCreateMemoryObjectsEXT_(1, &s.gl_memory_object);
#ifdef _WIN32
    glImportMemoryWin32HandleEXT_(s.gl_memory_object,
                                  mem_reqs.size,
                                  platform::kGlHandleType,
                                  s.memory_handle);
#else
    glImportMemoryFdEXT_(s.gl_memory_object,
                         mem_reqs.size,
                         platform::kGlHandleType,
                         s.memory_handle);
    // fd is consumed by import on Linux — mark as invalid to prevent double-close
    s.memory_handle = platform::kInvalidHandle;
#endif

    // Create GL texture backed by imported memory
    glCreateTextures(GL_TEXTURE_2D, 1, &s.gl_texture);
    glTextureParameteri(s.gl_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(s.gl_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(s.gl_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(s.gl_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureStorageMem2DEXT_(s.gl_texture, 1, use_16bit_ ? GL_RGBA16 : GL_RGBA8,
                              width_, height_, s.gl_memory_object, 0);

    // Import VK semaphore into GL
    glGenSemaphoresEXT_(1, &s.gl_semaphore);
#ifdef _WIN32
    glImportSemaphoreWin32HandleEXT_(s.gl_semaphore,
                                     platform::kGlHandleType,
                                     s.semaphore_handle);
#else
    glImportSemaphoreFdEXT_(s.gl_semaphore,
                            platform::kGlHandleType,
                            s.semaphore_handle);
    // fd is consumed by import on Linux — mark as invalid to prevent double-close
    s.semaphore_handle = platform::kInvalidHandle;
#endif
}

void shared_texture_pool::destroy_slot(slot& s)
{
    // GL cleanup
    if (s.gl_semaphore)
        glDeleteSemaphoresEXT_(1, &s.gl_semaphore);
    if (s.gl_texture)
        glDeleteTextures(1, &s.gl_texture);
    if (s.gl_memory_object)
        glDeleteMemoryObjectsEXT_(1, &s.gl_memory_object);

    // VK + handle cleanup
    destroy_slot_vk_only(s);
}

void shared_texture_pool::destroy_slot_vk_only(slot& s)
{
    auto dev = vk_device_.device();

    // VK cleanup
    if (s.vk_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(dev, s.vk_image_view, nullptr);
        s.vk_image_view = VK_NULL_HANDLE;
    }
    if (s.vk_image != VK_NULL_HANDLE) {
        vkDestroyImage(dev, s.vk_image, nullptr);
        s.vk_image = VK_NULL_HANDLE;
    }
    if (s.vk_memory != VK_NULL_HANDLE) {
        vkFreeMemory(dev, s.vk_memory, nullptr);
        s.vk_memory = VK_NULL_HANDLE;
    }
    if (s.vk_semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(dev, s.vk_semaphore, nullptr);
        s.vk_semaphore = VK_NULL_HANDLE;
    }

    // Handle cleanup
#ifdef _WIN32
    if (s.memory_handle) {
        CloseHandle(s.memory_handle);
        s.memory_handle = nullptr;
    }
    if (s.semaphore_handle) {
        CloseHandle(s.semaphore_handle);
        s.semaphore_handle = nullptr;
    }
#else
    if (s.memory_handle != platform::kInvalidHandle) {
        close(s.memory_handle);
        s.memory_handle = platform::kInvalidHandle;
    }
    if (s.semaphore_handle != platform::kInvalidHandle) {
        close(s.semaphore_handle);
        s.semaphore_handle = platform::kInvalidHandle;
    }
#endif
}

void shared_texture_pool::blit_from_texture(GLuint source_texture_id, int width, int height)
{
    // Copy from the mixer's target texture into our shared texture
    // This must be called on the OGL device thread
    auto& s = slots_[write_index_];

    // Clamp to pool dimensions to prevent out-of-bounds copy
    int clamped_w = (std::min)(width, static_cast<int>(width_));
    int clamped_h = (std::min)(height, static_cast<int>(height_));

    if (use_linear_tiling_) {
        // LINEAR tiling path: use glCopyImageSubData. On Pascal GPUs with
        // VK-external-memory-backed textures, glBlitFramebuffer (FBO rendering)
        // silently fails (black output). glCopyImageSubData bypasses the
        // rendering pipeline and performs a direct memory copy, which is
        // compatible with LINEAR tiling's contiguous memory layout.
        while (glGetError() != GL_NO_ERROR) {} // drain stale errors
        glCopyImageSubData(source_texture_id, GL_TEXTURE_2D, 0, 0, 0, 0,
                           s.gl_texture, GL_TEXTURE_2D, 0, 0, 0, 0,
                           clamped_w, clamped_h, 1);
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            static bool warned = false;
            if (!warned) {
                CASPAR_LOG(warning) << L"[shared_texture_pool] glCopyImageSubData failed: GL error 0x"
                                    << std::hex << err << std::dec
                                    << L" src=" << source_texture_id << L" dst=" << s.gl_texture
                                    << L" " << clamped_w << L"x" << clamped_h
                                    << L" write_idx=" << write_index_;
                warned = true;
            }
        }
    } else {
        // OPTIMAL tiling path: use glBlitFramebuffer. Goes through the rendering
        // pipeline which correctly handles vendor-specific optimal tiling layouts.
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, source_texture_id, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo_);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.gl_texture, 0);
        glBlitFramebuffer(0, 0, clamped_w, clamped_h, 0, 0, clamped_w, clamped_h,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void shared_texture_pool::upload_from_pbo(GLuint pbo, int width, int height, GLenum pixel_format, GLenum pixel_type)
{
    auto& s = slots_[write_index_];

    int clamped_w = (std::min)(width, static_cast<int>(width_));
    int clamped_h = (std::min)(height, static_cast<int>(height_));

    while (glGetError() != GL_NO_ERROR) {} // drain stale errors
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, s.gl_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, clamped_w, clamped_h,
                    pixel_format, pixel_type, nullptr);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        static bool warned = false;
        if (!warned) {
            CASPAR_LOG(warning) << L"[shared_texture_pool] upload_from_pbo glTexSubImage2D failed: GL error 0x"
                                << std::hex << err << std::dec
                                << L" pbo=" << pbo << L" tex=" << s.gl_texture
                                << L" " << clamped_w << L"x" << clamped_h
                                << L" fmt=0x" << std::hex << pixel_format << std::dec
                                << L" write_idx=" << write_index_;
            warned = true;
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void shared_texture_pool::signal_gl()
{
    auto& s = slots_[write_index_];

    // Signal the semaphore after the blit completes
    while (glGetError() != GL_NO_ERROR) {} // drain stale errors
    GLenum dst_layout = GL_LAYOUT_GENERAL_EXT; // 0x958D from GL_EXT_semaphore
    glSignalSemaphoreEXT_(s.gl_semaphore, 0, nullptr, 1, &s.gl_texture, &dst_layout);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        static bool warned = false;
        if (!warned) {
            CASPAR_LOG(warning) << L"[shared_texture_pool] glSignalSemaphoreEXT failed: GL error 0x"
                                << std::hex << err << std::dec
                                << L" sem=" << s.gl_semaphore << L" tex=" << s.gl_texture
                                << L" write_idx=" << write_index_;
            warned = true;
        }
    }

    // Flush to ensure the signal is submitted to the GPU
    glFlush();
}

VkSemaphore shared_texture_pool::wait_semaphore_vk() const
{
    return slots_[read_index_.load(std::memory_order_acquire)].vk_semaphore;
}

VkImage shared_texture_pool::current_vk_image() const
{
    return slots_[read_index_.load(std::memory_order_acquire)].vk_image;
}

VkImageView shared_texture_pool::current_vk_image_view() const
{
    return slots_[read_index_.load(std::memory_order_acquire)].vk_image_view;
}

void shared_texture_pool::swap()
{
    read_index_.store(write_index_, std::memory_order_release);
    write_index_ = (write_index_ + 1) % BUFFER_COUNT;
}

}} // namespace caspar::vulkan_output
