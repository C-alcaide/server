/*
 * Copyright 2025
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
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */

#include "image_kernel.h"

#include "../util/device.h"
#include "../util/pipeline.h"
#include "../util/platform_config.h"
#include "../util/renderpass.h"
#include "../util/texture.h"

#include <common/assert.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif

#include <core/frame/frame_transform.h>
#include <core/frame/pixel_format.h>

#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace caspar::accelerator::vulkan {

float get_precision_factor(common::bit_depth depth)
{
    switch (depth) {
        case common::bit_depth::bit8:
            return 1.0f;
        case common::bit_depth::bit10:
            return 64.0f;
        case common::bit_depth::bit12:
            return 16.0f;
        case common::bit_depth::bit16:
        default:
            return 1.0f;
    }
}

bool is_above_screen(double y) { return y < 0.0; }

bool is_below_screen(double y) { return y > 1.0; }

bool is_left_of_screen(double x) { return x < 0.0; }

bool is_right_of_screen(double x) { return x > 1.0; }

bool is_outside_screen(const std::vector<core::frame_geometry::coord>& coords)
{
    auto x_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_x; });
    auto y_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_y; });

    return boost::algorithm::all_of(x_coords, &is_left_of_screen) ||
           boost::algorithm::all_of(x_coords, &is_right_of_screen) ||
           boost::algorithm::all_of(y_coords, &is_above_screen) || boost::algorithm::all_of(y_coords, &is_below_screen);
}

static const double epsilon = 0.001;

// ── Tone Curve LUT builder (Fritsch-Carlson monotone cubic hermite) ─────────
static std::array<float, 256> build_curve_lut(const core::curve_channel& cc)
{
    std::array<float, 256> lut;
    if (cc.count < 2) {
        for (int i = 0; i < 256; ++i) lut[i] = i / 255.0f;
        return lut;
    }
    std::vector<std::pair<double, double>> pts;
    pts.reserve(cc.count);
    for (int i = 0; i < cc.count; ++i)
        pts.push_back({cc.points[i].x, cc.points[i].y});
    std::sort(pts.begin(), pts.end());

    int n = static_cast<int>(pts.size());
    std::vector<double> dx(n - 1), dy(n - 1), delta(n - 1), m(n);
    for (int i = 0; i < n - 1; ++i) {
        dx[i]    = pts[i + 1].first  - pts[i].first;
        dy[i]    = pts[i + 1].second - pts[i].second;
        delta[i] = (dx[i] > 1e-10) ? dy[i] / dx[i] : 0.0;
    }
    m[0]     = delta[0];
    m[n - 1] = delta[n - 2];
    for (int i = 1; i < n - 1; ++i)
        m[i] = (delta[i - 1] + delta[i]) * 0.5;
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(delta[i]) < 1e-10) { m[i] = m[i + 1] = 0.0; continue; }
        double a = m[i]     / delta[i];
        double b = m[i + 1] / delta[i];
        double h = std::sqrt(a * a + b * b);
        if (h > 3.0) { m[i] *= 3.0 / h; m[i + 1] *= 3.0 / h; }
    }
    for (int k = 0; k < 256; ++k) {
        double t = k / 255.0;
        if (t <= pts.front().first) { lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, pts.front().second))); continue; }
        if (t >= pts.back().first)  { lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, pts.back().second)));  continue; }
        int seg = 0;
        for (int i = 0; i < n - 2; ++i)
            if (t >= pts[i].first && t < pts[i + 1].first) { seg = i; break; }
        double h_   = dx[seg];
        double t_   = (h_ > 1e-10) ? (t - pts[seg].first) / h_ : 0.0;
        double t2   = t_ * t_;
        double t3   = t2 * t_;
        double h00  = 2*t3 - 3*t2 + 1;
        double h10  = t3  - 2*t2 + t_;
        double h01  = -2*t3 + 3*t2;
        double h11  = t3  - t2;
        double val  = h00 * pts[seg].second  + h10 * h_ * m[seg]
                    + h01 * pts[seg+1].second + h11 * h_ * m[seg+1];
        lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, val)));
    }
    return lut;
}
// ─────────────────────────────────────────────────────────────────────────────

static const uint32_t frame_buffer_size = 3;

// ── Vulkan LUT texture helper ────────────────────────────────────────────────
// Manages a small GPU-only image + staging buffer for uploading LUT data.
struct vk_lut_texture
{
    vk::Device       device    = nullptr;
    vk::Image        image     = nullptr;
    vk::ImageView    view      = nullptr;
    vk::DeviceMemory memory    = nullptr;
    vk::Buffer       staging   = nullptr;
    vk::DeviceMemory staging_mem = nullptr;
    void*            mapped    = nullptr;
    vk::DeviceSize   data_size = 0;

    void destroy()
    {
        if (!device) return;
        if (view)        device.destroyImageView(view);
        if (image)       device.destroyImage(image);
        if (memory)      device.freeMemory(memory);
        if (staging)     device.destroyBuffer(staging);
        if (staging_mem) { device.unmapMemory(staging_mem); device.freeMemory(staging_mem); }
        *this = {};
    }
};

struct image_kernel::impl
{
    spl::shared_ptr<device> vulkan_;
    common::bit_depth       depth_;
    int32_t                 frame_counter_ = 0;

    // ── Persistent LUT textures ──────────────────────────────────────────
    vk_lut_texture          lut3d_tex_{};
    const core::lut3d_data* lut3d_data_ptr_ = nullptr;  // tracks which data is uploaded

    vk_lut_texture          hue_curve_tex_{};

    vk_lut_texture          curve_lut_tex_{};

    vk_lut_texture               blend_mask_tex_{};
    const core::blend_mask_data* blend_mask_data_ptr_ = nullptr;  // tracks which data is uploaded

    lut_views               current_lut_views_{};
    // ─────────────────────────────────────────────────────────────────────

    struct frame_data : public frame_context
    {
        image_kernel::impl* parent = nullptr;

        vk::Buffer       buffer = nullptr;
        void*            data   = nullptr;
        vk::DeviceMemory memory = nullptr;
        size_t           size   = 0;

        vk::CommandBuffer cmd_buffer = nullptr;
        vk::Fence         fence      = nullptr;
        vk::Semaphore     render_sem = nullptr;   // exportable timeline semaphore for GPU-side wait
        platform::native_handle_t render_sem_handle = platform::kInvalidHandle; // cached handle
        uint64_t          render_sem_value  = 0;       // current timeline value

        explicit frame_data(image_kernel::impl* parent)
            : parent(parent)
        {
        }

        virtual vk::Buffer upload_vertex_data(const std::vector<float>& src)
        {
            return parent->upload_vertex_buffer(*this, (void*)src.data(), src.size() * sizeof(float));
        }
        virtual draw_data create_draw_data(const draw_params& params) { return parent->draw(params); }
        virtual lut_views get_lut_views() const override { return parent->current_lut_views_; }
        virtual void upload_pending_luts(vk::CommandBuffer cmd) override { parent->do_upload_pending_luts(cmd); }
        virtual std::shared_ptr<class pipeline> get_pipeline() { return parent->vulkan_->get_pipeline(parent->depth_); }
        virtual vk::CommandBuffer               get_command_buffer() { return cmd_buffer; }
        virtual void                            submit()
        {
            auto vk_device = parent->vulkan_->getVkDevice();
            if (!fence)
                fence = vk_device.createFence({});

            // Create an exportable timeline semaphore for GPU-side wait by CUDA consumers.
            if (!render_sem) {
                vk::ExportSemaphoreCreateInfo exportInfo{};
                exportInfo.handleTypes = static_cast<vk::ExternalSemaphoreHandleTypeFlagBits>(platform::kExternalSemaphoreHandleType);
                vk::SemaphoreTypeCreateInfo typeInfo{};
                typeInfo.semaphoreType = vk::SemaphoreType::eTimeline;
                typeInfo.initialValue  = 0;
                typeInfo.pNext         = &exportInfo;
                vk::SemaphoreCreateInfo semInfo{};
                semInfo.pNext = &typeInfo;
                render_sem = vk_device.createSemaphore(semInfo);
            }

            render_sem_value++;

            vk::TimelineSemaphoreSubmitInfo timelineInfo{};
            uint64_t signalValue = render_sem_value;
            timelineInfo.signalSemaphoreValueCount = 1;
            timelineInfo.pSignalSemaphoreValues    = &signalValue;

            vk::SubmitInfo submitInfo{};
            submitInfo.setCommandBuffers(cmd_buffer);
            if (render_sem) {
                submitInfo.setSignalSemaphores(render_sem);
                submitInfo.pNext = &timelineInfo;
            }
            parent->vulkan_->submit(submitInfo, fence);
        }

        void*                           render_complete_semaphore_handle() override
        {
            if (render_sem && render_sem_handle == platform::kInvalidHandle) {
                auto vk_device = parent->vulkan_->getVkDevice();
#ifdef _WIN32
                auto pfn = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
                    vk_device.getProcAddr("vkGetSemaphoreWin32HandleKHR"));
                if (!pfn) return nullptr;

                VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
                handleInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
                handleInfo.semaphore = static_cast<VkSemaphore>(render_sem);
                handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

                HANDLE handle = nullptr;
                VkResult result = pfn(static_cast<VkDevice>(vk_device), &handleInfo, &handle);
                if (result == VK_SUCCESS && handle)
                    render_sem_handle = handle;
#else
                auto pfn = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
                    vk_device.getProcAddr("vkGetSemaphoreFdKHR"));
                if (!pfn) return nullptr;

                VkSemaphoreGetFdInfoKHR fdInfo{};
                fdInfo.sType     = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                fdInfo.semaphore = static_cast<VkSemaphore>(render_sem);
                fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

                int fd = -1;
                VkResult result = pfn(static_cast<VkDevice>(vk_device), &fdInfo, &fd);
                if (result == VK_SUCCESS && fd >= 0)
                    render_sem_handle = fd;
#endif
            }
#ifdef _WIN32
            return render_sem_handle;
#else
            return render_sem_handle == platform::kInvalidHandle
                ? nullptr
                : reinterpret_cast<void*>(static_cast<intptr_t>(render_sem_handle));
#endif
        }

        uint64_t render_complete_semaphore_value() override { return render_sem_value; }
        virtual void                            wait_for_completion()
        {
            if (fence) {
                auto result = parent->vulkan_->getVkDevice().waitForFences(fence, true, 1000000000);
                if (result == vk::Result::eTimeout) {
                    CASPAR_LOG(warning) << L"[Vulkan image_kernel] Timeout waiting for render completion";
                }
            }
        }
        virtual std::shared_ptr<class texture>
        create_attachment(uint32_t width, uint32_t height, uint32_t components_count)
        {
            // Reuse an attachment texture from a previous frame on this slot
            // if the consumer has released its reference (use_count == 1 means
            // only our pool holds it).  This keeps the underlying VkDeviceMemory
            // and its exported Win32 HANDLE stable, which in turn keeps any
            // CUDA import on the consumer side valid — avoiding the extremely
            // expensive cudaImportExternalMemory call (~10-150 ms) every frame.
            for (auto& tex : attachment_pool_) {
                if (tex && tex.use_count() == 1 &&
                    static_cast<uint32_t>(tex->width()) == width &&
                    static_cast<uint32_t>(tex->height()) == height) {
                    return tex;
                }
            }
            auto tex = parent->vulkan_->create_attachment(width, height, parent->depth_, components_count);
            // Cap pool to prevent unbounded VRAM growth when consumers hold refs.
            static constexpr size_t MAX_ATTACHMENT_POOL = 4;
            if (attachment_pool_.size() < MAX_ATTACHMENT_POOL)
                attachment_pool_.push_back(tex);
            return tex;
        }

        // Pool of attachment textures for this slot.
        std::vector<std::shared_ptr<class texture>> attachment_pool_;
    };

    frame_data frames_[frame_buffer_size];
    uint32_t   current_frame_index_ = 0;

    explicit impl(const spl::shared_ptr<device>& vulkan, common::bit_depth depth)
        : vulkan_(vulkan)
        , depth_(depth)
        , frames_{frame_data{this}, frame_data{this}, frame_data{this}}
    {
        auto cmd_buffers = vulkan_->allocateCommandBuffers(frame_buffer_size);
        for (uint32_t i = 0; i < frame_buffer_size; ++i) {
            frames_[i].cmd_buffer = cmd_buffers[i];
        }
    }

    ~impl()
    {
        auto vk_device = vulkan_->getVkDevice();

        lut3d_tex_.destroy();
        hue_curve_tex_.destroy();
        curve_lut_tex_.destroy();
        blend_mask_tex_.destroy();

        for (auto& frame : frames_) {
            if (frame.buffer) {
                vk_device.unmapMemory(frame.memory);
                vk_device.destroyBuffer(frame.buffer);
                vk_device.freeMemory(frame.memory);
                if (frame.fence) {
                    vk_device.destroyFence(frame.fence);
                }
                if (frame.render_sem_handle != platform::kInvalidHandle) {
                    platform::close_handle(frame.render_sem_handle);
                }
                if (frame.render_sem) {
                    vk_device.destroySemaphore(frame.render_sem);
                }
            }
        }
    }

    spl::shared_ptr<renderpass> create_renderpass(uint32_t width, uint32_t height)
    {
        auto  device = vulkan_->getVkDevice();
        auto& ctx    = frames_[(++current_frame_index_) % frame_buffer_size];
        if (ctx.fence) {
            auto result = device.waitForFences(ctx.fence, true, 1000000000); // wait up to one second
            if (result == vk::Result::eTimeout) {
                CASPAR_LOG(warning) << L"[Vulkan image_kernel] Timeout waiting for fence";
            }
            device.resetFences(ctx.fence);
        }

        ctx.cmd_buffer.reset({});
        return spl::make_shared<renderpass>(&ctx, width, height);
    }

    uint32_t findDedicatedMemoryType(uint32_t typeMask, vk::MemoryPropertyFlags properties)
    {
        auto memProperties = vulkan_->getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeMask & (1 << i)) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
                return i;
            }
        }
        throw std::runtime_error("[Vulkan image_kernel] Failed to find suitable memory type");
    }

    vk::Buffer upload_vertex_buffer(frame_data& vb, void* data, size_t size)
    {
        if (vb.size < size) {
            auto vk_device = vulkan_->getVkDevice();

            if (vb.buffer) {
                vk_device.unmapMemory(vb.memory);
                vk_device.destroyBuffer(vb.buffer);
                vk_device.freeMemory(vb.memory);
            }

            // staging buffer
            vk::BufferCreateInfo stagingInfo{};
            stagingInfo.size        = size;
            stagingInfo.usage       = vk::BufferUsageFlagBits::eVertexBuffer;
            stagingInfo.sharingMode = vk::SharingMode::eExclusive;

            vb.buffer = vk_device.createBuffer(stagingInfo);

            auto stagingMemReq = vk_device.getBufferMemoryRequirements(vb.buffer);

            vk::MemoryAllocateInfo stagingAlloc{};
            stagingAlloc.allocationSize  = stagingMemReq.size;
            stagingAlloc.memoryTypeIndex = findDedicatedMemoryType(stagingMemReq.memoryTypeBits,
                                                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                                                       vk::MemoryPropertyFlagBits::eHostCoherent);

            vb.memory = vk_device.allocateMemory(stagingAlloc);
            vk_device.bindBufferMemory(vb.buffer, vb.memory, 0);

            vb.data = vk_device.mapMemory(vb.memory, 0, size);
            vb.size = size;
        }
        memcpy(vb.data, data, size);

        return vb.buffer;
    }

    // ── LUT texture management ───────────────────────────────────────────
    // Creates a VkImage + VkImageView + staging buffer for a LUT texture.
    // 3D LUT: imageType=3D, width=height=depth=size, format=R32G32B32Sfloat
    // 2D LUT: imageType=2D, width=256, height=1, format=R32G32B32A32Sfloat

    void create_lut_image_3d(vk_lut_texture& tex, uint32_t size)
    {
        auto vk_device = vulkan_->getVkDevice();
        tex.destroy();
        tex.device = vk_device;

        // Use RGBA32F (universally supported for sampling) — RGB data will be padded with alpha=1.0
        vk::DeviceSize byte_size = size * size * size * 4 * sizeof(float);
        tex.data_size = byte_size;

        // Image
        vk::ImageCreateInfo img_info{};
        img_info.imageType     = vk::ImageType::e3D;
        img_info.format        = vk::Format::eR32G32B32A32Sfloat;
        img_info.extent        = vk::Extent3D(size, size, size);
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = vk::SampleCountFlagBits::e1;
        img_info.tiling        = vk::ImageTiling::eOptimal;
        img_info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        img_info.sharingMode   = vk::SharingMode::eExclusive;
        img_info.initialLayout = vk::ImageLayout::eUndefined;
        tex.image = vk_device.createImage(img_info);

        auto mem_req = vk_device.getImageMemoryRequirements(tex.image);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = mem_req.size;
        alloc.memoryTypeIndex = findDedicatedMemoryType(mem_req.memoryTypeBits,
                                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
        tex.memory = vk_device.allocateMemory(alloc);
        vk_device.bindImageMemory(tex.image, tex.memory, 0);

        // View
        vk::ImageViewCreateInfo view_info{};
        view_info.image    = tex.image;
        view_info.viewType = vk::ImageViewType::e3D;
        view_info.format   = vk::Format::eR32G32B32A32Sfloat;
        view_info.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        tex.view = vk_device.createImageView(view_info);

        // Staging buffer
        vk::BufferCreateInfo buf_info{};
        buf_info.size  = byte_size;
        buf_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        tex.staging    = vk_device.createBuffer(buf_info);

        auto buf_req = vk_device.getBufferMemoryRequirements(tex.staging);
        vk::MemoryAllocateInfo buf_alloc{};
        buf_alloc.allocationSize  = buf_req.size;
        buf_alloc.memoryTypeIndex = findDedicatedMemoryType(buf_req.memoryTypeBits,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
        tex.staging_mem = vk_device.allocateMemory(buf_alloc);
        vk_device.bindBufferMemory(tex.staging, tex.staging_mem, 0);
        tex.mapped = vk_device.mapMemory(tex.staging_mem, 0, byte_size);
    }

    void create_lut_image_2d(vk_lut_texture& tex, uint32_t width, vk::Format format, vk::DeviceSize byte_size)
    {
        auto vk_device = vulkan_->getVkDevice();
        tex.destroy();
        tex.device = vk_device;
        tex.data_size = byte_size;

        // Image
        vk::ImageCreateInfo img_info{};
        img_info.imageType     = vk::ImageType::e2D;
        img_info.format        = format;
        img_info.extent        = vk::Extent3D(width, 1, 1);
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = vk::SampleCountFlagBits::e1;
        img_info.tiling        = vk::ImageTiling::eOptimal;
        img_info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        img_info.sharingMode   = vk::SharingMode::eExclusive;
        img_info.initialLayout = vk::ImageLayout::eUndefined;
        tex.image = vk_device.createImage(img_info);

        auto mem_req = vk_device.getImageMemoryRequirements(tex.image);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = mem_req.size;
        alloc.memoryTypeIndex = findDedicatedMemoryType(mem_req.memoryTypeBits,
                                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
        tex.memory = vk_device.allocateMemory(alloc);
        vk_device.bindImageMemory(tex.image, tex.memory, 0);

        // View
        vk::ImageViewCreateInfo view_info{};
        view_info.image    = tex.image;
        view_info.viewType = vk::ImageViewType::e2D;
        view_info.format   = format;
        view_info.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        tex.view = vk_device.createImageView(view_info);

        // Staging buffer
        vk::BufferCreateInfo buf_info{};
        buf_info.size  = byte_size;
        buf_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        tex.staging    = vk_device.createBuffer(buf_info);

        auto buf_req = vk_device.getBufferMemoryRequirements(tex.staging);
        vk::MemoryAllocateInfo buf_alloc{};
        buf_alloc.allocationSize  = buf_req.size;
        buf_alloc.memoryTypeIndex = findDedicatedMemoryType(buf_req.memoryTypeBits,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
        tex.staging_mem = vk_device.allocateMemory(buf_alloc);
        vk_device.bindBufferMemory(tex.staging, tex.staging_mem, 0);
        tex.mapped = vk_device.mapMemory(tex.staging_mem, 0, byte_size);
    }

    // Like create_lut_image_2d but with an explicit height (used for the
    // arbitrary-resolution projection blend mask).
    void create_image_2d_wh(vk_lut_texture& tex, uint32_t width, uint32_t height, vk::Format format,
                            vk::DeviceSize byte_size)
    {
        auto vk_device = vulkan_->getVkDevice();
        tex.destroy();
        tex.device = vk_device;
        tex.data_size = byte_size;

        vk::ImageCreateInfo img_info{};
        img_info.imageType     = vk::ImageType::e2D;
        img_info.format        = format;
        img_info.extent        = vk::Extent3D(width, height, 1);
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = vk::SampleCountFlagBits::e1;
        img_info.tiling        = vk::ImageTiling::eOptimal;
        img_info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        img_info.sharingMode   = vk::SharingMode::eExclusive;
        img_info.initialLayout = vk::ImageLayout::eUndefined;
        tex.image = vk_device.createImage(img_info);

        auto mem_req = vk_device.getImageMemoryRequirements(tex.image);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = mem_req.size;
        alloc.memoryTypeIndex = findDedicatedMemoryType(mem_req.memoryTypeBits,
                                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
        tex.memory = vk_device.allocateMemory(alloc);
        vk_device.bindImageMemory(tex.image, tex.memory, 0);

        vk::ImageViewCreateInfo view_info{};
        view_info.image    = tex.image;
        view_info.viewType = vk::ImageViewType::e2D;
        view_info.format   = format;
        view_info.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        tex.view = vk_device.createImageView(view_info);

        vk::BufferCreateInfo buf_info{};
        buf_info.size  = byte_size;
        buf_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        tex.staging    = vk_device.createBuffer(buf_info);

        auto buf_req = vk_device.getBufferMemoryRequirements(tex.staging);
        vk::MemoryAllocateInfo buf_alloc{};
        buf_alloc.allocationSize  = buf_req.size;
        buf_alloc.memoryTypeIndex = findDedicatedMemoryType(buf_req.memoryTypeBits,
                                                             vk::MemoryPropertyFlagBits::eHostVisible |
                                                             vk::MemoryPropertyFlagBits::eHostCoherent);
        tex.staging_mem = vk_device.allocateMemory(buf_alloc);
        vk_device.bindBufferMemory(tex.staging, tex.staging_mem, 0);
        tex.mapped = vk_device.mapMemory(tex.staging_mem, 0, byte_size);
    }

    void upload_lut_data(vk_lut_texture& tex, const void* data, vk::CommandBuffer cmd,
                         uint32_t width, uint32_t height, uint32_t depth_z)
    {
        if (data)
            memcpy(tex.mapped, data, tex.data_size);

        // Transition: undefined → transfer dst
        vk::ImageMemoryBarrier2 barrier{};
        barrier.oldLayout     = vk::ImageLayout::eUndefined;
        barrier.newLayout     = vk::ImageLayout::eTransferDstOptimal;
        barrier.image         = tex.image;
        barrier.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
        barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;

        vk::DependencyInfo dep{};
        dep.setImageMemoryBarriers(barrier);
        cmd.pipelineBarrier2(dep);

        // Copy staging → image
        vk::BufferImageCopy region{};
        region.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        region.imageExtent      = vk::Extent3D{width, height, depth_z};
        cmd.copyBufferToImage(tex.staging, tex.image, vk::ImageLayout::eTransferDstOptimal, region);

        // Transition: transfer dst → shader read
        barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        cmd.pipelineBarrier2(dep);
    }

    // Pending upload state — set during draw(), executed during commit()
    bool lut3d_upload_pending_     = false;
    uint32_t lut3d_pending_size_   = 0;
    bool hue_curve_upload_pending_ = false;
    bool curve_lut_upload_pending_ = false;
    std::vector<float> curve_lut_pending_data_;
    bool     blend_mask_upload_pending_ = false;
    uint32_t blend_mask_pending_w_      = 0;
    uint32_t blend_mask_pending_h_      = 0;

    /// Prepare LUT textures from draw_params transforms.
    /// Called during draw() — writes staging buffers and sets pending flags.
    void prepare_lut_textures(const draw_params& params)
    {
        const auto& transforms = params.transforms;

        // ── 3D LUT ───────────────────────────────────────────────────────
        const auto& lut = transforms.image_transform.lut3d;
        if (lut && lut->size > 0 && !lut->data.empty()) {
            if (lut.get() != lut3d_data_ptr_) {
                uint32_t sz = static_cast<uint32_t>(lut->size);
                vk::DeviceSize expected = sz * sz * sz * 4 * sizeof(float);
                if (!lut3d_tex_.image || lut3d_tex_.data_size != expected) {
                    create_lut_image_3d(lut3d_tex_, sz);
                }
                // Pad RGB → RGBA (source is size³×3 floats, staging is size³×4 floats)
                auto* dst = static_cast<float*>(lut3d_tex_.mapped);
                const float* src = lut->data.data();
                uint32_t count = sz * sz * sz;
                for (uint32_t i = 0; i < count; ++i) {
                    dst[i * 4 + 0] = src[i * 3 + 0];
                    dst[i * 4 + 1] = src[i * 3 + 1];
                    dst[i * 4 + 2] = src[i * 3 + 2];
                    dst[i * 4 + 3] = 1.0f;
                }
                lut3d_data_ptr_ = lut.get();
                lut3d_upload_pending_ = true;
                lut3d_pending_size_ = sz;
            }
            current_lut_views_.lut3d = lut3d_tex_.view;
        } else {
            current_lut_views_.lut3d = nullptr;
            if (!lut) lut3d_data_ptr_ = nullptr;
        }

        // ── Hue Curves ───────────────────────────────────────────────────
        const auto& hc = transforms.image_transform.hue_curves;
        if (hc && !hc->data.empty()) {
            vk::DeviceSize byte_size = 256 * 4 * sizeof(float);
            if (!hue_curve_tex_.image) {
                create_lut_image_2d(hue_curve_tex_, 256, vk::Format::eR32G32B32A32Sfloat, byte_size);
            }
            memcpy(hue_curve_tex_.mapped, hc->data.data(), byte_size);
            hue_curve_upload_pending_ = true;
            current_lut_views_.hue_curve = hue_curve_tex_.view;
        } else {
            current_lut_views_.hue_curve = nullptr;
        }

        // ── Tone Curves ──────────────────────────────────────────────────
        const auto& cv = transforms.image_transform.curves;
        if (cv.enable) {
            auto lut_r = build_curve_lut(cv.red);
            auto lut_g = build_curve_lut(cv.green);
            auto lut_b = build_curve_lut(cv.blue);
            auto lut_m = build_curve_lut(cv.master);

            curve_lut_pending_data_.resize(256 * 4);
            for (int i = 0; i < 256; ++i) {
                curve_lut_pending_data_[i * 4 + 0] = lut_b[i];
                curve_lut_pending_data_[i * 4 + 1] = lut_g[i];
                curve_lut_pending_data_[i * 4 + 2] = lut_r[i];
                curve_lut_pending_data_[i * 4 + 3] = lut_m[i];
            }

            vk::DeviceSize byte_size = 256 * 4 * sizeof(float);
            if (!curve_lut_tex_.image) {
                create_lut_image_2d(curve_lut_tex_, 256, vk::Format::eR32G32B32A32Sfloat, byte_size);
            }
            memcpy(curve_lut_tex_.mapped, curve_lut_pending_data_.data(), byte_size);
            curve_lut_upload_pending_ = true;
            current_lut_views_.curve_lut = curve_lut_tex_.view;
        } else {
            current_lut_views_.curve_lut = nullptr;
        }

        // ── Blend Mask ───────────────────────────────────────────────────
        const auto& mask = transforms.image_transform.blend_mask;
        if (mask && mask->width > 0 && mask->height > 0 && !mask->data.empty()) {
            if (mask.get() != blend_mask_data_ptr_) {
                uint32_t w = static_cast<uint32_t>(mask->width);
                uint32_t h = static_cast<uint32_t>(mask->height);
                vk::DeviceSize byte_size = static_cast<vk::DeviceSize>(w) * h * 4 * sizeof(float);
                if (!blend_mask_tex_.image || blend_mask_tex_.data_size != byte_size) {
                    create_image_2d_wh(blend_mask_tex_, w, h, vk::Format::eR32G32B32A32Sfloat, byte_size);
                }
                // Pad RGB → RGBA (source is w*h*3 floats, staging is w*h*4 floats)
                auto*        dst   = static_cast<float*>(blend_mask_tex_.mapped);
                const float* src   = mask->data.data();
                uint32_t     count = w * h;
                for (uint32_t i = 0; i < count; ++i) {
                    dst[i * 4 + 0] = src[i * 3 + 0];
                    dst[i * 4 + 1] = src[i * 3 + 1];
                    dst[i * 4 + 2] = src[i * 3 + 2];
                    dst[i * 4 + 3] = 1.0f;
                }
                blend_mask_data_ptr_       = mask.get();
                blend_mask_upload_pending_ = true;
                blend_mask_pending_w_      = w;
                blend_mask_pending_h_      = h;
            }
            current_lut_views_.blend_mask = blend_mask_tex_.view;
        } else {
            current_lut_views_.blend_mask = nullptr;
            if (!mask) blend_mask_data_ptr_ = nullptr;
        }
    }

    /// Record GPU upload commands for any LUTs that were prepared.
    /// Called at the start of commit() before any rendering.
    void do_upload_pending_luts(vk::CommandBuffer cmd)
    {
        if (lut3d_upload_pending_) {
            upload_lut_data(lut3d_tex_, nullptr, cmd, lut3d_pending_size_, lut3d_pending_size_, lut3d_pending_size_);
            lut3d_upload_pending_ = false;
        }
        if (hue_curve_upload_pending_) {
            upload_lut_data(hue_curve_tex_, nullptr, cmd, 256, 1, 1);
            hue_curve_upload_pending_ = false;
        }
        if (curve_lut_upload_pending_) {
            upload_lut_data(curve_lut_tex_, nullptr, cmd, 256, 1, 1);
            curve_lut_upload_pending_ = false;
        }
        if (blend_mask_upload_pending_) {
            upload_lut_data(blend_mask_tex_, nullptr, cmd, blend_mask_pending_w_, blend_mask_pending_h_, 1);
            blend_mask_upload_pending_ = false;
        }
    }
    // ─────────────────────────────────────────────────────────────────────

    std::pair<std::vector<core::frame_geometry::coord>, uniform_block> draw(const draw_params& params)
    {
        CASPAR_ASSERT(params.pix_desc.planes.size() == params.textures.size());

        if (params.textures.empty() || !params.background) {
            return {};
        }

        if (params.transforms.image_transform.opacity < epsilon) {
            return {};
        }

        if (params.geometry.data().empty()) {
            return {};
        }

        auto coords     = params.geometry.data();
        auto transforms = params.transforms;

        auto const first_plane = params.pix_desc.planes.at(0);
        if (params.geometry.mode() != core::frame_geometry::scale_mode::stretch && first_plane.width > 0 &&
            first_plane.height > 0) {
            auto width_scale  = static_cast<double>(params.target_width) / static_cast<double>(first_plane.width);
            auto height_scale = static_cast<double>(params.target_height) / static_cast<double>(first_plane.height);

            core::image_transform transform;
            double                target_scale;
            switch (params.geometry.mode()) {
                case core::frame_geometry::scale_mode::fit:
                    target_scale = std::min(width_scale, height_scale);

                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::fill:
                    target_scale = std::max(width_scale, height_scale);
                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::original:
                    transform.fill_scale[0] /= width_scale;
                    transform.fill_scale[1] /= height_scale;
                    break;

                case core::frame_geometry::scale_mode::hfill:
                    transform.fill_scale[1] *= width_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::vfill:
                    transform.fill_scale[0] *= height_scale / width_scale;
                    break;

                default:;
            }

            transforms = transforms.combine_transform(transform, params.aspect_ratio);
        }

        coords = transforms.transform_coords(coords);

        // Skip drawing if all the coordinates will be outside the screen.
        if (coords.size() < 3 || is_outside_screen(coords)) {
            return {};
        }

        uniform_block uniforms;

        for (int n = 0; n < params.textures.size(); ++n) {
            uniforms.precision_factor[n] = get_precision_factor(params.textures[n]->depth());
        }

        const auto is_hd           = params.pix_desc.planes.at(0).height > 700;
        const auto color_space     = is_hd ? params.pix_desc.color_space : core::color_space::bt601;
        // YCbCr decode: only indices 0-2 (bt601/bt709/bt2020) are valid in the shader arrays.
        // Wide-gamut spaces (P3, Adobe RGB) use BT.709 coefficients as fallback,
        // because if the source had BT.2020 matrix, av_color.h would have returned bt2020 directly.
        uniforms.color_space_index = static_cast<uint32_t>(color_space) > 2u ? 1u : static_cast<uint32_t>(color_space);

        if (params.pix_desc.is_straight_alpha) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_straight_alpha);
        }

        if (static_cast<bool>(params.local_key)) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::has_local_key);
        }
        if (static_cast<bool>(params.layer_key)) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::has_layer_key);
        }
        uniforms.pixel_format = static_cast<uint32_t>(params.pix_desc.format);

        // 8-bit render targets use BGRA output swizzle in the shader to match
        // VK_FORMAT_B8G8R8A8_UNORM import expectation.  16-bit renders RGBA
        // directly since VK_FORMAT_B16G16R16A16_UNORM does not exist in Vulkan.
        if (depth_ == common::bit_depth::bit8) {
            uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::output_bgra);
        }

        uniforms.opacity =
            transforms.image_transform.is_key ? 1.0f : static_cast<float>(transforms.image_transform.opacity);

        if (transforms.image_transform.chroma.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::chroma);

            if (transforms.image_transform.chroma.show_mask)
                uniforms.flags |= static_cast<uint32_t>(shader_flags::chroma_show_mask);

            uniforms.chroma_target_hue     = static_cast<float>(transforms.image_transform.chroma.target_hue) / 360.0f;
            uniforms.chroma_hue_width      = static_cast<float>(transforms.image_transform.chroma.hue_width);
            uniforms.chroma_min_saturation = static_cast<float>(transforms.image_transform.chroma.min_saturation);
            uniforms.chroma_min_brightness = static_cast<float>(transforms.image_transform.chroma.min_brightness);
            uniforms.chroma_softness       = 1.0f + static_cast<float>(transforms.image_transform.chroma.softness);
            uniforms.chroma_spill_suppress =
                static_cast<float>(transforms.image_transform.chroma.spill_suppress) / 360.0f;
            uniforms.chroma_spill_suppress_saturation =
                static_cast<float>(transforms.image_transform.chroma.spill_suppress_saturation);
        }

        // Setup blend_func
        auto blend_mode = params.blend_mode;
        if (transforms.image_transform.is_key) {
            blend_mode = core::blend_mode::normal;
        }

        uniforms.blend_mode = static_cast<uint32_t>(blend_mode);
        uniforms.keyer      = static_cast<uint32_t>(params.keyer);

        if (transforms.image_transform.invert) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::invert);
        }

        if (transforms.image_transform.levels.min_input > epsilon ||
            transforms.image_transform.levels.max_input < 1.0 - epsilon ||
            transforms.image_transform.levels.min_output > epsilon ||
            transforms.image_transform.levels.max_output < 1.0 - epsilon ||
            std::abs(transforms.image_transform.levels.gamma - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::levels);
            uniforms.min_input  = static_cast<float>(transforms.image_transform.levels.min_input);
            uniforms.max_input  = static_cast<float>(transforms.image_transform.levels.max_input);
            uniforms.min_output = static_cast<float>(transforms.image_transform.levels.min_output);
            uniforms.max_output = static_cast<float>(transforms.image_transform.levels.max_output);
            uniforms.gamma      = static_cast<float>(transforms.image_transform.levels.gamma);
        }

        if (std::abs(transforms.image_transform.brightness - 1.0) > epsilon ||
            std::abs(transforms.image_transform.saturation - 1.0) > epsilon ||
            std::abs(transforms.image_transform.contrast - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::csb);

            uniforms.brt = static_cast<float>(transforms.image_transform.brightness);
            uniforms.sat = static_cast<float>(transforms.image_transform.saturation);
            uniforms.con = static_cast<float>(transforms.image_transform.contrast);
        }

        // ── Target size (needed by blur, sharpening, grain) ────────────
        uniforms.target_size[0] = static_cast<float>(params.target_width);
        uniforms.target_size[1] = static_cast<float>(params.target_height);
        uniforms.aspect_ratio   = static_cast<float>(params.aspect_ratio);

        // ── Flip H/V ──────────────────────────────────────────────────
        if (transforms.image_transform.flip_h)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::flip_h);
        if (transforms.image_transform.flip_v)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::flip_v);

        // ── 360° Projection ───────────────────────────────────────────
        if (transforms.image_transform.projection.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_360);
            uniforms.view_yaw       = static_cast<float>(transforms.image_transform.projection.yaw);
            uniforms.view_pitch     = static_cast<float>(transforms.image_transform.projection.pitch);
            uniforms.view_roll      = static_cast<float>(transforms.image_transform.projection.roll);
            uniforms.view_fov       = static_cast<float>(transforms.image_transform.projection.fov);
            uniforms.view_offset_x  = static_cast<float>(transforms.image_transform.projection.offset_x);
            uniforms.view_offset_y  = static_cast<float>(transforms.image_transform.projection.offset_y);
            uniforms.frustum_h      = std::clamp(static_cast<float>(transforms.image_transform.projection.frustum_h), -1.0f, 1.0f);
            uniforms.frustum_v      = std::clamp(static_cast<float>(transforms.image_transform.projection.frustum_v), -1.0f, 1.0f);
            uniforms.lens_k1        = static_cast<float>(transforms.image_transform.projection.lens_k1);
            uniforms.lens_k2        = static_cast<float>(transforms.image_transform.projection.lens_k2);
            uniforms.lens_k3        = static_cast<float>(transforms.image_transform.projection.lens_k3);
            uniforms.lens_p1        = static_cast<float>(transforms.image_transform.projection.lens_p1);
            uniforms.lens_p2        = static_cast<float>(transforms.image_transform.projection.lens_p2);
            uniforms.source_lens    = static_cast<int32_t>(transforms.image_transform.projection.source_lens);
        }

        // ── Curved Screen ─────────────────────────────────────────────
        if (transforms.image_transform.projection.curve_enable)
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_curved);
        uniforms.screen_curve_type = static_cast<int32_t>(transforms.image_transform.projection.curve_type);
        uniforms.screen_arc        = std::clamp(static_cast<float>(transforms.image_transform.projection.screen_arc), -6.2831853f, 6.2831853f);
        uniforms.screen_arc_v      = std::clamp(static_cast<float>(transforms.image_transform.projection.screen_arc_v), -6.2831853f, 6.2831853f);
        uniforms.eye_distance      = std::max(static_cast<float>(transforms.image_transform.projection.eye_distance), 0.05f);

        // ── Edge Blending ─────────────────────────────────────────────
        {
            float ebl = std::clamp(static_cast<float>(transforms.image_transform.projection.edge_blend_left),   0.0f, 1.0f);
            float ebr = std::clamp(static_cast<float>(transforms.image_transform.projection.edge_blend_right),  0.0f, 1.0f);
            float ebt = std::clamp(static_cast<float>(transforms.image_transform.projection.edge_blend_top),    0.0f, 1.0f);
            float ebb = std::clamp(static_cast<float>(transforms.image_transform.projection.edge_blend_bottom), 0.0f, 1.0f);
            if (ebl > epsilon || ebr > epsilon || ebt > epsilon || ebb > epsilon) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::edge_blend);
                uniforms.edge_blend_left   = ebl;
                uniforms.edge_blend_right  = ebr;
                uniforms.edge_blend_top    = ebt;
                uniforms.edge_blend_bottom = ebb;
                uniforms.edge_blend_gamma  = std::clamp(static_cast<float>(transforms.image_transform.projection.edge_blend_gamma), 0.5f, 4.0f);
            }
        }

        // ── Projection blend mask ─────────────────────────────────────
        {
            const auto& mask = transforms.image_transform.blend_mask;
            if (mask && mask->width > 0 && mask->height > 0 && !mask->data.empty()) {
                uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::blend_mask);
            }
        }

        // ── ICVFX inner/outer frustum ─────────────────────────────────
        if (transforms.image_transform.projection.icvfx_enable) {
            const auto& proj = transforms.image_transform.projection;
            uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::icvfx_enable);
            uniforms.inner_yaw       = static_cast<float>(proj.inner_yaw);
            uniforms.inner_pitch     = static_cast<float>(proj.inner_pitch);
            uniforms.inner_roll      = static_cast<float>(proj.inner_roll);
            uniforms.inner_fov       = static_cast<float>(proj.inner_fov);
            uniforms.inner_offset_x  = static_cast<float>(proj.inner_offset_x);
            uniforms.inner_offset_y  = static_cast<float>(proj.inner_offset_y);
            uniforms.icvfx_q0x       = static_cast<float>(proj.icvfx_q0x);
            uniforms.icvfx_q0y       = static_cast<float>(proj.icvfx_q0y);
            uniforms.icvfx_q1x       = static_cast<float>(proj.icvfx_q1x);
            uniforms.icvfx_q1y       = static_cast<float>(proj.icvfx_q1y);
            uniforms.icvfx_q2x       = static_cast<float>(proj.icvfx_q2x);
            uniforms.icvfx_q2y       = static_cast<float>(proj.icvfx_q2y);
            uniforms.icvfx_q3x       = static_cast<float>(proj.icvfx_q3x);
            uniforms.icvfx_q3y       = static_cast<float>(proj.icvfx_q3y);
            uniforms.icvfx_feather   = std::max(static_cast<float>(proj.icvfx_feather), 1e-4f);
            uniforms.icvfx_outer_dim = std::clamp(static_cast<float>(proj.icvfx_outer_dim), 0.0f, 1.0f);
            uniforms.icvfx_inner_dim = std::clamp(static_cast<float>(proj.icvfx_inner_dim), 0.0f, 1.0f);
            uniforms.icvfx_inner_gain_r = std::max(static_cast<float>(proj.icvfx_inner_gain_r), 0.0f);
            uniforms.icvfx_inner_gain_g = std::max(static_cast<float>(proj.icvfx_inner_gain_g), 0.0f);
            uniforms.icvfx_inner_gain_b = std::max(static_cast<float>(proj.icvfx_inner_gain_b), 0.0f);
            uniforms.icvfx_outer_gain_r = std::max(static_cast<float>(proj.icvfx_outer_gain_r), 0.0f);
            uniforms.icvfx_outer_gain_g = std::max(static_cast<float>(proj.icvfx_outer_gain_g), 0.0f);
            uniforms.icvfx_outer_gain_b = std::max(static_cast<float>(proj.icvfx_outer_gain_b), 0.0f);
        }

        // ── Blur ──────────────────────────────────────────────────────
        if (transforms.image_transform.blur.enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::blur_enable);
            uniforms.blur_radius    = static_cast<float>(transforms.image_transform.blur.radius);
            uniforms.blur_type      = static_cast<int32_t>(transforms.image_transform.blur.type);
            uniforms.blur_angle     = static_cast<float>(transforms.image_transform.blur.angle);
            uniforms.blur_center[0] = static_cast<float>(transforms.image_transform.blur.center[0]);
            uniforms.blur_center[1] = static_cast<float>(transforms.image_transform.blur.center[1]);
            uniforms.blur_tilt[0]   = static_cast<float>(transforms.image_transform.blur.tilt_y);
            uniforms.blur_tilt[1]   = static_cast<float>(transforms.image_transform.blur.tilt_h);
        }

        // ── Color Grading ─────────────────────────────────────────────
        {
            static const float k_to_working[7][9] = {
                {0.6131516f,0.3395148f,0.0472947f, 0.0701011f,0.9162792f,0.0136197f, 0.0206177f,0.1095763f,0.8698060f},
                {0.7951281f,0.1643585f,0.0405134f, 0.0234399f,0.9415642f,0.0349959f, 0.0036186f,0.0613513f,0.9350301f},
                {0.8224549f,0.1774521f,-0.0000070f, 0.0332021f,0.9618927f,0.0049052f, 0.0170512f,0.0723025f,0.9106463f},
                {1.4514393f,-0.2365107f,-0.2149286f, -0.0765538f,1.1762297f,-0.0996759f, 0.0083161f,-0.0060324f,0.9977163f},
                {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f},
                {0.6954522f,0.1446577f,0.1598901f, 0.0439823f,0.8591788f,0.0968389f, -0.0055023f,0.0040678f,1.0014345f},
                {0.7112957f,0.1903613f,0.0983436f, 0.0406952f,0.8550396f,0.1042651f, -0.0025079f,0.0085993f,0.9939086f}
            };
            static const float k_to_output[7][9] = {
                {1.7050585f,-0.6217876f,-0.0832709f, -0.1302597f,1.1407927f,-0.0105330f, -0.0240003f,-0.1289711f,1.1529714f},
                {1.2746843f,-0.2692490f,-0.0054353f, -0.0293524f,1.0763680f,-0.0470156f, -0.0160993f,-0.0606079f,1.0767072f},
                {1.2239840f,-0.2239840f,0.0000000f, -0.0421197f,1.0421197f,0.0000000f, -0.0196576f,-0.0787093f,1.0983669f},
                {0.6954522f,0.1406787f,0.1638691f, 0.0447946f,0.8596711f,0.0955343f, -0.0055259f,0.0040252f,1.0015007f},
                {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f, 0.0f,0.0f,1.0f},
                {1.4516608f,-0.2434265f,-0.2082343f, -0.0752455f,1.1770530f,-0.1018075f, 0.0082817f,-0.0061186f,0.9978370f},
                {1.4235761f,-0.3158537f,-0.1077233f, -0.0682645f,1.1859178f,-0.1176531f, 0.0041827f,-0.0110575f,1.0068749f}
            };

            // Helper: expand row-major mat3 (9 floats) to 3×vec4 columns (12 floats, std140).
            // GLSL mat3(c0,c1,c2) treats arguments as COLUMNS.  The C++ tables store
            // matrices in row-major order, so we must transpose when packing for the shader.
            auto set_mat3 = [](float dst[12], const float src[9]) {
                dst[0]=src[0]; dst[1]=src[3]; dst[2]=src[6]; dst[3]=0;  // column 0
                dst[4]=src[1]; dst[5]=src[4]; dst[6]=src[7]; dst[7]=0;  // column 1
                dst[8]=src[2]; dst[9]=src[5]; dst[10]=src[8]; dst[11]=0; // column 2
            };

            const auto& cg = transforms.image_transform.color_grade;
            if (cg.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                uniforms.input_transfer  = cg.input_transfer;
                uniforms.output_transfer = cg.output_transfer;
                uniforms.tone_mapping_op = cg.tone_mapping;
                uniforms.display_peak_luminance = params.display_peak_luminance;
                // Combine user exposure with BT.2408 luminance adaptation.
                // For PQ (absolute): simple ratio 100/10000.
                // For HLG (scene-referred, OOTF γ=1.2): SDR white at 75% HLG
                // signal per BT.2408 §3.2 → scene-linear factor 0.265.
                auto get_luminance_scale = [](int src_t, int tgt_t) -> float {
                    constexpr float k_sdr_hlg = 0.265f;
                    bool src_sdr = (src_t <= 2);
                    bool tgt_sdr = (tgt_t <= 2);
                    bool src_hlg = (src_t == 4);
                    bool tgt_hlg = (tgt_t == 4);
                    bool src_pq  = (src_t == 3);
                    bool tgt_pq  = (tgt_t == 3);
                    if (src_sdr && tgt_hlg) return k_sdr_hlg;
                    if (src_hlg && tgt_sdr) return 1.0f / k_sdr_hlg;
                    if (src_sdr && tgt_pq)  return 0.01f;
                    if (src_pq  && tgt_sdr) return 100.0f;
                    if (src_hlg && tgt_pq)  return 0.1f;
                    if (src_pq  && tgt_hlg) return 10.0f;
                    return 1.0f;
                };
                float lum_scale = get_luminance_scale(cg.input_transfer, cg.output_transfer);
                uniforms.exposure = static_cast<float>(cg.exposure) * lum_scale;
                int ig = std::min(std::max(cg.input_gamut,  0), 6);
                int og = std::min(std::max(cg.output_gamut, 0), 6);

                // When no artistic tone mapping is applied and both gamuts are D65-based
                // (BT.709=0, BT.2020=1), use direct ITU-R BT.2087 matrices to avoid
                // chromatic adaptation artifacts from the ACEScg (D60) intermediate.
                static const float k_direct_cg[2][2][9] = {
                    { // from bt709
                        {1,0,0, 0,1,0, 0,0,1}, // → bt709 (identity)
                        {0.6274039f,0.3292830f,0.0433131f, 0.0690972f,0.9195404f,0.0113623f, 0.0163914f,0.0880133f,0.8955953f}, // → bt2020
                    },
                    { // from bt2020
                        {1.6604910f,-0.5876411f,-0.0728499f, -0.1245505f,1.1328999f,-0.0083494f, -0.0181508f,-0.1005789f,1.1187297f}, // → bt709
                        {1,0,0, 0,1,0, 0,0,1}, // → bt2020 (identity)
                    },
                };
                static const float k_identity_cg[9] = {1,0,0, 0,1,0, 0,0,1};

                if (cg.tone_mapping == 0 && ig <= 1 && og <= 1) {
                    // Direct D65↔D65 conversion — no ACEScg intermediate needed
                    set_mat3(uniforms.input_to_working,  k_direct_cg[ig][og]);
                    set_mat3(uniforms.working_to_output, k_identity_cg);
                } else {
                    // Full ACES grading pipeline through ACEScg working space
                    set_mat3(uniforms.input_to_working,  k_to_working[ig]);
                    set_mat3(uniforms.working_to_output, k_to_output[og]);
                }
            } else if (params.auto_color_convert &&
                       (params.pix_desc.color_space != params.target_color_space ||
                        params.pix_desc.color_transfer != params.target_color_transfer)) {
                // Auto color conversion: source differs from channel output.
                // Gamut indices for the k_direct matrix (0=bt709, 1=bt2020, 2=p3_d65, 3=p3_dci, 4=adobe_rgb)
                auto gamut_index = [](core::color_space cs) -> int {
                    switch (cs) {
                        case core::color_space::bt2020:    return 1;
                        case core::color_space::p3_d65:   return 2;
                        case core::color_space::p3_dci:   return 3;
                        case core::color_space::adobe_rgb:return 4;
                        default:                          return 0; // bt601/bt709 → index 0
                    }
                };
                // EOTF indices (apply_eotf): 1=srgb,2=rec709,3=pq,4=hlg,5=logc3,6=slog3,7=linear,8=gamma24,9=gamma26
                auto eotf_index = [](core::color_transfer ct) -> int {
                    switch (ct) {
                        case core::color_transfer::pq:      return 3;
                        case core::color_transfer::hlg:     return 4;
                        case core::color_transfer::linear:  return 7;
                        case core::color_transfer::gamma24: return 8;
                        case core::color_transfer::gamma26: return 9;
                        default:                            return 2; // sdr → rec709 (BT.1886)
                    }
                };
                // OETF indices (apply_oetf): 1=srgb,2=rec709,3=pq,4=hlg,5=linear,6=gamma24,7=gamma26
                auto oetf_index = [](core::color_transfer ct) -> int {
                    switch (ct) {
                        case core::color_transfer::pq:      return 3;
                        case core::color_transfer::hlg:     return 4;
                        case core::color_transfer::linear:  return 5;
                        case core::color_transfer::gamma24: return 6;
                        case core::color_transfer::gamma26: return 7;
                        default:                            return 2; // sdr → rec709 (BT.1886)
                    }
                };
                int ig = gamut_index(params.pix_desc.color_space);
                int og = gamut_index(params.target_color_space);
                // Skip if the mapped indices are identical (e.g. bt601 source on bt709 channel)
                if (ig != og || params.pix_desc.color_transfer != params.target_color_transfer) {
                    int it = eotf_index(params.pix_desc.color_transfer);
                    int ot = oetf_index(params.target_color_transfer);
                    // Use channel's configured auto tone-map operator (default: hard clamp).
                    int tm = params.auto_tone_map;
                    uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                    uniforms.input_transfer  = it;
                    uniforms.output_transfer = ot;
                    uniforms.tone_mapping_op = tm;
                    uniforms.display_peak_luminance = params.display_peak_luminance;

                    // BT.2408 luminance adaptation: scene-referred mapping
                    // for SDR↔HLG conversions (75% signal for ref white).
                    // Note: src_t uses EOTF indices, tgt_t uses OETF indices.
                    // Linear/gamma24/gamma26 are treated as SDR-level for luminance.
                    auto get_luminance_scale = [](int src_t, int tgt_t) -> float {
                        constexpr float k_sdr_hlg = 0.265f;
                        bool src_sdr = (src_t <= 2 || src_t >= 7); // rec709/srgb or linear/gamma24/gamma26
                        bool tgt_sdr = (tgt_t <= 2 || tgt_t >= 5); // rec709/srgb or linear/gamma24/gamma26
                        bool src_hlg = (src_t == 4);
                        bool tgt_hlg = (tgt_t == 4);
                        bool src_pq  = (src_t == 3);
                        bool tgt_pq  = (tgt_t == 3);
                        if (src_sdr && tgt_hlg) return k_sdr_hlg;
                        if (src_hlg && tgt_sdr) return 1.0f / k_sdr_hlg;
                        if (src_sdr && tgt_pq)  return 0.01f;
                        if (src_pq  && tgt_sdr) return 100.0f;
                        if (src_hlg && tgt_pq)  return 0.1f;
                        if (src_pq  && tgt_hlg) return 10.0f;
                        return 1.0f;
                    };
                    uniforms.exposure = get_luminance_scale(it, ot);

                    // Direct gamut matrices for auto conversion.
                    // 5 gamuts: 0=bt709, 1=bt2020, 2=p3_d65, 3=p3_dci, 4=adobe_rgb
                    // All D65-based pairs use ITU-R BT.2087 style direct matrices.
                    // P3-DCI (D50-ish white) uses Bradford-adapted matrices.
                    // Row-major 3×3 stored as 9 floats.
                    static const float k_direct[5][5][9] = {
                        { // from bt709
                            {1,0,0, 0,1,0, 0,0,1}, // → bt709
                            {0.6274039f,0.3292830f,0.0433131f, 0.0690972f,0.9195404f,0.0113623f, 0.0163914f,0.0880133f,0.8955953f}, // → bt2020
                            {0.8224620f,0.1775380f,0.0000000f, 0.0331942f,0.9668058f,0.0000000f, 0.0170826f,0.0723974f,0.9105200f}, // → p3_d65
                            {0.8685170f,0.1283810f,0.0031015f, 0.0344530f,0.9618840f,0.0036629f, 0.0167662f,0.0710578f,0.9121760f}, // → p3_dci
                            {0.7151583f,0.2848417f,0.0000000f, 0.0000000f,1.0000000f,0.0000000f, 0.0000000f,0.0411539f,0.9588461f}, // → adobe_rgb
                        },
                        { // from bt2020
                            {1.6604910f,-0.5876411f,-0.0728499f, -0.1245505f,1.1328999f,-0.0083494f, -0.0181508f,-0.1005789f,1.1187297f}, // → bt709
                            {1,0,0, 0,1,0, 0,0,1}, // → bt2020
                            {1.3434780f,-0.2820610f,-0.0614170f, -0.0652590f,1.0757410f,-0.0104820f, -0.0028170f,-0.0195750f,1.0223920f}, // → p3_d65
                            {1.3807920f,-0.3315990f,-0.0491930f, -0.0586050f,1.0680060f,-0.0094010f, -0.0029440f,-0.0196200f,1.0225640f}, // → p3_dci
                            {1.1512080f,-0.1512080f,0.0000000f, -0.1136290f,1.1136290f,0.0000000f, -0.0175840f,-0.0563370f,1.0739210f}, // → adobe_rgb
                        },
                        { // from p3_d65
                            {1.2249401f,-0.2249401f,0.0000000f, -0.0420489f,1.0420489f,0.0000000f, -0.0196376f,-0.0786498f,1.0982874f}, // → bt709
                            {0.7538780f,0.1986920f,0.0474300f, 0.0457440f,0.9413590f,0.0128970f, 0.0011430f,0.0156030f,0.9832540f}, // → bt2020
                            {1,0,0, 0,1,0, 0,0,1}, // → p3_d65
                            {1.0342270f,-0.0318520f,0.0000000f, 0.0012440f,0.9987560f,0.0000000f, -0.0002610f,0.0000000f,1.0028360f}, // → p3_dci (approx, Bradford)
                            {0.8681240f,0.1318760f,0.0000000f, -0.0339850f,1.0339850f,0.0000000f, -0.0178610f,0.0279920f,0.9898690f}, // → adobe_rgb
                        },
                        { // from p3_dci
                            {1.1827920f,-0.1831980f,0.0004060f, -0.0418540f,1.0418540f,0.0000000f, -0.0183830f,-0.0778470f,1.0962300f}, // → bt709
                            {0.7325850f,0.2146490f,0.0527660f, 0.0441100f,0.9427470f,0.0131430f, 0.0013630f,0.0156860f,0.9829510f}, // → bt2020
                            {0.9668230f,0.0307660f,0.0024110f, -0.0012100f,1.0012100f,0.0000000f, 0.0002530f,0.0000000f,0.9997470f}, // → p3_d65 (approx, Bradford)
                            {1,0,0, 0,1,0, 0,0,1}, // → p3_dci
                            {0.8399020f,0.1599740f,0.0001240f, -0.0340460f,1.0340460f,0.0000000f, -0.0176370f,0.0280000f,0.9896370f}, // → adobe_rgb
                        },
                        { // from adobe_rgb
                            {1.3982450f,-0.3982450f,0.0000000f, 0.0000000f,1.0000000f,0.0000000f, 0.0000000f,-0.0429180f,1.0429180f}, // → bt709
                            {0.8737660f,0.2049790f,0.0000000f, 0.0968570f,0.9031430f,0.0000000f, 0.0180060f,0.0824960f,0.8994980f}, // → bt2020 (via inv(adobe)×bt2020)
                            {1.1519280f,-0.1519280f,0.0000000f, 0.0391260f,0.9608740f,0.0000000f, 0.0205670f,0.0523530f,0.9270800f}, // → p3_d65
                            {1.1907120f,-0.1877990f,0.0000000f, 0.0405500f,0.9594500f,0.0000000f, 0.0197940f,0.0511990f,0.9290070f}, // → p3_dci (approx, Bradford)
                            {1,0,0, 0,1,0, 0,0,1}, // → adobe_rgb
                        },
                    };
                    static const float k_identity[9] = {1,0,0, 0,1,0, 0,0,1};
                    set_mat3(uniforms.input_to_working,  k_direct[ig][og]);
                    set_mat3(uniforms.working_to_output, k_identity);
                }
            }
        }

        // ── White Balance ─────────────────────────────────────────────
        if (std::abs(transforms.image_transform.temperature) > epsilon ||
            std::abs(transforms.image_transform.tint) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::white_balance);
            uniforms.wb_temperature = static_cast<float>(transforms.image_transform.temperature);
            uniforms.wb_tint        = static_cast<float>(transforms.image_transform.tint);
        }

        // ── Lift / Midtone / Gain ─────────────────────────────────────
        {
            const auto& lift    = transforms.image_transform.lift;
            const auto& midtone = transforms.image_transform.midtone;
            const auto& gain    = transforms.image_transform.gain;
            bool lmg_active =
                std::abs(lift[0]) > epsilon || std::abs(lift[1]) > epsilon || std::abs(lift[2]) > epsilon ||
                std::abs(midtone[0]-1.0) > epsilon || std::abs(midtone[1]-1.0) > epsilon || std::abs(midtone[2]-1.0) > epsilon ||
                std::abs(gain[0]-1.0) > epsilon || std::abs(gain[1]-1.0) > epsilon || std::abs(gain[2]-1.0) > epsilon;
            if (lmg_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::lmg_enable);
                uniforms.lmg_lift[0] = static_cast<float>(lift[0]);
                uniforms.lmg_lift[1] = static_cast<float>(lift[1]);
                uniforms.lmg_lift[2] = static_cast<float>(lift[2]);
                uniforms.lmg_midtone[0] = static_cast<float>(midtone[0]);
                uniforms.lmg_midtone[1] = static_cast<float>(midtone[1]);
                uniforms.lmg_midtone[2] = static_cast<float>(midtone[2]);
                uniforms.lmg_gain[0] = static_cast<float>(gain[0]);
                uniforms.lmg_gain[1] = static_cast<float>(gain[1]);
                uniforms.lmg_gain[2] = static_cast<float>(gain[2]);
            }
        }

        // ── Hue Shift ─────────────────────────────────────────────────
        if (std::abs(transforms.image_transform.hue_shift) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::hue_shift_enable);
            uniforms.hue_shift_degrees = static_cast<float>(transforms.image_transform.hue_shift);
        }

        // ── Tonal Balance ─────────────────────────────────────────────
        if (std::abs(transforms.image_transform.shadows) > epsilon ||
            std::abs(transforms.image_transform.highlights) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::tonebalance_enable);
            uniforms.tb_shadows    = static_cast<float>(transforms.image_transform.shadows);
            uniforms.tb_highlights = static_cast<float>(transforms.image_transform.highlights);
        }

        // ── Linear Saturation ─────────────────────────────────────────
        if (std::abs(transforms.image_transform.linear_saturation - 1.0) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::linear_sat_enable);
            uniforms.linear_sat_value = static_cast<float>(transforms.image_transform.linear_saturation);
        }

        // ── ASC CDL ───────────────────────────────────────────────────
        {
            const auto& s  = transforms.image_transform.cdl_slope;
            const auto& o  = transforms.image_transform.cdl_offset;
            const auto& p  = transforms.image_transform.cdl_power;
            double      cs = transforms.image_transform.cdl_saturation;
            bool cdl_active =
                std::abs(s[0]-1.0) > epsilon || std::abs(s[1]-1.0) > epsilon || std::abs(s[2]-1.0) > epsilon ||
                std::abs(o[0]) > epsilon     || std::abs(o[1]) > epsilon     || std::abs(o[2]) > epsilon     ||
                std::abs(p[0]-1.0) > epsilon || std::abs(p[1]-1.0) > epsilon || std::abs(p[2]-1.0) > epsilon ||
                std::abs(cs-1.0) > epsilon;
            if (cdl_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::cdl_enable);
                uniforms.cdl_slope[0] = static_cast<float>(s[0]);
                uniforms.cdl_slope[1] = static_cast<float>(s[1]);
                uniforms.cdl_slope[2] = static_cast<float>(s[2]);
                uniforms.cdl_saturation = static_cast<float>(cs);
                uniforms.cdl_offset[0] = static_cast<float>(o[0]);
                uniforms.cdl_offset[1] = static_cast<float>(o[1]);
                uniforms.cdl_offset[2] = static_cast<float>(o[2]);
                uniforms.cdl_power[0] = static_cast<float>(p[0]);
                uniforms.cdl_power[1] = static_cast<float>(p[1]);
                uniforms.cdl_power[2] = static_cast<float>(p[2]);
            }
        }

        // ── Split Toning ──────────────────────────────────────────────
        {
            const auto& sc = transforms.image_transform.split_shadow_color;
            const auto& hc = transforms.image_transform.split_highlight_color;
            bool split_active =
                std::abs(sc[0]) > epsilon || std::abs(sc[1]) > epsilon || std::abs(sc[2]) > epsilon ||
                std::abs(hc[0]) > epsilon || std::abs(hc[1]) > epsilon || std::abs(hc[2]) > epsilon;
            if (split_active) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::split_tone_enable);
                uniforms.split_shadow_color[0] = static_cast<float>(sc[0]);
                uniforms.split_shadow_color[1] = static_cast<float>(sc[1]);
                uniforms.split_shadow_color[2] = static_cast<float>(sc[2]);
                uniforms.split_balance = static_cast<float>(transforms.image_transform.split_balance);
                uniforms.split_highlight_color[0] = static_cast<float>(hc[0]);
                uniforms.split_highlight_color[1] = static_cast<float>(hc[1]);
                uniforms.split_highlight_color[2] = static_cast<float>(hc[2]);
            }
        }

        // ── Gamut Compression ─────────────────────────────────────────
        if (transforms.image_transform.gamut_compress) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::gamut_compress);
            // RGBA order: .r=Red(cyan limit), .g=Green(magenta limit), .b=Blue(yellow limit)
            uniforms.gc_limit[0] = static_cast<float>(transforms.image_transform.gc_cyan);
            uniforms.gc_limit[1] = static_cast<float>(transforms.image_transform.gc_magenta);
            uniforms.gc_limit[2] = static_cast<float>(transforms.image_transform.gc_yellow);
        }

        // ── 3D LUT ───────────────────────────────────────────────────
        {
            const auto& lut = transforms.image_transform.lut3d;
            if (lut && lut->size > 0 && !lut->data.empty()) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::lut3d_enable);
                uniforms.lut3d_strength = static_cast<float>(transforms.image_transform.lut3d_strength);
            }
        }

        // ── Hue Curves ───────────────────────────────────────────────
        {
            const auto& hc = transforms.image_transform.hue_curves;
            if (hc && !hc->data.empty()) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::hue_curve_enable);
            }
        }

        // ── Sharpening ───────────────────────────────────────────────
        if (std::abs(transforms.image_transform.sharpen_amount) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::sharpen_enable);
            uniforms.sharpen_amount = static_cast<float>(transforms.image_transform.sharpen_amount);
            uniforms.sharpen_radius = static_cast<float>(transforms.image_transform.sharpen_radius);
        }

        // ── Film Grain ───────────────────────────────────────────────
        if (std::abs(transforms.image_transform.grain_intensity) > epsilon) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::grain_enable);
            uniforms.grain_intensity = static_cast<float>(transforms.image_transform.grain_intensity);
            uniforms.grain_size      = static_cast<float>(transforms.image_transform.grain_size);
            uniforms.grain_frame     = frame_counter_++;
        }

        // ── Secondary Qualifier ──────────────────────────────────────
        if (transforms.image_transform.qualifier_enable) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::qualifier_enable);
            uniforms.qual_target_hue = static_cast<float>(transforms.image_transform.qual_target_hue);
            uniforms.qual_hue_width  = static_cast<float>(transforms.image_transform.qual_hue_width);
            uniforms.qual_min_sat    = static_cast<float>(transforms.image_transform.qual_min_sat);
            uniforms.qual_max_sat    = static_cast<float>(transforms.image_transform.qual_max_sat);
            uniforms.qual_min_lum    = static_cast<float>(transforms.image_transform.qual_min_lum);
            uniforms.qual_max_lum    = static_cast<float>(transforms.image_transform.qual_max_lum);
            uniforms.qual_softness   = static_cast<float>(transforms.image_transform.qual_softness);
            uniforms.qual_exposure   = static_cast<float>(transforms.image_transform.qual_exposure);
            uniforms.qual_sat_offset = static_cast<float>(transforms.image_transform.qual_sat_offset);
            uniforms.qual_hue_offset = static_cast<float>(transforms.image_transform.qual_hue_offset);
        }

        // ── Per-Channel RGB Levels ───────────────────────────────────
        {
            const auto& rl = transforms.image_transform.per_channel_levels;
            if (rl.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::rgb_levels_enable);
                uniforms.rgb_levels_min_input[0]  = static_cast<float>(rl.r.min_input);
                uniforms.rgb_levels_min_input[1]  = static_cast<float>(rl.g.min_input);
                uniforms.rgb_levels_min_input[2]  = static_cast<float>(rl.b.min_input);
                uniforms.rgb_levels_max_input[0]  = static_cast<float>(rl.r.max_input);
                uniforms.rgb_levels_max_input[1]  = static_cast<float>(rl.g.max_input);
                uniforms.rgb_levels_max_input[2]  = static_cast<float>(rl.b.max_input);
                uniforms.rgb_levels_gamma[0]      = static_cast<float>(rl.r.gamma);
                uniforms.rgb_levels_gamma[1]      = static_cast<float>(rl.g.gamma);
                uniforms.rgb_levels_gamma[2]      = static_cast<float>(rl.b.gamma);
                uniforms.rgb_levels_min_output[0] = static_cast<float>(rl.r.min_output);
                uniforms.rgb_levels_min_output[1] = static_cast<float>(rl.g.min_output);
                uniforms.rgb_levels_min_output[2] = static_cast<float>(rl.b.min_output);
                uniforms.rgb_levels_max_output[0] = static_cast<float>(rl.r.max_output);
                uniforms.rgb_levels_max_output[1] = static_cast<float>(rl.g.max_output);
                uniforms.rgb_levels_max_output[2] = static_cast<float>(rl.b.max_output);
            }
        }

        // ── Tone Curves ──────────────────────────────────────────────
        {
            const auto& cv = transforms.image_transform.curves;
            if (cv.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::curves_enable);
            }
        }

        // ── Shape Overlay ────────────────────────────────────────────
        {
            const auto& sh = transforms.image_transform.shape;
            if (sh.enable) {
                uniforms.flags |= static_cast<uint32_t>(shader_flags::shape_enable);
                uniforms.shape_type      = static_cast<int32_t>(sh.type);
                uniforms.shape_fill_type = static_cast<int32_t>(sh.fill_type);
                uniforms.shape_center[0] = static_cast<float>(sh.center[0]);
                uniforms.shape_center[1] = static_cast<float>(sh.center[1]);
                uniforms.shape_size[0]   = static_cast<float>(sh.size[0]);
                uniforms.shape_size[1]   = static_cast<float>(sh.size[1]);
                uniforms.shape_corner_radius = static_cast<float>(sh.corner_radius);
                uniforms.shape_softness      = static_cast<float>(sh.edge_softness);
                uniforms.shape_color1[0] = static_cast<float>(sh.color1[0]);
                uniforms.shape_color1[1] = static_cast<float>(sh.color1[1]);
                uniforms.shape_color1[2] = static_cast<float>(sh.color1[2]);
                uniforms.shape_color1[3] = static_cast<float>(sh.color1[3]);
                uniforms.shape_color2[0] = static_cast<float>(sh.color2[0]);
                uniforms.shape_color2[1] = static_cast<float>(sh.color2[1]);
                uniforms.shape_color2[2] = static_cast<float>(sh.color2[2]);
                uniforms.shape_color2[3] = static_cast<float>(sh.color2[3]);
                uniforms.shape_gradient_angle     = static_cast<float>(sh.gradient_angle);
                uniforms.shape_gradient_center[0] = static_cast<float>(sh.gradient_center[0]);
                uniforms.shape_gradient_center[1] = static_cast<float>(sh.gradient_center[1]);
                if (sh.stroke_enable) {
                    uniforms.flags |= static_cast<uint32_t>(shader_flags::shape_stroke);
                    uniforms.shape_stroke_width    = static_cast<float>(sh.stroke_width);
                    uniforms.shape_stroke_color[0] = static_cast<float>(sh.stroke_color[0]);
                    uniforms.shape_stroke_color[1] = static_cast<float>(sh.stroke_color[1]);
                    uniforms.shape_stroke_color[2] = static_cast<float>(sh.stroke_color[2]);
                    uniforms.shape_stroke_color[3] = static_cast<float>(sh.stroke_color[3]);
                }
            }
        }

        // Prepare LUT texture data in staging buffers (uploaded at commit time)
        prepare_lut_textures(params);

        return {std::move(coords), uniforms};
    }
};

image_kernel::image_kernel(const spl::shared_ptr<device>& device, common::bit_depth depth)
    : impl_(new impl(device, depth))
{
}
image_kernel::~image_kernel() {}

spl::shared_ptr<renderpass> image_kernel::create_renderpass(uint32_t width, uint32_t height)
{
    return impl_->create_renderpass(width, height);
}

} // namespace caspar::accelerator::vulkan
