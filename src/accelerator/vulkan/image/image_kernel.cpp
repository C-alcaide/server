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
#include "../util/glsl_compiler.h"
#include "../util/pipeline.h"
#include "../util/platform_config.h"
#include "../util/renderpass.h"
#include "../util/texture.h"

// The mixer's own fragment shader as GLSL text, for splicing a generated transform into.
// The SPIR-V glslc built at configure time is what every other draw uses and is unchanged.
#include "vk_image_fragment_src.h"

#include <accelerator/ocio/ocio_config.h>
#include <common/assert.h>
#include <common/log.h>
#include <common/utf.h>

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
#include <cstring>
#include <map>
#include <string>
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
        // `n - 1`, not `n - 2` — same fix as the OpenGL kernel, which carries the same
        // duplicated builder. See the longer note there; in short, the last of the
        // n-1 intervals was unreachable, so `seg` stayed 0 and the end of every
        // three-point-or-longer curve was evaluated with the first segment's data.
        int seg = 0;
        for (int i = 0; i < n - 1; ++i)
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
    // False right after create_lut_image_*() — the image is genuinely
    // eUndefined then (its actual initialLayout). Set true after the first
    // upload_lut_data() call, once it's actually in eShaderReadOnlyOptimal.
    bool             ever_uploaded = false;

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
    // The format of every attachment this kernel allocates, and so also which pipeline it
    // binds. Independent of depth_, which stays the channel's output depth.
    common::render_format   render_format_ = common::render_format::unorm;
    int32_t                 frame_counter_ = 0;

    // ── Persistent LUT textures ──────────────────────────────────────────
    vk_lut_texture          lut3d_tex_{};
    const core::lut3d_data* lut3d_data_ptr_ = nullptr;  // tracks which data is uploaded

    vk_lut_texture          hue_curve_tex_{};

    vk_lut_texture          curve_lut_tex_{};

    vk_lut_texture               blend_mask_tex_{};
    const core::blend_mask_data* blend_mask_data_ptr_ = nullptr;  // tracks which data is uploaded

    lut_views               current_lut_views_{};

    // ── OCIO input-transform variants ────────────────────────────────────
    //
    // One entry per colour space in use, keyed by OCIO's processor cache ID -- the same key
    // the OGL kernel uses, and safe to share the concept with because the cache ID leads
    // with the shading language, so the GLSL 4.0 and Vulkan builds of one colour space are
    // different entries by construction.
    //
    // Holds the LUT images and the pipeline that samples them. The pipeline itself is owned
    // by the variant pipeline cache (device::get_variant_pipeline) and merely referenced
    // here, because a Vulkan pipeline is bound to its shader module and outlives any one
    // kernel; the images are this device's and are destroyed with the kernel.
    /// One generated-transform LUT: its image, and the extent to copy into it. The extent is
    /// kept here because vk_lut_texture carries only a byte count, and a 4096x17 image and a
    /// 69632x1 one are the same number of bytes.
    struct ocio_lut
    {
        vk_lut_texture tex{};
        uint32_t       width   = 0;
        uint32_t       height  = 1;
        uint32_t       depth   = 1;
        bool           nearest = false; ///< OCIO asked for INTERP_NEAREST on this table
    };

    struct ocio_variant
    {
        std::vector<ocio_lut> textures;
        /// The views again, indexed by binding-1 so they can be handed to the pipeline
        /// without it having to know which slot came from which LUT.
        ocio_texture_views views{};
        /// Which of those need point sampling, same indexing. Carried alongside rather than
        /// inferred from the image, because nothing about a 363x1 R32_SFLOAT says whether
        /// interpolating it is meaningful.
        ocio_texture_filters nearest{};
        /// The pipeline whose fragment shader contains this transform's spliced source.
        /// Null only when the variant failed.
        std::shared_ptr<class pipeline> pipeline;
        bool failed = false; ///< do not retry a broken transform every frame
    };
    std::map<std::string, ocio_variant> ocio_variants_;

    /// Variants whose staging buffers are filled but whose images have not been copied into
    /// yet. Drained by do_upload_pending_luts() inside the frame's command buffer, because
    /// that is the only command buffer ordered against the draw that samples them.
    std::vector<ocio_variant*> ocio_upload_queue_;

    /// The pipeline the layer currently being prepared needs, or null for the base one.
    ///
    /// Set alongside current_lut_views_ and read by the renderpass immediately afterwards,
    /// for the same reason: draw() prepares one layer at a time, and both are properties of
    /// that layer rather than of the pass. A pass composites layers with different colour
    /// spaces, so a pipeline chosen once for the whole pass would apply one layer's transform
    /// to all of them.
    std::shared_ptr<class pipeline> current_ocio_pipeline_;
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
        virtual std::shared_ptr<class pipeline> get_layer_pipeline() const override
        {
            return parent->current_ocio_pipeline_;
        }
        virtual void upload_pending_luts(vk::CommandBuffer cmd) override { parent->do_upload_pending_luts(cmd); }
        virtual std::shared_ptr<class pipeline> get_pipeline()
        {
            return parent->vulkan_->get_pipeline(parent->depth_, parent->render_format_);
        }
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
            return create_attachment_as(width, height, components_count, parent->render_format_);
        }

        virtual std::shared_ptr<class texture>
        create_attachment_as(uint32_t width, uint32_t height, uint32_t components_count, common::render_format format)
        {
            // Reuse an attachment texture from a previous frame on this slot
            // if the consumer has released its reference (use_count == 1 means
            // only our pool holds it).  This keeps the underlying VkDeviceMemory
            // and its exported Win32 HANDLE stable, which in turn keeps any
            // CUDA import on the consumer side valid — avoiding the extremely
            // expensive cudaImportExternalMemory call (~10-150 ms) every frame.
            for (auto& tex : attachment_pool_) {
                // Format, not just dimensions: a VkImage's format is fixed at creation, so
                // matching on size alone would hand a caller asking for the resolve
                // target's unorm image back an fp16 one from the working space (or the
                // reverse). Same hazard as the device attachment pool.
                if (tex && tex.use_count() == 1 &&
                    static_cast<uint32_t>(tex->width()) == width &&
                    static_cast<uint32_t>(tex->height()) == height && tex->format() == format) {
                    // This texture was left in whatever layout the previous frame's
                    // last use put it in (e.g. eShaderReadOnlyOptimal after being
                    // sampled, or eTransferSrcOptimal after a readback) — it must be
                    // transitioned back to eRenderingLocalRead before the render pass
                    // declares that layout for it. create_attachment() below does this
                    // for a freshly-created/device-pooled texture; this cache bypasses
                    // that call entirely, so it must do the transition itself.
                    parent->vulkan_->reset_attachment_layout(tex);
                    return tex;
                }
            }
            auto tex =
                parent->vulkan_->create_attachment(width, height, parent->depth_, components_count, format);
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

    explicit impl(const spl::shared_ptr<device>& vulkan,
                  common::bit_depth               depth,
                  common::render_format           render_format)
        : vulkan_(vulkan)
        , depth_(depth)
        , render_format_(render_format)
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

        for (auto& [id, variant] : ocio_variants_) {
            for (auto& lut : variant.textures)
                lut.tex.destroy();
        }

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
        // tex may still be sampled by a command buffer from one of the other
        // frames-in-flight (up to frame_buffer_size); destroying it here would
        // otherwise be a use-after-free on the GPU. Only reached when a LUT's
        // resolution actually changes (not every frame), so a full device wait
        // is an acceptable one-time stall rather than a per-frame cost.
        vk_device.waitIdle();
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
        // See create_lut_image_3d — tex may still be in use by another
        // frame-in-flight; wait before destroying its old resources.
        vk_device.waitIdle();
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

    // A genuinely 1D image, for a generated transform's sampler1D tables. Not an Nx1 2D
    // image: the view type has to match what the shader declares.
    void create_image_1d(vk_lut_texture& tex, uint32_t width, vk::Format format, vk::DeviceSize byte_size)
    {
        auto vk_device = vulkan_->getVkDevice();
        // See create_lut_image_3d — tex may still be in use by another frame-in-flight.
        vk_device.waitIdle();
        tex.destroy();
        tex.device    = vk_device;
        tex.data_size = byte_size;

        vk::ImageCreateInfo img_info{};
        img_info.imageType     = vk::ImageType::e1D;
        img_info.format        = format;
        img_info.extent        = vk::Extent3D(width, 1, 1);
        img_info.mipLevels     = 1;
        img_info.arrayLayers   = 1;
        img_info.samples       = vk::SampleCountFlagBits::e1;
        img_info.tiling        = vk::ImageTiling::eOptimal;
        img_info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        img_info.sharingMode   = vk::SharingMode::eExclusive;
        img_info.initialLayout = vk::ImageLayout::eUndefined;
        tex.image              = vk_device.createImage(img_info);

        auto                   mem_req = vk_device.getImageMemoryRequirements(tex.image);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize  = mem_req.size;
        alloc.memoryTypeIndex = findDedicatedMemoryType(mem_req.memoryTypeBits,
                                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
        tex.memory = vk_device.allocateMemory(alloc);
        vk_device.bindImageMemory(tex.image, tex.memory, 0);

        vk::ImageViewCreateInfo view_info{};
        view_info.image            = tex.image;
        view_info.viewType         = vk::ImageViewType::e1D;
        view_info.format           = format;
        view_info.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        tex.view                   = vk_device.createImageView(view_info);

        vk::BufferCreateInfo buf_info{};
        buf_info.size  = byte_size;
        buf_info.usage = vk::BufferUsageFlagBits::eTransferSrc;
        tex.staging    = vk_device.createBuffer(buf_info);

        auto                   buf_req = vk_device.getBufferMemoryRequirements(tex.staging);
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
        // See create_lut_image_3d — tex may still be in use by another
        // frame-in-flight; wait before destroying its old resources.
        vk_device.waitIdle();
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
        // NOTE: this memcpy into the persistently-mapped staging buffer can
        // still race a previous frame's in-flight cmd.copyBufferToImage read
        // of the same staging buffer (up to frame_buffer_size frames may be in
        // flight). Fully closing that requires per-frame-slot staging buffers;
        // the barrier below only protects the destination *image*, not this
        // staging buffer, against a value update arriving while still being
        // consumed by an older in-flight frame.
        if (data)
            memcpy(tex.mapped, data, tex.data_size);

        // Transition: shader-read (from a previous frame's sampling, if any) →
        // transfer dst. Using eUndefined/eTopOfPipe unconditionally here
        // (discarding whatever layout the image was actually in) would not
        // wait for an in-flight fragment shader's read of this same image to
        // finish before this transfer overwrites it — a race that can produce
        // one torn frame whenever a LUT's value is updated while still in
        // flight. The image really is eUndefined the first time (right after
        // create_lut_image_*), so only use eShaderReadOnlyOptimal as the old
        // layout once a previous upload has actually put it there.
        vk::ImageMemoryBarrier2 barrier{};
        barrier.oldLayout = tex.ever_uploaded ? vk::ImageLayout::eShaderReadOnlyOptimal
                                              : vk::ImageLayout::eUndefined;
        barrier.newLayout     = vk::ImageLayout::eTransferDstOptimal;
        barrier.image         = tex.image;
        barrier.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        barrier.srcStageMask  = tex.ever_uploaded ? vk::PipelineStageFlagBits2::eFragmentShader
                                                  : vk::PipelineStageFlagBits2::eTopOfPipe;
        barrier.srcAccessMask = tex.ever_uploaded ? vk::AccessFlagBits2::eShaderSampledRead
                                                  : vk::AccessFlagBits2::eNone;
        barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
        barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        tex.ever_uploaded     = true;

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

    /// The uploaded LUT set for this item's OCIO transform, or nullptr when the item does
    /// not use one (or the transform could not be built or uploaded).
    ///
    /// Builds on a cache miss, which means on the frame path: one device stall to create the
    /// images plus OCIO's generation, logged as a warning. Every later frame is a map lookup.
    /// Pre-warming at MIXER OCIO command time is the proper fix -- see
    /// docs/OCIO_INTEGRATION_STUDY.md section 8.7 -- and is not done here.
    /// Build and cache an OCIO program WITHOUT drawing -- see the OGL kernel for the
    /// measurement that motivates it. Routed through `select_ocio_variant` so the cache key
    /// is by construction the one the later draw computes.
    void prewarm_ocio(const std::string& source_space, const std::string& display, const std::string& view)
    {
        select_ocio_variant(source_space, display, view, /*on_frame_path=*/false);
    }

    /// `on_frame_path` is what the log reports. This function is the only place a variant
    /// is built, so the warning fires for a pre-warm too -- and reporting "on the frame
    /// path" when the compile has been moved OFF it makes the log unable to answer the
    /// question the pre-warm exists to settle.
    const ocio_variant* select_ocio_variant(const std::string& source_space,
                                            const std::string& display,
                                            const std::string& view,
                                            bool               on_frame_path = true)
    {
        namespace ocio_ns = caspar::accelerator::ocio;

        const bool want_input   = !source_space.empty();
        const bool want_display = !display.empty() && !view.empty();
        if (!want_input && !want_display)
            return nullptr;

        // Cache hits inside OCIO after the AMCP commands already built these once for
        // validation, so they cost a lookup rather than a rebuild -- and they are what yield
        // the cache IDs that key everything below.
        ocio_ns::gpu_shader in_shader;
        ocio_ns::gpu_shader out_shader;
        if (want_input &&
            !ocio_ns::build_input_transform(source_space, in_shader, ocio_ns::gpu_target::vulkan))
            return nullptr;
        if (want_display &&
            !ocio_ns::build_display_transform(display, view, out_shader, ocio_ns::gpu_target::vulkan))
            return nullptr;

        // One program holds both halves, so the key names the pair. Two source spaces through
        // one display are two programs, and keying on either alone would serve a layer the
        // other's transform. The disjoint binding ranges (input 1..4, display 5..8) are what
        // let their textures share descriptor set 1 without colliding.
        const auto key = in_shader.cache_id + "|" + out_shader.cache_id;

        // One list, both halves. Each texture already carries the binding OCIO wrote into the
        // source, so order here is irrelevant -- the binding is what places it.
        ocio_ns::gpu_shader generated = in_shader;
        generated.cache_id = key;
        generated.textures.insert(generated.textures.end(),
                                  out_shader.textures.begin(), out_shader.textures.end());

        auto it = ocio_variants_.find(key);
        if (it != ocio_variants_.end())
            return it->second.failed ? nullptr : &it->second;

        // Warn only when there is actually something to upload. Most colour spaces emit no
        // LUT at all -- every camera log encoding in this config is pure arithmetic -- and
        // for those this costs a map insert, not a device stall. A warning there would train
        // the reader to ignore the one that matters.
        if (generated.textures.empty()) {
            CASPAR_LOG(debug) << L"[vk_kernel] OCIO transform for '" << u16(source_space)
                              << L"' needs no LUT texture";
        } else if (on_frame_path) {
            CASPAR_LOG(warning) << L"[vk_kernel] uploading " << generated.textures.size()
                                << L" OCIO LUT(s) ON THE FRAME PATH for '" << u16(source_space)
                                << L"'. Expect one dropped frame; every later frame is a cache hit.";
        } else {
            CASPAR_LOG(info) << L"[vk_kernel] pre-warming " << generated.textures.size()
                             << L" OCIO LUT(s) (off the frame path) for '" << u16(source_space)
                             << L"' -> '" << u16(display.empty() ? std::string("-") : display + " / " + view)
                             << L"'.";
        }

        ocio_variant v;
        try {
            v.textures.reserve(generated.textures.size());
            for (const auto& t : generated.textures) {
                // The binding is OCIO's, written into the generated source. Anything outside
                // the range the set layout declares would be a descriptor the shader reads
                // from an unbound slot, so refuse rather than silently write elsewhere.
                if (t.binding < 1 || t.binding > static_cast<int>(OCIO_MAX_TEXTURES)) {
                    CASPAR_LOG(error) << L"[vk_kernel] OCIO declared '" << u16(t.sampler_name)
                                      << L"' at binding " << t.binding << L", outside the 1.."
                                      << OCIO_MAX_TEXTURES << L" reserved in descriptor set 1";
                    v.failed = true;
                    break;
                }

                v.textures.emplace_back();
                auto& lut = v.textures.back();
                auto& tex = lut.tex;

                if (t.dimensions == 3) {
                    // Not reachable with the pinned studio config -- no colour space in it
                    // emits a 3D LUT, all 55 measured. The path is here because MIXER LUT3D
                    // already exercises these same two helpers every time a cube file is
                    // loaded, so a custom config that does emit one lands on proven code
                    // rather than on a refusal.
                    const uint32_t sz = static_cast<uint32_t>(t.edge_len);
                    create_lut_image_3d(tex, sz); // sets data_size to the padded RGBA size
                    pad_rgb_to_rgba(static_cast<float*>(tex.mapped), t.values.data(),
                                    static_cast<size_t>(sz) * sz * sz);
                    lut.width = lut.height = lut.depth = sz;
                } else {
                    // 1D and 2D differ by more than a name. A Vulkan image view's type must
                    // match the sampler the shader declares, and an ACES display transform
                    // declares sampler1D for its reach and gamut-cusp tables. Creating those
                    // as an Nx1 2D image -- which is what "not 3D means 2D" produced until
                    // display transforms existed -- is a type mismatch, not a layout detail.
                    const bool     one_d = t.dimensions == 1;
                    const uint32_t w     = static_cast<uint32_t>(t.width);
                    const uint32_t h     = one_d ? 1u : static_cast<uint32_t>(std::max(1, t.height));

                    // Single-channel uploads as-is into R32_SFLOAT because the generated
                    // source reads only .r; three-channel pads to RGBA, since 3-component
                    // formats are not reliably supported as sampled images.
                    const vk::Format     format    = t.channels == 1 ? vk::Format::eR32Sfloat
                                                                     : vk::Format::eR32G32B32A32Sfloat;
                    const uint32_t       comps     = t.channels == 1 ? 1u : 4u;
                    const vk::DeviceSize byte_size = static_cast<vk::DeviceSize>(w) * h * comps * sizeof(float);

                    if (one_d)
                        create_image_1d(tex, w, format, byte_size);
                    else
                        create_image_2d_wh(tex, w, h, format, byte_size);

                    if (t.channels == 1)
                        memcpy(tex.mapped, t.values.data(), byte_size);
                    else
                        pad_rgb_to_rgba(static_cast<float*>(tex.mapped), t.values.data(),
                                        static_cast<size_t>(w) * h);

                    lut.width  = w;
                    lut.height = h;
                }

                // An ACES display transform's 1D tables are INTERP_NEAREST. Sampling them
                // linearly interpolates between entries that were never meant to be
                // interpolated, which is wrong rather than merely soft -- and every
                // input-transform LUT was linear, so this could be taken for granted until
                // display transforms arrived.
                lut.nearest = !t.interpolate_linear;

                v.views[static_cast<size_t>(t.binding) - 1] = tex.view;
                v.nearest[static_cast<size_t>(t.binding) - 1] = lut.nearest;
            }

            if (!v.failed) {
                // Splice, compile, and hand the SPIR-V to the variant cache. All three are
                // expensive and all three happen once per colour space, not per frame.
                const auto spliced = splice_ocio(
                    std::string(reinterpret_cast<const char*>(fragment_shader_src)),
                    in_shader.source + out_shader.source,
                    want_input ? in_shader.function_name : std::string(),
                    want_display ? out_shader.function_name : std::string());
                if (spliced.empty()) {
                    v.failed = true;
                } else {
                    const auto spirv = compile_glsl_fragment_to_spirv(spliced, generated.cache_id);
                    if (spirv.empty()) {
                        // The compiler has already logged its diagnostic. Refuse rather than
                        // fall back to the base pipeline silently: an untransformed layer that
                        // looks merely wrong is harder to diagnose than one that says why.
                        CASPAR_LOG(error) << L"[vk_kernel] the spliced shader for '"
                                          << u16(source_space) << L"' did not compile";
                        v.failed = true;
                    } else {
                        v.pipeline = vulkan_->get_variant_pipeline(depth_, render_format_,
                                                                   generated.cache_id, spirv);
                    }
                }
            }
        } catch (...) {
            // Remember the failure so a broken transform costs one attempt, not one per
            // frame. The layer then renders untransformed -- visibly wrong, and preferable
            // to a device stall on every tick.
            CASPAR_LOG_CURRENT_EXCEPTION();
            v.failed = true;
        }

        if (v.failed) {
            for (auto& lut : v.textures)
                lut.tex.destroy();
            v.textures.clear();
            v.views   = {};
            v.pipeline = nullptr;
        }

        auto [pos, inserted] = ocio_variants_.emplace(generated.cache_id, std::move(v));
        if (pos->second.failed)
            return nullptr;
        if (!pos->second.textures.empty())
            ocio_upload_queue_.push_back(&pos->second);
        return &pos->second;
    }

    /// Splice a generated transform into the mixer's own fragment shader source.
    ///
    /// Two markers, both replaced, and the second one is where the channel-order trap lives.
    /// **This shader carries true RGB**, so the call is `col.rgb = f(vec4(col.rgb, col.a)).rgb`
    /// -- no swizzle. The OGL kernel's equivalent uses `.bgr` because that shader grades in
    /// BGR, and copying it here would mirror the hue wheel while leaving every grey correct,
    /// which is exactly the class of defect a grey ramp cannot catch.
    ///
    /// Returns empty if either marker is missing, which would otherwise compile cleanly into
    /// a shader that silently ignores the transform.
    static std::string splice_ocio(const std::string& base,
                                   const std::string& declarations,
                                   const std::string& input_function,
                                   const std::string& display_function)
    {
        static constexpr const char* DECL_MARKER    = "//__CASPAR_OCIO_DECLARATIONS__";
        static constexpr const char* CALL_MARKER    = "//__CASPAR_OCIO_TRANSFORM__";
        static constexpr const char* DISPLAY_MARKER = "//__CASPAR_OCIO_DISPLAY__";

        const auto decl_at    = base.find(DECL_MARKER);
        const auto call_at    = base.find(CALL_MARKER);
        const auto display_at = base.find(DISPLAY_MARKER);
        if (decl_at == std::string::npos || call_at == std::string::npos ||
            display_at == std::string::npos) {
            CASPAR_LOG(error) << L"[vk_kernel] the fragment shader is missing an OCIO splice "
                                 L"marker; refusing to build a transform that would be ignored";
            return {};
        }

        // A call for each half that is configured; an empty function name leaves the marker
        // replaced by nothing, which is what keeps the other half's built-in block in charge.
        const auto call = [](const std::string& fn) {
            return fn.empty() ? std::string() : "col.rgb = " + fn + "(vec4(col.rgb, col.a)).rgb;";
        };

        std::string out = base;
        // Back to front. Every replacement shifts the offsets after it, and the declarations
        // are by far the longest; the display marker sits after the transform marker, which
        // sits after the declarations.
        out.replace(display_at, std::strlen(DISPLAY_MARKER), call(display_function));
        out.replace(call_at, std::strlen(CALL_MARKER), call(input_function));
        out.replace(decl_at, std::strlen(DECL_MARKER), declarations);
        return out;
    }

    /// Widen a tightly-packed RGB float array into the RGBA staging layout Vulkan images
    /// use here. Three-channel formats are not reliably supported as sampled images, which
    /// is why the 3D LUT and blend mask already do exactly this.
    static void pad_rgb_to_rgba(float* dst, const float* src, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            dst[i * 4 + 0] = src[i * 3 + 0];
            dst[i * 4 + 1] = src[i * 3 + 1];
            dst[i * 4 + 2] = src[i * 3 + 2];
            dst[i * 4 + 3] = 1.0f;
        }
    }

    /// Prepare LUT textures from draw_params transforms.
    /// Called during draw() — writes staging buffers and sets pending flags.
    void prepare_lut_textures(const draw_params& params, const ocio_variant* ocio)
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

            // Pack in RGBA order, NOT the OpenGL kernel's BGRA order.
            //
            // The OpenGL kernel stores (B, G, R, M) on purpose: it carries the pixel
            // through grading in BGRA, so its `.r` slot is the blue-displayed
            // channel. This backend grades in RGB, and `apply_curves` reads slot 0
            // for `c.r` — so copying that packing verbatim applied the user's BLUE
            // curve to RED and vice versa. Measured before the fix: 20.8 LSB on the
            // single-op row and ~21 on every stack containing it, OpenGL clean.
            //
            // Green and master are their own inverse under the exchange, which is
            // why only the red and blue curves were visibly wrong and why a grey
            // ramp would not have caught it.
            curve_lut_pending_data_.resize(256 * 4);
            for (int i = 0; i < 256; ++i) {
                curve_lut_pending_data_[i * 4 + 0] = lut_r[i];
                curve_lut_pending_data_[i * 4 + 1] = lut_g[i];
                curve_lut_pending_data_[i * 4 + 2] = lut_b[i];
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

        // ── OCIO input transform ─────────────────────────────────────────
        // Cleared first: these views land in descriptor set 1, and a layer without an OCIO
        // transform must leave every one of its sampler bindings unwritten rather than
        // inherit the previous layer's.
        current_lut_views_.ocio = {};
        current_ocio_pipeline_  = nullptr;
        if (ocio) {
            current_lut_views_.ocio = ocio->views;
            current_ocio_pipeline_  = ocio->pipeline;
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
        // A generated transform's LUTs are uploaded once, when its variant is first built.
        // They are immutable afterwards: OCIO derives them from the config and the colour
        // space, neither of which can change without producing a different cache ID and so a
        // different variant.
        for (auto* variant : ocio_upload_queue_) {
            for (auto& lut : variant->textures)
                upload_lut_data(lut.tex, nullptr, cmd, lut.width, lut.height, lut.depth);
        }
        ocio_upload_queue_.clear();
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

        // Selected once for this layer and used twice below: the uniform block needs to know
        // whether OCIO owns the input half, and prepare_lut_textures needs its LUTs and its
        // pipeline. Selecting twice would regenerate OCIO's shader source and re-copy its LUT
        // values twice per layer per frame -- a cache hit inside OCIO is not a free call.
        const auto& ocio_tf = transforms.image_transform.ocio;
        const bool          ocio_in  = ocio_tf.enable && !ocio_tf.source_space.empty();
        const bool          ocio_out = !params.ocio_display.empty() && !params.ocio_view.empty();
        // A working-space composite converts every layer INTO ACEScg and none of them out of
        // it. The input halves must therefore take the ACEScg route even where they would
        // otherwise shortcut -- `k_direct_cg`, `k_direct` and the "source already matches the
        // target" skip each leave the pixel somewhere other than AP1, and a composite of
        // layers in different spaces is not in any space.
        //
        // Declared beside ocio_out rather than next to `cg`, because the output-half override
        // near the end of this function needs it and that sits in this scope.
        const bool          ws_composite = params.working_space_composite;

        // Did the pixel REACH the working space, by any route?
        //
        // Gamut compression now runs outside the input-conversion block (see
        // ogl/image/shader.frag for the account), so this is what decides whether running it
        // means anything: compressing a pixel that is still display-encoded, because the
        // layer had no conversion at all, would not be a gamut operation.
        bool                in_working_space = false;
        const ocio_variant* ocio     = select_ocio_variant(ocio_in ? ocio_tf.source_space : std::string(),
                                                           params.ocio_display,
                                                           params.ocio_view);

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

        // The SD convention as a FALLBACK, not an override — see the longer note in the
        // OpenGL kernel. Short version: untagged sub-720 YCbCr is conventionally
        // BT.601, but a source that declared its colour space must be honoured whatever
        // its size, and a CUSTOM channel format is an LED wall or projector rather than
        // an SD broadcast destination, so a small raster there implies nothing about
        // colour space.
        const auto is_hd = params.pix_desc.planes.at(0).height > 700;
        const auto color_space =
            (params.pix_desc.color_space_specified || is_hd || params.target_is_custom_format)
                ? params.pix_desc.color_space
                : core::color_space::bt601;
        // YCbCr decode: only indices 0-2 (bt601/bt709/bt2020) are valid in the shader arrays.
        // Wide-gamut spaces (P3, Adobe RGB) use BT.709 coefficients as fallback,
        // because if the source had BT.2020 matrix, av_color.h would have returned bt2020 directly.
        uniforms.color_space_index = static_cast<uint32_t>(color_space) > 2u ? 1u : static_cast<uint32_t>(color_space);

        if (params.pix_desc.is_straight_alpha) {
            uniforms.flags |= static_cast<uint32_t>(shader_flags::is_straight_alpha);
        }

        // Channel-level, and in flags2 because the first word is full (edge_blend is bit
        // 31). The shader reads it at two sites: it decides whether to unpremultiply above
        // the colour chain, and whether to re-premultiply below it.
        if (params.straight_alpha_grading) {
            uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::straight_alpha_grading);
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
        //
        // Not for a key pass. A key is not a picture that will be scanned out --
        // it is sampled back through textures[LOCAL_KEY].r by whichever item the
        // key applies to. Swizzling on the way in put the key in blue and left
        // red at zero, so that sample returned 0 and multiplied the next item's
        // alpha away entirely. The OpenGL backend never had this to get wrong:
        // it allocates a single-channel texture for the key, so there is nothing
        // to swizzle.
        //
        // Seen as a Spout frame with alpha rendering opaque-over-nothing on the
        // Vulkan mixer while OpenGL composited it correctly -- a Spout frame
        // arrives as two items, a key followed by the picture, so it takes this
        // path where an ordinary clip does not.
        if (depth_ == common::bit_depth::bit8 && !transforms.image_transform.is_key) {
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
            // Gamut matrices, regenerated 2026-08-12 from OCIO 2.5.2 through the pinned studio
            // config -- the same library and config the server links. Four of the seven original
            // rows were not the matrices they claimed to be (bt2020, dcip3 d65, arri wide gamut 3
            // and sony sgamut3.cine; worst deviation 0.41 per element). See
            // docs/GAMUT_MATRIX_DEFECT_2026-08-12.md.
            //
            // Indices 0..6 are the MIXER COLORSPACE gamut enum and must not be reordered.
            // 7 and 8 are new, so a channel configured p3-dci or adobe-rgb has a row at all:
            // `working_gamut_index()` maps a core::color_space here, and it is NOT the same index
            // space as `gamut_index()`, which addresses k_direct.
            static const float k_to_working[9][9] = {
                // bt709 -> ACEScg
                {0.6130974f, 0.3395231f, 0.0473795f, 0.0701937f, 0.9163539f, 0.0134524f, 0.0206156f, 0.1095698f, 0.8698146f},
                // bt2020 -> ACEScg
                {0.9748950f, 0.0195991f, 0.0055059f, 0.0021796f, 0.9955355f, 0.0022850f, 0.0047972f, 0.0245320f, 0.9706708f},
                // dcip3 d65 -> ACEScg
                {0.7357979f, 0.2121665f, 0.0520356f, 0.0471799f, 0.9380457f, 0.0147744f, 0.0035637f, 0.0411419f, 0.9552944f},
                // aces_ap0 -> ACEScg
                {1.4514393f, -0.2365108f, -0.2149286f, -0.0765538f, 1.1762297f, -0.0996759f, 0.0083161f, -0.0060324f, 0.9977163f},
                // aces_ap1 identity -> ACEScg
                {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f},
                // arri wide gamut 3 -> ACEScg
                {0.9666334f, 0.1155416f, -0.0821751f, 0.0481904f, 1.1849383f, -0.2331287f, 0.0071933f, -0.0665937f, 1.0594004f},
                // sony sgamut3.cine -> ACEScg
                {0.9345170f, 0.1436417f, -0.0781587f, -0.0505267f, 1.2616092f, -0.2110825f, -0.0245030f, -0.0306710f, 1.0551741f},
                // dcip3 (DCI white) -> ACEScg
                {0.6947249f, 0.2562645f, 0.0490107f, 0.0429788f, 0.9461758f, 0.0108454f, 0.0036155f, 0.0430195f, 0.9533650f},
                // adobe rgb -> ACEScg
                {0.8573283f, 0.0932583f, 0.0494134f, 0.0981558f, 0.8878143f, 0.0140299f, 0.0288279f, 0.0640172f, 0.9071549f},
            };
            static const float k_to_output[9][9] = {
                // ACEScg -> bt709
                {1.7050509f, -0.6217921f, -0.0832589f, -0.1302564f, 1.1408048f, -0.0105483f, -0.0240034f, -0.1289690f, 1.1529723f},
                // ACEScg -> bt2020
                {1.0258248f, -0.0200532f, -0.0057716f, -0.0022344f, 1.0045865f, -0.0023521f, -0.0050134f, -0.0252901f, 1.0303035f},
                // ACEScg -> dcip3 d65
                {1.3792142f, -0.3088641f, -0.0703500f, -0.0693349f, 1.0822967f, -0.0129619f, -0.0021590f, -0.0454593f, 1.0476184f},
                // ACEScg -> aces_ap0
                {0.6954522f, 0.1406787f, 0.1638691f, 0.0447946f, 0.8596711f, 0.0955343f, -0.0055259f, 0.0040252f, 1.0015007f},
                // ACEScg -> aces_ap1 identity
                {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f},
                // ACEScg -> arri wide gamut 3
                {1.0389643f, -0.0979906f, 0.0590263f, -0.0441881f, 0.8586612f, 0.1855270f, -0.0098322f, 0.0546406f, 0.9551915f},
                // ACEScg -> sony sgamut3.cine
                {1.0650269f, -0.1199250f, 0.0548981f, 0.0470203f, 0.7912176f, 0.1617622f, 0.0260986f, 0.0202137f, 0.9536877f},
                // ACEScg -> dcip3 (DCI white)
                {1.4641202f, -0.3933270f, -0.0707931f, -0.0664765f, 1.0752915f, -0.0088150f, -0.0025529f, -0.0470296f, 1.0495825f},
                // ACEScg -> adobe rgb
                {1.1822189f, -0.1196734f, -0.0625455f, -0.1302564f, 1.1408047f, -0.0105483f, -0.0283769f, -0.0767026f, 1.1050796f},
            };

            // Helper: expand row-major mat3 (9 floats) to 3×vec4 columns (12 floats, std140).
            // GLSL mat3(c0,c1,c2) treats arguments as COLUMNS.  The C++ tables store
            // matrices in row-major order, so we must transpose when packing for the shader.
            auto set_mat3 = [](float dst[12], const float src[9]) {
                dst[0]=src[0]; dst[1]=src[3]; dst[2]=src[6]; dst[3]=0;  // column 0
                dst[4]=src[1]; dst[5]=src[4]; dst[6]=src[7]; dst[7]=0;  // column 1
                dst[8]=src[2]; dst[9]=src[5]; dst[10]=src[8]; dst[11]=0; // column 2
            };

            // Enum -> shader index mappings, shared by the OCIO branch and the auto branch
            // below. Hoisted rather than duplicated: two copies of the same mapping drift,
            // and this repo has paid for that three times (see the harness's
            // test_single_source_of_truth.py). Matches ogl/image/image_kernel.cpp, which
            // hoisted them for exactly this reason when its OCIO branch landed.
            //
            // Gamut indices (0=bt709, 1=bt2020, 2=p3_d65, 3=p3_dci, 4=adobe_rgb)
            auto gamut_index = [](core::color_space cs) -> int {
                switch (cs) {
                    case core::color_space::bt2020:    return 1;
                    case core::color_space::p3_d65:   return 2;
                    case core::color_space::p3_dci:   return 3;
                    case core::color_space::adobe_rgb:return 4;
                    default:                          return 0; // bt601/bt709 → index 0
                }
            };
            // The WORKING-table index space: addresses k_to_working / k_to_output, whose order is
            // the MIXER COLORSPACE gamut enum (0=bt709, 1=bt2020, 2=dcip3-d65, 3=ap0, 4=ap1,
            // 5=arri-wg3, 6=sgamut3.cine) plus 7=p3-dci and 8=adobe-rgb added 2026-08-12.
            //
            // NOT interchangeable with gamut_index() above, which addresses k_direct and reads
            // 3=p3_dci, 4=adobe_rgb. Using one for the other is the defect fixed on 2026-08-12:
            // `k_to_output[gamut_index(target)]` gave a p3-dci channel the ACEScg->AP0 matrix and
            // an adobe-rgb channel the identity, on every MIXER OCIO layer. Two index spaces over
            // one vocabulary is exactly the duplication this tree keeps paying for -- if a third
            // table ever appears, give it a named accessor too rather than an int.
            auto working_gamut_index = [](core::color_space cs) -> int {
                switch (cs) {
                    case core::color_space::bt2020:    return 1;
                    case core::color_space::p3_d65:    return 2;
                    case core::color_space::p3_dci:    return 7;
                    case core::color_space::adobe_rgb: return 8;
                    default:                           return 0; // bt601/bt709 -> index 0
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


            const auto& cg = transforms.image_transform.color_grade;
            if (ocio_in || params.output_convert_only) {
                // OCIO produced the working-space pixel, so the shader's own input conversion
                // is off -- its spliced call replaces that block rather than following it. The
                // output half still has to run, driven by the channel's target: without it the
                // layer would reach the render target in scene-linear ACEScg with no OETF and
                // the wrong primaries.
                uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::output_convert);
                uniforms.output_transfer = oetf_index(params.target_color_transfer);
                uniforms.tone_mapping_op = params.auto_tone_map;
                uniforms.display_peak_luminance = params.display_peak_luminance;
                // This backend folds the BT.2408 luminance scale into `exposure`; there is no
                // separate uniform for it, unlike the OGL shader. Neutral on this path.
                uniforms.exposure = 1.0f;
                // OCIO put it in the working space; output_convert_only means it was already
                // there. Either way gamut compression below is now reachable here.
                in_working_space = true;
                // Still not available on this path: user exposure lives in the color_grade
                // struct inside the input block OCIO replaces, and its only setter -- MIXER
                // COLORSPACE's 6th argument -- is mutually exclusive with MIXER OCIO. So it
                // is UNREACHABLE here rather than silently ignored.
                //
                // This backend applying exposure BEFORE the matrix while OGL applies it
                // after is not a divergence: a scalar commutes with a linear matrix, and
                // with `apply_gamut_compress`, which is homogeneous of degree one.
                // `cli.py conformance --exposure` at 0.5, 1.6 and 2.5 -- both mixers,
                // within 1 LSB of the same model. See the OGL kernel for the full note.
                set_mat3(uniforms.working_to_output, k_to_output[working_gamut_index(params.target_color_space)]);
            } else if (cg.enable) {
                // MIXER COLORSPACE owns both halves of the conversion.
                in_working_space = true;
                uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::input_convert) |
                                   static_cast<uint32_t>(shader_flags2::output_convert);
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

                if (cg.tone_mapping == 0 && ig <= 1 && og <= 1 && !ws_composite) {
                    // Direct D65↔D65 conversion — no ACEScg intermediate needed.
                    // Unavailable under a working-space composite: it leaves the pixel in
                    // the OUTPUT gamut, and the composite has to be in AP1.
                    set_mat3(uniforms.input_to_working,  k_direct_cg[ig][og]);
                    set_mat3(uniforms.working_to_output, k_identity_cg);
                } else {
                    // Full ACES grading pipeline through ACEScg working space
                    set_mat3(uniforms.input_to_working,  k_to_working[ig]);
                    set_mat3(uniforms.working_to_output, k_to_output[og]);
                }
            } else if (params.auto_color_convert &&
                       (ws_composite ||
                        params.pix_desc.color_space != params.target_color_space ||
                        params.pix_desc.color_transfer != params.target_color_transfer)) {
                // Auto color conversion: source differs from channel output.
                // gamut_index / eotf_index / oetf_index are the hoisted copies above.
                int ig = gamut_index(params.pix_desc.color_space);
                int og = gamut_index(params.target_color_space);
                // Skip if the mapped indices are identical (e.g. bt601 source on bt709 channel)
                // Never skip under a working-space composite -- see the OGL kernel for why.
                if (ws_composite || ig != og ||
                    params.pix_desc.color_transfer != params.target_color_transfer) {
                    int it = eotf_index(params.pix_desc.color_transfer);
                    int ot = oetf_index(params.target_color_transfer);
                    // Use channel's configured auto tone-map operator (default: hard clamp).
                    int tm = params.auto_tone_map;
                    // auto-color-convert owns both halves too.
                    in_working_space = true;
                    uniforms.flags |= static_cast<uint32_t>(shader_flags::color_grading);
                    uniforms.flags2 |= static_cast<uint32_t>(shader_flags2::input_convert) |
                                       static_cast<uint32_t>(shader_flags2::output_convert);
                    uniforms.input_transfer  = it;
                    uniforms.output_transfer = ot;
                    uniforms.tone_mapping_op = tm;
                    uniforms.display_peak_luminance = params.display_peak_luminance;

                    // BT.2408 luminance adaptation: scene-referred mapping
                    // for SDR↔HLG conversions (75% signal for ref white).
                    // Note: src_t uses EOTF indices, tgt_t uses OETF indices.
                    // Linear/gamma24/gamma26 are treated as SDR-level for luminance.
                    auto get_luminance_scale = [&](int src_t, int tgt_t) -> float {
                        // SDR→PQ uses configurable sdr_reference_white per BT.2408 Amd.4.
                        constexpr float k_sdr_hlg = 0.265f;
                        float sdr_pq_scale = params.sdr_reference_white / 10000.0f;
                        float pq_sdr_scale = 10000.0f / params.sdr_reference_white;
                        bool src_sdr = (src_t <= 2 || src_t >= 7); // rec709/srgb or linear/gamma24/gamma26
                        bool tgt_sdr = (tgt_t <= 2 || tgt_t >= 5); // rec709/srgb or linear/gamma24/gamma26
                        bool src_hlg = (src_t == 4);
                        bool tgt_hlg = (tgt_t == 4);
                        bool src_pq  = (src_t == 3);
                        bool tgt_pq  = (tgt_t == 3);
                        if (src_sdr && tgt_hlg) return k_sdr_hlg;
                        if (src_hlg && tgt_sdr) return 1.0f / k_sdr_hlg;
                        if (src_sdr && tgt_pq)  return sdr_pq_scale;
                        if (src_pq  && tgt_sdr) return pq_sdr_scale;
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
                            {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f}, // -> bt709
                            {0.6274039f, 0.3292830f, 0.0433131f, 0.0690973f, 0.9195404f, 0.0113623f, 0.0163914f, 0.0880133f, 0.8955953f}, // -> bt2020
                            {0.8224620f, 0.1775380f, 0.0000000f, 0.0331942f, 0.9668058f, 0.0000000f, 0.0170826f, 0.0723974f, 0.9105200f}, // -> p3_d65
                            {0.8685797f, 0.1289191f, 0.0025011f, 0.0345404f, 0.9618114f, 0.0036482f, 0.0167714f, 0.0710400f, 0.9121886f}, // -> p3_dci
                            {0.7151256f, 0.2848744f, -0.0000000f, 0.0000000f, 1.0000000f, -0.0000000f, -0.0000000f, 0.0411619f, 0.9588381f}, // -> adobe_rgb
                        },
                        { // from bt2020
                            {1.6604910f, -0.5876411f, -0.0728499f, -0.1245505f, 1.1328999f, -0.0083494f, -0.0181508f, -0.1005789f, 1.1187297f}, // -> bt709
                            {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f}, // -> bt2020
                            {1.3435782f, -0.2821797f, -0.0613986f, -0.0652975f, 1.0757879f, -0.0104905f, 0.0028218f, -0.0195985f, 1.0167767f}, // -> p3_d65
                            {1.4261665f, -0.3646123f, -0.0615542f, -0.0625062f, 1.0689717f, -0.0064655f, 0.0024438f, -0.0211213f, 1.0186775f}, // -> p3_dci
                            {1.1519784f, -0.0975031f, -0.0544753f, -0.1245505f, 1.1328999f, -0.0083494f, -0.0225304f, -0.0498065f, 1.0723369f}, // -> adobe_rgb
                        },
                        { // from p3_d65
                            {1.2249402f, -0.2249402f, 0.0000000f, -0.0420570f, 1.0420569f, 0.0000000f, -0.0196376f, -0.0786360f, 1.0982736f}, // -> bt709
                            {0.7538331f, 0.1985974f, 0.0475696f, 0.0457438f, 0.9417772f, 0.0124789f, -0.0012103f, 0.0176017f, 0.9836086f}, // -> bt2020
                            {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f}, // -> p3_d65
                            {1.0584872f, -0.0612341f, 0.0027469f, 0.0017874f, 0.9942058f, 0.0040067f, -0.0003569f, -0.0014757f, 1.0018326f}, // -> p3_dci
                            {0.8640051f, 0.1359949f, -0.0000000f, -0.0420570f, 1.0420570f, 0.0000000f, -0.0205604f, -0.0325061f, 1.0530665f}, // -> adobe_rgb
                        },
                        { // from p3_dci
                            {1.1575164f, -0.1549624f, -0.0025540f, -0.0415001f, 1.0455679f, -0.0040679f, -0.0180500f, -0.0785783f, 1.0966283f}, // -> bt709
                            {0.7117832f, 0.2436603f, 0.0445565f, 0.0416152f, 0.9498416f, 0.0085432f, -0.0008447f, 0.0191095f, 0.9817352f}, // -> bt2020
                            {0.9446454f, 0.0581774f, -0.0028228f, -0.0016997f, 1.0057173f, -0.0040176f, 0.0003340f, 0.0015022f, 0.9981638f}, // -> p3_d65
                            {1.0000000f, 0.0000000f, -0.0000000f, -0.0000000f, 1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f}, // -> p3_dci
                            {0.8159473f, 0.1870380f, -0.0029853f, -0.0415001f, 1.0455679f, -0.0040679f, -0.0190153f, -0.0323062f, 1.0513215f}, // -> adobe_rgb
                        },
                        { // from adobe_rgb
                            {1.3983557f, -0.3983557f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, -0.0000000f, -0.0429290f, 1.0429290f}, // -> bt709
                            {0.8773338f, 0.0774937f, 0.0451725f, 0.0966226f, 0.8915273f, 0.0118501f, 0.0229211f, 0.0430367f, 0.9340423f}, // -> bt2020
                            {1.1500944f, -0.1500944f, -0.0000000f, 0.0464173f, 0.9535827f, 0.0000000f, 0.0238876f, 0.0265048f, 0.9496076f}, // -> p3_d65
                            {1.2145835f, -0.2171920f, 0.0026085f, 0.0482998f, 0.9478954f, 0.0038048f, 0.0234524f, 0.0251997f, 0.9513479f}, // -> p3_dci
                            {1.0000000f, 0.0000000f, 0.0000000f, 0.0000000f, 1.0000000f, 0.0000000f, -0.0000000f, 0.0000000f, 1.0000000f}, // -> adobe_rgb
                        },
                    };
                    static const float k_identity[9] = {1,0,0, 0,1,0, 0,0,1};
                    if (ws_composite) {
                        // Into ACEScg, not to the display. The output half is suppressed
                        // below and the channel's post-composite pass supplies k_to_output.
                        set_mat3(uniforms.input_to_working,
                                 k_to_working[working_gamut_index(params.pix_desc.color_space)]);
                    } else {
                        set_mat3(uniforms.input_to_working,  k_direct[ig][og]);
                    }
                    set_mat3(uniforms.working_to_output, k_identity);

                    // Auto gamut compression: enable ACES-style soft compress for
                    // wide→narrow gamut conversions to prevent hard-clipping of
                    // out-of-gamut colors (e.g. BT.2020→BT.709).
                    if (params.auto_gamut_compress && ig != og) {
                        uniforms.flags |= static_cast<uint32_t>(shader_flags::gamut_compress);
                        // Default ACES 1.3 gamut compress limits (cyan, magenta, yellow)
                        uniforms.gc_limit[0] = 1.147f;
                        uniforms.gc_limit[1] = 1.264f;
                        uniforms.gc_limit[2] = 1.312f;
                    }
                }
            }
        }

        // A channel display transform owns the output half outright, whichever branch above
        // ran. Applied after the whole chain and unconditionally, because it is orthogonal to
        // how the INPUT half was decided: the layer may have reached the working space via
        // MIXER OCIO, via MIXER COLORSPACE, via auto-convert or not at all, and in every one
        // of those cases the display transform is what encodes it for output.
        //
        // Outside the chain rather than inside it -- an earlier attempt put this between two
        // `else if` arms, which silently gated auto-convert on there being no display
        // transform.
        if (ocio_out || ws_composite) {
            uniforms.flags2 &= ~static_cast<uint32_t>(shader_flags2::output_convert);
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

        // ── Exposure ─────────────────────────────────────────────────
        //
        // `MIXER EXPOSURE` composes with whatever the conversion path already put here --
        // on this backend that includes the folded-in BT.2408 luminance scale. Both are
        // scalars, so multiplying is the only answer that is not arbitrary.
        //
        // Gated on having reached the working space, like gamut compression below: a
        // "linear" gain on a pixel that is still display-encoded is not a gain on light.
        if (in_working_space)
            uniforms.exposure *= static_cast<float>(transforms.image_transform.exposure);

        // ── Gamut Compression ─────────────────────────────────────────
        if (transforms.image_transform.gamut_compress && in_working_space) {
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
            // Degrees in, normalised hue out — /360 for the centre, /180 for the
            // width, which the shader compares against `AngleDiff(...)*2`. Same fix
            // as the OpenGL kernel; see the longer note there. Uploaded raw, the hue
            // mask evaluated to 1 for every pixel and the key ignored hue.
            uniforms.qual_target_hue = static_cast<float>(transforms.image_transform.qual_target_hue / 360.0);
            uniforms.qual_hue_width  = static_cast<float>(transforms.image_transform.qual_hue_width / 180.0);
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
        prepare_lut_textures(params, ocio);

        return {std::move(coords), uniforms};
    }
};

image_kernel::image_kernel(const spl::shared_ptr<device>& device,
                           common::bit_depth              depth,
                           common::render_format          render_format)
    : impl_(new impl(device, depth, render_format))
{
}
image_kernel::~image_kernel() {}

spl::shared_ptr<renderpass> image_kernel::create_renderpass(uint32_t width, uint32_t height)
{
    return impl_->create_renderpass(width, height);
}

void image_kernel::prewarm_ocio(const std::string& source_space,
                                const std::string& display,
                                const std::string& view)
{
    impl_->prewarm_ocio(source_space, display, view);
}

} // namespace caspar::accelerator::vulkan
