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

#include "../StdAfx.h"

#ifdef CASPAR_GST_CUDA_EGRESS

#include "gst_cuda_egress.h"

#include <common/log.h>
#include <common/utf.h>

#include <ffmpeg/consumer/cuda_vk_upload.h>

#include <gst/cuda/gstcuda.h>
#include <gst/cuda/gstcudabufferpool.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace caspar { namespace gstreamer {

namespace {

} // namespace

struct gst_cuda_egress::impl
{
    GstCudaContext*   context_   = nullptr;
    GstCudaAllocator* allocator_ = nullptr;
    /// A pool, not a fresh allocation per frame. Measured without one: the CUDA route cost
    /// **2.24 cores against the host readback's 1.98**, i.e. it was worse than the thing it
    /// exists to replace. Allocating device memory fifty times a second is not free, and a
    /// zero-copy path that pays for an allocator every frame has given the saving back before
    /// it starts.
    GstBufferPool*    pool_      = nullptr;
    GstCaps*          caps_      = nullptr;
    GstVideoInfo      info_{};

    ffmpeg::cuda_vk_uploader uploader_;

    bool active_    = false;
    bool disabled_  = false;
    bool complained_ = false;

    ~impl() { close(); }

    void close()
    {
        if (pool_) {
            gst_buffer_pool_set_active(pool_, FALSE);
            gst_object_unref(pool_);
            pool_ = nullptr;
        }
        if (allocator_ && active_) {
            gst_cuda_allocator_set_active(allocator_, FALSE);
            active_ = false;
        }

        if (caps_) {
            gst_caps_unref(caps_);
            caps_ = nullptr;
        }
        if (allocator_) {
            gst_object_unref(allocator_);
            allocator_ = nullptr;
        }
        if (context_) {
            gst_object_unref(context_);
            context_ = nullptr;
        }
    }

    /// Logs a reason once and latches off. A GPU route that reports its refusal every frame is
    /// as unusable as one that reports it never.
    void give_up(const std::wstring& reason)
    {
        if (!complained_) {
            complained_ = true;
            CASPAR_LOG(info) << L"[gstreamer] CUDA egress off: " << reason
                             << L". Frames go through host memory, which is correct and costs a "
                                L"readback.";
        }
        disabled_ = true;
    }
};

gst_cuda_egress::gst_cuda_egress()
    : impl_(new impl())
{
}

gst_cuda_egress::~gst_cuda_egress() = default;

bool gst_cuda_egress::available() { return ffmpeg::cuda_vk_uploader::available(); }

bool gst_cuda_egress::open(const core::video_format_desc& format_desc, std::wstring& reason)
{
    auto& m = *impl_;

    if (!available()) {
        reason = L"no usable CUDA device";
        return false;
    }

    // GStreamer creates the context and we borrow its CUcontext, never the reverse. See the
    // header: a device pointer from another context is an access violation rather than an
    // error, and this makes the mismatch unrepresentable.
    m.context_ = gst_cuda_context_new(0);
    if (m.context_ == nullptr) {
        reason = L"GStreamer could not create a CUDA context";
        return false;
    }
    auto* cu_ctx = gst_cuda_context_get_handle(m.context_);
    if (cu_ctx == nullptr) {
        reason = L"the GstCudaContext exposes no CUcontext";
        m.close();
        return false;
    }
    m.uploader_.set_context(cu_ctx);

    // BGRA, because that is what the channel already is. The encoder accepts it directly, so
    // there is no conversion anywhere on this path -- which is the point of taking it.
    gst_video_info_set_format(&m.info_, GST_VIDEO_FORMAT_BGRA,
                              static_cast<guint>(format_desc.width),
                              static_cast<guint>(format_desc.height));
    GST_VIDEO_INFO_FPS_N(&m.info_) = format_desc.framerate.numerator();
    GST_VIDEO_INFO_FPS_D(&m.info_) = format_desc.framerate.denominator();

    m.caps_ = gst_video_info_to_caps(&m.info_);
    if (m.caps_ == nullptr) {
        reason = L"could not build caps for the CUDA buffers";
        m.close();
        return false;
    }
    gst_caps_set_features(m.caps_, 0, gst_caps_features_new("memory:CUDAMemory", nullptr));

    // The registered singleton, not one we make: there is no `gst_cuda_allocator_new`, and
    // GStreamer's CUDA elements all resolve the same registered allocator by name. Taking a
    // reference keeps it alive for as long as the buffers we hand out.
    auto* found = gst_allocator_find(GST_CUDA_MEMORY_TYPE_NAME);
    if (found == nullptr) {
        reason = L"the CUDA allocator is not registered (is the nvcodec plugin loaded?)";
        m.close();
        return false;
    }
    m.allocator_ = GST_CUDA_ALLOCATOR_CAST(found);

    // **GStreamer allocates the device memory, not us**, and that is the fix rather than a
    // simplification. Allocating our own pointer -- with either the driver or the runtime API,
    // both tried -- and copying into it produced `cudaMemcpy2DFromArray: invalid argument` on
    // the first frame every time: the runtime binds to a device's PRIMARY context, and
    // `gst_cuda_context_new` creates its own, so a pointer made under one is not a pointer the
    // copy made under the other will accept.
    //
    // Letting the allocator that owns the context make the allocation removes the question. It
    // is also the memory the encoder expects, so nothing downstream has to be told about it.
    if (!gst_cuda_allocator_set_active(m.allocator_, TRUE)) {
        reason = L"the CUDA allocator would not activate";
        m.close();
        return false;
    }
    m.active_ = true;

    m.pool_ = gst_cuda_buffer_pool_new(m.context_);
    if (m.pool_ == nullptr) {
        reason = L"could not create a CUDA buffer pool";
        m.close();
        return false;
    }
    {
        auto* config = gst_buffer_pool_get_config(m.pool_);
        // A minimum of 3: the encoder holds one, the appsrc queue holds one, and this fills the
        // next. Zero maximum, so a downstream element that briefly holds more does not stall
        // the channel waiting for a buffer -- which would be a far worse failure than the
        // allocation it saves.
        gst_buffer_pool_config_set_params(config, m.caps_, GST_VIDEO_INFO_SIZE(&m.info_), 3, 0);
        if (!gst_buffer_pool_set_config(m.pool_, config) || !gst_buffer_pool_set_active(m.pool_, TRUE)) {
            reason = L"the CUDA buffer pool would not accept its configuration";
            m.close();
            return false;
        }
    }

    CASPAR_LOG(info) << L"[gstreamer] CUDA egress active: " << format_desc.width << L"x"
                     << format_desc.height << L" BGRA handed to the encoder without a host copy.";
    return true;
}

GstCaps* gst_cuda_egress::caps() const { return impl_->caps_; }

GstBuffer* gst_cuda_egress::wrap(const core::const_frame& frame)
{
    auto& m = *impl_;
    if (m.disabled_ || !m.active_)
        return nullptr;

    const auto& textures = frame.textures();
    if (textures.empty() || !textures.front()) {
        m.give_up(L"the channel frame carries no GPU texture");
        return nullptr;
    }

    GstBuffer* buffer = nullptr;
    if (gst_buffer_pool_acquire_buffer(m.pool_, &buffer, nullptr) != GST_FLOW_OK || buffer == nullptr) {
        m.give_up(L"the CUDA buffer pool would not give up a buffer");
        return nullptr;
    }

    // GST_MAP_CUDA asks for the DEVICE pointer rather than a host mapping; without it this
    // would silently hand back staged host memory and the copy would be the readback all over
    // again, only less obviously.
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, static_cast<GstMapFlags>(GST_MAP_WRITE | GST_MAP_CUDA))) {
        gst_buffer_unref(buffer);
        m.give_up(L"could not map the CUDA buffer for device writing");
        return nullptr;
    }

    const auto pitch = GST_VIDEO_INFO_PLANE_STRIDE(&m.info_, 0);
    // Waits for the mixer's render internally, so the copy cannot read a half-drawn frame.
    const bool copied =
        m.uploader_.copy_to_device(textures.front(), map.data, static_cast<std::size_t>(pitch));
    gst_buffer_unmap(buffer, &map);

    if (!copied) {
        gst_buffer_unref(buffer);
        m.give_up(u16(std::string(m.uploader_.last_error() ? m.uploader_.last_error() : "unknown")));
        return nullptr;
    }

    return buffer;
}

}} // namespace caspar::gstreamer

#endif // CASPAR_GST_CUDA_EGRESS
