/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 * Author: Robert Nagy, ronag89@gmail.com
 */

#pragma once

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/memory.h>
#include <common/render_format.h>

#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <core/mixer/image/image_mixer.h>
#include <core/video_format.h>

#include <future>
#include <memory>

namespace caspar { namespace accelerator { namespace ogl {

class previz_renderer;
class channel_texture_store;

class image_mixer final : public core::image_mixer
{
  public:
    /// `render_format` is the numeric format of the mixer's *internal* render targets, and
    /// is independent of `depth`, which stays the channel's output depth. unorm is
    /// bit-identical to the behaviour before this parameter existed; fp16 buys range
    /// (negatives and values above 1.0) for a linear working space. See
    /// common/render_format.h and docs/OCIO_INTEGRATION_STUDY.md section 4.3.
    image_mixer(const spl::shared_ptr<class device>& ogl,
                int                                  channel_id,
                const size_t                         max_frame_size,
                common::bit_depth                    depth,
                common::render_format                render_format = common::render_format::unorm);
    image_mixer(const image_mixer&) = delete;

    ~image_mixer();

    image_mixer& operator=(const image_mixer&) = delete;

    std::future<core::render_output> render(const core::video_format_desc& format_desc) override;
    core::mutable_frame create_frame(const void* tag, const core::pixel_format_desc& desc) override;
    core::mutable_frame
    create_frame(const void* video_stream_tag, const core::pixel_format_desc& desc, common::bit_depth depth) override;

    void update_aspect_ratio(double aspect_ratio) override;

    // core::image_mixer

    void              push(const core::frame_transform& frame) override;
    void              visit(const core::const_frame& frame) override;
    void              pop() override;
    common::bit_depth depth() const override;

    // Expose the underlying OGL device for CUDA-GL interop producers.
    std::shared_ptr<class device> get_ogl_device() const;

    /// Producers use this to discover the mixer's GL device (for WGL/CUDA
    /// interop). impl overrode it but this class did not forward it, so it
    /// always returned the base's nullptr and every interop path that asks for
    /// it silently declined -- including the whole D3D11->GL GPU-direct decode
    /// path, which therefore never ran.
    void* gpu_device_handle() const override;

    core::gpu_backend gpu_device_backend() const override;

    void* native_gl_context() const override;
    void* native_egl_display() const override;

    void set_cpu_readback_needed(bool needed) override;

    // Previz 3D rendering
    previz_renderer&  get_previz_renderer();
    void              set_channel_texture_store(std::shared_ptr<channel_texture_store> store);

    void set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_reference_white, bool auto_gamut_compress, bool straight_alpha_grading, bool working_space_composite) override;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path) override;
    void set_calibration_bypass(bool bypass) override;
    void set_ocio_display(const std::string& display, const std::string& view) override;
    core::ocio_display_state get_ocio_display() const override;
    void set_consumer_views(std::vector<core::ocio_view_key> views) override;
    bool composites_in_working_space() const override;
    core::calibration_lut_state get_calibration_state() const override;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::ogl
