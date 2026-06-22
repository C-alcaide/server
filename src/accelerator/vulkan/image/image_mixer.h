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

#pragma once

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/memory.h>

#include <core/frame/frame.h>
#include <core/frame/pixel_format.h>
#include <core/mixer/image/image_mixer.h>
#include <core/video_format.h>

#include <future>

namespace caspar { namespace accelerator {

namespace ogl {
class device;
class channel_texture_store;
class previz_renderer;
} // namespace ogl

namespace vulkan {

class previz_texture_bridge;

class image_mixer final : public core::image_mixer
{
  public:
    image_mixer(const spl::shared_ptr<class device>& vulkan,
                int                                  channel_id,
                const size_t                         max_frame_size,
                common::bit_depth                    depth);
    image_mixer(const image_mixer&) = delete;

    ~image_mixer();

    image_mixer& operator=(const image_mixer&) = delete;

    std::future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>>>
                        render(const core::video_format_desc& format_desc) override;
    core::mutable_frame create_frame(const void* tag, const core::pixel_format_desc& desc) override;
    core::mutable_frame
    create_frame(const void* video_stream_tag, const core::pixel_format_desc& desc, common::bit_depth depth) override;

    void update_aspect_ratio(double aspect_ratio) override;

    // core::image_mixer

    void              push(const core::frame_transform& frame) override;
    void              visit(const core::const_frame& frame) override;
    void              pop() override;
    common::bit_depth depth() const override;

    bool is_vulkan() const override { return true; }

    void set_cpu_readback_needed(bool needed) override;

    std::shared_ptr<class device> get_vk_device() const;

    /// Previz support — called by accelerator after construction.
    void set_previz_ogl_device(const std::shared_ptr<ogl::device>& ogl_dev);
    void set_channel_texture_store(const std::shared_ptr<ogl::channel_texture_store>& store);
    void set_previz_bridge(const std::shared_ptr<class previz_texture_bridge>& bridge);
    ogl::previz_renderer* get_previz_renderer();

    void set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_reference_white, bool auto_gamut_compress) override;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path) override;
    void set_calibration_bypass(bool bypass) override;
    core::calibration_lut_state get_calibration_state() const override;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::vulkan
