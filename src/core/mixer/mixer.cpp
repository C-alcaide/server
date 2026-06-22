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

#include "../StdAfx.h"

#include "mixer.h"

#include "../frame/frame.h"

#include "audio/audio_mixer.h"
#include "image/image_mixer.h"

#include <common/diagnostics/graph.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame_transform.h>
#include <core/frame/pixel_format.h>
#include <core/video_format.h>

#include <queue>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core {

struct mixer::impl
{
    monitor::state                       state_;
    int                                  channel_index_;
    spl::shared_ptr<diagnostics::graph>  graph_;
    audio_mixer                          audio_mixer_{graph_};
    spl::shared_ptr<image_mixer>         image_mixer_;
    core::color_space                    default_color_space_{core::color_space::bt709};
    core::color_transfer                 default_color_transfer_{core::color_transfer::sdr};
    std::queue<std::future<const_frame>> buffer_;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

    impl(int channel_index, spl::shared_ptr<diagnostics::graph> graph, spl::shared_ptr<image_mixer> image_mixer, core::color_space default_color_space, core::color_transfer default_color_transfer, bool auto_color_convert = true, int auto_tone_map = 0, float display_peak_luminance = 1000.0f, float sdr_reference_white = 100.0f, bool auto_gamut_compress = false)
        : channel_index_(channel_index)
        , graph_(std::move(graph))
        , image_mixer_(std::move(image_mixer))
        , default_color_space_(default_color_space)
        , default_color_transfer_(default_color_transfer)
    {
        image_mixer_->set_target_color(default_color_space_, default_color_transfer_, auto_color_convert, auto_tone_map, display_peak_luminance, sdr_reference_white, auto_gamut_compress);
    }

    const_frame operator()(std::vector<draw_frame> frames, const video_format_desc& format_desc, int nb_samples)
    {
        // Evaluate the previous tick's deferred result BEFORE rendering the
        // current tick.  This ensures the image_mixer's still-frame cache is
        // up-to-date when render() checks it, preventing a 1-tick stale-cache
        // race that caused frames to display out of order.
        const_frame prev_result;
        bool        have_prev = static_cast<int>(buffer_.size()) >= format_desc.field_count;
        if (have_prev) {
            auto f = std::move(buffer_.front());
            buffer_.pop();
            prev_result = std::move(f.get());
        }

        image_mixer_->update_aspect_ratio(static_cast<double>(format_desc.square_width) /
                                          static_cast<double>(format_desc.square_height));

        for (auto& frame : frames) {
            frame.accept(audio_mixer_);
            frame.transform().image_transform.layer_depth = 1;
            frame.accept(*image_mixer_);
        }

        auto result = image_mixer_->render(format_desc);
        auto audio  = audio_mixer_(format_desc, nb_samples);

        state_["audio"] = audio_mixer_.state();

        auto depth     = image_mixer_->depth();
        auto is_vulkan = image_mixer_->is_vulkan();

        buffer_.push(std::async(
            std::launch::deferred,
            [result = std::move(result),
             audio  = std::move(audio),
             graph  = graph_,
             depth,
             is_vulkan,
             format_desc,
             default_color_space    = default_color_space_,
             default_color_transfer = default_color_transfer_,
             tag = this]() mutable {
                // VK mixer: 8-bit outputs BGRA (shader .bgra swizzle into R8G8B8A8),
                //           16-bit outputs RGBA directly (no B16G16R16A16 format exists).
                // OGL mixer: always BGRA (GL_BGRA readback format swaps R/B from internal RGBA).
                auto pf = (is_vulkan && depth != common::bit_depth::bit8) ? pixel_format::rgba : pixel_format::bgra;
                auto desc = pixel_format_desc(pf, default_color_space, default_color_transfer);
                desc.planes.push_back(pixel_format_desc::plane(format_desc.width, format_desc.height, 4, depth));
                auto tuple = std::move(result.get());
                auto& tex_ptr = std::get<1>(tuple);
                // Pass the shared_future<array<const uint8_t>> to const_frame for lazy readback.
                // GPU→CPU copy is deferred until a consumer actually calls image_data().
                auto frame = const_frame(tag, std::move(std::get<0>(tuple)), std::move(audio), desc, std::move(tex_ptr));
                return frame;
            }));

        return have_prev ? prev_result : const_frame{};
    }

    void set_master_volume(float volume) { audio_mixer_.set_master_volume(volume); }

    float get_master_volume() { return audio_mixer_.get_master_volume(); }

    void flush()
    {
        // Drain any stale deferred frames left in the 1-frame delay buffer.
        // This prevents the next consumer from receiving a frame that was
        // rendered for a previous producer.
        while (!buffer_.empty()) {
            buffer_.pop();
        }
    }
};

mixer::mixer(int channel_index, spl::shared_ptr<diagnostics::graph> graph, spl::shared_ptr<image_mixer> image_mixer, core::color_space default_color_space, core::color_transfer default_color_transfer, bool auto_color_convert, int auto_tone_map, float display_peak_luminance, float sdr_reference_white, bool auto_gamut_compress)
    : impl_(new impl(channel_index, std::move(graph), std::move(image_mixer), default_color_space, default_color_transfer, auto_color_convert, auto_tone_map, display_peak_luminance, sdr_reference_white, auto_gamut_compress))
{
}
void        mixer::set_master_volume(float volume) { impl_->set_master_volume(volume); }
float       mixer::get_master_volume() { return impl_->get_master_volume(); }
void        mixer::flush() { impl_->flush(); }
const_frame mixer::operator()(std::vector<draw_frame> frames, const video_format_desc& format_desc, int nb_samples)
{
    return (*impl_)(std::move(frames), format_desc, nb_samples);
}
mutable_frame mixer::create_frame(const void* tag, const pixel_format_desc& desc)
{
    return impl_->image_mixer_->create_frame(tag, desc);
}
core::monitor::state mixer::state() const { return impl_->state_; }

common::bit_depth mixer::depth() const { return impl_->image_mixer_->depth(); }

spl::shared_ptr<image_mixer> mixer::get_image_mixer() const { return impl_->image_mixer_; }

}} // namespace caspar::core
