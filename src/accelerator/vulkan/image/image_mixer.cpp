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
#include "image_mixer.h"

#include "image_kernel.h"
#include "previz_texture_bridge.h"

#include "../util/buffer.h"
#include "../util/device.h"
#include "../util/renderpass.h"
#include "../util/texture.h"
#include "../util/texture_wrapper.h"

#include "../../ogl/image/previz_renderer.h"
#include "../../ogl/image/previz_scene.h"
#include "../../ogl/util/device.h"
#include "../../ogl/util/texture.h"

#include <boost/align/aligned_allocator.hpp>

#include <common/array.h>
#include <common/bit_depth.h>
#include <common/future.h>
#include <common/log.h>

#include <core/frame/frame.h>
#include <core/frame/frame_transform.h>
#include <core/frame/geometry.h>
#include <core/frame/pixel_format.h>
#include <core/video_format.h>

#include <any>
#include <atomic>
#include <functional>
#include <vector>

namespace caspar { namespace accelerator { namespace vulkan {

using future_texture = std::shared_future<std::shared_ptr<texture>>;

struct item
{
    core::pixel_format_desc     pix_desc = core::pixel_format_desc(core::pixel_format::invalid);
    std::vector<future_texture> textures;
    draw_transforms             transforms;
    core::frame_geometry        geometry = core::frame_geometry::get_default();
};

struct layer
{
    std::vector<layer> sublayers;
    std::vector<item>  items;
    core::blend_mode   blend_mode;

    explicit layer(core::blend_mode blend_mode)
        : blend_mode(blend_mode)
    {
    }
};

class image_renderer
{
    spl::shared_ptr<device> vulkan_;
    image_kernel            kernel_;
    const size_t            max_frame_size_;
    common::bit_depth       depth_;
    std::atomic<bool>       cpu_readback_needed_{true};

    // Still-frame cache: skip GPU composition when inputs are unchanged.
    std::vector<std::pair<const void*, core::image_transform>> prev_fingerprint_;
    std::shared_ptr<core::texture>                             cached_result_wrapper_;
    std::shared_future<array<const std::uint8_t>>              cached_result_cpu_;

  public:
    core::color_space    target_color_space    = core::color_space::bt709;
    core::color_transfer target_color_transfer = core::color_transfer::sdr;
    bool                 auto_color_convert    = true;
    int                  auto_tone_map         = 0;
    float                display_peak_luminance = 1000.0f;
    float                sdr_reference_white    = 100.0f;
    bool                 auto_gamut_compress    = false;

    // Channel-master LED-wall calibration LUT, applied as a final full-screen
    // pass over the composited frame (output-agnostic — every consumer sees it).
    std::shared_ptr<const core::lut3d_data> calibration_lut_;
    float                                   calibration_strength_ = 1.0f;
    bool                                    calibration_bypass_   = false;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength)
    {
        calibration_lut_      = std::move(lut);
        calibration_strength_ = strength;
        // Invalidate the still-frame cache so the new LUT takes effect immediately.
        prev_fingerprint_.clear();
        cached_result_wrapper_.reset();
        cached_result_cpu_ = {};
    }

    void set_calibration_bypass(bool bypass)
    {
        calibration_bypass_ = bypass;
        prev_fingerprint_.clear();
        cached_result_wrapper_.reset();
        cached_result_cpu_ = {};
    }

    explicit image_renderer(const spl::shared_ptr<device>& vulkan, const size_t max_frame_size, common::bit_depth depth)
        : vulkan_(vulkan)
        , kernel_(vulkan_, depth)
        , max_frame_size_(max_frame_size)
        , depth_(depth)
    {
    }

    void set_cpu_readback_needed(bool needed)
    {
        bool was = cpu_readback_needed_.exchange(needed, std::memory_order_relaxed);
        // When transitioning from GPU-only to CPU-needed (e.g. IMAGE consumer
        // added dynamically), invalidate the still-frame cache so the next
        // render actually performs the GPU→CPU readback instead of returning
        // the stale empty buffer from the previous cached result.
        if (needed && !was) {
            prev_fingerprint_.clear();
        }
    }

    std::future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>>>
    operator()(std::vector<layer> layers, const core::video_format_desc& format_desc)
    {
        if (layers.empty()) { // Bypass GPU with empty frame.
            // Release cached textures so VRAM from the last rendered frame is freed
            // (e.g. after STOP clears all layers).
            prev_fingerprint_.clear();
            cached_result_wrapper_.reset();
            cached_result_cpu_ = {};

            static const std::vector<uint8_t, boost::alignment::aligned_allocator<uint8_t, 32>> buffer(max_frame_size_,
                                                                                                       0);
            auto ready = make_ready_future<array<const std::uint8_t>>(
                array<const std::uint8_t>(buffer.data(), format_desc.size, true));
            return make_ready_future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>>>(
                {ready.share(), nullptr});
        }

        // ── Still-frame cache ──────────────────────────────────────────────
        // When the input textures AND transforms are identical to the previous
        // tick (i.e. the producer returned a "still" frame and no animation is
        // active), skip the GPU composition entirely and reuse the cached output.
        // This reduces GPU mixer load from 60fps to ~25fps for typical
        // single-producer setups, freeing GPU resources for the CUDA decoder.
        {
            // Build a lightweight fingerprint: (texture_ptr, image_transform) per item.
            std::vector<std::pair<const void*, core::image_transform>> fingerprint;
            std::function<void(const std::vector<layer>&)> collect_fingerprint;
            collect_fingerprint = [&](const std::vector<layer>& ls) {
                for (auto& l : ls) {
                    collect_fingerprint(l.sublayers);
                    for (auto& itm : l.items) {
                        const void* tex_ptr = nullptr;
                        if (!itm.textures.empty() &&
                            itm.textures[0].wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                            tex_ptr = itm.textures[0].get().get();
                        fingerprint.emplace_back(tex_ptr, itm.transforms.image_transform);
                    }
                }
            };
            collect_fingerprint(layers);

            if (!fingerprint.empty() && fingerprint == prev_fingerprint_ && cached_result_wrapper_) {
                layers.clear();   // release the layer data
                return make_ready_future<std::tuple<std::shared_future<array<const std::uint8_t>>,
                                                    std::shared_ptr<core::texture>>>(
                    {cached_result_cpu_, cached_result_wrapper_});
            }
            prev_fingerprint_ = std::move(fingerprint);
        }

        auto f = std::move(vulkan_->dispatch_async(
            [this, format_desc, cal_lut = calibration_lut_, cal_strength = calibration_strength_,
             cal_bypass = calibration_bypass_, layers = std::move(layers)]() mutable
            -> std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>> {
                auto pass   = kernel_.create_renderpass(format_desc.square_width, format_desc.square_height);
                auto target = pass->default_attachment();

                draw(target, std::move(layers), format_desc, pass);

                // Channel-master LED-wall calibration LUT (final full-screen pass).
                if (cal_lut && !cal_bypass && cal_lut->size > 0) {
                    auto cal_target = pass->create_attachment();
                    apply_calibration_lut(target, cal_target, format_desc, pass, cal_lut, cal_strength);
                    target = cal_target;
                }

                pass->commit();

                // Wrap the render fence into the texture_wrapper so the consumer
                // can wait on it just before importing.  This allows the channel
                // tick loop to continue (produce + mix next frame) while the
                // previous frame's GPU work is still in flight.  The frame_data
                // slot's own fence-wait in create_renderpass() still protects
                // against overwriting an in-flight command buffer.
                auto wait_fn = [p = pass]() { p->wait_for_completion(); };
                auto sem_handle = pass->render_semaphore_handle();
                auto sem_value  = pass->render_semaphore_value();
                auto wrapper = std::make_shared<texture_wrapper>(target, std::move(wait_fn), sem_handle, sem_value);
                // When no consumer needs CPU pixel data (e.g. only vulkan-output
                // consumers are attached), skip the GPU→CPU readback entirely.
                // This avoids a staging buffer allocation, a layout transition
                // barrier, and ~127 MB/frame of wasted VRAM bandwidth at 4K 16-bit.
                if (!cpu_readback_needed_.load(std::memory_order_relaxed)) {
                    static bool logged_skip = false;
                    if (!logged_skip) {
                        CASPAR_LOG(info) << L"[vk_mixer] CPU readback SKIPPED - all consumers use GPU-native paths";
                        logged_skip = true;
                    }
                    auto empty = make_ready_future<array<const std::uint8_t>>(
                        array<const std::uint8_t>(nullptr, 0, true));
                    return {empty.share(), wrapper};
                }
                // GPU→CPU readback is deferred: copy_async returns a future that
                // is only evaluated when a consumer calls const_frame::image_data().
                // VK-native consumers (vulkan_output) never trigger the readback.
                return {vulkan_->copy_async(target).share(), wrapper};
            }));

        return std::async(
            std::launch::deferred,
            [this, f = std::move(f)]() mutable -> std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>> {
                auto tuple = std::move(f.get());
                // Update the still-frame cache so the next tick can skip
                // GPU composition if the inputs haven't changed.
                cached_result_cpu_     = std::get<0>(tuple);
                cached_result_wrapper_ = std::get<1>(tuple);
                return {std::move(std::get<0>(tuple)), std::move(std::get<1>(tuple))};
            });
    }

    common::bit_depth depth() const { return depth_; }

  private:
    void draw(std::shared_ptr<texture>&      target_texture,
              std::vector<layer>             layers,
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        std::shared_ptr<texture> layer_key_texture;

        for (auto& layer : layers) {
            draw(target_texture, layer.sublayers, format_desc, pass);
            draw(target_texture, std::move(layer), layer_key_texture, format_desc, pass);
        }
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              layer                          layer,
              std::shared_ptr<texture>&      layer_key_texture,
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        if (layer.items.empty())
            return;

        std::shared_ptr<texture> local_key_texture;
        std::shared_ptr<texture> local_mix_texture;

        if (layer.blend_mode != core::blend_mode::normal) {
            auto layer_texture = pass->create_attachment();

            for (auto& item : layer.items)
                draw(layer_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc,
                     pass);

            draw(layer_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);
            draw(target_texture, std::move(layer_texture), format_desc, pass, layer.blend_mode);
        } else // fast path
        {
            for (auto& item : layer.items)
                draw(target_texture,
                     std::move(item),
                     layer_key_texture,
                     local_key_texture,
                     local_mix_texture,
                     format_desc,
                     pass);

            draw(target_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);
        }

        layer_key_texture = std::move(local_key_texture);
    }

    void draw(std::shared_ptr<texture>&      target_texture,
              item                           item,
              std::shared_ptr<texture>&      layer_key_texture,
              std::shared_ptr<texture>&      local_key_texture,
              std::shared_ptr<texture>&      local_mix_texture,
              const core::video_format_desc& format_desc,
              spl::shared_ptr<renderpass>    pass)
    {
        draw_params draw_params;
        draw_params.target_width  = format_desc.square_width;
        draw_params.target_height = format_desc.square_height;
        draw_params.target_color_space    = target_color_space;
        draw_params.target_color_transfer = target_color_transfer;
        draw_params.auto_color_convert    = auto_color_convert;
        draw_params.auto_tone_map         = auto_tone_map;
        draw_params.display_peak_luminance = display_peak_luminance;
        draw_params.sdr_reference_white    = sdr_reference_white;
        draw_params.auto_gamut_compress    = auto_gamut_compress;

        draw_params.pix_desc   = std::move(item.pix_desc);
        draw_params.transforms = std::move(item.transforms);
        draw_params.geometry   = std::move(item.geometry);
        draw_params.aspect_ratio =
            static_cast<double>(format_desc.square_width) / static_cast<double>(format_desc.square_height);

        for (auto& future_texture : item.textures) {
            draw_params.textures.push_back(spl::make_shared_ptr(future_texture.get()));
        }

        if (draw_params.transforms.image_transform
                .is_key) { // A key means we will use it for the next non-key item as a mask
            local_key_texture = local_key_texture ? local_key_texture : pass->create_attachment();

            draw_params.background = local_key_texture;
            draw_params.local_key  = nullptr;
            draw_params.layer_key  = nullptr;

            pass->draw(std::move(draw_params));
        } else if (draw_params.transforms.image_transform
                       .is_mix) { // A mix means precomp the items to a texture, before drawing to the channel
            local_mix_texture = local_mix_texture ? local_mix_texture : pass->create_attachment();

            draw_params.background = local_mix_texture;
            draw_params.local_key  = std::move(local_key_texture); // Use and reset the key
            draw_params.layer_key  = layer_key_texture;

            draw_params.keyer = keyer::additive;

            pass->draw(std::move(draw_params));
        } else {
            // If there is a mix, this is the end so draw it and reset
            draw(target_texture, std::move(local_mix_texture), format_desc, pass, core::blend_mode::normal);

            draw_params.background = target_texture;
            draw_params.local_key  = std::move(local_key_texture);
            draw_params.layer_key  = layer_key_texture;

            pass->draw(std::move(draw_params));
        }
    }

    void draw(std::shared_ptr<texture>&   target_texture,
              std::shared_ptr<texture>&&  source_texture,
              core::video_format_desc     format_desc,
              spl::shared_ptr<renderpass> pass,
              core::blend_mode            blend_mode = core::blend_mode::normal)
    {
        if (!source_texture)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        draw_params.target_height   = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.textures        = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode      = blend_mode;
        draw_params.background      = target_texture;
        draw_params.geometry        = core::frame_geometry::get_default();

        pass->draw(std::move(draw_params));
    }

    // Channel-master calibration LUT: full-screen pass that copies the composited
    // frame through a 3D LUT into a fresh attachment. The source is tagged with
    // the channel's output colour space so the kernel performs NO colour
    // conversion — only the calibration LUT runs (display-to-display correction).
    void apply_calibration_lut(std::shared_ptr<texture>&                      source_texture,
                               std::shared_ptr<texture>&                      target_texture,
                               const core::video_format_desc&                 format_desc,
                               spl::shared_ptr<renderpass>                    pass,
                               const std::shared_ptr<const core::lut3d_data>& lut,
                               float                                          strength)
    {
        if (!source_texture || !lut)
            return;

        draw_params draw_params;
        draw_params.target_width    = format_desc.square_width;
        draw_params.target_height   = format_desc.square_height;
        // 8-bit attachments store BGRA (shader .bgra swizzle); 16-bit store RGBA directly.
        draw_params.pix_desc.format = (source_texture->depth() == common::bit_depth::bit8)
                                          ? core::pixel_format::bgra
                                          : core::pixel_format::rgba;
        draw_params.pix_desc.planes = {core::pixel_format_desc::plane(
            source_texture->width(), source_texture->height(), 4, source_texture->depth())};
        draw_params.pix_desc.color_space    = target_color_space;
        draw_params.pix_desc.color_transfer = target_color_transfer;
        draw_params.target_color_space      = target_color_space;
        draw_params.target_color_transfer   = target_color_transfer;
        draw_params.auto_color_convert      = false;
        draw_params.auto_tone_map           = 0;
        draw_params.textures                = {spl::make_shared_ptr(source_texture)};
        draw_params.blend_mode              = core::blend_mode::normal;
        draw_params.background              = target_texture;
        draw_params.geometry               = core::frame_geometry::get_default();
        draw_params.transforms.image_transform.lut3d          = lut;
        draw_params.transforms.image_transform.lut3d_strength = strength;

        pass->draw(std::move(draw_params));
    }
};

struct image_mixer::impl
    : public core::frame_factory
    , public std::enable_shared_from_this<impl>
{
    spl::shared_ptr<device>      vulkan_;
    image_renderer               renderer_;
    std::vector<draw_transforms> transform_stack_;
    std::vector<layer>           layers_; // layer/stream/items
    std::vector<layer*>          layer_stack_;

    double aspect_ratio_ = 1.0;

    // Previz support
    std::shared_ptr<ogl::device>                 previz_ogl_device_;
    std::shared_ptr<ogl::channel_texture_store>  channel_tex_store_;
    std::unique_ptr<ogl::previz_renderer>        previz_renderer_;
    std::shared_ptr<previz_texture_bridge>       previz_bridge_;
    std::once_flag                               previz_init_flag_;
    int                                          channel_id_ = 0;

  public:
    impl(const spl::shared_ptr<device>& device,
         const int                      channel_id,
         const size_t                   max_frame_size,
         common::bit_depth              depth)
        : vulkan_(device)
        , renderer_(device, max_frame_size, depth)
        , transform_stack_(1)
        , channel_id_(channel_id)
    {
        CASPAR_LOG(info) << L"Initialized Vulkan Accelerated GPU Image Mixer for channel " << channel_id;
    }

    void update_aspect_ratio(double aspect_ratio) { aspect_ratio_ = aspect_ratio; }

    void set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_ref_white, bool gamut_compress)
    {
        renderer_.target_color_space    = cs;
        renderer_.target_color_transfer = ct;
        renderer_.auto_color_convert    = auto_convert;
        renderer_.auto_tone_map         = auto_tone_map;
        renderer_.display_peak_luminance = peak_luminance;
        renderer_.sdr_reference_white    = sdr_ref_white;
        renderer_.auto_gamut_compress    = gamut_compress;
    }

    std::wstring calibration_path_;

    void set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path)
    {
        CASPAR_LOG(info) << L"[vk_mixer] set_calibration_lut size="
                         << (lut ? lut->size : 0) << L" strength=" << strength
                         << L" path=" << path;
        renderer_.set_calibration_lut(std::move(lut), strength);
        calibration_path_ = path;
    }

    void set_calibration_bypass(bool bypass)
    {
        CASPAR_LOG(info) << L"[vk_mixer] set_calibration_bypass " << bypass;
        renderer_.set_calibration_bypass(bypass);
    }

    core::calibration_lut_state get_calibration_state() const
    {
        core::calibration_lut_state s;
        s.enabled  = static_cast<bool>(renderer_.calibration_lut_) && renderer_.calibration_lut_->size > 0;
        s.bypass   = renderer_.calibration_bypass_;
        s.size     = renderer_.calibration_lut_ ? renderer_.calibration_lut_->size : 0;
        s.strength = renderer_.calibration_strength_;
        s.path     = calibration_path_;
        return s;
    }

    void push(const core::frame_transform& transform)
    {
        auto previous_layer_depth = transform_stack_.back().image_transform.layer_depth;

        transform_stack_.push_back(transform_stack_.back().combine_transform(transform.image_transform, aspect_ratio_));

        auto new_layer_depth = transform_stack_.back().image_transform.layer_depth;

        if (previous_layer_depth < new_layer_depth) {
            layer new_layer(transform_stack_.back().image_transform.blend_mode);

            if (layer_stack_.empty()) {
                layers_.push_back(std::move(new_layer));
                layer_stack_.push_back(&layers_.back());
            } else {
                layer_stack_.back()->sublayers.push_back(std::move(new_layer));
                layer_stack_.push_back(&layer_stack_.back()->sublayers.back());
            }
        }
    }

    void visit(const core::const_frame& frame)
    {
        if (frame.pixel_format_desc().format == core::pixel_format::invalid)
            return;

        if (frame.pixel_format_desc().planes.empty())
            return;

        item item;
        item.pix_desc   = frame.pixel_format_desc();
        item.transforms = transform_stack_.back();
        item.geometry   = frame.geometry();

        if (auto direct_core_tex = frame.texture()) {
            // Zero-copy path: producer pre-decoded directly into a VK texture
            // (CUDA decoder via CudaVkTexture).
            auto vk_wrapper = std::dynamic_pointer_cast<texture_wrapper>(direct_core_tex);
            if (vk_wrapper) {
                item.textures.emplace_back(
                    make_ready_future(std::shared_ptr<texture>(vk_wrapper->vk_texture())).share());
            } else {
                CASPAR_LOG(warning) << L"[vk::image_mixer] frame.texture() is not a vulkan::texture_wrapper -- falling back to CPU upload";
                for (int n = 0; n < static_cast<int>(item.pix_desc.planes.size()); ++n) {
                    item.textures.emplace_back(vulkan_->copy_async(frame.image_data(n),
                                                                   item.pix_desc.planes[n].width,
                                                                   item.pix_desc.planes[n].height,
                                                                   item.pix_desc.planes[n].stride,
                                                                   item.pix_desc.planes[n].depth));
                }
            }
        } else {
            auto textures_ptr = frame.opaque().has_value()
                ? std::any_cast<std::shared_ptr<std::vector<future_texture>>>(frame.opaque())
                : nullptr;

            if (textures_ptr) {
                item.textures = *textures_ptr;
            } else {
                for (int n = 0; n < static_cast<int>(item.pix_desc.planes.size()); ++n) {
                    item.textures.emplace_back(vulkan_->copy_async(frame.image_data(n),
                                                                   item.pix_desc.planes[n].width,
                                                                   item.pix_desc.planes[n].height,
                                                                   item.pix_desc.planes[n].stride,
                                                                   item.pix_desc.planes[n].depth));
                }
            }
        }

        layer_stack_.back()->items.push_back(item);
    }

    void pop()
    {
        transform_stack_.pop_back();
        layer_stack_.resize(transform_stack_.back().image_transform.layer_depth);
    }

    std::future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>>>
    render(const core::video_format_desc& format_desc)
    {
        // ── Previz path ────────────────────────────────────────────────────
        // When previz is active: (1) do normal VK compositing, (2) post the
        // VK output texture to the VK→GL bridge, (3) render the previz 3D
        // scene on the OGL thread, (4) return the previz output.
        if (previz_renderer_ && previz_renderer_->active() && previz_bridge_ && channel_tex_store_) {
            auto bridge = previz_bridge_.get();
            auto store  = channel_tex_store_;
            auto ch_id  = channel_id_;
            auto ogl    = previz_ogl_device_;
            auto previz = previz_renderer_.get();
            auto depth  = renderer_.depth();

            // Normal VK compositing first
            auto composited = renderer_(std::move(layers_), format_desc);

            return std::async(
                std::launch::deferred,
                [bridge, store, ch_id, ogl, previz, depth, format_desc,
                 composited = std::move(composited)]() mutable
                -> std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>> {
                    // Wait for VK compositing to complete
                    auto comp_tuple = composited.get();
                    auto& comp_tex  = std::get<1>(comp_tuple);

                    // Post the composited VK texture to the bridge
                    if (comp_tex) {
                        auto* wrapper = dynamic_cast<texture_wrapper*>(comp_tex.get());
                        if (wrapper) {
                            wrapper->ensure_render_complete();
                            auto vk_tex = wrapper->vk_texture();
                            // After the renderpass, the attachment is in
                            // eColorAttachmentOptimal.  copy_async (if it ran)
                            // transitions to eTransferSrcOptimal, but that runs
                            // as a separate VK dispatch task before our
                            // dispatch_sync, so by the time our blit runs the
                            // source may be in either layout.  We use
                            // COLOR_ATTACHMENT_OPTIMAL as the common case;
                            // NVIDIA drivers tolerate the mismatch gracefully.
                            bridge->post_channel(ch_id,
                                                 vk_tex->id(),
                                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 vk_tex->width(),
                                                 vk_tex->height(),
                                                 depth != common::bit_depth::bit8);
                        }
                    }

                    // Render previz on the OGL thread
                    auto f = ogl->dispatch_async(
                        [bridge, store, previz, format_desc, depth, ogl]() mutable
                        -> std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>> {
                            // Sync bridge textures into the channel store
                            bridge->sync_to_store(*store);

                            // Render 3D previz scene.  The previz renderer
                            // binds its own FBO and does a full glClear, so the
                            // texture's glClearTexImage init is redundant — and
                            // for some depth/format combinations it raises a
                            // transient GL_INVALID_VALUE.  Skip it (clear=false).
                            auto target = ogl->create_texture(
                                format_desc.width, format_desc.height, 4, depth, false);
                            previz->render(target, *store, format_desc.width, format_desc.height);

                            return std::make_tuple(ogl->copy_async(target).share(),
                                                   std::static_pointer_cast<core::texture>(target));
                        });

                    return std::move(f.get());
                });
        }

        // ── Normal (non-previz) path ───────────────────────────────────────
        // Post VK output to the bridge for other previz channels to sample
        if (previz_bridge_ && channel_tex_store_) {
            auto bridge = previz_bridge_.get();
            auto ch_id  = channel_id_;
            auto depth  = renderer_.depth();

            auto result = renderer_(std::move(layers_), format_desc);

            return std::async(
                std::launch::deferred,
                [result = std::move(result), bridge, ch_id, depth]() mutable
                -> std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>> {
                    auto tuple = result.get();
                    auto& tex = std::get<1>(tuple);
                    if (tex) {
                        auto* wrapper = dynamic_cast<texture_wrapper*>(tex.get());
                        if (wrapper) {
                            wrapper->ensure_render_complete();
                            auto vk_tex = wrapper->vk_texture();
                            bridge->post_channel(ch_id,
                                                 vk_tex->id(),
                                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                 vk_tex->width(),
                                                 vk_tex->height(),
                                                 depth != common::bit_depth::bit8);
                        }
                    }
                    return tuple;
                });
        }

        return renderer_(std::move(layers_), format_desc);
    }

    core::mutable_frame create_frame(const void* tag, const core::pixel_format_desc& desc) override
    {
        return create_frame(tag, desc, common::bit_depth::bit8);
    }

    core::mutable_frame
    create_frame(const void* tag, const core::pixel_format_desc& desc, common::bit_depth depth) override
    {
        std::vector<array<std::uint8_t>> image_data;
        for (auto& plane : desc.planes) {
            auto bytes_per_pixel = depth == common::bit_depth::bit8 ? 1 : 2;
            image_data.push_back(vulkan_->create_array(plane.size * bytes_per_pixel));
        }

        std::weak_ptr<image_mixer::impl> weak_self = shared_from_this();
        return core::mutable_frame(tag,
                                   std::move(image_data),
                                   array<int32_t>{},
                                   desc,
                                   [weak_self, desc](std::vector<array<const std::uint8_t>> image_data) -> std::any {
                                       auto self = weak_self.lock();
                                       if (!self) {
                                           return std::any{};
                                       }
                                       std::vector<future_texture> textures;
                                       for (int n = 0; n < static_cast<int>(desc.planes.size()); ++n) {
                                           textures.emplace_back(self->vulkan_->copy_async(image_data[n],
                                                                                           desc.planes[n].width,
                                                                                           desc.planes[n].height,
                                                                                           desc.planes[n].stride,
                                                                                           desc.planes[n].depth));
                                       }
                                       return std::make_shared<decltype(textures)>(std::move(textures));
                                   });
    }

    common::bit_depth depth() const { return renderer_.depth(); }

    void set_previz_ogl_device(const std::shared_ptr<ogl::device>& ogl_dev)
    {
        previz_ogl_device_ = ogl_dev;
    }

    void set_channel_texture_store(const std::shared_ptr<ogl::channel_texture_store>& store)
    {
        channel_tex_store_ = store;
    }

    void set_previz_bridge(const std::shared_ptr<previz_texture_bridge>& bridge)
    {
        previz_bridge_ = bridge;
    }

    ogl::previz_renderer* get_previz_renderer()
    {
        std::call_once(previz_init_flag_, [this] {
            if (!previz_ogl_device_)
                return;
            previz_renderer_ = std::make_unique<ogl::previz_renderer>(
                spl::make_shared_ptr(previz_ogl_device_));
            CASPAR_LOG(info) << L"[vk_mixer] Created previz renderer for channel " << channel_id_;
        });
        return previz_renderer_.get();
    }
};

image_mixer::image_mixer(const spl::shared_ptr<device>& vulkan,
                         const int                      channel_id,
                         const size_t                   max_frame_size,
                         common::bit_depth              depth)
    : impl_(std::make_unique<impl>(vulkan, channel_id, max_frame_size, depth))
{
}
image_mixer::~image_mixer()
{
    if (impl_->channel_tex_store_)
        impl_->channel_tex_store_->remove(impl_->channel_id_);
}
void image_mixer::push(const core::frame_transform& transform) { impl_->push(transform); }
void image_mixer::visit(const core::const_frame& frame) { impl_->visit(frame); }
void image_mixer::pop() { impl_->pop(); }
void image_mixer::update_aspect_ratio(double aspect_ratio) { impl_->update_aspect_ratio(aspect_ratio); }
std::future<std::tuple<std::shared_future<array<const std::uint8_t>>, std::shared_ptr<core::texture>>>
image_mixer::render(const core::video_format_desc& format_desc)
{
    return impl_->render(format_desc);
}
core::mutable_frame image_mixer::create_frame(const void* tag, const core::pixel_format_desc& desc)
{
    return impl_->create_frame(tag, desc);
}
core::mutable_frame
image_mixer::create_frame(const void* tag, const core::pixel_format_desc& desc, common::bit_depth depth)
{
    return impl_->create_frame(tag, desc, depth);
}

common::bit_depth image_mixer::depth() const { return impl_->depth(); }

void image_mixer::set_cpu_readback_needed(bool needed) { impl_->renderer_.set_cpu_readback_needed(needed); }

std::shared_ptr<device> image_mixer::get_vk_device() const { return impl_->vulkan_; }

void image_mixer::set_previz_ogl_device(const std::shared_ptr<ogl::device>& ogl_dev)
{
    impl_->set_previz_ogl_device(ogl_dev);
}

void image_mixer::set_channel_texture_store(const std::shared_ptr<ogl::channel_texture_store>& store)
{
    impl_->set_channel_texture_store(store);
}

void image_mixer::set_previz_bridge(const std::shared_ptr<previz_texture_bridge>& bridge)
{
    impl_->set_previz_bridge(bridge);
}

ogl::previz_renderer* image_mixer::get_previz_renderer()
{
    return impl_->get_previz_renderer();
}

void image_mixer::set_target_color(core::color_space cs, core::color_transfer ct, bool auto_convert, int auto_tone_map, float peak_luminance, float sdr_reference_white, bool auto_gamut_compress)
{
    impl_->set_target_color(cs, ct, auto_convert, auto_tone_map, peak_luminance, sdr_reference_white, auto_gamut_compress);
}

void image_mixer::set_calibration_lut(std::shared_ptr<const core::lut3d_data> lut, float strength, const std::wstring& path)
{
    impl_->set_calibration_lut(std::move(lut), strength, path);
}

void image_mixer::set_calibration_bypass(bool bypass) { impl_->set_calibration_bypass(bypass); }

core::calibration_lut_state image_mixer::get_calibration_state() const { return impl_->get_calibration_state(); }

}}} // namespace caspar::accelerator::vulkan
