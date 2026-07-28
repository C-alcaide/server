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
 */

#include "ofx_producer.h"

#include "../bridge/ofx_image_bridge.h"
#include "../host/ofx_host.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/array.h>
#include <common/log.h>
#include <common/tweener.h>
#include <common/utf.h>

#if defined(CASPAR_OFX_VULKAN_CUDA)
#include <accelerator/vulkan/image/image_mixer.h>
#include <accelerator/vulkan/util/device.h>
#include "../../cuda_vk_texture.h"
#include <cuda_runtime.h>
#endif

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/frame_visitor.h>
#include <core/frame/pixel_format.h>
#include <core/producer/frame_producer_registry.h>

#include <common/bit_depth.h>

#include <boost/algorithm/string.hpp>

#include <algorithm>
#include <cstdint>
#include <future>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace caspar { namespace ofx {

namespace {

/// Extracts the first (top-most) const_frame carrying image data from a draw_frame.
class extract_visitor : public core::frame_visitor
{
  public:
    core::const_frame frame;

    void push(const core::frame_transform&) override {}
    void pop() override {}
    void visit(const core::const_frame& f) override
    {
        if (!frame && f)
            frame = f;
    }
};

/// Map CasparCG's field to the OFX field-to-render. Progressive is unfielded; interlaced fields
/// are signalled as "both" (we hand the plug-in the full frame; CasparCG interlaces at output).
inline field_kind to_field_kind(core::video_field f)
{
    return f == core::video_field::progressive ? field_kind::none : field_kind::both;
}

/// Format a double for AMCP metadata output: fixed-point, trailing zeros trimmed.
inline std::wstring fmt_num(double v)
{
    std::wostringstream os;
    os << std::fixed << std::setprecision(6) << v;
    std::wstring s = os.str();
    if (s.find(L'.') != std::wstring::npos) {
        s.erase(s.find_last_not_of(L'0') + 1);
        if (!s.empty() && s.back() == L'.')
            s.pop_back();
    }
    return s;
}

/// Format a range limit: values at/beyond the float range are reported as unbounded ("*").
inline std::wstring fmt_lim(double v)
{
    return (v <= -1e37 || v >= 1e37) ? std::wstring(L"*") : fmt_num(v);
}

class ofx_producer : public core::frame_producer
{
    spl::shared_ptr<core::frame_producer> source_;
    std::shared_ptr<core::frame_producer> source_to_; ///< second input, transition context only (nullable)
    spl::shared_ptr<core::frame_factory>  frame_factory_;
    std::wstring                          plugin_id_;

    /// The OGL mixer device (non-null only when the channel uses the OpenGL mixer). When present
    /// and the plug-in supports OpenGL, filter frames render zero-copy into a mixer-owned texture.
    std::shared_ptr<accelerator::ogl::device> ogl_device_;

#if defined(CASPAR_OFX_VULKAN_CUDA)
    /// The Vulkan mixer device (non-null only on the Vulkan mixer). When present and the plug-in
    /// supports CUDA, filter frames render zero-copy into an exportable VK texture via CUDA interop.
    std::shared_ptr<accelerator::vulkan::device> vk_device_;
    /// A small pool of CUDA-Vulkan interop textures reused across frames (avoids per-frame VK alloc
    /// + cudaImportExternalMemory). A texture is reusable once no draw_frame still references it.
    std::vector<std::shared_ptr<CudaVkTexture>> vk_cuda_pool_;
#endif

    std::unique_ptr<effect>    effect_;
    std::vector<std::uint8_t>  src_rgba_;
    std::vector<std::uint8_t>  src_to_rgba_; ///< bridged SourceTo buffer (transition context)
    std::vector<std::uint8_t>  dst_rgba_;
    int                        width_  = 0;
    int                        height_ = 0;
    int                        bytes_per_channel_ = 0; ///< 0 = effect not yet created
    int                        working_bytes_     = 1; ///< negotiated OFX depth (1/2/4)
    double                     fps_               = 25.0;
    effect_context             context_           = effect_context::filter;
    int                        transition_frames_ = 25; ///< frames over which the transition ramps 0->1
    std::uint32_t              frame_number_of_effect_ = 0;
    std::mutex                 effect_mutex_;

    /// Per-parameter keyframe animation (decoupled from the MIXER keyframe engine, which
    /// targets image_transform). Keyframes hold one value per component; interpolation reuses
    /// CasparCG's tweener. Evaluated each frame and pushed via effect_->set_param().
    struct keyframe
    {
        double              frame = 0.0;
        std::vector<double> values;
    };
    std::map<std::string, std::vector<keyframe>> anim_;   ///< keyframes per param, sorted by frame
    std::map<std::string, tweener>               tween_;  ///< tween function per param

    std::vector<double> evaluate_anim(const std::vector<keyframe>& keys, const tweener& tw, double frame) const
    {
        if (keys.empty())
            return {};
        if (frame <= keys.front().frame)
            return keys.front().values;
        if (frame >= keys.back().frame)
            return keys.back().values;

        for (std::size_t i = 1; i < keys.size(); ++i) {
            if (frame <= keys[i].frame) {
                const auto&  k0  = keys[i - 1];
                const auto&  k1  = keys[i];
                const double dur = k1.frame - k0.frame;
                const double t   = dur > 0.0 ? (frame - k0.frame) / dur : 0.0;

                const std::size_t   n = std::max(k0.values.size(), k1.values.size());
                std::vector<double> out(n, 0.0);
                for (std::size_t c = 0; c < n; ++c) {
                    const double b = c < k0.values.size() ? k0.values[c] : 0.0;
                    const double e = c < k1.values.size() ? k1.values[c] : 0.0;
                    out[c]         = tw(t, b, e - b, 1.0);
                }
                return out;
            }
        }
        return keys.back().values;
    }

    void apply_animation(double frame)
    {
        for (const auto& [name, keys] : anim_) {
            auto it     = tween_.find(name);
            auto values = evaluate_anim(keys, it != tween_.end() ? it->second : tweener(L"linear"), frame);
            if (!values.empty())
                effect_->set_param(name, values, frame);
        }
    }

    /// (Re)create the effect for the given frame size and bit depth. Must hold effect_mutex_.
    bool ensure_effect(int w, int h, int bytes)
    {
        if (effect_ && effect_->valid() && w == width_ && h == height_ && bytes == bytes_per_channel_)
            return true;

        effect_ = global_host().create_effect(u8(plugin_id_), context_, w, h, fps_, bytes);
        if (!effect_ || !effect_->valid()) {
            effect_.reset();
            return false;
        }

        width_             = w;
        height_            = h;
        bytes_per_channel_ = bytes;
        working_bytes_     = effect_->working_bytes(); // negotiated OFX depth: 1, 2 or 4 (float)
        // Working buffers are sized by the OFX working depth (float = 16 bytes/pixel).
        src_rgba_.assign(static_cast<std::size_t>(w) * h * 4 * working_bytes_, 0);
        dst_rgba_.assign(static_cast<std::size_t>(w) * h * 4 * working_bytes_, 0);
        if (context_ == effect_context::transition)
            src_to_rgba_.assign(static_cast<std::size_t>(w) * h * 4 * working_bytes_, 0);
        const wchar_t* depth = working_bytes_ == 4 ? L"float32" : (working_bytes_ == 2 ? L"16-bit" : L"8-bit");
        CASPAR_LOG(info) << L"[ofx] Created effect '" << plugin_id_ << L"' at " << w << L"x" << h << L" (source "
                         << (bytes == 2 ? L"16-bit" : L"8-bit") << L", working " << depth << L").";
        return true;
    }

#if defined(CASPAR_OFX_VULKAN_CUDA)
    /// Acquire a CUDA-Vulkan interop texture (w x h) for zero-copy CUDA output: reuse a pooled one
    /// no longer referenced by any in-flight frame, else create a new one (pool capped).
    std::shared_ptr<CudaVkTexture> acquire_vk_cuda_texture(int w, int h)
    {
        for (auto& t : vk_cuda_pool_) {
            if (t && t->is_free() && t->vk_texture()->width() == w && t->vk_texture()->height() == h)
                return t;
        }
        if (vk_cuda_pool_.size() < 4) {
            try {
                auto vk_tex = vk_device_->create_exportable_texture(w, h, 4, common::bit_depth::bit8);
                auto cvt = std::make_shared<CudaVkTexture>(vk_tex, static_cast<VkDevice>(vk_device_->getVkDevice()));
                vk_cuda_pool_.push_back(cvt);
                return cvt;
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << L"[ofx] CUDA-VK interop texture creation failed: " << u16(e.what());
                return nullptr;
            }
        }
        return nullptr; // all pooled textures still in flight -> skip zero-copy this frame
    }
#endif

  public:
    ofx_producer(const core::frame_producer_dependencies&    dependencies,
                 std::wstring                                 plugin_id,
                 const spl::shared_ptr<core::frame_producer>& source,
                 effect_context                               context)
        : source_(source)
        , frame_factory_(dependencies.frame_factory)
        , plugin_id_(std::move(plugin_id))
        , width_(dependencies.format_desc.width)
        , height_(dependencies.format_desc.height)
        , context_(context)
    {
        fps_ = dependencies.format_desc.duration != 0
                   ? static_cast<double>(dependencies.format_desc.time_scale) /
                         static_cast<double>(dependencies.format_desc.duration)
                   : 25.0;
        // Resolve the OGL mixer device for the zero-copy GL path (null on the Vulkan mixer, in
        // which case filter frames use the CPU/self-contained GL readback path).
        if (auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get()))
            ogl_device_ = ogl_mixer->get_ogl_device();
#if defined(CASPAR_OFX_VULKAN_CUDA)
        if (auto* vk_mixer = dynamic_cast<accelerator::vulkan::image_mixer*>(frame_factory_.get()))
            vk_device_ = vk_mixer->get_vk_device();
#endif
        // The effect is created lazily on the first frame, once its true bit depth is known.
    }

    /// Transition constructor: two sources (From/To) blended by the plug-in's "Transition" param,
    /// which ramps 0->1 over transition_frames.
    ofx_producer(const core::frame_producer_dependencies&    dependencies,
                 std::wstring                                 plugin_id,
                 const spl::shared_ptr<core::frame_producer>& source_from,
                 const spl::shared_ptr<core::frame_producer>& source_to,
                 int                                          transition_frames)
        : source_(source_from)
        , source_to_(source_to)
        , frame_factory_(dependencies.frame_factory)
        , plugin_id_(std::move(plugin_id))
        , width_(dependencies.format_desc.width)
        , height_(dependencies.format_desc.height)
        , context_(effect_context::transition)
        , transition_frames_(transition_frames > 0 ? transition_frames : 25)
    {
        fps_ = dependencies.format_desc.duration != 0
                   ? static_cast<double>(dependencies.format_desc.time_scale) /
                         static_cast<double>(dependencies.format_desc.duration)
                   : 25.0;
        if (auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get()))
            ogl_device_ = ogl_mixer->get_ogl_device();
    }

    core::draw_frame receive_transition(const core::video_field field, int nb_samples)
    {
        auto from_df = source_->receive(field, nb_samples);
        auto to_df   = source_to_->receive(field, nb_samples);
        if (!from_df)
            return from_df;
        if (!to_df)
            return from_df; // To not ready yet -> show From

        extract_visitor vf;
        from_df.accept(vf);
        extract_visitor vt;
        to_df.accept(vt);
        const auto cf_from = vf.frame;
        const auto cf_to   = vt.frame;
        if (!cf_from || !cf_to)
            return from_df;

        const auto& pf = cf_from.pixel_format_desc();
        const auto& pt = cf_to.pixel_format_desc();
        auto ok = [](const core::pixel_format_desc& d) {
            return (d.format == core::pixel_format::bgra || d.format == core::pixel_format::rgba) &&
                   d.planes.size() == 1 && d.planes[0].depth == common::bit_depth::bit8;
        };
        if (!ok(pf) || !ok(pt))
            return from_df;

        const int w = static_cast<int>(cf_from.width());
        const int h = static_cast<int>(cf_from.height());
        if (static_cast<int>(cf_to.width()) != w || static_cast<int>(cf_to.height()) != h)
            return from_df; // both inputs must share dimensions

        const bool from_rgba = pf.format == core::pixel_format::rgba;
        const bool to_rgba   = pt.format == core::pixel_format::rgba;

        bool rendered = false;
        {
            std::lock_guard<std::mutex> lock(effect_mutex_);
            if (!ensure_effect(w, h, 1) || working_bytes_ != 1)
                return from_df;

            const int ws = w * 4;
            if (from_rgba)
                rgba_top_down_to_rgba_bottom_up(cf_from.image_data(0).data(), pf.planes[0].linesize, src_rgba_.data(), ws, w, h);
            else
                bgra_top_down_to_rgba_bottom_up(cf_from.image_data(0).data(), pf.planes[0].linesize, src_rgba_.data(), ws, w, h);
            if (pf.is_straight_alpha)
                premultiply_rgba8(src_rgba_.data(), ws, w, h);

            if (to_rgba)
                rgba_top_down_to_rgba_bottom_up(cf_to.image_data(0).data(), pt.planes[0].linesize, src_to_rgba_.data(), ws, w, h);
            else
                bgra_top_down_to_rgba_bottom_up(cf_to.image_data(0).data(), pt.planes[0].linesize, src_to_rgba_.data(), ws, w, h);
            if (pt.is_straight_alpha)
                premultiply_rgba8(src_to_rgba_.data(), ws, w, h);

            const double frame = static_cast<double>(frame_number_of_effect_);
            double       t     = transition_frames_ > 0 ? frame / static_cast<double>(transition_frames_) : 1.0;
            t                  = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);

            apply_animation(frame);
            rendered = effect_->render_transition(
                src_rgba_.data(), src_to_rgba_.data(), dst_rgba_.data(), w, h, frame, t, to_field_kind(field));
            ++frame_number_of_effect_;
        }
        if (!rendered)
            return from_df;

        core::pixel_format_desc out_pfd = pf;
        out_pfd.is_straight_alpha       = !effect_->output_premultiplied();
        auto      out        = frame_factory_->create_frame(this, out_pfd);
        const int dst_stride = out.pixel_format_desc().planes[0].linesize;
        const int ws         = w * 4;
        if (from_rgba)
            rgba_bottom_up_to_rgba_top_down(dst_rgba_.data(), ws, out.image_data(0).data(), dst_stride, w, h);
        else
            rgba_bottom_up_to_bgra_top_down(dst_rgba_.data(), ws, out.image_data(0).data(), dst_stride, w, h);

        // Audio: pass the From source's audio through (audio crossfade is out of scope).
        const auto& src_audio = cf_from.audio_data();
        if (src_audio.size() > 0)
            out.audio_data() = std::vector<std::int32_t>(src_audio.data(), src_audio.data() + src_audio.size());

        return core::draw_frame(std::move(out));
    }

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        // Generator context: no source clip; render straight into a fresh frame.
        if (context_ == effect_context::generator) {
            core::pixel_format_desc pfd(core::pixel_format::bgra);
            pfd.planes.push_back(core::pixel_format_desc::plane(width_, height_, 4));

            bool rendered = false;
            {
                std::lock_guard<std::mutex> lock(effect_mutex_);
                if (!ensure_effect(width_, height_, 1))
                    return core::draw_frame::empty();
                apply_animation(frame_number_of_effect_);
                rendered = effect_->render(
                    nullptr, dst_rgba_.data(), width_, height_, frame_number_of_effect_++, to_field_kind(field));
            }
            if (!rendered)
                return core::draw_frame::empty();

            auto out = frame_factory_->create_frame(this, pfd);
            rgba_bottom_up_to_bgra_top_down(
                dst_rgba_.data(), width_ * 4, out.image_data(0).data(), pfd.planes[0].linesize, width_, height_);
            return core::draw_frame(std::move(out));
        }

        // Transition context: blend two sources (From/To) by the plug-in's "Transition" param.
        if (context_ == effect_context::transition) {
            return receive_transition(field, nb_samples);
        }

        auto source_frame = source_->receive(field, nb_samples);

        if (!source_frame)
            return source_frame;

        extract_visitor v;
        source_frame.accept(v);

        const auto cf = v.frame;
        if (!cf)
            return source_frame;

        const auto& pfd = cf.pixel_format_desc();
        const bool  src_rgba = pfd.format == core::pixel_format::rgba;
        if ((pfd.format != core::pixel_format::bgra && pfd.format != core::pixel_format::rgba) ||
            pfd.planes.size() != 1) {
            static bool warned_fmt = false;
            if (!warned_fmt) {
                warned_fmt = true;
                CASPAR_LOG(warning) << L"[ofx] source frame format=" << static_cast<int>(pfd.format)
                                    << L" planes=" << pfd.planes.size()
                                    << L" not single-plane BGRA/RGBA; passing source through.";
            }
            return source_frame; // Only single-plane BGRA/RGBA frames are handled (8- or 16-bit).
        }

        const int  w     = static_cast<int>(cf.width());
        const int  h     = static_cast<int>(cf.height());
        const int  bytes = pfd.planes[0].depth == common::bit_depth::bit8 ? 1 : 2;

        bool rendered = false;
        {
            std::lock_guard<std::mutex> lock(effect_mutex_);

            if (!ensure_effect(w, h, bytes))
                return source_frame; // effect unavailable -> pass through

            // OFX time follows the source producer's playback position.
            const double t = static_cast<double>(source_->frame_number());

            // If the plug-in reports this frame is a no-op, skip the round-trip entirely.
            if (effect_->is_identity(w, h, t))
                return source_frame;

#if defined(CASPAR_OFX_VULKAN_CUDA)
            // Zero-copy CUDA fast path: Vulkan mixer + CUDA-capable plug-in + 8-bit. The plug-in
            // renders into a CUDA device buffer which is copied device-to-device into an exportable
            // VK texture the mixer consumes directly — no CPU readback.
            if (vk_device_ && bytes == 1 && working_bytes_ == 1 && effect_->cuda_capable()) {
                const int work_stride = w * 4;
                if (src_rgba)
                    rgba_top_down_to_rgba_bottom_up(cf.image_data(0).data(), pfd.planes[0].linesize, src_rgba_.data(), work_stride, w, h);
                else
                    bgra_top_down_to_rgba_bottom_up(cf.image_data(0).data(), pfd.planes[0].linesize, src_rgba_.data(), work_stride, w, h);
                if (pfd.is_straight_alpha)
                    premultiply_rgba8(src_rgba_.data(), work_stride, w, h);

                apply_animation(t);
                void* out_dev = effect_->render_cuda(src_rgba_.data(), w, h, t, to_field_kind(field));
                if (out_dev) {
                    auto cvt = acquire_vk_cuda_texture(w, h);
                    if (cvt) {
                        cudaError_t e = cudaMemcpy2DToArray(cvt->array(), 0, 0, out_dev,
                                                            static_cast<size_t>(w) * 4, static_cast<size_t>(w) * 4,
                                                            static_cast<size_t>(h), cudaMemcpyDeviceToDevice);
                        if (e == cudaSuccess)
                            e = cudaDeviceSynchronize();
                        if (e == cudaSuccess) {
                            core::pixel_format_desc tex_pfd(core::pixel_format::bgra, pfd.color_space, pfd.color_transfer);
                            tex_pfd.is_straight_alpha = !effect_->output_premultiplied();
                            tex_pfd.planes.push_back(core::pixel_format_desc::plane(w, h, 4, common::bit_depth::bit8));

                            auto empty_store = std::make_shared<std::vector<std::uint8_t>>(0);
                            array<const std::uint8_t> dummy_img(empty_store->data(), 0, std::move(empty_store));
                            std::vector<array<const std::uint8_t>> img_vec;
                            img_vec.push_back(std::move(dummy_img));

                            const auto& zaud = cf.audio_data();
                            auto audio_store = std::make_shared<std::vector<std::int32_t>>(zaud.data(), zaud.data() + zaud.size());
                            array<const std::int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                            static bool logged = false;
                            if (!logged) { logged = true;
                                CASPAR_LOG(info) << L"[ofx] CUDA-Vulkan zero-copy producer path active (no readback)."; }

                            return core::draw_frame(core::const_frame(
                                this, std::move(img_vec), std::move(audio_arr), tex_pfd, cvt->core_texture()));
                        }
                    }
                }
                return source_frame; // CUDA zero-copy failed -> pass source through
            }
#endif

            // Zero-copy OpenGL fast path: OGL mixer + GL-capable plug-in + 8-bit source/working
            // depth. The plug-in renders directly into a mixer texture — no CPU readback, no
            // re-upload. Falls through to the CPU path for the Vulkan mixer, non-GL plug-ins, or
            // 16-bit/float working depths. A legacy fixed-function plug-in (which cannot run on the
            // mixer's core-profile device) is auto-detected on its first frame and thereafter uses
            // the self-contained compatibility path.
            if (ogl_device_ && bytes == 1 && working_bytes_ == 1 && effect_->opengl_capable() &&
                effect_->zerocopy_gl_supported()) {
                const int work_stride = w * 4;
                if (src_rgba)
                    rgba_top_down_to_rgba_bottom_up(
                        cf.image_data(0).data(), pfd.planes[0].linesize, src_rgba_.data(), work_stride, w, h);
                else
                    bgra_top_down_to_rgba_bottom_up(
                        cf.image_data(0).data(), pfd.planes[0].linesize, src_rgba_.data(), work_stride, w, h);
                if (pfd.is_straight_alpha)
                    premultiply_rgba8(src_rgba_.data(), work_stride, w, h);

                apply_animation(t);
                auto out_tex = effect_->render_gl_zerocopy(
                    *ogl_device_, src_rgba_.data(), w, h, t, to_field_kind(field));
                if (out_tex) {
                    // The OFX plug-in renders standard RGBA-ordered pixels into the texture. The OGL
                    // mixer's shader works in a BGRA convention (its "rgba" case samples straight and
                    // treats .r as blue, which is only correct for BGRA-ordered textures like the
                    // cuda_prores decoder produces). Labelling our RGBA-ordered texture as bgra makes
                    // the shader apply its .bgra swizzle, mapping our channels correctly.
                    core::pixel_format_desc tex_pfd(core::pixel_format::bgra, pfd.color_space, pfd.color_transfer);
                    tex_pfd.is_straight_alpha = !effect_->output_premultiplied();
                    tex_pfd.planes.push_back(core::pixel_format_desc::plane(w, h, 4, common::bit_depth::bit8));

                    // Texture-backed const_frame: one dummy (empty) CPU plane; the mixer consumes the
                    // GPU texture directly (matches the cuda_prores zero-copy producer pattern).
                    auto empty_store = std::make_shared<std::vector<std::uint8_t>>(0);
                    array<const std::uint8_t> dummy_img(empty_store->data(), 0, std::move(empty_store));
                    std::vector<array<const std::uint8_t>> img_vec;
                    img_vec.push_back(std::move(dummy_img));

                    const auto& zsrc_audio = cf.audio_data();
                    auto        audio_store = std::make_shared<std::vector<std::int32_t>>(
                        zsrc_audio.data(), zsrc_audio.data() + zsrc_audio.size());
                    array<const std::int32_t> audio_arr(audio_store->data(), audio_store->size(), std::move(audio_store));

                    return core::draw_frame(core::const_frame(
                        this, std::move(img_vec), std::move(audio_arr), tex_pfd, std::move(out_tex)));
                }
                // Zero-copy declined. If the plug-in was flagged as needing a compatibility profile,
                // fall through to the self-contained readback path (which renders it correctly).
                // Otherwise it was a genuine render failure -> pass the source through.
                if (effect_->zerocopy_gl_supported())
                    return source_frame;
            }

            const int src_stride = pfd.planes[0].linesize;
            const int wb         = working_bytes_;      // negotiated OFX depth (1/2/4)
            const int work_stride = w * 4 * wb;         // tight working row bytes

            if (wb == 4) {
                // Float working depth (plug-in requires/prefers float).
                if (src_rgba) {
                    static bool warned = false;
                    if (!warned) { warned = true;
                        CASPAR_LOG(warning) << L"[ofx] '" << plugin_id_ << L"' needs float depth on an RGBA "
                                            << L"source; not converted yet -> passing source through."; }
                    return source_frame;
                }
                bgra_to_rgbaf_bottom_up(
                    cf.image_data(0).data(), src_stride, bytes, reinterpret_cast<float*>(src_rgba_.data()), work_stride, w, h);
                if (pfd.is_straight_alpha)
                    premultiply_rgbaf(reinterpret_cast<float*>(src_rgba_.data()), work_stride, w, h);
            } else if (wb == bytes) {
                // Integer working depth matching the source.
                if (bytes == 2) {
                    if (src_rgba) {
                        static bool warned = false;
                        if (!warned) { warned = true;
                            CASPAR_LOG(warning) << L"[ofx] '" << plugin_id_ << L"' 16-bit RGBA source not "
                                                << L"converted yet -> passing source through."; }
                        return source_frame;
                    }
                    bgra16_top_down_to_rgba16_bottom_up(cf.image_data(0).data(), src_stride, src_rgba_.data(), work_stride, w, h);
                } else if (src_rgba) {
                    rgba_top_down_to_rgba_bottom_up(cf.image_data(0).data(), src_stride, src_rgba_.data(), work_stride, w, h);
                } else {
                    bgra_top_down_to_rgba_bottom_up(cf.image_data(0).data(), src_stride, src_rgba_.data(), work_stride, w, h);
                }
                if (pfd.is_straight_alpha) {
                    if (bytes == 2)
                        premultiply_rgba16(src_rgba_.data(), work_stride, w, h);
                    else
                        premultiply_rgba8(src_rgba_.data(), work_stride, w, h);
                }
            } else {
                // Rare: plug-in wants an integer depth different from the source (e.g. 16-bit-only
                // plug-in with an 8-bit source). Not converted yet -> pass source through.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    CASPAR_LOG(warning) << L"[ofx] '" << plugin_id_ << L"' needs an unsupported working depth; "
                                        << L"passing source through.";
                }
                return source_frame;
            }

            // OFX time follows the source producer's playback position (temporal plug-ins + the
            // per-parameter keyframe timeline both key off this).
            apply_animation(t);
            rendered = effect_->render(src_rgba_.data(), dst_rgba_.data(), w, h, t, to_field_kind(field));
        }
        if (!rendered)
            return source_frame;

        // Honor the plug-in's declared output premultiplication.
        core::pixel_format_desc out_pfd = pfd;
        out_pfd.is_straight_alpha       = !effect_->output_premultiplied();

        auto      out         = frame_factory_->create_frame(this, out_pfd);
        const int dst_stride  = out.pixel_format_desc().planes[0].linesize;
        const int work_stride = w * 4 * working_bytes_;
        if (working_bytes_ == 4)
            rgbaf_bottom_up_to_bgra(
                reinterpret_cast<const float*>(dst_rgba_.data()), work_stride, out.image_data(0).data(), dst_stride, bytes, w, h);
        else if (bytes == 2)
            rgba16_bottom_up_to_bgra16_top_down(dst_rgba_.data(), work_stride, out.image_data(0).data(), dst_stride, w, h);
        else if (src_rgba)
            rgba_bottom_up_to_rgba_top_down(dst_rgba_.data(), work_stride, out.image_data(0).data(), dst_stride, w, h);
        else
            rgba_bottom_up_to_bgra_top_down(dst_rgba_.data(), work_stride, out.image_data(0).data(), dst_stride, w, h);

        // Preserve the source frame's audio (OFX processes video only).
        const auto& src_audio = cf.audio_data();
        if (src_audio.size() > 0)
            out.audio_data() = std::vector<std::int32_t>(src_audio.data(), src_audio.data() + src_audio.size());

        return core::draw_frame(std::move(out));
    }

    core::monitor::state state() const override { return source_->state(); }

    std::wstring print() const override { return L"ofx[" + plugin_id_ + L"|" + source_->print() + L"]"; }

    std::wstring name() const override { return L"ofx"; }

    uint32_t frame_number() const override { return source_->frame_number(); }

    uint32_t nb_frames() const override { return source_->nb_frames(); }

    bool is_ready() override
    {
        return source_->is_ready() && (context_ != effect_context::transition || source_to_->is_ready());
    }

    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        // OFX control protocol (via AMCP CALL):
        //   CALL <ch-layer> OFX LIST                 -> list parameters
        //   CALL <ch-layer> OFX SET <name> <v...>    -> set a parameter from scalar value(s)
        //   CALL <ch-layer> OFX KEY <name> <frame> <v...> [tween]
        //   CALL <ch-layer> OFX CLEARKEYS <name>
        // Anything else is forwarded to the wrapped source producer.
        if (!params.empty() && boost::iequals(params.at(0), L"OFX")) {
            const std::wstring sub = params.size() > 1 ? params.at(1) : L"";

            if (boost::iequals(sub, L"LIST")) {
                std::wstring result;
                if (effect_) {
                    for (const auto& p : effect_->params()) {
                        // Backward-compatible prefix (name type "label") plus metadata for client UX:
                        //   <name> <type> "<label>" dim=<n> [min=.. max=.. def=..] [choices="a","b"]
                        result += u16(p.name) + L" " + u16(p.type) + L" \"" + u16(p.label) + L"\"";
                        result += L" dim=" + std::to_wstring(p.dimension);
                        if (p.has_range) {
                            result += L" min=" + fmt_lim(p.min) + L" max=" + fmt_lim(p.max) +
                                      L" def=" + fmt_num(p.def);
                        }
                        if (!p.choices.empty()) {
                            result += L" choices=";
                            for (std::size_t i = 0; i < p.choices.size(); ++i) {
                                if (i)
                                    result += L",";
                                result += L"\"" + u16(p.choices[i]) + L"\"";
                            }
                        }
                        result += L"\r\n";
                    }
                }
                std::promise<std::wstring> pr;
                pr.set_value(result);
                return pr.get_future();
            }

            if (boost::iequals(sub, L"SET") && params.size() >= 4) {
                const std::string        name = u8(params.at(2));
                std::vector<double>       values;
                for (std::size_t i = 3; i < params.size(); ++i) {
                    try {
                        values.push_back(std::stod(params.at(i)));
                    } catch (...) {
                        values.push_back(0.0);
                    }
                }

                bool ok = false;
                {
                    std::lock_guard<std::mutex> lock(effect_mutex_);
                    ok = effect_ && effect_->set_param(name, values, frame_number_of_effect_);
                }

                std::promise<std::wstring> pr;
                pr.set_value(ok ? L"" : L"402 CALL ERROR (unknown OFX parameter or effect not ready)\r\n");
                return pr.get_future();
            }

            // CALL <ch-layer> OFX SETSTR <name> <string...>  (string parameters)
            if (boost::iequals(sub, L"SETSTR") && params.size() >= 4) {
                const std::string name = u8(params.at(2));
                std::wstring       value;
                for (std::size_t i = 3; i < params.size(); ++i) {
                    if (i > 3)
                        value += L" ";
                    value += params.at(i);
                }
                bool ok = false;
                {
                    std::lock_guard<std::mutex> lock(effect_mutex_);
                    ok = effect_ && effect_->set_param_string(name, u8(value), frame_number_of_effect_);
                }
                std::promise<std::wstring> pr;
                pr.set_value(ok ? L"" : L"402 CALL ERROR (unknown OFX string parameter or effect not ready)\r\n");
                return pr.get_future();
            }

            // CALL <ch-layer> OFX KEY <name> <frame> <v0> [v1 v2 v3] [tweener]
            if (boost::iequals(sub, L"KEY") && params.size() >= 5) {
                const std::string name = u8(params.at(2));

                keyframe            kf;
                std::wstring        tween_name;
                try {
                    kf.frame = std::stod(params.at(3));
                } catch (...) {
                    kf.frame = 0.0;
                }
                for (std::size_t i = 4; i < params.size(); ++i) {
                    try {
                        kf.values.push_back(std::stod(params.at(i)));
                    } catch (...) {
                        // A non-numeric trailing token is treated as the tweener name.
                        tween_name = params.at(i);
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(effect_mutex_);
                    auto& keys = anim_[name];
                    keys.push_back(kf);
                    std::sort(keys.begin(), keys.end(), [](const keyframe& a, const keyframe& b) {
                        return a.frame < b.frame;
                    });
                    if (!tween_name.empty())
                        tween_[name] = tweener(tween_name);
                }

                std::promise<std::wstring> pr;
                pr.set_value(L"");
                return pr.get_future();
            }

            // CALL <ch-layer> OFX CLEARKEYS <name>
            if (boost::iequals(sub, L"CLEARKEYS") && params.size() >= 3) {
                const std::string name = u8(params.at(2));
                {
                    std::lock_guard<std::mutex> lock(effect_mutex_);
                    anim_.erase(name);
                    tween_.erase(name);
                }
                std::promise<std::wstring> pr;
                pr.set_value(L"");
                return pr.get_future();
            }

            std::promise<std::wstring> pr;
            pr.set_value(L"403 CALL ERROR (usage: OFX LIST | OFX SET <name> <v...> | OFX KEY <name> "
                         L"<frame> <v...> [tween] | OFX CLEARKEYS <name>)\r\n");
            return pr.get_future();
        }

        return source_->call(params);
    }
};

} // namespace

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || !boost::iequals(params.at(0), L"[OFX]"))
        return core::frame_producer::empty();

    if (params.size() < 2) {
        CASPAR_LOG(warning) << L"[ofx] Usage: [OFX] <plugin-id> [<source-producer...>]";
        return core::frame_producer::empty();
    }

    const auto                plugin_id = params.at(1);
    std::vector<std::wstring> source_params(params.begin() + 2, params.end());

    // Transition context: [OFX] <plugin> TRANSITION <fromSource> <toSource> [<frames>]
    if (!source_params.empty() && boost::iequals(source_params.at(0), L"TRANSITION")) {
        if (source_params.size() < 3) {
            CASPAR_LOG(warning) << L"[ofx] Usage: [OFX] <plugin-id> TRANSITION <from-source> <to-source> [frames]";
            return core::frame_producer::empty();
        }
        auto from = dependencies.producer_registry->create_producer(dependencies, {source_params.at(1)});
        auto to   = dependencies.producer_registry->create_producer(dependencies, {source_params.at(2)});
        if (from == core::frame_producer::empty() || to == core::frame_producer::empty()) {
            CASPAR_LOG(warning) << L"[ofx] Could not create both transition sources for '" << plugin_id << L"'.";
            return core::frame_producer::empty();
        }
        int frames = 25;
        if (source_params.size() >= 4) {
            try {
                frames = std::stoi(source_params.at(3));
            } catch (...) {
                frames = 25;
            }
        }
        return spl::make_shared<ofx_producer>(dependencies, plugin_id, from, to, frames);
    }

    // Filter context: a source producer is supplied and wrapped.
    if (!source_params.empty()) {
        auto source = dependencies.producer_registry->create_producer(dependencies, source_params);
        if (source == core::frame_producer::empty()) {
            CASPAR_LOG(warning) << L"[ofx] Could not create source producer for effect '" << plugin_id << L"'.";
            return core::frame_producer::empty();
        }
        return spl::make_shared<ofx_producer>(dependencies, plugin_id, source, effect_context::filter);
    }

    // No source: only valid if the plug-in supports the Generator context.
    const std::string id_utf8 = u8(plugin_id);
    for (const auto& p : global_host().plugins()) {
        if (p.identifier != id_utf8)
            continue;
        for (const auto& ctx : p.contexts) {
            if (ctx == "OfxImageEffectContextGenerator")
                return spl::make_shared<ofx_producer>(
                    dependencies, plugin_id, core::frame_producer::empty(), effect_context::generator);
        }
    }

    CASPAR_LOG(warning) << L"[ofx] '" << plugin_id
                        << L"' needs a source producer (not a generator). Usage: [OFX] <plugin-id> <source...>";
    return core::frame_producer::empty();
}

}} // namespace caspar::ofx
