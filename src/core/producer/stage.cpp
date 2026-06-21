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

#include "stage.h"

#include "layer.h"

#include "../frame/draw_frame.h"

#include <common/diagnostics/graph.h>
#include <common/executor.h>
#include <common/future.h>

#include <core/frame/frame_transform.h>
#include <core/producer/route/route_producer.h>
#include <modules/keyframes/keyframe_data.h>
#include <modules/keyframes/keyframe_fields.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/adaptors.hpp>

#include <functional>
#include <future>
#include <map>
#include <set>
#include <vector>

namespace caspar { namespace core {

struct stage::impl : public std::enable_shared_from_this<impl>
{
    int                                 channel_index_;
    spl::shared_ptr<diagnostics::graph> graph_;
    monitor::state                      state_;
    std::map<int, layer>                layers_;
    std::map<int, tweened_transform>    tweens_;
    std::set<int>                       routeSources;

    mutable std::mutex      format_desc_mutex_;
    core::video_format_desc format_desc_;

    // ── Keyframe state (all accessed only on the stage executor — no mutex) ──
    std::map<int, caspar::keyframes::keyframe_timeline> kf_timelines_;
    std::set<int>                                       kf_armed_;
    std::map<int, caspar::keyframes::kf_values>         kf_last_values_; // for change detection
    std::map<int, double>                               kf_media_time_override_; // from CALL SEEK
    std::map<int, uint32_t>                             kf_last_frame_number_;   // for override clearing

    executor   executor_{L"stage " + std::to_wstring(channel_index_)};
    std::mutex lock_;

  private:
    void orderSourceLayers(std::vector<std::pair<int, bool>>&        layerVec,
                           const std::map<int, std::pair<int, int>>& routed_layers,
                           int                                       l,
                           int                                       depth)
    {
        if (0 == depth)
            routeSources.clear();

        if (std::find_if(layerVec.begin(), layerVec.end(), [l](std::pair<int, bool> p) { return p.first == l; }) !=
            layerVec.end()) {
            return;
        }

        auto routeIt = routed_layers.find(l);
        if (routed_layers.end() == routeIt) {
            layerVec.push_back(std::make_pair(l, true));
            return;
        }

        std::pair<int, int> routeSrc(routeIt->second);
        if (channel_index_ != routeSrc.first) {
            layerVec.push_back(std::make_pair(l, true));
            return;
        }

        // check for circular route setup - skip recursion if found
        routeSources.emplace(routeSrc.second);
        bool layerOK = true;
        if (routeSources.find(l) == routeSources.end()) {
            orderSourceLayers(layerVec, routed_layers, routeSrc.second, ++depth);
        } else {
            layerOK = false;
        }

        if (std::find_if(layerVec.begin(), layerVec.end(), [l](std::pair<int, bool> p) { return p.first == l; }) ==
            layerVec.end()) {
            layerVec.push_back(std::make_pair(l, layerOK));
        }
    }

    layer& get_layer(int index)
    {
        auto it = layers_.find(index);
        if (it == std::end(layers_)) {
            it = layers_.emplace(index, layer(video_format_desc())).first;
        }
        return it->second;
    }

  public:
    impl(int channel_index, spl::shared_ptr<diagnostics::graph> graph, const core::video_format_desc& format_desc)
        : channel_index_(channel_index)
        , graph_(std::move(graph))
        , format_desc_(format_desc)
    {
    }

    const stage_frames operator()(uint64_t                                     frame_number,
                                  std::vector<int>&                            fetch_background,
                                  std::function<void(int, const layer_frame&)> routesCb)
    {
        return executor_.invoke([=] {
            std::map<int, layer_frame> frames;
            stage_frames               result = {};

            result.format_desc = video_format_desc();
            result.nb_samples =
                result.format_desc.audio_cadence[frame_number % result.format_desc.audio_cadence.size()];

            auto is_interlaced = format_desc_.field_count == 2;
            auto field1        = is_interlaced ? video_field::a : video_field::progressive;

            try {
                for (auto& t : tweens_)
                    t.second.tick(1);

                  // ── Keyframe evaluation ────────────────────────────────────
                  // Compute media time per layer and interpolate armed
                  // timelines directly on the stage executor, BEFORE rendering.
                  {
                      double fps = 25.0;
                      if (format_desc_.framerate.numerator() > 0 && format_desc_.framerate.denominator() > 0)
                          fps = static_cast<double>(format_desc_.framerate.numerator()) /
                                format_desc_.framerate.denominator();

                      for (int armed_layer : kf_armed_) {
                          auto tl_it = kf_timelines_.find(armed_layer);
                          if (tl_it == kf_timelines_.end() || tl_it->second.empty())
                              continue;

                          auto layer_it = layers_.find(armed_layer);
                          if (layer_it == layers_.end())
                              continue;

                          auto producer = layer_it->second.foreground();
                          uint32_t fn = (producer.get() != nullptr) ? producer->frame_number() : 0;

                          // Determine media time: use seek override if available
                          double media_time;
                          auto override_it = kf_media_time_override_.find(armed_layer);
                          if (override_it != kf_media_time_override_.end()) {
                              auto last_fn_it = kf_last_frame_number_.find(armed_layer);
                              if (last_fn_it != kf_last_frame_number_.end() &&
                                  fn != 0 && fn != last_fn_it->second) {
                                  // Producer frame_number advanced — producer has caught up
                                  kf_media_time_override_.erase(override_it);
                                  media_time = static_cast<double>(fn) / fps;
                              } else {
                                  media_time = override_it->second;
                              }
                          } else {
                              media_time = static_cast<double>(fn) / fps;
                          }
                          kf_last_frame_number_[armed_layer] = fn;

                          // Interpolate
                          auto values = tl_it->second.interpolate(media_time);
                          if (values.empty())
                              continue;

                          // Skip if unchanged from last frame
                          auto& last = kf_last_values_[armed_layer];
                          if (values == last)
                              continue;
                          last = values;

                          // Apply to tween
                          auto src = tweens_[armed_layer].fetch();
                          auto dst = src;
                          caspar::keyframes::apply_kf_to_transform(values, dst.image_transform);
                          tweens_[armed_layer] = tweened_transform(src, dst, 0, tweener(L"linear"));
                      }
                  }

                // build a map of layers that are sourced from route producers
                std::map<int, std::pair<int, int>> routed_layers;
                for (auto& p : layers_) {
                    auto producer = std::move(p.second.foreground());
                    if (0 == producer->name().compare(L"route")) {
                        try {
                            auto rc       = spl::dynamic_pointer_cast<core::route_control>(producer);
                            auto srcChan  = rc->get_source_channel();
                            auto srcLayer = rc->get_source_layer();
                            routed_layers.emplace(p.first, std::make_pair(srcChan, srcLayer));
                            rc->set_cross_channel(channel_index_ != srcChan);
                        } catch (std::bad_cast) {
                            CASPAR_LOG(error) << "Failed to cast route producer";
                        }
                    }
                }

                // sort layer order so that sources get pulled before routes
                std::vector<std::pair<int, bool>> layerVec;
                for (auto& p : layers_)
                    orderSourceLayers(layerVec, routed_layers, p.first, 0);

                // when running interlaced, both fields are be pulled at once.
                // This will risk some stutter for freshly created producers, but it lets us tick at 25hz and avoids
                // amcp changes starting on the second field

                for (auto& l : layerVec) {
                    auto p = layers_.find(l.first);
                    if (p == layers_.end())
                        continue;

                    auto& layer = p->second;
                    auto& tween = tweens_[p->first];

                    auto has_background_route =
                        std::find(fetch_background.begin(), fetch_background.end(), p->first) != fetch_background.end();

                    layer_frame res = {};
                    if (l.second) {
                        res.foreground1_raw = layer.receive(field1, result.nb_samples);
                        res.foreground1     = draw_frame::push(res.foreground1_raw, tween.fetch());
                        res.foreground1.transform().image_transform.enable_geometry_modifiers = true;
                    }

                    res.has_background = layer.has_background();
                    if (has_background_route)
                        res.background1 = layer.receive_background(field1, result.nb_samples);

                    if (is_interlaced) {
                        res.is_interlaced = true;
                        if (l.second) {
                            res.foreground2_raw = layer.receive(video_field::b, result.nb_samples);
                            res.foreground2     = draw_frame::push(res.foreground2_raw, tween.fetch());
                            res.foreground2.transform().image_transform.enable_geometry_modifiers = true;
                        }
                        if (has_background_route)
                            res.background2 = layer.receive_background(video_field::b, result.nb_samples);
                    }

                    frames[p->first] = res;

                    // push received foreground frame to any configured route producer
                    routesCb(p->first, res);
                }

                for (auto& p : frames) {
                    result.frames.push_back(p.second.foreground1);
                    if (is_interlaced)
                        result.frames2.push_back(p.second.foreground2);
                }

                {
                    // push stage_frames to support any channel routes that have been set
                    layer_frame chan_lf   = {};
                    chan_lf.is_interlaced = is_interlaced;
                    chan_lf.foreground1   = wrap_layer_frames_for_route(result.frames);
                    if (is_interlaced)
                        chan_lf.foreground2 = wrap_layer_frames_for_route(result.frames2);

                    routesCb(-1, chan_lf);
                }

                monitor::state state;
                for (auto& p : layers_) {
                    state["layer"][p.first] = p.second.state();

                    // Publish the full projection/curve state per layer so OSC
                    // subscribers (and virtual-production tooling) can mirror the
                    // server's projection model in real time.
                    auto tw = tweens_.find(p.first);
                    if (tw != tweens_.end()) {
                        const auto pr = tw->second.fetch().image_transform.projection;
                        auto       ps = state["layer"][p.first]["projection"];
                        ps["enable"]       = pr.enable;
                        ps["yaw"]          = pr.yaw;
                        ps["pitch"]        = pr.pitch;
                        ps["roll"]         = pr.roll;
                        ps["fov"]          = pr.fov;
                        ps["offset_x"]     = pr.offset_x;
                        ps["offset_y"]     = pr.offset_y;
                        ps["frustum_h"]    = pr.frustum_h;
                        ps["frustum_v"]    = pr.frustum_v;
                        ps["lens_k1"]      = pr.lens_k1;
                        ps["lens_k2"]      = pr.lens_k2;
                        ps["lens_k3"]      = pr.lens_k3;
                        ps["lens_p1"]      = pr.lens_p1;
                        ps["lens_p2"]      = pr.lens_p2;
                        ps["source_lens"]  = static_cast<int>(pr.source_lens);
                        ps["curve_enable"] = pr.curve_enable;
                        ps["curve_auto"]   = pr.curve_auto;
                        ps["curve_type"]   = static_cast<int>(pr.curve_type);
                        ps["screen_arc"]   = pr.screen_arc;
                        ps["screen_arc_v"] = pr.screen_arc_v;
                        ps["eye_distance"] = pr.eye_distance;
                        ps["edge_blend"]   = {pr.edge_blend_left,
                                              pr.edge_blend_right,
                                              pr.edge_blend_top,
                                              pr.edge_blend_bottom,
                                              pr.edge_blend_gamma};
                        ps["icvfx_enable"]    = pr.icvfx_enable;
                        ps["inner_fov"]       = pr.inner_fov;
                        ps["icvfx_feather"]   = pr.icvfx_feather;
                        ps["icvfx_outer_dim"] = pr.icvfx_outer_dim;
                    }
                }
                state_ = std::move(state);
            } catch (...) {
                layers_.clear();
                CASPAR_LOG_CURRENT_EXCEPTION();
            }

            return result;
        });
    }

    core::draw_frame wrap_layer_frames_for_route(std::vector<core::draw_frame> frames)
    {
        // Note: this must not mutate the vector used for the layer
        for (auto& frame : frames) {
            // Tell the compositor that these are layers, matching what normal rendering does
            frame.transform().image_transform.layer_depth = 1;
        }
        return core::draw_frame(frames);
    }

    std::future<void>
    apply_transforms(const std::vector<std::tuple<int, stage::transform_func_t, unsigned int, tweener>>& transforms)
    {
        return executor_.begin_invoke([=] {
            for (auto& transform : transforms) {
                auto& tween = tweens_[std::get<0>(transform)];
                auto  src   = tween.fetch();
                auto  dst   = std::get<1>(transform)(tween.dest());
                tweens_[std::get<0>(transform)] =
                    tweened_transform(src, dst, std::get<2>(transform), std::get<3>(transform));
            }
        });
    }

    std::future<void> apply_transform(int                            index,
                                      const stage::transform_func_t& transform,
                                      unsigned int                   mix_duration,
                                      const tweener&                 tween)
    {
        return executor_.begin_invoke([=] {
            auto src       = tweens_[index].fetch();
            auto dst       = transform(src);
            tweens_[index] = tweened_transform(src, dst, mix_duration, tween);
        });
    }

    std::future<void> clear_transforms(int index)
    {
        return executor_.begin_invoke([=] { tweens_.erase(index); });
    }

    std::future<void> clear_transforms()
    {
        return executor_.begin_invoke([=] { tweens_.clear(); });
    }

    std::future<frame_transform> get_current_transform(int index)
    {
        return executor_.begin_invoke([=] { return tweens_[index].fetch(); });
    }

    std::future<void> load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
    {
        return executor_.begin_invoke([=] { get_layer(index).load(producer, preview, auto_play); });
    }

    std::future<void> preview(int index)
    {
        return executor_.begin_invoke([=] { get_layer(index).preview(); });
    }

    std::future<void> pause(int index)
    {
        return executor_.begin_invoke([=] { get_layer(index).pause(); });
    }

    std::future<void> resume(int index)
    {
        return executor_.begin_invoke([=] { get_layer(index).resume(); });
    }

    std::future<void> play(int index)
    {
        return executor_.begin_invoke([=] { get_layer(index).play(); });
    }

    std::future<void> stop(int index)
    {
        return executor_.begin_invoke([=] { get_layer(index).stop(); });
    }

    std::future<void> clear(int index)
    {
        return executor_.begin_invoke([=] { layers_.erase(index); });
    }

    std::future<void> clear()
    {
        return executor_.begin_invoke([=] { layers_.clear(); });
    }

    std::future<void> swap_layers(const std::shared_ptr<stage>& other, bool swap_transforms)
    {
        auto other_impl = other->impl_;

        if (other_impl.get() == this) {
            return make_ready_future();
        }

        auto func = [=] {
            auto layers       = layers_ | boost::adaptors::map_values;
            auto other_layers = other_impl->layers_ | boost::adaptors::map_values;

            std::swap(layers_, other_impl->layers_);

            if (swap_transforms)
                std::swap(tweens_, other_impl->tweens_);
        };

        return invoke_both(other, func);
    }

    std::future<void> swap_layer(int index, int other_index, bool swap_transforms)
    {
        return executor_.begin_invoke([=] {
            std::swap(get_layer(index), get_layer(other_index));

            if (swap_transforms)
                std::swap(tweens_[index], tweens_[other_index]);
        });
    }

    std::future<void> swap_layer(int index, int other_index, const std::shared_ptr<stage>& other, bool swap_transforms)
    {
        auto other_impl = other->impl_;

        if (other_impl.get() == this)
            return swap_layer(index, other_index, swap_transforms);
        auto func = [=] {
            auto& my_layer    = get_layer(index);
            auto& other_layer = other_impl->get_layer(other_index);

            std::swap(my_layer, other_layer);

            if (swap_transforms) {
                auto& my_tween    = tweens_[index];
                auto& other_tween = other_impl->tweens_[other_index];
                std::swap(my_tween, other_tween);
            }
        };

        return invoke_both(other, func);
    }

    std::future<void> invoke_both(const std::shared_ptr<stage>& other, std::function<void()> func)
    {
        auto other_impl = other->impl_;

        if (other_impl->channel_index_ < channel_index_) {
            return other_impl->executor_.begin_invoke([=] { executor_.invoke(func); });
        }

        return executor_.begin_invoke([=] { other_impl->executor_.invoke(func); });
    }

    std::future<std::shared_ptr<frame_producer>> foreground(int index)
    {
        return executor_.begin_invoke(
            [=]() -> std::shared_ptr<frame_producer> { return get_layer(index).foreground(); });
    }

    std::future<std::shared_ptr<frame_producer>> background(int index)
    {
        return executor_.begin_invoke(
            [=]() -> std::shared_ptr<frame_producer> { return get_layer(index).background(); });
    }

    std::future<std::wstring> call(int index, const std::vector<std::wstring>& params)
    {
        return flatten(executor_.begin_invoke([=] {
            auto result = get_layer(index).foreground()->call(params).share();

            // Detect seek commands for keyframe media time tracking.
            // When a CALL SEEK happens while paused, frame_number() won't update,
            // so we set an override that the KF render loop uses until the producer
            // catches up.
            if (params.size() >= 2 && boost::iequals(params[0], L"seek")) {
                double fps = 25.0;
                if (format_desc_.framerate.numerator() > 0 && format_desc_.framerate.denominator() > 0)
                    fps = static_cast<double>(format_desc_.framerate.numerator()) /
                          format_desc_.framerate.denominator();
                try {
                    int64_t seek_frame = boost::lexical_cast<int64_t>(params[1]);
                    if (params.size() > 2)
                        seek_frame += boost::lexical_cast<int64_t>(params[2]);
                    kf_media_time_override_[index] = static_cast<double>(seek_frame) / fps;
                } catch (...) {
                    // Keyword seeks ("rel","in","out","end") — we can't know
                    // the target frame before the async seek completes, so
                    // don't set an override.  The render loop will pick up
                    // the new frame_number naturally once the producer catches up.
                }
            }

            return result;
        }));
    }
    std::future<std::wstring> callbg(int index, const std::vector<std::wstring>& params)
    {
        return flatten(executor_.begin_invoke([=] { return get_layer(index).background()->call(params).share(); }));
    }

    std::unique_lock<std::mutex> get_lock() { return std::move(std::unique_lock<std::mutex>(lock_)); }

    // ── Keyframe management (all on executor) ─────────────────────────────

    std::future<void> kf_set(int layer, std::shared_ptr<void> data)
    {
        return executor_.begin_invoke([=] {
            auto tl = std::static_pointer_cast<caspar::keyframes::keyframe_timeline>(data);
            kf_timelines_[layer] = *tl;
            kf_armed_.erase(layer);       // disarm on new SET (protocol rule)
            kf_last_values_.erase(layer);
            kf_media_time_override_.erase(layer);
            kf_last_frame_number_.erase(layer);
        });
    }

    std::future<bool> kf_arm(int layer)
    {
        return executor_.begin_invoke([=] {
            if (kf_timelines_.count(layer)) {
                kf_armed_.insert(layer);
                return true;
            }
            return false;
        });
    }

    std::future<void> kf_disarm(int layer)
    {
        return executor_.begin_invoke([=] {
            kf_armed_.erase(layer);
            kf_last_values_.erase(layer);
        });
    }

    std::future<void> kf_clear(int layer)
    {
        return executor_.begin_invoke([=] {
            kf_timelines_.erase(layer);
            kf_armed_.erase(layer);
            kf_last_values_.erase(layer);
            kf_media_time_override_.erase(layer);
            kf_last_frame_number_.erase(layer);
        });
    }

    std::future<std::shared_ptr<void>> kf_get(int layer)
    {
        return executor_.begin_invoke([=]() -> std::shared_ptr<void> {
            auto it = kf_timelines_.find(layer);
            if (it == kf_timelines_.end())
                return nullptr;
            return std::make_shared<caspar::keyframes::keyframe_timeline>(it->second);
        });
    }

    std::future<bool> kf_has(int layer)
    {
        return executor_.begin_invoke([=] { return kf_timelines_.count(layer) > 0; });
    }

    std::future<bool> kf_is_armed(int layer)
    {
        return executor_.begin_invoke([=] { return kf_armed_.count(layer) > 0; });
    }

    std::future<bool> kf_patch(int layer, double time_secs, std::shared_ptr<void> patch_data)
    {
        return executor_.begin_invoke([=] {
            auto it = kf_timelines_.find(layer);
            if (it == kf_timelines_.end())
                return false;
            auto vals = std::static_pointer_cast<caspar::keyframes::kf_values>(patch_data);
            bool ok = it->second.patch_at_time(time_secs, *vals);
            if (ok)
                kf_last_values_.erase(layer); // force re-evaluation
            return ok;
        });
    }

    std::future<void> kf_set_media_time(int layer, double time_secs)
    {
        return executor_.begin_invoke([=] {
            kf_media_time_override_[layer] = time_secs;
            kf_last_values_.erase(layer); // force re-evaluation
        });
    }

    std::future<std::shared_ptr<void>> kf_get_status(int layer)
    {
        return executor_.begin_invoke([=]() -> std::shared_ptr<void> {
            auto s = std::make_shared<caspar::keyframes::kf_status>();
            auto it = kf_timelines_.find(layer);
            s->has_timeline   = (it != kf_timelines_.end());
            s->armed          = kf_armed_.count(layer) > 0;
            s->keyframe_count = s->has_timeline ? it->second.size() : 0;
            return std::static_pointer_cast<void>(s);
        });
    }

    core::video_format_desc video_format_desc() const
    {
        std::lock_guard<std::mutex> lock(format_desc_mutex_);
        return format_desc_;
    }

    std::future<void> video_format_desc(const core::video_format_desc& format_desc)
    {
        return executor_.begin_invoke([=] {
            {
                std::lock_guard<std::mutex> lock(format_desc_mutex_);
                format_desc_ = format_desc;
            }

            layers_.clear();
        });
    }
};

stage::stage(int channel_index, spl::shared_ptr<diagnostics::graph> graph, const core::video_format_desc& format_desc)
    : impl_(new impl(channel_index, std::move(graph), format_desc))
{
}
std::future<std::wstring> stage::call(int index, const std::vector<std::wstring>& params)
{
    return impl_->call(index, params);
}
std::future<std::wstring> stage::callbg(int index, const std::vector<std::wstring>& params)
{
    return impl_->callbg(index, params);
}
std::future<void> stage::apply_transforms(const std::vector<stage::transform_tuple_t>& transforms)
{
    return impl_->apply_transforms(transforms);
}
std::future<void> stage::apply_transform(int                                                                index,
                                         const std::function<core::frame_transform(core::frame_transform)>& transform,
                                         unsigned int   mix_duration,
                                         const tweener& tween)
{
    return impl_->apply_transform(index, transform, mix_duration, tween);
}
std::future<void>            stage::clear_transforms(int index) { return impl_->clear_transforms(index); }
std::future<void>            stage::clear_transforms() { return impl_->clear_transforms(); }
std::future<frame_transform> stage::get_current_transform(int index) { return impl_->get_current_transform(index); }
std::future<void> stage::load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
{
    return impl_->load(index, producer, preview, auto_play);
}
std::future<void> stage::preview(int index) { return impl_->preview(index); }
std::future<void> stage::pause(int index) { return impl_->pause(index); }
std::future<void> stage::resume(int index) { return impl_->resume(index); }
std::future<void> stage::play(int index) { return impl_->play(index); }
std::future<void> stage::stop(int index) { return impl_->stop(index); }
std::future<void> stage::clear(int index) { return impl_->clear(index); }
std::future<void> stage::clear() { return impl_->clear(); }
std::future<void> stage::swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage>(other);
    return impl_->swap_layers(other2, swap_transforms);
}
std::future<void> stage::swap_layer(int index, int other_index, bool swap_transforms)
{
    return impl_->swap_layer(index, other_index, swap_transforms);
}
std::future<void>
stage::swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage>(other);
    return impl_->swap_layer(index, other_index, other2, swap_transforms);
}
std::future<std::shared_ptr<frame_producer>> stage::foreground(int index) { return impl_->foreground(index); }
std::future<std::shared_ptr<frame_producer>> stage::background(int index) { return impl_->background(index); }
const stage_frames                           stage::operator()(uint64_t                                     frame_number,
                                     std::vector<int>&                            fetch_background,
                                     std::function<void(int, const layer_frame&)> routesCb)
{
    return (*impl_)(frame_number, fetch_background, routesCb);
}
core::monitor::state    stage::state() const { return impl_->state_; }
core::video_format_desc stage::video_format_desc() const { return impl_->video_format_desc(); }
std::future<void>       stage::video_format_desc(const core::video_format_desc& format_desc)
{
    return impl_->video_format_desc(format_desc);
}
std::unique_lock<std::mutex> stage::get_lock() const { return impl_->get_lock(); }
std::future<void>            stage::execute(std::function<void()> func)
{
    func();
    return make_ready_future();
}

// ── Keyframe management (stage wrappers) ─────────────────────────────────
std::future<void>                  stage::set_keyframe_data(int layer, std::shared_ptr<void> data) { return impl_->kf_set(layer, std::move(data)); }
std::future<bool>                  stage::arm_keyframes(int layer)     { return impl_->kf_arm(layer); }
std::future<void>                  stage::disarm_keyframes(int layer)  { return impl_->kf_disarm(layer); }
std::future<void>                  stage::clear_keyframes(int layer)   { return impl_->kf_clear(layer); }
std::future<std::shared_ptr<void>> stage::get_keyframe_data(int layer) { return impl_->kf_get(layer); }
std::future<bool>                  stage::has_keyframe_data(int layer) { return impl_->kf_has(layer); }
std::future<bool>                  stage::is_keyframes_armed(int layer) { return impl_->kf_is_armed(layer); }
std::future<bool>                  stage::patch_keyframe(int layer, double time_secs, std::shared_ptr<void> patch_data) { return impl_->kf_patch(layer, time_secs, std::move(patch_data)); }
std::future<void>                  stage::set_media_time_override(int layer, double time_secs) { return impl_->kf_set_media_time(layer, time_secs); }
std::future<std::shared_ptr<void>> stage::get_keyframe_status(int layer) { return impl_->kf_get_status(layer); }

// STAGE DELAYED (For batching operations)
stage_delayed::stage_delayed(std::shared_ptr<stage>& st, int index)
    : executor_{L"batch stage " + boost::lexical_cast<std::wstring>(index)}
    , stage_(st)
{
    // Start the executor blocked on a future that will complete when we are ready for it to execute
    executor_.begin_invoke([=]() -> void { waiter_.get_future().get(); });
}

std::future<std::wstring> stage_delayed::call(int index, const std::vector<std::wstring>& params)
{
    return executor_.begin_invoke([=]() -> std::wstring { return stage_->call(index, params).get(); });
}
std::future<std::wstring> stage_delayed::callbg(int index, const std::vector<std::wstring>& params)
{
    return executor_.begin_invoke([=]() -> std::wstring { return stage_->callbg(index, params).get(); });
}
std::future<void> stage_delayed::apply_transforms(const std::vector<stage_delayed::transform_tuple_t>& transforms)
{
    return executor_.begin_invoke([=]() { return stage_->apply_transforms(transforms).get(); });
}
std::future<void>
stage_delayed::apply_transform(int                                                                index,
                               const std::function<core::frame_transform(core::frame_transform)>& transform,
                               unsigned int                                                       mix_duration,
                               const tweener&                                                     tween)
{
    return executor_.begin_invoke(
        [=]() { return stage_->apply_transform(index, transform, mix_duration, tween).get(); });
}
std::future<void> stage_delayed::clear_transforms(int index)
{
    return executor_.begin_invoke([=]() { return stage_->clear_transforms(index).get(); });
}
std::future<void> stage_delayed::clear_transforms()
{
    return executor_.begin_invoke([=]() { return stage_->clear_transforms().get(); });
}
std::future<frame_transform> stage_delayed::get_current_transform(int index)
{
    return executor_.begin_invoke([=]() { return stage_->get_current_transform(index).get(); });
}
std::future<void>
stage_delayed::load(int index, const spl::shared_ptr<frame_producer>& producer, bool preview, bool auto_play)
{
    return executor_.begin_invoke([=]() { return stage_->load(index, producer, preview, auto_play).get(); });
}
std::future<void> stage_delayed::preview(int index)
{
    return executor_.begin_invoke([=]() { return stage_->preview(index).get(); });
}
std::future<void> stage_delayed::pause(int index)
{
    return executor_.begin_invoke([=]() { return stage_->pause(index).get(); });
}
std::future<void> stage_delayed::resume(int index)
{
    return executor_.begin_invoke([=]() { return stage_->resume(index).get(); });
}
std::future<void> stage_delayed::play(int index)
{
    return executor_.begin_invoke([=]() { return stage_->play(index).get(); });
}
std::future<void> stage_delayed::stop(int index)
{
    return executor_.begin_invoke([=]() { return stage_->stop(index).get(); });
}
std::future<void> stage_delayed::clear(int index)
{
    return executor_.begin_invoke([=]() { return stage_->clear(index).get(); });
}
std::future<void> stage_delayed::clear()
{
    return executor_.begin_invoke([=]() { return stage_->clear().get(); });
}
std::future<void> stage_delayed::swap_layers(const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage_delayed>(other);
    return executor_.begin_invoke([=]() { return stage_->swap_layers(other2->stage_, swap_transforms).get(); });
}
std::future<void> stage_delayed::swap_layer(int index, int other_index, bool swap_transforms)
{
    return executor_.begin_invoke([=]() { return stage_->swap_layer(index, other_index, swap_transforms).get(); });
}
std::future<void>
stage_delayed::swap_layer(int index, int other_index, const std::shared_ptr<stage_base>& other, bool swap_transforms)
{
    const auto other2 = std::static_pointer_cast<stage_delayed>(other);

    // Something so that we know to lock the channel
    other2->executor_.begin_invoke([]() {});

    return executor_.begin_invoke(
        [=]() { return stage_->swap_layer(index, other_index, other2->stage_, swap_transforms).get(); });
}

std::future<std::shared_ptr<frame_producer>> stage_delayed::foreground(int index)
{
    return executor_.begin_invoke([=]() -> std::shared_ptr<frame_producer> { return stage_->foreground(index).get(); });
}
std::future<std::shared_ptr<frame_producer>> stage_delayed::background(int index)
{
    return executor_.begin_invoke([=]() -> std::shared_ptr<frame_producer> { return stage_->background(index).get(); });
}

std::future<void> stage_delayed::execute(std::function<void()> func)
{
    return executor_.begin_invoke([=]() { return stage_->execute(func).get(); });
}

// ── Keyframe management (stage_delayed forwarding) ───────────────────────
std::future<void> stage_delayed::set_keyframe_data(int layer, std::shared_ptr<void> data)
{
    return executor_.begin_invoke([=]() { return stage_->set_keyframe_data(layer, data).get(); });
}
std::future<bool> stage_delayed::arm_keyframes(int layer)
{
    return executor_.begin_invoke([=]() { return stage_->arm_keyframes(layer).get(); });
}
std::future<void> stage_delayed::disarm_keyframes(int layer)
{
    return executor_.begin_invoke([=]() { return stage_->disarm_keyframes(layer).get(); });
}
std::future<void> stage_delayed::clear_keyframes(int layer)
{
    return executor_.begin_invoke([=]() { return stage_->clear_keyframes(layer).get(); });
}
std::future<std::shared_ptr<void>> stage_delayed::get_keyframe_data(int layer)
{
    return executor_.begin_invoke([=]() -> std::shared_ptr<void> { return stage_->get_keyframe_data(layer).get(); });
}
std::future<bool> stage_delayed::has_keyframe_data(int layer)
{
    return executor_.begin_invoke([=]() { return stage_->has_keyframe_data(layer).get(); });
}
std::future<bool> stage_delayed::is_keyframes_armed(int layer)
{
    return executor_.begin_invoke([=]() { return stage_->is_keyframes_armed(layer).get(); });
}
std::future<bool> stage_delayed::patch_keyframe(int layer, double time_secs, std::shared_ptr<void> patch_data)
{
    return executor_.begin_invoke([=]() { return stage_->patch_keyframe(layer, time_secs, patch_data).get(); });
}
std::future<void> stage_delayed::set_media_time_override(int layer, double time_secs)
{
    return executor_.begin_invoke([=]() { return stage_->set_media_time_override(layer, time_secs).get(); });
}
std::future<std::shared_ptr<void>> stage_delayed::get_keyframe_status(int layer)
{
    return executor_.begin_invoke([=]() -> std::shared_ptr<void> { return stage_->get_keyframe_status(layer).get(); });
}

}} // namespace caspar::core
