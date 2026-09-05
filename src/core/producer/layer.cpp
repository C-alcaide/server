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

#include "layer.h"

#include <limits>

#include "frame_producer.h"

#include "../frame/draw_frame.h"
#include "../video_format.h"

namespace caspar { namespace core {

struct layer::impl
{
    monitor::state                state_;
    const core::video_format_desc format_desc_;

    spl::shared_ptr<frame_producer> foreground_ = frame_producer::empty();
    spl::shared_ptr<frame_producer> background_ = frame_producer::empty();

    bool auto_play_ = false;
    bool paused_    = false;

  public:
    impl(const core::video_format_desc format_desc)
        : format_desc_(format_desc)
    {
    }

    void pause() { paused_ = true; }

    void resume() { paused_ = false; }

    void load(spl::shared_ptr<frame_producer> producer, bool preview_producer, bool auto_play)
    {
        background_ = std::move(producer);
        auto_play_  = auto_play;

        if (auto_play_ && foreground_ == frame_producer::empty()) {
            play();
        } else if (preview_producer) {
            preview(true);
        }
    }

    void preview(bool force)
    {
        if (force || background_ != frame_producer::empty()) {
            play();
            paused_ = true;
        }
    }

    void play()
    {
        if (background_ != frame_producer::empty()) {
            if (!paused_) {
                background_->leading_producer(foreground_);
            } else {
                if (format_desc_.field_count == 2) {
                    auto frame1 = foreground_->last_frame(core::video_field::a);
                    auto frame2 = foreground_->last_frame(core::video_field::b);
                    background_->leading_producer(
                        spl::make_shared<core::const_producer>(std::move(frame1), std::move(frame2)));
                } else {
                    auto frame = foreground_->last_frame(core::video_field::progressive);
                    background_->leading_producer(spl::make_shared<core::const_producer>(frame, frame));
                }
            }

            foreground_ = std::move(background_);
            background_ = frame_producer::empty();

            auto_play_ = false;
        }

        paused_ = false;
    }

    void stop()
    {
        foreground_ = frame_producer::empty();
        auto_play_  = false;
    }

    draw_frame receive(const video_field field, int nb_samples)
    {
        try {
            if (foreground_->following_producer() != core::frame_producer::empty() && field != video_field::b) {
                foreground_ = foreground_->following_producer();
            }

            int64_t frames_left = 0;
            if (auto_play_) {
                auto auto_play_delta = background_->auto_play_delta();
                if (auto_play_delta) {
                    auto time     = static_cast<std::int64_t>(foreground_->frame_number());
                    auto duration = static_cast<std::int64_t>(foreground_->nb_frames());
                    frames_left   = duration - time - *auto_play_delta;
                    if (frames_left < 1 && field != video_field::b) {
                        play();
                    }
                }
            }

            auto frame = paused_ ? core::draw_frame{} : foreground_->receive(field, nb_samples);
            if (!frame) {
                frame = foreground_->last_frame(field);
            }

            state_                           = {};
            state_["foreground"]             = foreground_->state();
            state_["foreground"]["producer"] = foreground_->name();
            state_["foreground"]["paused"]   = paused_;

            // What a client otherwise has to guess.
            //
            // `casparcg-state`, the Sofie-ecosystem library that drives upstream CasparCG,
            // carries `MIN_TIME_SINCE_PLAY = 150 // [ms]` -- a hardcoded quarantine after a
            // PLAY, during which it holds commands rather than issuing them. It exists
            // because the client cannot ask whether the producer is ready, so the number was
            // arrived at empirically. `is_ready()` has been on every producer all along and
            // was published nowhere: no OSC key, no INFO field, no command.
            //
            // Called here, on the tick thread, and nowhere else -- `av_producer::is_ready`
            // takes the same `buffer_mutex_` the decode thread does. `transition_producer`
            // already calls it once per tick per layer, so this is within precedent.
            const bool empty = foreground_ == frame_producer::empty();
            const bool ready = !empty && foreground_->is_ready();

            state_["foreground"]["empty"] = empty;
            state_["foreground"]["ready"] = ready;

            // A transport state, so "what is this layer doing" is one read rather than an
            // inference over paused, frames_left and a duration the client had to fetch
            // separately. `nb_frames()` is UINT32_MAX while looping, so a loop never reads eof.
            const auto nb  = foreground_->nb_frames();
            const auto fn  = foreground_->frame_number();
            const bool eof = !empty && nb != std::numeric_limits<uint32_t>::max() && fn >= nb;

            state_["foreground"]["transport"] = empty    ? std::string("stopped")
                                               : paused_ ? std::string("paused")
                                               : !ready  ? std::string("loading")
                                               : eof     ? std::string("eof")
                                                         : std::string("playing");

            if (frames_left > 0) {
                state_["foreground"]["frames_left"] = frames_left;
            }

            state_["background"]             = background_->state();
            state_["background"]["producer"] = background_->name();

            return frame;
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
            stop();
            return draw_frame{};
        }
    }

    draw_frame receive_background(const video_field field, int nb_samples)
    {
        try {
            return background_->first_frame(field);

        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
            background_ = frame_producer::empty();
            return draw_frame{};
        }
    }
};

layer::layer(const core::video_format_desc format_desc)
    : impl_(new impl(format_desc))
{
}
layer::layer(layer&& other)
    : impl_(std::move(other.impl_))
{
}
layer& layer::operator=(layer&& other)
{
    impl_ = std::move(other.impl_);
    return *this;
}
void layer::swap(layer& other) { impl_.swap(other.impl_); }
void layer::load(spl::shared_ptr<frame_producer> frame_producer, bool preview, bool auto_play)
{
    return impl_->load(std::move(frame_producer), preview, auto_play);
}
void       layer::play() { impl_->play(); }
void       layer::preview() { impl_->preview(false); }
void       layer::pause() { impl_->pause(); }
void       layer::resume() { impl_->resume(); }
void       layer::stop() { impl_->stop(); }
draw_frame layer::receive(const video_field field, int nb_samples) { return impl_->receive(field, nb_samples); }
draw_frame layer::receive_background(const video_field field, int nb_samples)
{
    return impl_->receive_background(field, nb_samples);
}
spl::shared_ptr<frame_producer> layer::foreground() const { return impl_->foreground_; }
spl::shared_ptr<frame_producer> layer::background() const { return impl_->background_; }
bool                            layer::has_background() const { return impl_->background_ != frame_producer::empty(); }
core::monitor::state            layer::state() const { return impl_->state_; }
}} // namespace caspar::core
