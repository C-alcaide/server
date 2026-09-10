/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// THE PLAYHEAD, and the one thing in the timeline that has a clock.
//
// THE CLOCK IS THE CHANNEL'S FRAME COUNTER, never a producer's. That is D2 of the plan and it is
// the single largest behavioural difference from the `KEYFRAMES` engine this replaces, which took
// its time from `producer->frame_number()` on the animated layer. The consequences of that were
// not edge cases:
//
//   a COLOUR layer's frame number is 0 forever, so an animated grade on a colour fill never
//   moved;
//   an EMPTY layer pins t = 0, so a document that animates a layer before its clip loads starts
//   over when the clip arrives;
//   a PAUSED clip freezes the animation with it, so a hold on a still frame stops the grade
//   ramping under it -- which is the one time an operator most wants it to keep going;
//   and `SEEK` was unobservable, because the next tick recomputed the position from the producer
//   and overwrote whatever the seek had set.
//
// The frame counter has none of those failure modes, and it is the same NOMINAL clock
// `evaluate_bindings` already chose: `position_at` advances `flicks_per_frame x rate` per tick
// from the `frame_number` `video_channel` passes into `stage::operator()`. A dropped frame does
// not make the show run fast, and a fitted trajectory is reproducible -- which is what every gate
// in the harness reads.
//
// Shows that must not slip against house time get timecode chase (D9), which is a CORRECTION on
// top of this rather than a different clock.

#include "time.h"

#include <boost/rational.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace caspar { namespace core { namespace timeline {

enum class transport_state
{
    stopped, //< position 0, nothing owned, nothing published as active
    paused,  //< position held; ownership KEPT (D4: pause is not release)
    playing
};

const char* to_string(transport_state s);

/// One thing a client can ask of a playhead.
struct transport_command
{
    enum class verb
    {
        play,
        pause,
        stop,
        seek,
        rate,
        loop,
        clear_loop,
        go
    };

    verb                                    v = verb::play;
    flicks                                  at = 0;                //< for seek
    boost::rational<std::int64_t>           new_rate{1};           //< for rate
    std::pair<flicks, flicks>               region{0, 0};          //< for loop
    std::string                             trigger;               //< for go
    std::optional<std::uint64_t>            at_frame;              //< schedule it
};

class transport
{
  public:
    transport_state               state() const { return st_; }
    boost::rational<std::int64_t> rate() const { return rate_; }
    std::optional<std::pair<flicks, flicks>> loop() const { return loop_; }

    /// The position at `frame`, given the channel's flicks per frame.
    ///
    /// PURE and monotonic in `frame`: same inputs, same answer, so a client can predict where a
    /// document will be at a frame it has not reached, and a battery can compute the expected
    /// value for a frame-stamped sample rather than for a wall-clock instant.
    flicks position_at(std::uint64_t frame, flicks per_frame) const;

    /// Apply one command as of `frame`. Returns whether anything changed.
    ///
    /// STOP > PAUSE > RUN within one tick, which is WATCHOUT's rank and the only one that is
    /// safe: several clients may act in the same frame, and a stop that loses to a play is a show
    /// that keeps running after somebody stopped it.
    bool apply(const transport_command& c, std::uint64_t frame, flicks per_frame);

    /// Apply a whole tick's worth, in the safe order rather than arrival order.
    bool apply_all(std::vector<transport_command>& pending, std::uint64_t frame, flicks per_frame);

    /// The triggers a GO has fired, in order, with the position at which each fired.
    const std::vector<std::pair<std::string, flicks>>& fired() const { return fired_; }
    void                                               clear_fired() { fired_.clear(); }

  private:
    transport_state               st_       = transport_state::stopped;
    boost::rational<std::int64_t> rate_{1};

    /// The playhead is not stored -- it is DERIVED from the frame counter, from an anchor pair.
    /// Storing it and adding to it every tick would accumulate whatever the last rate change
    /// rounded, and would make `position_at(frame)` unanswerable for any frame but the current
    /// one. An anchor makes both exact.
    std::uint64_t anchor_frame_    = 0;
    flicks        anchor_position_ = 0;

    std::optional<std::pair<flicks, flicks>>    loop_;
    std::vector<std::pair<std::string, flicks>> fired_;

    void reanchor(std::uint64_t frame, flicks per_frame);
};

/// Aborts on a transport disagreement. Called at boot.
void transport_self_test();

}}} // namespace caspar::core::timeline
