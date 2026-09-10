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

#include "transport.h"

#include <common/log.h>

#include <algorithm>
#include <cstdlib>

namespace caspar { namespace core { namespace timeline {

const char* to_string(transport_state s)
{
    switch (s) {
        case transport_state::playing:
            return "playing";
        case transport_state::paused:
            return "paused";
        default:
            return "stopped";
    }
}

flicks transport::position_at(std::uint64_t frame, flicks per_frame) const
{
    if (st_ != transport_state::playing)
        return anchor_position_;

    // Signed, because `rate` may be negative and `frame` is unsigned: computing
    // `frame - anchor_frame_` in unsigned arithmetic after a seek backwards wraps to something
    // near 2^64 and the playhead lands ten billion years out.
    const auto elapsed = static_cast<std::int64_t>(frame) - static_cast<std::int64_t>(anchor_frame_);

    // The whole point of a rational rate: `elapsed * per_frame * num / den` in that order keeps
    // the product exact for every rate a client can express, so `RATE 1/3` does not drift.
    const auto advanced = (elapsed * per_frame * rate_.numerator()) / rate_.denominator();

    auto pos = anchor_position_ + advanced;

    if (loop_ && loop_->second > loop_->first) {
        const auto span = loop_->second - loop_->first;
        auto       rel  = pos - loop_->first;
        // Floor-modulo, so a negative rate wraps to the END of the region rather than to a
        // position before its start. `%` truncates toward zero, which would give a position
        // outside the region for every backwards pass.
        rel = ((rel % span) + span) % span;
        pos = loop_->first + rel;
    }

    return pos;
}

void transport::reanchor(std::uint64_t frame, flicks per_frame)
{
    anchor_position_ = position_at(frame, per_frame);
    anchor_frame_    = frame;
}

bool transport::apply(const transport_command& c, std::uint64_t frame, flicks per_frame)
{
    const auto before_state = st_;
    const auto before_pos   = anchor_position_;
    const auto before_rate  = rate_;
    const auto before_loop  = loop_;

    switch (c.v) {
        case transport_command::verb::play:
            // Re-anchor FIRST, so playing from paused resumes where it was rather than jumping
            // to wherever the frame counter has reached since.
            reanchor(frame, per_frame);
            st_ = transport_state::playing;
            break;

        case transport_command::verb::pause:
            reanchor(frame, per_frame);
            st_ = transport_state::paused;
            break;

        case transport_command::verb::stop:
            st_              = transport_state::stopped;
            anchor_position_ = 0;
            anchor_frame_    = frame;
            break;

        case transport_command::verb::seek:
            anchor_position_ = c.at;
            anchor_frame_    = frame;
            break;

        case transport_command::verb::rate:
            if (c.new_rate == 0)
                return false; //< rate 0 is `pause` under another name; refused so there is one way
            reanchor(frame, per_frame);
            rate_ = c.new_rate;
            break;

        case transport_command::verb::loop:
            if (c.region.second <= c.region.first)
                return false;
            loop_ = c.region;
            break;

        case transport_command::verb::clear_loop:
            loop_.reset();
            break;

        case transport_command::verb::go:
            fired_.emplace_back(c.trigger, position_at(frame, per_frame));
            return true;
    }

    return st_ != before_state || anchor_position_ != before_pos || rate_ != before_rate ||
           loop_ != before_loop;
}

bool transport::apply_all(std::vector<transport_command>& pending, std::uint64_t frame,
                          flicks per_frame)
{
    if (pending.empty())
        return false;

    // STOP > PAUSE > RUN, and the implementation is "apply the strongest ONE" rather than
    // "apply them all in an order". Sorting and applying every one was the first attempt and it
    // is wrong in the obvious way: whichever rank goes last wins, so ordering stop-then-play
    // leaves it PLAYING -- the exact failure the rank exists to prevent, and the self-test said
    // so at boot.
    //
    // Several clients may act in the same frame, and a stop that loses to a play is a show that
    // keeps running after somebody stopped it. That has to be impossible rather than unlikely,
    // which is WATCHOUT's rank and now this one.
    const auto is_run_state = [](const transport_command& c) {
        return c.v == transport_command::verb::stop || c.v == transport_command::verb::pause ||
               c.v == transport_command::verb::play;
    };
    const auto strength = [](const transport_command& c) {
        switch (c.v) {
            case transport_command::verb::stop:
                return 3;
            case transport_command::verb::pause:
                return 2;
            default:
                return 1; // play
        }
    };

    bool changed = false;

    // POSITIONAL COMMANDS FIRST, in arrival order. Two seeks in one tick genuinely mean "the
    // last one wins", and reordering them would be inventing a rule. They run before the run
    // state so that `SEEK 100` and `STOP` in one tick still end stopped at zero -- stop wins,
    // including over a seek.
    for (const auto& c : pending)
        if (!is_run_state(c))
            changed = apply(c, frame, per_frame) || changed;

    const transport_command* winner = nullptr;
    for (const auto& c : pending) {
        if (!is_run_state(c))
            continue;
        if (!winner || strength(c) > strength(*winner))
            winner = &c;
    }
    if (winner)
        changed = apply(*winner, frame, per_frame) || changed;

    pending.clear();
    return changed;
}

void transport_self_test()
{
    const auto req = [](bool ok, const char* what) {
        if (!ok) {
            CASPAR_LOG(fatal) << L"timeline::transport_self_test: " << what;
            std::abort();
        }
    };

    const auto fps      = boost::rational<int>(25, 1);
    const auto per_frame = flicks_per_frame(fps);

    const auto cmd = [](transport_command::verb v) {
        transport_command c;
        c.v = v;
        return c;
    };

    // ---- stopped stays at zero, whatever the frame counter does -------------------------
    {
        transport t;
        req(t.state() == transport_state::stopped, "a new transport is stopped");
        req(t.position_at(0, per_frame) == 0, "at zero");
        req(t.position_at(100000, per_frame) == 0,
            "and a stopped playhead does not advance with the frame counter");
    }

    // ---- playing advances EXACTLY one frame per frame -----------------------------------
    {
        transport t;
        t.apply(cmd(transport_command::verb::play), 1000, per_frame);
        req(t.state() == transport_state::playing, "play plays");
        req(t.position_at(1000, per_frame) == 0, "from where it was");
        req(t.position_at(1001, per_frame) == per_frame, "one frame later, exactly one frame in");
        req(t.position_at(1025, per_frame) == from_seconds(1.0),
            "and twenty-five frames later, exactly one second at 25p");
        req(t.position_at(1000 + 25 * 3600, per_frame) == from_seconds(3600.0),
            "and an hour later, exactly an hour -- no accumulated rounding, because the position "
            "is DERIVED from an anchor rather than added to every tick");
    }

    // ---- pause holds, and resume does not jump ------------------------------------------
    {
        transport t;
        t.apply(cmd(transport_command::verb::play), 0, per_frame);
        t.apply(cmd(transport_command::verb::pause), 50, per_frame);
        req(t.state() == transport_state::paused, "pause pauses");
        req(t.position_at(50, per_frame) == 50 * per_frame, "at where it had reached");
        req(t.position_at(500, per_frame) == 50 * per_frame,
            "and it stays there while the frame counter runs on");

        t.apply(cmd(transport_command::verb::play), 500, per_frame);
        req(t.position_at(500, per_frame) == 50 * per_frame,
            "resuming starts from where it paused, NOT from wherever the counter reached -- "
            "otherwise a pause of ten seconds costs ten seconds of the show");
        req(t.position_at(501, per_frame) == 51 * per_frame, "and then advances normally");
    }

    // ---- stop is not pause ---------------------------------------------------------------
    {
        transport t;
        t.apply(cmd(transport_command::verb::play), 0, per_frame);
        t.apply(cmd(transport_command::verb::stop), 50, per_frame);
        req(t.state() == transport_state::stopped, "stop stops");
        req(t.position_at(50, per_frame) == 0, "and rewinds to zero");
    }

    // ---- seek, forwards and backwards ----------------------------------------------------
    {
        transport t;
        auto      c = cmd(transport_command::verb::seek);
        c.at        = from_seconds(12.0);
        t.apply(c, 100, per_frame);
        req(t.position_at(100, per_frame) == from_seconds(12.0), "seek lands where asked");
        req(t.state() == transport_state::stopped, "and does not start playback by itself");

        t.apply(cmd(transport_command::verb::play), 100, per_frame);
        req(t.position_at(125, per_frame) == from_seconds(13.0), "and playback continues from it");

        // A SEEK BACKWARDS, which is where unsigned arithmetic on the frame counter goes wrong.
        c.at = from_seconds(1.0);
        t.apply(c, 200, per_frame);
        req(t.position_at(200, per_frame) == from_seconds(1.0), "a backwards seek lands");
        req(t.position_at(199, per_frame) < from_seconds(1.0),
            "and a frame BEFORE the anchor is before it, not ten billion years out -- which is "
            "what an unsigned `frame - anchor` gives");
    }

    // ---- rate, including negative and fractional ------------------------------------------
    {
        transport t;
        t.apply(cmd(transport_command::verb::play), 0, per_frame);
        auto c      = cmd(transport_command::verb::rate);
        c.new_rate  = 2;
        t.apply(c, 0, per_frame);
        req(t.position_at(25, per_frame) == from_seconds(2.0), "rate 2 covers two seconds in one");

        c.new_rate = boost::rational<std::int64_t>(1, 3);
        t.apply(c, 0, per_frame);
        req(t.position_at(75, per_frame) == from_seconds(1.0),
            "rate 1/3 covers one second in three -- EXACTLY, which is why the rate is a rational");

        transport r;
        auto      s = cmd(transport_command::verb::seek);
        s.at        = from_seconds(10.0);
        r.apply(s, 0, per_frame);
        r.apply(cmd(transport_command::verb::play), 0, per_frame);
        c.new_rate = -1;
        r.apply(c, 0, per_frame);
        req(r.position_at(25, per_frame) == from_seconds(9.0), "rate -1 descends");
        req(r.position_at(250, per_frame) == from_seconds(0.0), "and passes zero into negative");
        req(r.position_at(275, per_frame) == from_seconds(-1.0), "which is a position, not a wrap");

        c.new_rate = 0;
        req(!r.apply(c, 0, per_frame),
            "rate 0 is refused -- it is `pause` under another name, and two ways to do one thing "
            "is two things to keep consistent");
    }

    // ---- a loop region, in both directions -----------------------------------------------
    {
        transport t;
        auto      l  = cmd(transport_command::verb::loop);
        l.region     = {from_seconds(20.0), from_seconds(45.0)};
        t.apply(l, 0, per_frame);
        auto s = cmd(transport_command::verb::seek);
        s.at   = from_seconds(20.0);
        t.apply(s, 0, per_frame);
        t.apply(cmd(transport_command::verb::play), 0, per_frame);

        req(t.position_at(0, per_frame) == from_seconds(20.0), "it starts at the region's start");
        req(t.position_at(25 * 24, per_frame) == from_seconds(44.0), "and runs to just before the end");
        req(t.position_at(25 * 25, per_frame) == from_seconds(20.0), "then wraps to the start");
        req(t.position_at(25 * 26, per_frame) == from_seconds(21.0), "and keeps going");
        req(t.position_at(25 * 50, per_frame) == from_seconds(20.0), "twice round is the same place");
        for (int f = 0; f < 25 * 200; f += 7) {
            const auto p = t.position_at(static_cast<std::uint64_t>(f), per_frame);
            req(p >= from_seconds(20.0) && p < from_seconds(45.0),
                "and the playhead NEVER leaves the region");
        }

        // Backwards through the region: floor-modulo, so it wraps to the END rather than to
        // somewhere before the start. `%` truncating toward zero gives the latter.
        auto c     = cmd(transport_command::verb::rate);
        c.new_rate = -1;
        t.apply(c, 0, per_frame);
        for (int f = 0; f < 25 * 200; f += 7) {
            const auto p = t.position_at(static_cast<std::uint64_t>(f), per_frame);
            req(p >= from_seconds(20.0) && p < from_seconds(45.0),
                "in reverse too -- which is what floor-modulo buys over `%`");
        }
    }

    // ---- STOP > PAUSE > RUN within one tick -----------------------------------------------
    {
        transport                      t;
        std::vector<transport_command> pending;
        pending.push_back(cmd(transport_command::verb::stop));
        pending.push_back(cmd(transport_command::verb::play));
        t.apply_all(pending, 0, per_frame);
        req(t.state() == transport_state::stopped,
            "a stop and a play in the same tick leaves it STOPPED, whatever order they arrived "
            "in -- a stop that loses to a play is a show that keeps running after somebody "
            "stopped it");

        std::vector<transport_command> p2;
        p2.push_back(cmd(transport_command::verb::play));
        p2.push_back(cmd(transport_command::verb::pause));
        t.apply_all(p2, 0, per_frame);
        req(t.state() == transport_state::paused, "and a pause outranks a play the same way");

        // A SEEK AND A PLAY in one tick: both take effect, because a seek is positional rather
        // than a run state and there is nothing for them to disagree about.
        transport                      u;
        std::vector<transport_command> p3;
        auto                           sk = cmd(transport_command::verb::seek);
        sk.at                              = from_seconds(30.0);
        p3.push_back(sk);
        p3.push_back(cmd(transport_command::verb::play));
        u.apply_all(p3, 0, per_frame);
        req(u.state() == transport_state::playing, "a seek does not suppress a play");
        req(u.position_at(0, per_frame) == from_seconds(30.0), "and the play starts from the seek");

        // ...and a STOP in the same tick as a seek still ends stopped at zero.
        std::vector<transport_command> p4;
        p4.push_back(sk);
        p4.push_back(cmd(transport_command::verb::stop));
        u.apply_all(p4, 0, per_frame);
        req(u.state() == transport_state::stopped && u.position_at(0, per_frame) == 0,
            "stop wins over a seek in the same tick, and rewinds");
    }

    // ---- go records its trigger with the position it fired at ------------------------------
    {
        transport t;
        t.apply(cmd(transport_command::verb::play), 0, per_frame);
        auto g    = cmd(transport_command::verb::go);
        g.trigger = "go";
        t.apply(g, 250, per_frame);
        req(t.fired().size() == 1, "a GO is recorded");
        req(t.fired().front().first == "go", "by name");
        req(t.fired().front().second == from_seconds(10.0),
            "with the position it fired at, which is what the resolver needs to place the "
            "objects waiting on it");
    }

    CASPAR_LOG(info) << L"[timeline-transport] self-test: all checks passed";
}

}}} // namespace caspar::core::timeline
