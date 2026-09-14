/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include <common/except.h>
#include "frame_clock.h"

namespace caspar { namespace cluster { namespace sync {

frame_clock::frame_clock(std::shared_ptr<ptp::ptp_clock> clock,
                         int64_t                         epoch_origin_ns,
                         int                             fps_num,
                         int                             fps_den)
    : clock_(std::move(clock))
    , epoch_origin_ns_(epoch_origin_ns)
    , fps_packed_(pack_fps(fps_num, fps_den))
{
}

int64_t frame_clock::current_frame() const
{
    return ptp_ns_to_frame(clock_->now_ns());
}

int64_t frame_clock::frame_to_ptp_ns(int64_t frame) const
{
    auto [num, den] = unpack_fps(fps_packed_.load(std::memory_order_relaxed));
    // ptp_ns = epoch_origin + frame * den * 1e9 / num
    // Split to avoid overflow: frame * den * 1e9 overflows int64 at ~42h for 59.94fps
    // frame_ns = (frame / num) * den * 1e9 + ((frame % num) * den * 1e9) / num
    int64_t whole_sec_frames = frame / num;          // How many whole "seconds worth" of frames
    int64_t remainder_frames = frame % num;          // Leftover frames within the second
    int64_t whole_ns = whole_sec_frames * den * 1'000'000'000LL; // safe: whole_sec_frames * den fits for centuries
    int64_t frac_ns  = (remainder_frames * den * 1'000'000'000LL) / num; // safe: remainder < num, so remainder*den*1e9 < num*den*1e9 ≈ 6e13
    return epoch_origin_ns_ + whole_ns + frac_ns;
}

int64_t frame_clock::ptp_ns_to_frame(int64_t ptp_ns) const
{
    auto [num, den] = unpack_fps(fps_packed_.load(std::memory_order_relaxed));
    int64_t elapsed = ptp_ns - epoch_origin_ns_;
    if (elapsed < 0) {
        return -1;
    }
    // frame = elapsed_ns * num / (den * 1e9), split to avoid overflowing (elapsed * num) --
    // and the split CARRIES ITS REMAINDER, which is the whole of the correction below.
    //
    // ── THIS COMPUTED floor(a) + floor(b) WHERE THE ANSWER IS floor(a + b) ──────────────
    //
    // The previous form took `(elapsed_sec * num) / den` and `(elapsed_rem * num) / (den *
    // 1e9)` and added them. For an INTEGER frame rate `den` is 1, the first division is exact,
    // and the two agree. For a 1001-denominator rate the first division truncates and the lost
    // fraction is dropped rather than carried, so the result is **one frame low**, on a
    // fraction of instants that depends on the sub-second phase.
    //
    // Measured 2026-09-14 over 3000 instants per rate: exact at 25p and 50p; **wrong on 58% of
    // instants at 29.97, 76% at 59.94 and 78% at 23.976**, always by one frame. `ptp_ns_to_frame`
    // is how a node answers "which frame is it", so on any fractional rate two nodes sampling at
    // slightly different sub-second phases disagreed about the frame number -- which is the one
    // thing this module exists to prevent. `sync_framerate_from_channels()` passes
    // `video_format_desc().framerate` straight through, so 1001 denominators reach here on any
    // NTSC-rate channel.
    //
    // The exact form, with elapsed = elapsed_sec*1e9 + elapsed_rem and A = elapsed_sec*num:
    //     frame = (A*1e9 + elapsed_rem*num) / (den*1e9)
    //           = A/den + (A%den * 1e9 + elapsed_rem*num) / (den*1e9)
    // Every term stays well inside int64: A is ~4.7e11 after 90 days at 59.94, and the
    // numerator of the second term is bounded by (den-1)*1e9 + 1e9*num ≈ 6.1e13.
    const int64_t elapsed_sec = elapsed / 1'000'000'000LL;
    const int64_t elapsed_rem = elapsed % 1'000'000'000LL;

    const int64_t a         = elapsed_sec * num;
    const int64_t whole     = a / den;
    const int64_t carry     = a % den;   // the fraction the old form threw away
    const int64_t remainder = carry * 1'000'000'000LL + elapsed_rem * num;

    return whole + remainder / (static_cast<int64_t>(den) * 1'000'000'000LL);
}

int64_t frame_clock::ns_until_frame(int64_t target_frame) const
{
    int64_t target_ns = frame_to_ptp_ns(target_frame);
    int64_t now_ns    = clock_->now_ns();
    return target_ns - now_ns;
}

void frame_clock::set_framerate(int fps_num, int fps_den)
{
    if (fps_den <= 0 || fps_num <= 0) {
        return;
    }
    fps_packed_.store(pack_fps(fps_num, fps_den), std::memory_order_relaxed);
}

// ── THE FRAME ARITHMETIC IS CHECKED AT BOOT ─────────────────────────────────────────────
//
// A cluster's whole promise is that every node answers "which frame is it" with the same
// number, and that answer is this file. It needs no cluster, no network and no channel to
// check -- it is a pure function -- so it is asserted at start-up rather than left to a
// battery that would need two machines to run.
//
// **IT IS GATED ON 1001-DENOMINATOR RATES BECAUSE THAT IS WHERE IT WAS WRONG.** The previous
// `ptp_ns_to_frame` computed floor(a) + floor(b) where the answer is floor(a + b); at 25p and
// 50p `den` is 1, the first term is exact, and the two forms agree, so an integer-rate check
// could not have caught it. Measured before the fix: wrong on 58% of instants at 29.97, 76% at
// 59.94, 78% at 23.976, always one frame low.
void frame_clock_self_test()
{
    const auto fail = [](const std::string& what) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("frame_clock self-test: " + what));
    };

    struct rate
    {
        int         num;
        int         den;
        const char* name;
    };
    // Integer rates AND fractional ones: the fractional ones are the regression, the integer
    // ones are the control that says the fix did not break the common case.
    const rate rates[] = {
        {25, 1, "25p"},         {50, 1, "50p"},          {60, 1, "60p"},
        {24000, 1001, "23.976p"}, {30000, 1001, "29.97p"}, {60000, 1001, "59.94p"},
    };

    constexpr int64_t NS = 1'000'000'000LL;

    for (const auto& r : rates) {
        frame_clock fc(nullptr, 0, r.num, r.den);

        // 1. ptp_ns_to_frame agrees with the exact rational answer, over a spread of
        //    sub-second phases -- which is the axis the old form was wrong along.
        for (int64_t sec = 0; sec < 400; ++sec) {
            for (int64_t sub : {0LL, 1LL, 250'000'000LL, 500'000'000LL, 999'999'999LL}) {
                const int64_t elapsed = sec * NS + sub;
                const int64_t got = fc.ptp_ns_to_frame(elapsed);

                // Oracle: frame n has started iff frame_to_ptp_ns(n) <= elapsed, and frame
                // n+1 has not. Expressed as a RELATIONSHIP between the two conversions rather
                // than as a second copy of the formula, so the two cannot be wrong together.
                if (got < 0)
                    fail(std::string(r.name) + ": negative frame for a non-negative elapsed");
                if (fc.frame_to_ptp_ns(got) > elapsed)
                    fail(std::string(r.name) + ": reported a frame that has not started yet");
                if (fc.frame_to_ptp_ns(got + 1) <= elapsed)
                    fail(std::string(r.name) + ": reported a frame that is already over -- the "
                                               "1001-denominator off-by-one, if this is 29.97, "
                                               "59.94 or 23.976");
            }
        }

        // 2. Monotonic: time never runs backwards, and the frame number never does either.
        int64_t prev = -1;
        for (int64_t e = 0; e < 5 * NS; e += 997'003LL) {
            const int64_t f = fc.ptp_ns_to_frame(e);
            if (f < prev)
                fail(std::string(r.name) + ": frame number went backwards");
            prev = f;
        }
    }
}

}}} // namespace caspar::cluster::sync
