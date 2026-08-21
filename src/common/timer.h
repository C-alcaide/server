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
 * Author: Helge Norberg, helge.norberg@svt.se
 */

#pragma once

#include <chrono>

namespace caspar {

class timer
{
    std::int_least64_t start_time_;

  public:
    timer() { start_time_ = now(); }

    void restart() { start_time_ = now(); }

    /// Seconds, as before -- only the RESOLUTION changed, from 1 ms to 1 us.
    ///
    /// `now()` used to be `duration_cast<milliseconds>`, so `elapsed()` could only ever
    /// return whole milliseconds and every sub-millisecond measurement in this tree read
    /// exactly **zero**. That is not a rounding nicety, it invalidated published numbers:
    /// `GPU_INTEROP_ARCHITECTURE.md` carried a DeckLink table quoting a 0.06 ms fence wait,
    /// a figure this instrument could not represent, and re-running it on 2026-08-21 gave
    /// `fence=0ms import=0ms wait=0ms launch=0ms total=0ms` for most 50-frame windows with
    /// occasional 0.04/0.78 -- which is one quantised 1 ms hit divided by 50, not a
    /// measurement of anything.
    ///
    /// Microseconds rather than nanoseconds because int64 microseconds since the epoch has
    /// no practical overflow and the extra three digits buy nothing on a frame path whose
    /// steps are tens of microseconds at the low end.
    double elapsed() const { return static_cast<double>(now() - start_time_) / 1'000'000.0; }

  private:
    static std::int_least64_t now()
    {
        using namespace std::chrono;

        return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }
};

} // namespace caspar
