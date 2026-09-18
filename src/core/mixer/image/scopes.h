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

#pragma once

#include <core/monitor/monitor.h>

#include <cstdint>
#include <vector>

namespace caspar { namespace core { namespace scopes {

/// The histogram the two mixers' compute shaders produce: 768 counts, r[0..255] then
/// g[0..255] then b[0..255], where a pixel's bin is round(v * 255) of the value the consumer
/// receives -- so a flat 8-bit patch lands in exactly one bin per channel.
constexpr std::size_t histogram_bins   = 256;
constexpr std::size_t histogram_counts = 3 * histogram_bins;

/// Fold a histogram into the tree a channel publishes under `mixer/scopes/`:
///
///     scopes/histogram/r   256 counts          scopes/pixels   width * height
///     scopes/histogram/g   256 counts          scopes/bins     256
///     scopes/histogram/b   256 counts
///
/// Identical on both backends by construction, because this is the only place the shape is
/// written down. Three 256-element vectors is three heap allocations per tick (DECISIONS
/// L166's caveat), which is why the feature is opt-in (L394) and why `pixels` is published
/// beside them: sum(bins) == pixels is the integrity check a client can run for free.
inline monitor::state fold_histogram(const std::vector<std::uint32_t>& counts, int width, int height)
{
    monitor::state s;
    if (counts.size() != histogram_counts)
        return s;
    const char* names[3] = {"r", "g", "b"};
    for (std::size_t c = 0; c < 3; ++c) {
        monitor::vector_t bins;
        bins.reserve(histogram_bins);
        for (std::size_t i = 0; i < histogram_bins; ++i)
            bins.push_back(static_cast<std::int32_t>(counts[c * histogram_bins + i]));
        s["histogram"][names[c]] = std::move(bins);
    }
    s["pixels"] = static_cast<std::int64_t>(width) * static_cast<std::int64_t>(height);
    s["bins"]   = static_cast<std::int32_t>(histogram_bins);
    return s;
}

}}} // namespace caspar::core::scopes
