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

#include <cstdint>
#include <vector>

namespace caspar { namespace image {

/// One 8-bit BGRA buffer -> PNG bytes, in memory.
///
/// WHY THIS EXISTS AS A FUNCTION rather than inside the consumer that already encodes PNGs: the
/// IMAGE consumer's encoder is inline and writes to an `fstream`, so nothing else can reach it.
/// The node-graph preview needs the same encode with the bytes handed back, and the alternative
/// -- writing a temporary file and reading it back -- would put a disk round trip and a
/// filesystem race on the path of an API request.
///
/// AND WHY IT LIVES IN THIS MODULE rather than beside the API that wants it: PNG encoding here
/// is FFmpeg, and `protocol_http` neither links FFmpeg nor should. The shell links everything,
/// so it calls this and hands `protocol_http` the finished bytes -- the same injection the
/// timeline uses to reach the producer registry without `core` depending on it.
///
/// BGRA IN, because that is what both mixers' readbacks produce (`device::copy_async` on an
/// 8-bit attachment), so a caller passes what it already has rather than converting first.
///
/// Returns an empty vector on any failure, having logged it. A preview that cannot be encoded
/// is a 500 to the client either way, and there is nothing a caller could do differently with
/// a code.
std::vector<std::uint8_t> encode_png_bgra8(const std::uint8_t* data, int width, int height);

}} // namespace caspar::image
