/*
 * Copyright (c) 2026 CasparCG Contributors
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

// NVIDIA GPUDirect for Video (DVP) capability probe for the DeckLink consumer.
//
// DVP provides a hardware-synchronised transfer between a GPU resource and a
// page-locked system-memory buffer that a video-I/O card (DeckLink) DMAs -
// the Tier-2 path. It is only available on professional NVIDIA GPUs with the
// DVP runtime. Unlike the SDK sample's naive strstr("Quadro") check (which
// wrongly rejects modern RTX A-series pro cards), we probe by actually
// initialising the DVP CUDA context, which is the authoritative capability test.
#pragma once

namespace caspar { namespace decklink {

// True when DVP initialised on CUDA device 0. Probed once (cached) and logged.
// Always safe to call; returns false on non-DVP builds or unsupported GPUs.
bool dvp_available();

}} // namespace caspar::decklink
