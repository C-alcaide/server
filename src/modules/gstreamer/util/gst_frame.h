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

#include <core/frame/draw_frame.h>
#include <core/frame/frame_metadata.h>

#include <gst/gst.h>   // GstBuffer, GstSample
#include <core/frame/frame_factory.h>

#include <memory>
#include <string>

struct AVFrame;
typedef struct _GstSample GstSample;

namespace caspar { namespace gstreamer {

/// The raw video formats this module accepts from a pipeline, as a caps fragment.
/// Anything else is converted by the videoconvert the producer appends, so this list
/// decides what crosses the boundary without a CPU conversion — not what can be played.
const char* supported_caps_formats();

/// Copies one sample into a mixer frame, with the audio the caller paired to it (which may be
/// null). Returns an empty draw_frame — never throws — when the sample carries a format this
/// module does not map; the caller logs and drops it.
/// Every closed-caption packet on this buffer, or an empty vector.
///
/// Pass-through, not decode: the bytes are carried as they arrived so a consumer can re-emit
/// them. See `core::frame_metadata` for why that distinction matters.
std::shared_ptr<const core::frame_metadata> captions_of(GstBuffer* buffer);

core::draw_frame make_frame(void*                    tag,
                            core::frame_factory&     frame_factory,
                            GstSample*               sample,
                            std::shared_ptr<AVFrame> audio = nullptr);

}} // namespace caspar::gstreamer
