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

// GPU-direct DeckLink output for the OpenGL mixer.
//
// The OpenGL mixer composites into a GL texture. Instead of reading that whole
// texture back to the CPU and packing v210/BGRA with AVX2, this strategy runs a
// GL compute shader on the mixer's own GL thread to pack the (subregion of the)
// texture into v210 or BGRA, then reads back only the packed result into a
// page-locked host buffer that DeckLink DMAs. The compute + readback run via the
// texture's ogl::device dispatch, so they execute on the GL thread where the
// texture and its context are valid and are naturally ordered after the mixer's
// composite (in-context command ordering) - no cross-context fence needed.
//
// Falls back to the wrapped CPU strategy when the frame carries no OGL texture
// (e.g. a different mixer), for interlaced output, or for key-only ports.
#pragma once

#include "format_strategy.h"

namespace caspar { namespace decklink {

class ogl_gl_strategy final : public format_strategy
{
  public:
    ogl_gl_strategy(bool                             is_hdr,
                    bool                             use_bt2020,
                    spl::shared_ptr<format_strategy> fallback,
                    bool                             needs_v210,
                    bool                             use_dvp);
    ~ogl_gl_strategy() override;

    BMDPixelFormat        get_pixel_format() override;
    int                   get_row_bytes(int width) override;
    std::shared_ptr<void> allocate_frame_data(const core::video_format_desc& format_desc) override;
    std::shared_ptr<void> convert_frame_for_port(const core::video_format_desc& channel_format_desc,
                                                 const core::video_format_desc& decklink_format_desc,
                                                 const port_configuration&      config,
                                                 const core::const_frame&       frame1,
                                                 const core::const_frame&       frame2,
                                                 BMDFieldDominance              field_dominance) override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

// Returns an ogl_gl_strategy, or the fallback if the OGL GPU path can't be built.
// use_dvp requests the NVIDIA GPUDirect-for-Video (DVP) readback tail instead of
// glGetBufferSubData; it self-probes at runtime and falls back if DVP is absent.
spl::shared_ptr<format_strategy> try_create_ogl_gl_strategy(bool                             is_hdr,
                                                            bool                             use_bt2020,
                                                            spl::shared_ptr<format_strategy> fallback,
                                                            bool                             needs_v210,
                                                            bool                             use_dvp);

}} // namespace caspar::decklink
