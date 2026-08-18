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

#pragma once

#include <common/array.h>
#include <common/bit_depth.h>

#ifdef WIN32
#include <core/frame/pixel_format.h>
#include <memory>
namespace caspar::accelerator::d3d {
class d3d_texture2d;
}
#endif

namespace caspar { namespace core {

class frame_factory
{
  public:
    frame_factory()                                = default;
    frame_factory& operator=(const frame_factory&) = delete;
    virtual ~frame_factory()                       = default;

    frame_factory(const frame_factory&) = delete;

    virtual class mutable_frame create_frame(const void* video_stream_tag, const struct pixel_format_desc& desc) = 0;
    virtual class mutable_frame
    create_frame(const void* video_stream_tag, const struct pixel_format_desc& desc, common::bit_depth depth) = 0;

#ifdef WIN32
    /// `audio` rides along because a frame built from an imported texture has nowhere else to
    /// put it: the const_frame that comes back owns an opaque holding the GPU textures, so it
    /// cannot be taken apart and rebuilt with samples added afterwards. Defaulted, so the CEF
    /// shared-texture path — the only caller before this — is unchanged.
    virtual class const_frame import_d3d_texture(const void* video_stream_tag,
                                                 const std::shared_ptr<accelerator::d3d::d3d_texture2d>& d3d_texture,
                                                 core::pixel_format                                      format,
                                                 common::bit_depth                                       depth,
                                                 array<std::int32_t> audio = array<std::int32_t>{}) = 0;
#endif
};

}} // namespace caspar::core
