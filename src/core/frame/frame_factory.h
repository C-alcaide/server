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

#include <common/bit_depth.h>

namespace caspar { namespace core {

/// Which accelerator a gpu_device_handle() belongs to. A producer that wants to
/// hand the mixer a GPU texture has to know which API to build it with, and the
/// handle alone cannot say -- it used to be assumed to be OpenGL, which is why
/// the Vulkan mixer silently lost the ffmpeg GPU-direct decode path entirely.
enum class gpu_backend
{
    none,
    opengl,
    vulkan
};

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

    /// Return an opaque handle to the underlying GPU device (e.g. ogl::device*
    /// or vulkan::device*, per gpu_device_backend()).
    /// Returns nullptr if the mixer has no GPU device or doesn't support direct import.
    virtual void* gpu_device_handle() const { return nullptr; }

    /// Which API gpu_device_handle() should be interpreted as. Callers must
    /// check this before casting: the two backends' device types are unrelated.
    virtual gpu_backend gpu_device_backend() const { return gpu_backend::none; }
};

}} // namespace caspar::core
