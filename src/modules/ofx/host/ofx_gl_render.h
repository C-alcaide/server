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

#include <cstdint>
#include <memory>

namespace caspar { namespace ofx {

/// A self-contained offscreen OpenGL context used by the OFX OpenGL render backend.
///
/// The OFX host uploads the source frame into a GL texture, hands the plug-in that texture
/// plus an output texture (via the OpenGL render suite), the plug-in renders into the output
/// texture, and the host reads it back to CPU. This keeps GL-capable plug-ins working without
/// coupling to the Vulkan/OpenGL mixer's own context (at the cost of an upload/readback).
///
/// All methods must be called on the same thread (the producer's frame thread). Construction
/// may throw if a context cannot be created.
class gl_backend
{
  public:
    gl_backend();
    ~gl_backend();

    gl_backend(const gl_backend&)            = delete;
    gl_backend& operator=(const gl_backend&) = delete;

    /// Make the offscreen context current on the calling thread. Returns false on failure.
    bool make_current();

    /// Upload an 8-bit RGBA image (bottom-up) into the reusable source texture; returns its id.
    unsigned int upload_source(const std::uint8_t* rgba, int width, int height);

    /// Ensure the reusable output texture exists at width x height (RGBA8); returns its id.
    unsigned int ensure_output(int width, int height);

    /// Read the output texture back into an 8-bit RGBA buffer (bottom-up).
    void readback_output(std::uint8_t* rgba, int width, int height);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::ofx
