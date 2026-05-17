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
 * Author: Carlos Fernandez
 */

#pragma once

#include <string>

namespace caspar { namespace core {

class const_frame;  // Forward declaration

/**
 * Write a const_frame to a PNG file on disk.
 *
 * The frame's pixel data (BGRA) is converted to RGBA before writing.
 * Creates parent directories if they don't exist.
 *
 * @param frame The frame to write (must have valid image_data).
 * @param path  Output file path (should end in .png).
 * @return true on success, false on failure.
 */
bool write_frame_png(const const_frame& frame, const std::wstring& path);

}} // namespace caspar::core
