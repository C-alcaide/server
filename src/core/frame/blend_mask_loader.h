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

#include "frame_transform.h"

#include <memory>
#include <string>

namespace caspar { namespace core {

/**
 * Load a per-pixel projection blend mask from a PNG file.
 *
 * The image is decoded to RGB float values in the 0..1 range and stored in a
 * blend_mask_data buffer.  The mask is sampled in the layer's output screen
 * space and multiplied into the final colour, enabling arbitrary soft-edge
 * overlap masks for multi-projector blending.
 *
 * @param filename  Path to a .png file.
 * @return shared blend_mask_data on success.
 * @throws file_not_found     if the file does not exist.
 * @throws file_read_error    if the PNG could not be decoded.
 */
std::shared_ptr<blend_mask_data> load_blend_mask(const std::wstring& filename);

}} // namespace caspar::core
