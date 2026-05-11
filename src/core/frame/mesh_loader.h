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

#include "geometry.h"

#include <string>

namespace caspar { namespace core {

/**
 * Load a mesh from a glTF/GLB or OBJ file and return it as a frame_geometry
 * with geometry_type::mesh.
 *
 * Vertex positions from the mesh are normalised to the 0-1 range (bounding-box
 * fit) so they work with the existing CasparCG 2D compositor pipeline.  Mesh UV
 * coordinates are passed through directly as texture coordinates.
 *
 * Only the first mesh/primitive in the file is loaded.
 *
 * @param filename  Path to a .glb, .gltf, or .obj file.
 * @return frame_geometry with type mesh and triangulated vertex data.
 * @throws file_not_found    if the file does not exist.
 * @throws invalid_argument  if the file cannot be parsed or contains no geometry.
 */
frame_geometry load_mesh(const std::wstring& filename);

}} // namespace caspar::core
