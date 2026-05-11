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

#include "mesh_loader.h"

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/filesystem.hpp>

// tinygltf — implementation compiled here
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tiny_gltf.h>

// tinyobjloader — implementation compiled here
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace caspar { namespace core {

namespace {

// ---- helpers ---------------------------------------------------------------

struct mesh_data
{
    std::vector<frame_geometry::coord> coords;
};

// Normalise vertex positions so they fit within 0-1 (bounding-box fit,
// preserving aspect ratio, centered).
void normalise_positions(std::vector<frame_geometry::coord>& coords)
{
    if (coords.empty())
        return;

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    for (const auto& c : coords) {
        min_x = std::min(min_x, c.vertex_x);
        min_y = std::min(min_y, c.vertex_y);
        max_x = std::max(max_x, c.vertex_x);
        max_y = std::max(max_y, c.vertex_y);
    }

    double w = max_x - min_x;
    double h = max_y - min_y;
    if (w < 1e-12)
        w = 1.0;
    if (h < 1e-12)
        h = 1.0;

    double scale = 1.0 / std::max(w, h);

    // Center the smaller axis
    double offset_x = (1.0 - w * scale) * 0.5;
    double offset_y = (1.0 - h * scale) * 0.5;

    for (auto& c : coords) {
        c.vertex_x = (c.vertex_x - min_x) * scale + offset_x;
        c.vertex_y = (c.vertex_y - min_y) * scale + offset_y;
    }
}

// ---- glTF loader -----------------------------------------------------------

template <typename T>
const T* get_buffer_ptr(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    const auto& bv = model.bufferViews[accessor.bufferView];
    return reinterpret_cast<const T*>(&model.buffers[bv.buffer].data[bv.byteOffset + accessor.byteOffset]);
}

int get_stride(const tinygltf::Model& model, const tinygltf::Accessor& accessor, int component_size)
{
    const auto& bv = model.bufferViews[accessor.bufferView];
    return bv.byteStride > 0 ? static_cast<int>(bv.byteStride)
                             : (component_size * tinygltf::GetNumComponentsInType(accessor.type));
}

mesh_data load_gltf(const std::string& path)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!warn.empty())
        CASPAR_LOG(warning) << L"[mesh_loader] glTF warning: " << u8(warn);
    if (!ok)
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("Failed to load glTF: " + (err.empty() ? path : err)));
    if (model.meshes.empty())
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("glTF file contains no meshes: " + path));

    // Use the first primitive of the first mesh
    const auto& mesh = model.meshes[0];
    if (mesh.primitives.empty())
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("glTF mesh has no primitives: " + path));

    const auto& prim = mesh.primitives[0];

    // Get POSITION accessor
    auto pos_it = prim.attributes.find("POSITION");
    if (pos_it == prim.attributes.end())
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("glTF primitive has no POSITION attribute: " + path));

    const auto& pos_acc = model.accessors[pos_it->second];
    if (pos_acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("glTF POSITION must be float: " + path));

    // Get TEXCOORD_0 accessor (optional)
    const tinygltf::Accessor* uv_acc = nullptr;
    auto                      uv_it  = prim.attributes.find("TEXCOORD_0");
    if (uv_it != prim.attributes.end())
        uv_acc = &model.accessors[uv_it->second];

    // Read position data
    const auto* pos_data   = get_buffer_ptr<uint8_t>(model, pos_acc);
    int         pos_stride = get_stride(model, pos_acc, sizeof(float));

    // Read UV data
    const uint8_t* uv_data   = nullptr;
    int            uv_stride  = 0;
    if (uv_acc) {
        uv_data   = get_buffer_ptr<uint8_t>(model, *uv_acc);
        uv_stride = get_stride(model, *uv_acc, sizeof(float));
    }

    auto read_pos = [&](int idx) -> std::pair<double, double> {
        const float* p = reinterpret_cast<const float*>(pos_data + idx * pos_stride);
        // glTF uses right-handed Y-up: X=right, Y=up, Z=towards viewer
        // We project to 2D by taking X and Y (ignore Z for now)
        return {static_cast<double>(p[0]), static_cast<double>(p[1])};
    };

    auto read_uv = [&](int idx) -> std::pair<double, double> {
        if (!uv_data)
            return {0.0, 0.0};
        const float* p = reinterpret_cast<const float*>(uv_data + idx * uv_stride);
        return {static_cast<double>(p[0]), static_cast<double>(p[1])};
    };

    mesh_data result;

    if (prim.indices >= 0) {
        // Indexed geometry
        const auto& idx_acc = model.accessors[prim.indices];
        const auto& idx_bv  = model.bufferViews[idx_acc.bufferView];
        const auto* idx_raw = &model.buffers[idx_bv.buffer].data[idx_bv.byteOffset + idx_acc.byteOffset];

        auto get_index = [&](size_t i) -> int {
            switch (idx_acc.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    return static_cast<int>(reinterpret_cast<const uint16_t*>(idx_raw)[i]);
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                    return static_cast<int>(reinterpret_cast<const uint32_t*>(idx_raw)[i]);
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    return static_cast<int>(idx_raw[i]);
                default:
                    return 0;
            }
        };

        // glTF only allows TRIANGLES (4), TRIANGLE_STRIP (5), TRIANGLE_FAN (6)
        // We only support TRIANGLES for now
        if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1 && prim.mode != 4)
            CASPAR_THROW_EXCEPTION(
                invalid_argument() << msg_info("Only TRIANGLES mode is supported for glTF meshes: " + path));

        result.coords.reserve(idx_acc.count);
        for (size_t i = 0; i < idx_acc.count; ++i) {
            int         idx = get_index(i);
            if (idx < 0 || idx >= static_cast<int>(pos_acc.count))
                continue;
            auto [vx, vy]   = read_pos(idx);
            auto [tx, ty]   = read_uv(idx);
            frame_geometry::coord c;
            c.vertex_x  = vx;
            c.vertex_y  = vy;
            c.texture_x = tx;
            c.texture_y = ty;
            c.texture_r = 0.0;
            c.texture_q = 1.0;
            result.coords.push_back(c);
        }
    } else {
        // Non-indexed
        result.coords.reserve(pos_acc.count);
        for (size_t i = 0; i < pos_acc.count; ++i) {
            auto [vx, vy] = read_pos(static_cast<int>(i));
            auto [tx, ty] = read_uv(static_cast<int>(i));
            frame_geometry::coord c;
            c.vertex_x  = vx;
            c.vertex_y  = vy;
            c.texture_x = tx;
            c.texture_y = ty;
            c.texture_r = 0.0;
            c.texture_q = 1.0;
            result.coords.push_back(c);
        }
    }

    // Pad to multiple of 3 if needed (shouldn't be for TRIANGLES mode)
    while (result.coords.size() % 3 != 0)
        result.coords.pop_back();

    return result;
}

// ---- OBJ loader ------------------------------------------------------------

mesh_data load_obj(const std::string& path)
{
    tinyobj::ObjReader       reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(path, config))
        CASPAR_THROW_EXCEPTION(invalid_argument()
                               << msg_info("Failed to load OBJ: " +
                                           (reader.Error().empty() ? path : reader.Error())));

    if (!reader.Warning().empty())
        CASPAR_LOG(warning) << L"[mesh_loader] OBJ warning: " << u8(reader.Warning());

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    if (shapes.empty())
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("OBJ file contains no shapes: " + path));

    mesh_data result;

    // Use the first shape
    const auto& mesh = shapes[0].mesh;

    result.coords.reserve(mesh.indices.size());
    for (const auto& idx : mesh.indices) {
        frame_geometry::coord c;

        // Vertex position (OBJ is right-handed, Y-up like glTF)
        if (idx.vertex_index >= 0 &&
            static_cast<size_t>(3 * idx.vertex_index + 2) < attrib.vertices.size()) {
            c.vertex_x = static_cast<double>(attrib.vertices[3 * idx.vertex_index + 0]);
            c.vertex_y = static_cast<double>(attrib.vertices[3 * idx.vertex_index + 1]);
            // Z ignored for 2D projection
        }

        // Texture coordinate
        if (idx.texcoord_index >= 0 &&
            static_cast<size_t>(2 * idx.texcoord_index + 1) < attrib.texcoords.size()) {
            c.texture_x = static_cast<double>(attrib.texcoords[2 * idx.texcoord_index + 0]);
            c.texture_y = static_cast<double>(attrib.texcoords[2 * idx.texcoord_index + 1]);
        }

        c.texture_r = 0.0;
        c.texture_q = 1.0;
        result.coords.push_back(c);
    }

    // Pad to multiple of 3 if needed
    while (result.coords.size() % 3 != 0)
        result.coords.pop_back();

    return result;
}

} // anonymous namespace

// ---- public API ------------------------------------------------------------

frame_geometry load_mesh(const std::wstring& filename)
{
    auto path_str = u8(filename);

    if (!boost::filesystem::exists(path_str))
        CASPAR_THROW_EXCEPTION(file_not_found() << msg_info("Mesh file not found: " + path_str));

    auto ext = boost::filesystem::path(path_str).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    mesh_data data;

    if (ext == ".glb" || ext == ".gltf") {
        data = load_gltf(path_str);
    } else if (ext == ".obj") {
        data = load_obj(path_str);
    } else {
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("Unsupported mesh format (use .glb, .gltf, or .obj): " +
                                                              ext));
    }

    if (data.coords.empty() || data.coords.size() < 3)
        CASPAR_THROW_EXCEPTION(invalid_argument() << msg_info("Mesh contains no valid geometry: " + path_str));

    // Normalise positions to 0-1 range for the CasparCG screen-space pipeline
    normalise_positions(data.coords);

    CASPAR_LOG(info) << L"[mesh_loader] Loaded mesh: " << filename << L" (" << data.coords.size() / 3
                     << L" triangles)";

    return frame_geometry(frame_geometry::geometry_type::mesh, frame_geometry::scale_mode::stretch,
                          std::move(data.coords));
}

}} // namespace caspar::core
