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

#include "previz_renderer.h"

#include <core/stage/stage_math.h>

#include "../util/device.h"
#include "../util/shader.h"
#include "../util/texture.h"

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include <tiny_gltf.h>

#include <tiny_obj_loader.h>

#pragma warning(push)
#pragma warning(disable : 4838 4309)
#include "ogl_previz_vertex.h"
#include "ogl_previz_fragment.h"
#pragma warning(pop)

#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caspar { namespace accelerator { namespace ogl {

namespace {

// ---- Math helpers ----------------------------------------------------------

// `mat4` and `compute_frustum` used to live here, in this anonymous namespace. They are pure
// maths over plain structs and did no GL at all, and that placement made them untestable
// without a display -- so they now live in `core/stage/stage_math.h`, verbatim. The
// `using` below keeps every call site in this file spelled as it was.
using core::mat4;
using core::compute_frustum;

// ---- glTF loader -----------------------------------------------------------

// A validated window onto one glTF accessor's data.
struct gltf_view
{
    const uint8_t* raw    = nullptr;
    std::size_t    stride = 0;
    std::size_t    count  = 0;
};

// Accessor indices, bufferView indices, buffer indices and byte offsets all come
// straight from the file, so each one has to be range-checked before it reaches
// pointer arithmetic -- a truncated or crafted model otherwise reads outside the
// buffer. `bufferView` in particular defaults to -1 for an accessor that has
// none (legal glTF, e.g. a sparse accessor), which indexes backwards. This is the
// same bounds discipline load_obj_scene() below already applies.
//
// `min_elem_size` is the number of bytes the caller reads per element.
// `tightly_packed` is for index accessors, which glTF forbids byteStride on.
bool resolve_gltf_accessor(const tinygltf::Model& model,
                           int                    acc_index,
                           std::size_t            min_elem_size,
                           bool                   tightly_packed,
                           gltf_view&             out)
{
    if (acc_index < 0 || static_cast<std::size_t>(acc_index) >= model.accessors.size())
        return false;
    const auto& acc = model.accessors[acc_index];

    if (acc.bufferView < 0 || static_cast<std::size_t>(acc.bufferView) >= model.bufferViews.size())
        return false;
    const auto& bv = model.bufferViews[acc.bufferView];

    if (bv.buffer < 0 || static_cast<std::size_t>(bv.buffer) >= model.buffers.size())
        return false;
    const auto& data = model.buffers[bv.buffer].data;

    const auto comp  = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(acc.componentType));
    const auto ncomp = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(acc.type));
    if (comp <= 0 || ncomp <= 0 || acc.count == 0)
        return false;
    const std::size_t elem = static_cast<std::size_t>(comp) * static_cast<std::size_t>(ncomp);
    if (elem < min_elem_size)
        return false;

    std::size_t stride = elem;
    if (!tightly_packed && bv.byteStride > 0)
        stride = bv.byteStride;
    if (stride < elem)
        return false;

    // begin + (count - 1) * stride + elem must fit in the buffer, computed so
    // that neither the offset sum nor the extent can wrap.
    const std::size_t begin = bv.byteOffset + acc.byteOffset;
    if (begin < bv.byteOffset || begin > data.size())
        return false;
    const std::size_t avail = data.size() - begin;
    if (avail < elem || (avail - elem) / stride < acc.count - 1)
        return false;

    // The accessor also has to stay inside its own buffer view.
    if (bv.byteLength > 0) {
        if (acc.byteOffset > bv.byteLength)
            return false;
        const std::size_t bv_avail = bv.byteLength - acc.byteOffset;
        if (bv_avail < elem || (bv_avail - elem) / stride < acc.count - 1)
            return false;
    }

    out.raw    = data.data() + begin;
    out.stride = stride;
    out.count  = acc.count;
    return true;
}

void load_gltf_scene(const std::string& path, previz_scene& scene)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    bool ok = false;
    auto ext = boost::filesystem::path(path).extension().string();
    if (ext == ".glb")
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    else
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty())
        CASPAR_LOG(warning) << L"[previz] glTF warning: " << u8(warn);
    if (!ok)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to load glTF: " + err));

    scene.meshes.clear();

    for (const auto& mesh : model.meshes) {
        for (size_t pi = 0; pi < mesh.primitives.size(); ++pi) {
            const auto& prim = mesh.primitives[pi];

            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end())
                continue;

            gltf_view pos_v;
            if (!resolve_gltf_accessor(model, pos_it->second, 3 * sizeof(float), false, pos_v)) {
                CASPAR_LOG(warning) << L"[previz] glTF: primitive skipped, POSITION accessor out of range.";
                continue;
            }

            // Normals
            gltf_view  nrm_v;
            auto       nrm_it   = prim.attributes.find("NORMAL");
            const bool have_nrm = nrm_it != prim.attributes.end() &&
                                  resolve_gltf_accessor(model, nrm_it->second, 3 * sizeof(float), false, nrm_v);

            // UVs
            gltf_view  uv_v;
            auto       uv_it   = prim.attributes.find("TEXCOORD_0");
            const bool have_uv = uv_it != prim.attributes.end() &&
                                 resolve_gltf_accessor(model, uv_it->second, 2 * sizeof(float), false, uv_v);

            // Each attribute is bounded by its own count: an index validated against
            // POSITION says nothing about a shorter NORMAL/TEXCOORD_0 accessor.
            auto read_pos = [&](std::size_t i) -> std::array<float, 3> {
                float p[3];
                std::memcpy(p, pos_v.raw + i * pos_v.stride, sizeof(p));
                return {p[0], p[1], p[2]};
            };
            auto read_nrm = [&](std::size_t i) -> std::array<float, 3> {
                if (!have_nrm || i >= nrm_v.count)
                    return {0.0f, 1.0f, 0.0f};
                float p[3];
                std::memcpy(p, nrm_v.raw + i * nrm_v.stride, sizeof(p));
                return {p[0], p[1], p[2]};
            };
            auto read_uv = [&](std::size_t i) -> std::array<float, 2> {
                if (!have_uv || i >= uv_v.count)
                    return {0.0f, 0.0f};
                float p[2];
                std::memcpy(p, uv_v.raw + i * uv_v.stride, sizeof(p));
                return {p[0], p[1]};
            };

            previz_mesh pm;
            pm.name = mesh.name;
            if (mesh.primitives.size() > 1)
                pm.name += "_" + std::to_string(pi);

            // Read material base color
            if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size())) {
                const auto& mat = model.materials[prim.material];
                const auto& pbr = mat.pbrMetallicRoughness;
                pm.base_color   = {static_cast<float>(pbr.baseColorFactor[0]),
                                   static_cast<float>(pbr.baseColorFactor[1]),
                                   static_cast<float>(pbr.baseColorFactor[2])};
            }

            if (prim.indices >= 0) {
                gltf_view idx_v;
                if (!resolve_gltf_accessor(model, prim.indices, 1, /*tightly_packed=*/true, idx_v)) {
                    CASPAR_LOG(warning) << L"[previz] glTF: primitive skipped, index accessor out of range.";
                    continue;
                }
                const int comp_type = model.accessors[prim.indices].componentType;

                auto get_index = [&](std::size_t i) -> long long {
                    const uint8_t* p = idx_v.raw + i * idx_v.stride;
                    switch (comp_type) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                            uint16_t v;
                            std::memcpy(&v, p, sizeof(v));
                            return static_cast<long long>(v);
                        }
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                            uint32_t v;
                            std::memcpy(&v, p, sizeof(v));
                            return static_cast<long long>(v);
                        }
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            return static_cast<long long>(*p);
                        default:
                            return -1;
                    }
                };

                // idx_v.count is now bounded by the real buffer size, so this
                // reserve can no longer be driven to an absurd allocation.
                pm.vertices.reserve(idx_v.count);
                for (std::size_t i = 0; i < idx_v.count; ++i) {
                    const long long idx = get_index(i);
                    if (idx < 0 || static_cast<std::size_t>(idx) >= pos_v.count)
                        continue;
                    auto [px, py, pz] = read_pos(static_cast<std::size_t>(idx));
                    auto [nx, ny, nz] = read_nrm(static_cast<std::size_t>(idx));
                    auto [tu, tv]     = read_uv(static_cast<std::size_t>(idx));
                    pm.vertices.push_back({px, py, pz, nx, ny, nz, tu, tv});
                }
            } else {
                pm.vertices.reserve(pos_v.count);
                for (std::size_t i = 0; i < pos_v.count; ++i) {
                    auto [px, py, pz] = read_pos(i);
                    auto [nx, ny, nz] = read_nrm(i);
                    auto [tu, tv]     = read_uv(i);
                    pm.vertices.push_back({px, py, pz, nx, ny, nz, tu, tv});
                }
            }

            scene.meshes.push_back(std::move(pm));
        }
    }

    CASPAR_LOG(info) << L"[previz] Loaded scene: " << u8(path) << L" (" << scene.meshes.size() << L" meshes)";
}

// ---- OBJ loader -----------------------------------------------------------

void load_obj_scene(const std::string& path, previz_scene& scene)
{
    tinyobj::ObjReader       reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(path, config))
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Failed to load OBJ: " + reader.Error()));

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    scene.meshes.clear();

    for (const auto& shape : shapes) {
        previz_mesh pm;
        pm.name = shape.name;

        pm.vertices.reserve(shape.mesh.indices.size());
        for (const auto& idx : shape.mesh.indices) {
            previz_vertex vtx;
            if (idx.vertex_index >= 0 &&
                static_cast<size_t>(3 * idx.vertex_index + 2) < attrib.vertices.size()) {
                vtx.px = static_cast<float>(attrib.vertices[3 * idx.vertex_index + 0]);
                vtx.py = static_cast<float>(attrib.vertices[3 * idx.vertex_index + 1]);
                vtx.pz = static_cast<float>(attrib.vertices[3 * idx.vertex_index + 2]);
            }
            if (idx.normal_index >= 0 &&
                static_cast<size_t>(3 * idx.normal_index + 2) < attrib.normals.size()) {
                vtx.nx = static_cast<float>(attrib.normals[3 * idx.normal_index + 0]);
                vtx.ny = static_cast<float>(attrib.normals[3 * idx.normal_index + 1]);
                vtx.nz = static_cast<float>(attrib.normals[3 * idx.normal_index + 2]);
            }
            if (idx.texcoord_index >= 0 &&
                static_cast<size_t>(2 * idx.texcoord_index + 1) < attrib.texcoords.size()) {
                vtx.u = static_cast<float>(attrib.texcoords[2 * idx.texcoord_index + 0]);
                vtx.v = static_cast<float>(attrib.texcoords[2 * idx.texcoord_index + 1]);
            }
            pm.vertices.push_back(vtx);
        }

        scene.meshes.push_back(std::move(pm));
    }

    CASPAR_LOG(info) << L"[previz] Loaded OBJ: " << u8(path) << L" (" << scene.meshes.size() << L" shapes)";
}

} // anonymous namespace

// ---- previz_renderer implementation ----------------------------------------

struct previz_renderer::impl
{
    spl::shared_ptr<device>     ogl_;
    previz_scene                scene_;
    std::shared_ptr<shader>     shader_;
    GLuint                      fbo_       = 0;
    GLuint                      depth_rbo_ = 0;
    int                         fbo_w_     = 0;
    int                         fbo_h_     = 0;
    GLuint                      grid_vao_  = 0;
    GLuint                      grid_vbo_  = 0;
    int                         grid_vert_count_ = 0;
    mutable std::mutex          scene_mutex_;
    projection_apply_fn         projection_fn_;

    explicit impl(const spl::shared_ptr<device>& ogl)
        : ogl_(ogl)
    {
    }

    ~impl()
    {
        ogl_->dispatch_sync([this] {
            if (fbo_)
                glDeleteFramebuffers(1, &fbo_);
            if (depth_rbo_)
                glDeleteRenderbuffers(1, &depth_rbo_);
            if (grid_vao_) {
                glDeleteVertexArrays(1, &grid_vao_);
                glDeleteBuffers(1, &grid_vbo_);
            }
            for (auto& mesh : scene_.meshes) {
                if (mesh.vao) {
                    glDeleteVertexArrays(1, &mesh.vao);
                    mesh.vao = 0;
                }
                if (mesh.vbo) {
                    glDeleteBuffers(1, &mesh.vbo);
                    mesh.vbo = 0;
                }
            }
        });
    }

    void ensure_shader()
    {
        if (shader_)
            return;
        shader_ = std::make_shared<shader>(std::string(reinterpret_cast<const char*>(previz_vertex_shader)), std::string(reinterpret_cast<const char*>(previz_fragment_shader)));
    }

    void ensure_fbo(int w, int h)
    {
        if (fbo_ && fbo_w_ == w && fbo_h_ == h)
            return;

        if (fbo_)
            glDeleteFramebuffers(1, &fbo_);
        if (depth_rbo_)
            glDeleteRenderbuffers(1, &depth_rbo_);

        glGenFramebuffers(1, &fbo_);
        glGenRenderbuffers(1, &depth_rbo_);

        glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        fbo_w_ = w;
        fbo_h_ = h;
    }

    void ensure_grid()
    {
        if (grid_vao_)
            return;

        // Generate a ground-plane grid at Y=0, spanning -20..+20 metres in X and Z
        const int   half  = 20;
        std::vector<previz_vertex> verts;

        for (int i = -half; i <= half; ++i) {
            float fi = static_cast<float>(i);
            // Line along Z (constant X)
            verts.push_back({fi, 0.0f, static_cast<float>(-half), 0, 1, 0, 0, 0});
            verts.push_back({fi, 0.0f, static_cast<float>(half),  0, 1, 0, 0, 0});
            // Line along X (constant Z)
            verts.push_back({static_cast<float>(-half), 0.0f, fi, 0, 1, 0, 0, 0});
            verts.push_back({static_cast<float>(half),  0.0f, fi, 0, 1, 0, 0, 0});
        }

        grid_vert_count_ = static_cast<int>(verts.size());

        glGenVertexArrays(1, &grid_vao_);
        glGenBuffers(1, &grid_vbo_);

        glBindVertexArray(grid_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, grid_vbo_);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(verts.size() * sizeof(previz_vertex)),
                     verts.data(), GL_STATIC_DRAW);

        auto stride = static_cast<GLsizei>(sizeof(previz_vertex));
        auto pos_loc = shader_->get_attrib_location("a_Position");
        glEnableVertexAttribArray(pos_loc);
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, stride, nullptr);

        auto nrm_loc = shader_->get_attrib_location("a_Normal");
        glEnableVertexAttribArray(nrm_loc);
        glVertexAttribPointer(nrm_loc, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));

        auto uv_loc = shader_->get_attrib_location("a_TexCoord");
        glEnableVertexAttribArray(uv_loc);
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(6 * sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void upload_mesh(previz_mesh& mesh)
    {
        if (!mesh.gpu_dirty)
            return;

        if (!mesh.vao) {
            glGenVertexArrays(1, &mesh.vao);
            glGenBuffers(1, &mesh.vbo);
        }

        glBindVertexArray(mesh.vao);
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(previz_vertex)),
                     mesh.vertices.data(),
                     GL_STATIC_DRAW);

        auto stride = static_cast<GLsizei>(sizeof(previz_vertex));

        // a_Position (vec3) at offset 0
        auto pos_loc = shader_->get_attrib_location("a_Position");
        glEnableVertexAttribArray(pos_loc);
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, stride, nullptr);

        // a_Normal (vec3) at offset 12
        auto nrm_loc = shader_->get_attrib_location("a_Normal");
        glEnableVertexAttribArray(nrm_loc);
        glVertexAttribPointer(nrm_loc, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));

        // a_TexCoord (vec2) at offset 24
        auto uv_loc = shader_->get_attrib_location("a_TexCoord");
        glEnableVertexAttribArray(uv_loc);
        glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(6 * sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        mesh.gpu_dirty = false;
    }

    void render(std::shared_ptr<texture>& target, const channel_texture_store& tex_store, int w, int h)
    {
        if (!scene_.active || scene_.meshes.empty())
            return;

        ensure_shader();
        ensure_fbo(w, h);

        shader_->use();

        // Choose the viewport POV: the navigation/view camera when an override is
        // active, otherwise the production virtual camera.  This NEVER affects
        // compute_frustum(), which always uses scene_.camera.
        const previz_camera& view_cam =
            scene_.has_view_override ? scene_.view_camera : scene_.camera;

        // Build view matrix (camera)
        auto view = mat4::rotate_z(-view_cam.roll) * mat4::rotate_x(-view_cam.pitch) *
                    mat4::rotate_y(-view_cam.yaw) *
                    mat4::translate(-view_cam.x, -view_cam.y, -view_cam.z);

        float aspect = static_cast<float>(w) / static_cast<float>(h);
        auto  proj   = mat4::perspective(view_cam.fov, aspect, view_cam.near_clip, view_cam.far_clip);

        // Flip Y so the FBO output matches CasparCG's top-down texture convention.
        // OpenGL FBOs render bottom-up; the downstream copy_async / screen consumer
        // expects origin at the top-left.
        proj.m[5] = -proj.m[5];

        auto model = mat4::identity();
        auto mvp   = proj * view * model;

        // Set uniforms
        shader_->set("u_light_dir", 0.32934, 0.76847, 0.54891); // normalized (0.3, 0.7, 0.5)
        shader_->set("u_ambient", 0.25);

        // Save the currently-bound FBO so we can restore it when done.
        // The OGL device keeps its own FBO permanently bound on the GL thread;
        // binding FBO 0 instead would break subsequent glFramebufferTexture2D
        // calls made by the image_kernel through the GL() macro.
        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

        // Bind FBO with depth buffer, attach target texture as color
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target->id(), 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_);

        glViewport(0, 0, w, h);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (scene_.show_wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // Draw ground-plane grid
        if (scene_.show_grid) {
            ensure_grid();
            glUniformMatrix4fv(glGetUniformLocation(shader_->id(), "u_mvp"), 1, GL_FALSE, mvp.m);
            glUniformMatrix4fv(glGetUniformLocation(shader_->id(), "u_model"), 1, GL_FALSE, model.m);
            shader_->set("u_is_screen", false);
            shader_->set("u_has_texture", false);
            shader_->set("u_base_color", 0.35, 0.35, 0.35);
            glBindVertexArray(grid_vao_);
            glDrawArrays(GL_LINES, 0, grid_vert_count_);
            glBindVertexArray(0);
        }

        // Render each mesh
        for (auto& mesh : scene_.meshes) {
            if (!mesh.visible)
                continue;

            upload_mesh(mesh);

            if (mesh.vertices.empty())
                continue;

            // Set MVP (all meshes use identity model matrix for now)
            glUniformMatrix4fv(glGetUniformLocation(shader_->id(), "u_mvp"), 1, GL_FALSE, mvp.m);
            glUniformMatrix4fv(glGetUniformLocation(shader_->id(), "u_model"), 1, GL_FALSE, model.m);

            // Determine if this mesh is a screen
            bool is_screen = mesh.is_screen;
            shader_->set("u_is_screen", is_screen);

            // Bind channel texture if mapped.
            // Keep tex_entry alive until after glDrawArrays so the shared_ptr
            // prevents the texture pool from recycling the GL texture.
            auto map_it = scene_.mesh_to_channel.find(mesh.name);
            bool has_tex = false;
            channel_texture_store::entry tex_entry;
            if (is_screen && map_it != scene_.mesh_to_channel.end()) {
                tex_entry = tex_store.get(map_it->second);
                if (tex_entry.tex_id != 0) {
                    // Drain any pending GL errors before binding the channel
                    // texture.  The CUDA-GL interop path (cuda_prores) may
                    // leave stale errors when the texture was recently
                    // unmapped from CUDA.
                    while (glGetError() != GL_NO_ERROR) {}

                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, tex_entry.tex_id);

                    // Verify the bind succeeded — CUDA interop textures can
                    // transiently fail if the decode thread hasn't finished
                    // unmapping.
                    if (glGetError() == GL_NO_ERROR) {
                        shader_->set("u_texture", 0);
                        has_tex = true;
                    } else {
                        // Bind failed — unbind and fall through to placeholder
                        glBindTexture(GL_TEXTURE_2D, 0);
                        while (glGetError() != GL_NO_ERROR) {}
                    }
                }
            }
            shader_->set("u_has_texture", has_tex);

            if (!is_screen) {
                shader_->set("u_base_color", static_cast<double>(mesh.base_color[0]),
                             static_cast<double>(mesh.base_color[1]), static_cast<double>(mesh.base_color[2]));
            }

            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(mesh.vertices.size()));
            glBindVertexArray(0);

            if (has_tex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, 0);
            }
        }

        if (scene_.show_wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

        // Drain any pending GL errors so they don't leak to the next
        // channel's rendering pass (which uses the GL() error-checking macro).
        while (glGetError() != GL_NO_ERROR) {}
    }

    void load_scene(const std::string& path)
    {
        // Clean up old GPU resources
        for (auto& mesh : scene_.meshes) {
            if (mesh.vao) {
                glDeleteVertexArrays(1, &mesh.vao);
                mesh.vao = 0;
            }
            if (mesh.vbo) {
                glDeleteBuffers(1, &mesh.vbo);
                mesh.vbo = 0;
            }
        }

        // Empty path = clear the scene
        if (path.empty()) {
            scene_.meshes.clear();
            scene_.mesh_to_channel.clear();
            scene_.screens.clear();
            scene_.scene_path.clear();
            scene_.active = false;
            CASPAR_LOG(info) << L"[previz] Scene cleared";
            return;
        }

        auto ext = boost::filesystem::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".glb" || ext == ".gltf")
            load_gltf_scene(path, scene_);
        else if (ext == ".obj")
            load_obj_scene(path, scene_);
        else
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Unsupported format: " + ext));

        if (scene_.meshes.size() > 500)
            CASPAR_LOG(warning) << L"[previz] Scene contains " << scene_.meshes.size()
                                << L" meshes \u2014 performance may be impacted";

        scene_.scene_path = path;
        scene_.active     = true;
    }

    void map_mesh(const std::string& name, int channel)
    {
        for (auto& mesh : scene_.meshes) {
            if (mesh.name == name) {
                mesh.is_screen = true;
                scene_.mesh_to_channel[name] = channel;
                CASPAR_LOG(info) << L"[previz] Mapped mesh \"" << u8(name) << L"\" to channel " << channel;
                return;
            }
        }
        CASPAR_LOG(warning) << L"[previz] Mesh not found: " << u8(name);
    }

    void unmap_mesh(const std::string& name)
    {
        scene_.mesh_to_channel.erase(name);
        for (auto& mesh : scene_.meshes) {
            if (mesh.name == name)
                mesh.is_screen = false;
        }
    }
};

// ---- public API ------------------------------------------------------------

previz_renderer::previz_renderer(const spl::shared_ptr<device>& ogl)
    : impl_(new impl(ogl))
{
}

previz_renderer::~previz_renderer() = default;

void previz_renderer::load_scene(const std::string& path)
{
    impl_->ogl_->dispatch_sync([this, &path] {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        impl_->load_scene(path);
    });
}

void previz_renderer::map_mesh(const std::string& mesh_name, int channel_index)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->map_mesh(mesh_name, channel_index);
}

void previz_renderer::unmap_mesh(const std::string& mesh_name)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->unmap_mesh(mesh_name);
}

void previz_renderer::set_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        impl_->scene_.camera = {x, y, z, yaw, pitch, roll, fov, 0.1f, 100.0f};
    }
    update_projections();
}

void previz_renderer::reset_camera()
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        impl_->scene_.camera = previz_camera{};
    }
    update_projections();
}

void previz_renderer::set_view_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.view_camera       = {x, y, z, yaw, pitch, roll, fov, 0.1f, 100.0f};
    impl_->scene_.has_view_override = true;
    // No update_projections(): the view camera never drives projection.
}

void previz_renderer::clear_view_camera()
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.has_view_override = false;
}

void previz_renderer::set_camera_locked(bool locked)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.camera_locked = locked;
}

bool previz_renderer::is_camera_locked() const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    return impl_->scene_.camera_locked;
}

void previz_renderer::save_layout(const std::string& path) const
{
    namespace pt = boost::property_tree;
    pt::ptree root;

    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        const auto& sc = impl_->scene_;

        auto write_cam = [](const previz_camera& c) {
            pt::ptree n;
            n.put("x", c.x);     n.put("y", c.y);     n.put("z", c.z);
            n.put("yaw", c.yaw); n.put("pitch", c.pitch); n.put("roll", c.roll);
            n.put("fov", c.fov);
            return n;
        };

        root.put("version", 1);
        root.put_child("camera", write_cam(sc.camera));
        root.put_child("view_camera", write_cam(sc.view_camera));
        root.put("has_view_override", sc.has_view_override);
        root.put("camera_locked", sc.camera_locked);
        root.put("auto_projection", sc.auto_projection);
        if (!sc.scene_path.empty())
            root.put("model_path", sc.scene_path);

        pt::ptree screens;
        for (const auto& [name, m] : sc.screens) {
            pt::ptree s;
            s.put("name", m.name);
            s.put("width_m", m.width_m);
            s.put("height_m", m.height_m);
            s.put("radius_m", m.radius_m);
            s.put("arc_deg", m.arc_deg);
            s.put("arc_v_deg", m.arc_v_deg);
            s.put("pos_x", m.pos_x); s.put("pos_y", m.pos_y); s.put("pos_z", m.pos_z);
            s.put("rot_yaw", m.rot_yaw); s.put("rot_pitch", m.rot_pitch); s.put("rot_roll", m.rot_roll);
            s.put("res_w", m.res_w); s.put("res_h", m.res_h);
            s.put("channel", m.channel);
            s.put("eye_mode", m.eye_mode);
            s.put("design_eye_x", m.design_eye_x);
            s.put("design_eye_y", m.design_eye_y);
            s.put("design_eye_z", m.design_eye_z);
            s.put("icvfx_enable", m.icvfx_enable);
            screens.push_back(std::make_pair("", s));
        }
        root.put_child("screens", screens);
    }

    pt::write_json(path, root);
    CASPAR_LOG(info) << L"[previz] Saved stage layout to " << u16(path);
}

void previz_renderer::load_layout(const std::string& path)
{
    namespace pt = boost::property_tree;
    pt::ptree root;
    pt::read_json(path, root);

    auto read_cam = [](const pt::ptree& n) {
        previz_camera c;
        c.x   = n.get("x", 0.0f);   c.y = n.get("y", 1.5f);   c.z = n.get("z", 5.0f);
        c.yaw = n.get("yaw", 0.0f); c.pitch = n.get("pitch", 0.0f); c.roll = n.get("roll", 0.0f);
        c.fov = n.get("fov", 60.0f);
        return c;
    };

    // Optionally (re)load the venue model first (outside the scene lock).
    auto model_path = root.get<std::string>("model_path", "");
    if (!model_path.empty() && boost::filesystem::exists(model_path)) {
        try { load_scene(model_path); } catch (...) {}
    }

    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto& sc = impl_->scene_;

        if (auto cam = root.get_child_optional("camera"))
            sc.camera = read_cam(*cam);
        if (auto vc = root.get_child_optional("view_camera"))
            sc.view_camera = read_cam(*vc);
        sc.has_view_override = root.get("has_view_override", false);
        sc.camera_locked     = root.get("camera_locked", false);
        sc.auto_projection   = root.get("auto_projection", sc.auto_projection);

        sc.screens.clear();
        if (auto screens = root.get_child_optional("screens")) {
            for (const auto& [_, s] : *screens) {
                screen_meta m;
                m.name      = s.get("name", std::string());
                if (m.name.empty())
                    continue;
                m.width_m   = s.get("width_m", 1.0f);
                m.height_m  = s.get("height_m", 1.0f);
                m.radius_m  = s.get("radius_m", 0.0f);
                m.arc_deg   = s.get("arc_deg", 0.0f);
                m.arc_v_deg = s.get("arc_v_deg", 0.0f);
                m.pos_x     = s.get("pos_x", 0.0f);
                m.pos_y     = s.get("pos_y", 0.0f);
                m.pos_z     = s.get("pos_z", 0.0f);
                m.rot_yaw   = s.get("rot_yaw", 0.0f);
                m.rot_pitch = s.get("rot_pitch", 0.0f);
                m.rot_roll  = s.get("rot_roll", 0.0f);
                m.res_w     = s.get("res_w", 0);
                m.res_h     = s.get("res_h", 0);
                m.channel   = s.get("channel", -1);
                m.eye_mode  = s.get("eye_mode", 0);
                m.design_eye_x = s.get("design_eye_x", 0.0f);
                m.design_eye_y = s.get("design_eye_y", 1.5f);
                m.design_eye_z = s.get("design_eye_z", 3.0f);
                m.icvfx_enable = s.get("icvfx_enable", false);
                sc.screens[m.name] = m;
            }
        }
    }

    // Rebuild meshes + channel mappings for every restored screen.
    {
        std::map<std::string, screen_meta> restored;
        {
            std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
            restored = impl_->scene_.screens;
        }
        for (const auto& [name, m] : restored) {
            if (m.radius_m > 1e-4f && m.arc_deg > 1e-4f)
                add_screen_curved(name, m.width_m, m.height_m, m.radius_m, m.arc_deg);
            else
                add_screen_flat(name, m.width_m, m.height_m);
            set_screen_position(name, m.pos_x, m.pos_y, m.pos_z);
            set_screen_rotation(name, m.rot_yaw, m.rot_pitch, m.rot_roll);
            if (m.res_w > 0 && m.res_h > 0)
                set_screen_resolution(name, m.res_w, m.res_h);
            set_screen_arc_v(name, m.arc_v_deg);
            set_screen_eye_mode(name, m.eye_mode, m.design_eye_x, m.design_eye_y, m.design_eye_z);
            set_screen_icvfx(name, m.icvfx_enable);
            if (m.channel >= 0)
                set_screen_channel(name, m.channel);
        }
    }
    update_projections();
    CASPAR_LOG(info) << L"[previz] Loaded stage layout from " << u16(path);
}

void previz_renderer::set_mesh_visible(const std::string& mesh_name, bool visible)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    for (auto& mesh : impl_->scene_.meshes) {
        if (mesh.name == mesh_name) {
            mesh.visible = visible;
            return;
        }
    }
}

void previz_renderer::set_grid(bool on)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.show_grid = on;
}

void previz_renderer::set_wireframe(bool on)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.show_wireframe = on;
}

void previz_renderer::set_gizmo(bool on)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.show_gizmo = on;
}

void previz_renderer::save_camera_preset(const std::string& name)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->scene_.camera_presets[name] = impl_->scene_.camera;
    CASPAR_LOG(info) << L"[previz] Saved camera preset: " << u8(name);
}

void previz_renderer::recall_camera_preset(const std::string& name)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    auto it = impl_->scene_.camera_presets.find(name);
    if (it != impl_->scene_.camera_presets.end()) {
        impl_->scene_.camera = it->second;
        CASPAR_LOG(info) << L"[previz] Recalled camera preset: " << u8(name);
    } else {
        CASPAR_LOG(warning) << L"[previz] Camera preset not found: " << u8(name);
    }
}

std::vector<std::string> previz_renderer::list_camera_presets() const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    std::vector<std::string> names;
    for (const auto& [k, v] : impl_->scene_.camera_presets)
        names.push_back(k);
    return names;
}

// ---- Procedural screen helpers -----------------------------------------------

namespace {

previz_mesh generate_flat_screen(const std::string& name, float w, float h)
{
    previz_mesh m;
    m.name      = name;
    m.is_screen = true;
    m.base_color = {0.0f, 0.0f, 0.0f}; // screens are black when no texture

    float hw = w * 0.5f;

    // Two triangles, normal facing +Z, origin at center-bottom
    m.vertices = {
        {-hw, 0, 0,  0, 0, 1,  0, 1},
        { hw, 0, 0,  0, 0, 1,  1, 1},
        { hw, h, 0,  0, 0, 1,  1, 0},
        {-hw, 0, 0,  0, 0, 1,  0, 1},
        { hw, h, 0,  0, 0, 1,  1, 0},
        {-hw, h, 0,  0, 0, 1,  0, 0},
    };
    m.gpu_dirty = true;
    return m;
}

previz_mesh generate_curved_screen(const std::string& name, float w, float h, float /*radius_hint*/, float arc_deg)
{
    previz_mesh m;
    m.name      = name;
    m.is_screen = true;
    m.base_color = {0.0f, 0.0f, 0.0f};

    const int   segments = std::max(8, static_cast<int>(arc_deg / 2.0f));
    const float arc_rad  = arc_deg * static_cast<float>(M_PI) / 180.0f;
    const float half_arc = arc_rad * 0.5f;

    // Derive radius from width and arc so chord width matches the specified width
    const float radius = (std::abs(std::sin(half_arc)) > 1e-6f)
                        ? (w * 0.5f / std::sin(half_arc))
                        : (w * 100.0f); // near-zero arc: approximate as very large radius

    m.vertices.reserve(segments * 6);

    for (int i = 0; i < segments; ++i) {
        float t0 = static_cast<float>(i)     / static_cast<float>(segments);
        float t1 = static_cast<float>(i + 1) / static_cast<float>(segments);

        float a0 = -half_arc + t0 * arc_rad;
        float a1 = -half_arc + t1 * arc_rad;

        float x0 = radius * std::sin(a0);
        float z0 = radius * std::cos(a0) - radius; // center of curvature behind screen
        float x1 = radius * std::sin(a1);
        float z1 = radius * std::cos(a1) - radius;

        // Outward-facing normals (away from center of curvature)
        float nx0 = std::sin(a0), nz0 = std::cos(a0);
        float nx1 = std::sin(a1), nz1 = std::cos(a1);

        float u0 = t0, u1 = t1;

        // Bottom-left, bottom-right, top-right
        m.vertices.push_back({x0, 0, z0, nx0, 0, nz0, u0, 1});
        m.vertices.push_back({x1, 0, z1, nx1, 0, nz1, u1, 1});
        m.vertices.push_back({x1, h, z1, nx1, 0, nz1, u1, 0});
        // Bottom-left, top-right, top-left
        m.vertices.push_back({x0, 0, z0, nx0, 0, nz0, u0, 1});
        m.vertices.push_back({x1, h, z1, nx1, 0, nz1, u1, 0});
        m.vertices.push_back({x0, h, z0, nx0, 0, nz0, u0, 0});
    }

    m.gpu_dirty = true;
    return m;
}

void apply_screen_transform(previz_mesh& mesh, const screen_meta& meta)
{
    // Re-generate the base mesh then apply rotation + translation
    previz_mesh base;
    if (meta.radius_m > 0.0f)
        base = generate_curved_screen(meta.name, meta.width_m, meta.height_m, meta.radius_m, meta.arc_deg);
    else
        base = generate_flat_screen(meta.name, meta.width_m, meta.height_m);

    // Build rotation matrix (yaw * pitch * roll)
    auto ry = mat4::rotate_y(meta.rot_yaw);
    auto rx = mat4::rotate_x(meta.rot_pitch);
    auto rz = mat4::rotate_z(meta.rot_roll);
    auto rot = ry * rx * rz;

    mesh.vertices.resize(base.vertices.size());
    for (size_t i = 0; i < base.vertices.size(); ++i) {
        auto& sv = base.vertices[i];
        auto& dv = mesh.vertices[i];

        // Rotate position
        float px = rot.m[0]*sv.px + rot.m[4]*sv.py + rot.m[8]*sv.pz;
        float py = rot.m[1]*sv.px + rot.m[5]*sv.py + rot.m[9]*sv.pz;
        float pz = rot.m[2]*sv.px + rot.m[6]*sv.py + rot.m[10]*sv.pz;
        dv.px = px + meta.pos_x;
        dv.py = py + meta.pos_y;
        dv.pz = pz + meta.pos_z;

        // Rotate normal
        dv.nx = rot.m[0]*sv.nx + rot.m[4]*sv.ny + rot.m[8]*sv.nz;
        dv.ny = rot.m[1]*sv.nx + rot.m[5]*sv.ny + rot.m[9]*sv.nz;
        dv.nz = rot.m[2]*sv.nx + rot.m[6]*sv.ny + rot.m[10]*sv.nz;

        dv.u = sv.u;
        dv.v = sv.v;
    }
    mesh.gpu_dirty = true;
}


} // anonymous namespace

void previz_renderer::add_screen_flat(const std::string& name, float width_m, float height_m)
{
    unsigned int old_vao = 0, old_vbo = 0;
    {
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);

    // Remove any existing mesh with the same name (prevents orphaned duplicates)
    auto& meshes = impl_->scene_.meshes;
    for (auto& m : meshes) {
        if (m.name == name) { old_vao = m.vao; old_vbo = m.vbo; m.vao = 0; m.vbo = 0; break; }
    }
    meshes.erase(std::remove_if(meshes.begin(), meshes.end(),
                                [&](const previz_mesh& m) { return m.name == name; }),
                 meshes.end());

    screen_meta sm;
    sm.name     = name;
    sm.width_m  = width_m;
    sm.height_m = height_m;
    impl_->scene_.screens[name] = sm;

    auto mesh = generate_flat_screen(name, width_m, height_m);
    impl_->scene_.meshes.push_back(std::move(mesh));
    impl_->scene_.active = true;
    } // unlock scene_mutex_

    // Free old GL resources (no lock held — no deadlock)
    if (old_vao || old_vbo) {
        impl_->ogl_->dispatch_async([old_vao, old_vbo] {
            if (old_vao) { auto v = old_vao; glDeleteVertexArrays(1, &v); }
            if (old_vbo) { auto v = old_vbo; glDeleteBuffers(1, &v); }
        });
    }

    CASPAR_LOG(info) << L"[previz] Added flat screen: " << u8(name)
                     << L" (" << width_m << L"x" << height_m << L"m)";
}

void previz_renderer::add_screen_curved(const std::string& name, float width_m, float height_m,
                                         float radius_m, float arc_deg)
{
    unsigned int old_vao = 0, old_vbo = 0;
    float saved_radius = 0.0f;
    {
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);

    // Remove any existing mesh with the same name (prevents orphaned duplicates)
    auto& meshes = impl_->scene_.meshes;
    for (auto& m : meshes) {
        if (m.name == name) { old_vao = m.vao; old_vbo = m.vbo; m.vao = 0; m.vbo = 0; break; }
    }
    meshes.erase(std::remove_if(meshes.begin(), meshes.end(),
                                [&](const previz_mesh& m) { return m.name == name; }),
                 meshes.end());

    screen_meta sm;
    sm.name      = name;
    sm.width_m   = width_m;
    sm.height_m  = height_m;
    sm.arc_deg   = arc_deg;

    // Derive actual radius from width and arc
    float arc_rad  = arc_deg * static_cast<float>(M_PI) / 180.0f;
    float half_arc = arc_rad * 0.5f;
    sm.radius_m = (std::abs(std::sin(half_arc)) > 1e-6f)
                ? (width_m * 0.5f / std::sin(half_arc))
                : radius_m;
    impl_->scene_.screens[name] = sm;

    auto mesh = generate_curved_screen(name, width_m, height_m, sm.radius_m, arc_deg);
    impl_->scene_.meshes.push_back(std::move(mesh));
    impl_->scene_.active = true;
    saved_radius = sm.radius_m;
    } // unlock scene_mutex_

    // Free old GL resources (no lock held — no deadlock)
    if (old_vao || old_vbo) {
        impl_->ogl_->dispatch_async([old_vao, old_vbo] {
            if (old_vao) { auto v = old_vao; glDeleteVertexArrays(1, &v); }
            if (old_vbo) { auto v = old_vbo; glDeleteBuffers(1, &v); }
        });
    }

    CASPAR_LOG(info) << L"[previz] Added curved screen: " << u8(name)
                     << L" (" << width_m << L"x" << height_m << L"m, r=" << saved_radius
                     << L"m, arc=" << arc_deg << L"°)";
}

void previz_renderer::set_screen_position(const std::string& name, float x, float y, float z)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;

        it->second.pos_x = x;
        it->second.pos_y = y;
        it->second.pos_z = z;

        for (auto& mesh : impl_->scene_.meshes) {
            if (mesh.name == name) {
                apply_screen_transform(mesh, it->second);
                break;
            }
        }
    }
    update_projections();
}

void previz_renderer::set_screen_rotation(const std::string& name, float yaw, float pitch, float roll)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;

        it->second.rot_yaw   = yaw;
        it->second.rot_pitch = pitch;
        it->second.rot_roll  = roll;

        for (auto& mesh : impl_->scene_.meshes) {
            if (mesh.name == name) {
                apply_screen_transform(mesh, it->second);
                break;
            }
        }
    }
    update_projections();
}

void previz_renderer::set_screen_resolution(const std::string& name, int width_px, int height_px)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);

    auto it = impl_->scene_.screens.find(name);
    if (it == impl_->scene_.screens.end()) return;
    it->second.res_w = width_px;
    it->second.res_h = height_px;
}

void previz_renderer::set_screen_eye_mode(const std::string& name, int eye_mode, float x, float y, float z)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;
        it->second.eye_mode = (eye_mode == 1) ? 1 : 0;
        if (it->second.eye_mode == 1) {
            it->second.design_eye_x = x;
            it->second.design_eye_y = y;
            it->second.design_eye_z = z;
        }
    }
    update_projections();
}

void previz_renderer::set_screen_arc_v(const std::string& name, float arc_v_deg)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;
        it->second.arc_v_deg = arc_v_deg;
    }
    update_projections();
}

void previz_renderer::set_screen_icvfx(const std::string& name, bool enable)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;
        it->second.icvfx_enable = enable;
    }
    update_projections();
    CASPAR_LOG(info) << L"[previz] Screen " << u8(name) << L" ICVFX "
                     << (enable ? L"enabled" : L"disabled");
}

void previz_renderer::set_screen_channel(const std::string& name, int channel)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        auto it = impl_->scene_.screens.find(name);
        if (it == impl_->scene_.screens.end()) return;
        it->second.channel = channel;

        if (channel >= 0) {
            impl_->map_mesh(name, channel);
        } else {
            impl_->unmap_mesh(name);
        }
    }
    update_projections();
}

void previz_renderer::remove_screen(const std::string& name)
{
    unsigned int saved_vao = 0, saved_vbo = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        impl_->scene_.screens.erase(name);
        impl_->scene_.mesh_to_channel.erase(name);

        auto& meshes = impl_->scene_.meshes;
        for (auto it = meshes.begin(); it != meshes.end(); ++it) {
            if (it->name == name) {
                it->is_screen = false;
                saved_vao = it->vao;
                saved_vbo = it->vbo;
                meshes.erase(it);
                break;
            }
        }
    }

    // Dispatch GL resource cleanup to the GL thread (no lock held — no deadlock)
    if (saved_vao || saved_vbo) {
        impl_->ogl_->dispatch_async([saved_vao, saved_vbo] {
            if (saved_vao) { auto v = saved_vao; glDeleteVertexArrays(1, &v); }
            if (saved_vbo) { auto v = saved_vbo; glDeleteBuffers(1, &v); }
        });
    }
    CASPAR_LOG(info) << L"[previz] Removed screen: " << u8(name);
}

std::vector<std::string> previz_renderer::list_screens() const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    std::vector<std::string> names;
    for (const auto& [k, v] : impl_->scene_.screens)
        names.push_back(k);
    return names;
}

void previz_renderer::render(std::shared_ptr<texture>&         target,
                             const channel_texture_store&       tex_store,
                             int                                width,
                             int                                height)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->render(target, tex_store, width, height);
}

previz_scene previz_renderer::scene() const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    return impl_->scene_;
}

bool previz_renderer::active() const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    return impl_->scene_.active;
}

// ---- Auto-projection API ---------------------------------------------------

void previz_renderer::set_auto_projection(bool on)
{
    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        impl_->scene_.auto_projection = on;
    }
    if (on)
        update_projections();
    CASPAR_LOG(info) << L"[previz] Auto-projection " << (on ? L"enabled" : L"disabled");
}

void previz_renderer::set_projection_callback(projection_apply_fn fn)
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    impl_->projection_fn_ = std::move(fn);
}

void previz_renderer::update_projections()
{
    struct proj_update
    {
        int               channel;
        screen_projection proj;
    };
    std::vector<proj_update> updates;
    projection_apply_fn      fn;

    {
        std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
        if (!impl_->scene_.auto_projection || !impl_->projection_fn_)
            return;
        fn = impl_->projection_fn_;
        for (const auto& [name, meta] : impl_->scene_.screens) {
            if (meta.channel < 1)
                continue;
            updates.push_back({meta.channel, compute_frustum(impl_->scene_.camera, meta)});
        }
    }
    // Lock released — safe to call external callbacks (stage transforms)
    for (const auto& u : updates)
        fn(u.channel, u.proj);
}

screen_projection previz_renderer::compute_screen_projection(const std::string& screen_name) const
{
    std::lock_guard<std::mutex> lock(impl_->scene_mutex_);
    auto it = impl_->scene_.screens.find(screen_name);
    if (it == impl_->scene_.screens.end())
        return {};
    return compute_frustum(impl_->scene_.camera, it->second);
}

}}} // namespace caspar::accelerator::ogl
