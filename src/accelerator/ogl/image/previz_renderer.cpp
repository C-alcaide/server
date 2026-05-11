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

#include "../util/device.h"
#include "../util/shader.h"
#include "../util/texture.h"

#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/filesystem.hpp>

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caspar { namespace accelerator { namespace ogl {

namespace {

// ---- Math helpers ----------------------------------------------------------

struct mat4
{
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    static mat4 identity() { return {}; }

    static mat4 perspective(float fov_deg, float aspect, float near_p, float far_p)
    {
        float f   = 1.0f / std::tan(fov_deg * static_cast<float>(M_PI) / 360.0f);
        float nf  = 1.0f / (near_p - far_p);
        mat4  r   = {};
        std::memset(r.m, 0, sizeof(r.m));
        r.m[0]  = f / aspect;
        r.m[5]  = f;
        r.m[10] = (far_p + near_p) * nf;
        r.m[11] = -1.0f;
        r.m[14] = 2.0f * far_p * near_p * nf;
        return r;
    }

    static mat4 translate(float x, float y, float z)
    {
        mat4 r;
        r.m[12] = x;
        r.m[13] = y;
        r.m[14] = z;
        return r;
    }

    static mat4 rotate_x(float deg)
    {
        float rad = deg * static_cast<float>(M_PI) / 180.0f;
        float c   = std::cos(rad);
        float s   = std::sin(rad);
        mat4  r;
        r.m[5]  = c;
        r.m[6]  = s;
        r.m[9]  = -s;
        r.m[10] = c;
        return r;
    }

    static mat4 rotate_y(float deg)
    {
        float rad = deg * static_cast<float>(M_PI) / 180.0f;
        float c   = std::cos(rad);
        float s   = std::sin(rad);
        mat4  r;
        r.m[0]  = c;
        r.m[2]  = -s;
        r.m[8]  = s;
        r.m[10] = c;
        return r;
    }

    static mat4 rotate_z(float deg)
    {
        float rad = deg * static_cast<float>(M_PI) / 180.0f;
        float c   = std::cos(rad);
        float s   = std::sin(rad);
        mat4  r;
        r.m[0] = c;
        r.m[1] = s;
        r.m[4] = -s;
        r.m[5] = c;
        return r;
    }

    mat4 operator*(const mat4& o) const
    {
        mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row) {
                r.m[c * 4 + row] = 0;
                for (int k = 0; k < 4; ++k)
                    r.m[c * 4 + row] += m[k * 4 + row] * o.m[c * 4 + k];
            }
        return r;
    }
};

// ---- glTF loader -----------------------------------------------------------

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

            const auto& pos_acc = model.accessors[pos_it->second];
            const auto& pos_bv  = model.bufferViews[pos_acc.bufferView];
            const auto* pos_raw = &model.buffers[pos_bv.buffer].data[pos_bv.byteOffset + pos_acc.byteOffset];
            int pos_stride = pos_bv.byteStride > 0 ? static_cast<int>(pos_bv.byteStride) : 12;

            // Normals
            const uint8_t* nrm_raw    = nullptr;
            int            nrm_stride = 12;
            auto           nrm_it     = prim.attributes.find("NORMAL");
            if (nrm_it != prim.attributes.end()) {
                const auto& nrm_acc = model.accessors[nrm_it->second];
                const auto& nrm_bv  = model.bufferViews[nrm_acc.bufferView];
                nrm_raw    = &model.buffers[nrm_bv.buffer].data[nrm_bv.byteOffset + nrm_acc.byteOffset];
                nrm_stride = nrm_bv.byteStride > 0 ? static_cast<int>(nrm_bv.byteStride) : 12;
            }

            // UVs
            const uint8_t* uv_raw    = nullptr;
            int            uv_stride = 8;
            auto           uv_it     = prim.attributes.find("TEXCOORD_0");
            if (uv_it != prim.attributes.end()) {
                const auto& uv_acc = model.accessors[uv_it->second];
                const auto& uv_bv  = model.bufferViews[uv_acc.bufferView];
                uv_raw    = &model.buffers[uv_bv.buffer].data[uv_bv.byteOffset + uv_acc.byteOffset];
                uv_stride = uv_bv.byteStride > 0 ? static_cast<int>(uv_bv.byteStride) : 8;
            }

            auto read_pos = [&](int i) -> std::array<float, 3> {
                const float* p = reinterpret_cast<const float*>(pos_raw + i * pos_stride);
                return {p[0], p[1], p[2]};
            };
            auto read_nrm = [&](int i) -> std::array<float, 3> {
                if (!nrm_raw)
                    return {0.0f, 1.0f, 0.0f};
                const float* p = reinterpret_cast<const float*>(nrm_raw + i * nrm_stride);
                return {p[0], p[1], p[2]};
            };
            auto read_uv = [&](int i) -> std::array<float, 2> {
                if (!uv_raw)
                    return {0.0f, 0.0f};
                const float* p = reinterpret_cast<const float*>(uv_raw + i * uv_stride);
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

                pm.vertices.reserve(idx_acc.count);
                for (size_t i = 0; i < idx_acc.count; ++i) {
                    int  idx      = get_index(i);
                    if (idx < 0 || idx >= static_cast<int>(pos_acc.count))
                        continue;
                    auto [px, py, pz] = read_pos(idx);
                    auto [nx, ny, nz] = read_nrm(idx);
                    auto [tu, tv]     = read_uv(idx);
                    pm.vertices.push_back({px, py, pz, nx, ny, nz, tu, tv});
                }
            } else {
                pm.vertices.reserve(pos_acc.count);
                for (size_t i = 0; i < pos_acc.count; ++i) {
                    auto [px, py, pz] = read_pos(static_cast<int>(i));
                    auto [nx, ny, nz] = read_nrm(static_cast<int>(i));
                    auto [tu, tv]     = read_uv(static_cast<int>(i));
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
        shader_ = std::make_shared<shader>(std::string(previz_vertex_shader), std::string(previz_fragment_shader));
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

        // Build view matrix (camera)
        auto view = mat4::rotate_z(-scene_.camera.roll) * mat4::rotate_x(-scene_.camera.pitch) *
                    mat4::rotate_y(-scene_.camera.yaw) *
                    mat4::translate(-scene_.camera.x, -scene_.camera.y, -scene_.camera.z);

        float aspect = static_cast<float>(w) / static_cast<float>(h);
        auto  proj   = mat4::perspective(scene_.camera.fov, aspect, scene_.camera.near_clip, scene_.camera.far_clip);

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

// ---- Compute frustum from screen geometry relative to camera ---------------
//
// LED-volume convention: the projection yaw/pitch are determined by the
// screen's NORMAL direction (which part of the 360° sphere this panel
// represents), NOT the camera-to-screen direction.  Rotating a screen
// changes which slice of the equirectangular it displays.  The camera
// position still affects the FOV (apparent angular size of the panel).

screen_projection compute_frustum(const previz_camera& cam, const screen_meta& meta)
{
    // Build screen rotation matrix (same Ry*Rx*Rz order as apply_screen_transform)
    auto ry  = mat4::rotate_y(meta.rot_yaw);
    auto rx  = mat4::rotate_x(meta.rot_pitch);
    auto rz  = mat4::rotate_z(meta.rot_roll);
    auto rot = ry * rx * rz;

    // Screen normal: un-rotated screen faces +Z → third column of rotation matrix
    // The normal points from the screen surface TOWARD the camera (inward).
    float snx = rot.m[8], sny = rot.m[9], snz = rot.m[10];

    // Screen up: rotated +Y → second column of rotation matrix
    float sux = rot.m[4], suy = rot.m[5], suz = rot.m[6];

    // ── Projection yaw/pitch from screen normal ────────────────────
    // The normal points inward (toward viewer).  The content behind the
    // screen (outward direction) is (-snx, -sny, -snz).
    // Convention: yaw=0 → looks along -Z, yaw=90 → looks along -X.
    // atan2(snx, snz) gives 0 for a default front screen (normal 0,0,1)
    // and ±90 for side screens.
    float proj_yaw = std::atan2(snx, snz) * 180.0f / static_cast<float>(M_PI);

    // Pitch: negative sny → content above (positive pitch = look up)
    float horiz_normal = std::sqrt(snx * snx + snz * snz);
    float proj_pitch = std::atan2(-sny, horiz_normal) * 180.0f / static_cast<float>(M_PI);

    // ── FOV from camera-to-screen perpendicular distance ───────────
    // Screen center: origin is center-bottom, so center is at local (0, h/2, 0)
    float hh = meta.height_m * 0.5f;
    float cx = meta.pos_x + rot.m[4] * hh;
    float cy = meta.pos_y + rot.m[5] * hh;
    float cz = meta.pos_z + rot.m[6] * hh;

    float dx = cx - cam.x;
    float dy = cy - cam.y;
    float dz = cz - cam.z;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f)
        return {}; // camera at screen center — degenerate

    // Perpendicular distance from camera to screen plane
    float perp_dist = std::abs(dx * snx + dy * sny + dz * snz);
    if (perp_dist < 1e-6f)
        perp_dist = dist; // camera in screen plane — use total distance

    // Vertical FOV: angular extent of screen height from camera
    float fov_v = 2.0f * std::atan2(meta.height_m * 0.5f, perp_dist) * 180.0f / static_cast<float>(M_PI);
    fov_v       = std::max(1.0f, std::min(170.0f, fov_v)); // clamp to sane range

    // ── Roll from screen up vs world up ────────────────────────────
    // View direction = outward from screen = -normal
    float vx = -snx, vy = -sny, vz = -snz;

    // Project screen up onto plane perpendicular to view direction
    float dot_uv = sux * vx + suy * vy + suz * vz;
    float pux = sux - dot_uv * vx;
    float puy = suy - dot_uv * vy;
    float puz = suz - dot_uv * vz;
    float pu_len = std::sqrt(pux * pux + puy * puy + puz * puz);

    float proj_roll = 0.0f;
    if (pu_len > 1e-6f) {
        pux /= pu_len;
        puy /= pu_len;
        puz /= pu_len;

        // Project world up (0,1,0) onto the same plane
        float wux = -vy * vx;
        float wuy = 1.0f - vy * vy;
        float wuz = -vy * vz;
        float wu_len = std::sqrt(wux * wux + wuy * wuy + wuz * wuz);

        if (wu_len > 1e-6f) {
            wux /= wu_len;
            wuy /= wu_len;
            wuz /= wu_len;

            float cos_r = std::max(-1.0f, std::min(1.0f, pux * wux + puy * wuy + puz * wuz));

            // Cross product for sign: cross(world_up_proj, screen_up_proj) · view
            float cross_v = (wuy * puz - wuz * puy) * vx + (wuz * pux - wux * puz) * vy +
                            (wux * puy - wuy * pux) * vz;

            proj_roll = std::acos(cos_r) * 180.0f / static_cast<float>(M_PI);
            if (cross_v < 0)
                proj_roll = -proj_roll;
        }
    }

    return {proj_yaw, proj_pitch, proj_roll, fov_v};
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
    static const double DEG2RAD = 3.141592653589793 / 180.0;

    struct proj_update
    {
        int    channel;
        double yaw, pitch, roll, fov;
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
            auto p = compute_frustum(impl_->scene_.camera, meta);
            updates.push_back({meta.channel,
                               static_cast<double>(p.yaw_deg) * DEG2RAD,
                               static_cast<double>(p.pitch_deg) * DEG2RAD,
                               static_cast<double>(p.roll_deg) * DEG2RAD,
                               static_cast<double>(p.fov_deg) * DEG2RAD});
        }
    }
    // Lock released — safe to call external callbacks (stage transforms)
    for (const auto& u : updates)
        fn(u.channel, u.yaw, u.pitch, u.roll, u.fov);
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
