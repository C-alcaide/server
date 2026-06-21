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

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cstdint>

namespace caspar { namespace core {
class texture;
}}

namespace caspar { namespace accelerator { namespace ogl {

// ---- Previz vertex (3D position + normal + UV) ----------------------------

struct previz_vertex
{
    float px = 0.0f, py = 0.0f, pz = 0.0f; // position
    float nx = 0.0f, ny = 0.0f, nz = 0.0f; // normal
    float u = 0.0f, v = 0.0f;              // texture coordinate
};

// ---- Previz mesh (single named surface) -----------------------------------

struct previz_mesh
{
    std::string                name;
    std::vector<previz_vertex> vertices; // triangle list
    std::array<float, 3>       base_color = {0.5f, 0.5f, 0.5f};
    bool                       is_screen  = false;
    bool                       visible    = true;

    // GL resources (uploaded on first use)
    unsigned int vao = 0;
    unsigned int vbo = 0;
    bool   gpu_dirty = true;
};

// ---- Previz camera --------------------------------------------------------

struct previz_camera
{
    float x = 0.0f, y = 1.5f, z = 5.0f; // position (metres)
    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f; // rotation (degrees)
    float fov = 60.0f;     // vertical field of view (degrees)
    float near_clip = 0.1f;
    float far_clip = 100.0f;
};

// ---- Procedural screen metadata -------------------------------------------

struct screen_meta
{
    std::string name;
    float width_m  = 1.0f;
    float height_m = 1.0f;
    float radius_m = 0.0f;   // 0 = flat
    float arc_deg  = 0.0f;
    float arc_v_deg = 0.0f;  // vertical arc (0 = single-curved cylinder)
    float pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;
    float rot_yaw = 0.0f, rot_pitch = 0.0f, rot_roll = 0.0f;
    int   res_w = 0;          // 0 = not set (use channel default)
    int   res_h = 0;
    int   channel = -1;       // mapped channel (-1 = unmapped)
    // Eye-point model for curve compensation & FOV:
    //   eye_mode 0 = CAMERA → eye follows the production virtual camera (in-camera VFX)
    //   eye_mode 1 = FIXED  → eye sits at a fixed audience design position (design_eye)
    int   eye_mode = 0;
    float design_eye_x = 0.0f, design_eye_y = 1.5f, design_eye_z = 3.0f;
    // ICVFX inner/outer frustum: when true, auto-projection computes a
    // camera-eye inner frustum + feathered camera-frustum mask for this screen.
    bool  icvfx_enable = false;
};

// ---- Previz scene (all meshes + camera + mappings) ------------------------

struct previz_scene
{
    std::vector<previz_mesh>         meshes;
    previz_camera                    camera;
    // Independent viewport/navigation camera.  When has_view_override is true the
    // render() POV uses view_camera, while compute_frustum() ALWAYS uses the
    // production 'camera' above.  This decouples operator viewport orbiting (Live
    // Link / nav) from the virtual production camera that drives projection.
    previz_camera                    view_camera;
    bool                             has_view_override = false;
    // Operator OVERRIDE/freeze: when true the bound tracker stops driving the
    // production camera, letting the operator hand-fly it (PREVIZ CAMERA SET still
    // works).  Cleared to hand control back to the tracker.
    bool                             camera_locked = false;
    std::map<std::string, int>       mesh_to_channel; // mesh name → channel index
    bool                             show_grid      = true;
    bool                             show_wireframe = false;
    bool                             show_gizmo     = true;
    bool                             active         = false;
    bool                             auto_projection = false;
    std::string                      scene_path;

    // Camera presets (name → camera state)
    std::map<std::string, previz_camera> camera_presets;

    // Procedural screens (name → metadata)
    std::map<std::string, screen_meta>   screens;
};

// ---- Channel texture store ------------------------------------------------
// Allows the previz renderer to look up the latest output texture from any
// CasparCG channel.  Each channel posts its output texture after rendering;
// the previz renderer reads them at draw time.  Thread-safe.

class channel_texture_store
{
  public:
    void update(int channel_id, std::shared_ptr<core::texture> tex, unsigned int texture_id, int width, int height)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[channel_id] = {std::move(tex), texture_id, width, height};
    }

    struct entry
    {
        std::shared_ptr<core::texture> tex_holder; // prevents pool recycling
        unsigned int tex_id = 0;
        int    width  = 0;
        int    height = 0;
    };

    entry get(int channel_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(channel_id);
        if (it != entries_.end())
            return it->second;
        return {};
    }

    void remove(int channel_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.erase(channel_id);
    }

  private:
    mutable std::mutex    mutex_;
    std::map<int, entry>  entries_;
};

}}} // namespace caspar::accelerator::ogl
