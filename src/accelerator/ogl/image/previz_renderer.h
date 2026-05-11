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

#include "previz_scene.h"

#include <common/memory.h>

#include <functional>
#include <memory>
#include <string>

namespace caspar { namespace accelerator { namespace ogl {

class device;
class texture;

/// Callback invoked when auto-projection recomputes a screen's frustum.
/// Parameters: channel_id, yaw_rad, pitch_rad, roll_rad, fov_rad
using projection_apply_fn =
    std::function<void(int channel, double yaw, double pitch, double roll, double fov)>;

/// Computed projection for one screen.
struct screen_projection
{
    float yaw_deg   = 0.0f;
    float pitch_deg = 0.0f;
    float roll_deg  = 0.0f;
    float fov_deg   = 60.0f;
};

class previz_renderer
{
  public:
    explicit previz_renderer(const spl::shared_ptr<device>& ogl);
    ~previz_renderer();

    /// Load a glTF/GLB or OBJ venue model into the scene.
    void load_scene(const std::string& path);

    /// Map a named mesh to a CasparCG channel (for texture sampling).
    void map_mesh(const std::string& mesh_name, int channel_index);

    /// Unmap a named mesh (revert to default material).
    void unmap_mesh(const std::string& mesh_name);

    /// Set the virtual camera.
    void set_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov);

    /// Reset camera to default.
    void reset_camera();

    /// Toggle visibility of a named mesh.
    void set_mesh_visible(const std::string& mesh_name, bool visible);

    /// Display toggles.
    void set_grid(bool on);
    void set_wireframe(bool on);
    void set_gizmo(bool on);

    /// Camera presets.
    void save_camera_preset(const std::string& name);
    void recall_camera_preset(const std::string& name);
    std::vector<std::string> list_camera_presets() const;

    /// Procedural screen creation.
    void add_screen_flat(const std::string& name, float width_m, float height_m);
    void add_screen_curved(const std::string& name, float width_m, float height_m, float radius_m, float arc_deg);
    void set_screen_position(const std::string& name, float x, float y, float z);
    void set_screen_rotation(const std::string& name, float yaw, float pitch, float roll);
    void set_screen_resolution(const std::string& name, int width_px, int height_px);
    void set_screen_channel(const std::string& name, int channel);
    void remove_screen(const std::string& name);
    std::vector<std::string> list_screens() const;

    /// Render the 3D scene to the given target texture.
    void render(std::shared_ptr<texture>&         target,
                const channel_texture_store&       tex_store,
                int                                width,
                int                                height);

    /// Get a snapshot of the current scene (thread-safe copy).
    previz_scene scene() const;

    /// Whether a scene is loaded and active.
    bool active() const;

    /// Auto-projection: derive MIXER PROJECTION parameters from screen geometry.
    void set_auto_projection(bool on);
    void set_projection_callback(projection_apply_fn fn);
    void update_projections();

    /// Compute the projection for a single screen (returns degrees).
    screen_projection compute_screen_projection(const std::string& screen_name) const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::accelerator::ogl
