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
/// Receives the full computed screen_projection (angles + fov + curve geometry).
struct screen_projection;
using projection_apply_fn =
    std::function<void(int channel, const screen_projection&)>;

/// Computed projection for one screen.
struct screen_projection
{
    float yaw_deg   = 0.0f;
    float pitch_deg = 0.0f;
    float roll_deg  = 0.0f;
    float fov_deg   = 60.0f;
    // Derived curved-screen compensation geometry.
    int   curve_type      = 0;     // 0=flat,1=cylinder,2=sphere,3=fisheye
    float screen_arc_deg  = 0.0f;  // horizontal arc subtended by the screen
    float screen_arc_v_deg = 0.0f; // vertical arc (0 = cylinder)
    float eye_distance    = 1.0f;  // viewer distance / screen radius (k)
    // ── ICVFX inner/outer frustum ──────────────────────────────────────
    bool  icvfx_enable    = false; // inner-frustum blend active for this screen
    float inner_yaw_deg   = 0.0f;  // inner (camera-eye) view orientation
    float inner_pitch_deg = 0.0f;
    float inner_roll_deg  = 0.0f;
    float inner_fov_deg   = 60.0f;
    float inner_eye_distance = 1.0f;
    // Camera-frustum mask quad in output NDC (-1..+1): 0=UL,1=UR,2=LR,3=LL
    float icvfx_q[8]      = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
    float icvfx_feather   = 0.05f; // mask edge feather (NDC units)
    float icvfx_outer_dim = 1.0f;  // outer-region brightness multiplier (0..1)
};

class previz_renderer
{
  public:
    explicit previz_renderer(const spl::shared_ptr<device>& ogl);
    ~previz_renderer();

    /// Load a glTF/GLB or OBJ venue model into the scene.
    void load_scene(const std::string& path);

    /// Persist / restore the procedural stage layout (screens + cameras +
    /// eye modes) to a JSON file.  Independent of the venue mesh model.
    void save_layout(const std::string& path) const;
    void load_layout(const std::string& path);

    /// Map a named mesh to a CasparCG channel (for texture sampling).
    void map_mesh(const std::string& mesh_name, int channel_index);

    /// Unmap a named mesh (revert to default material).
    void unmap_mesh(const std::string& mesh_name);

    /// Set the virtual camera.
    void set_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov);

    /// Reset camera to default.
    void reset_camera();

    /// Set an independent viewport/navigation camera (render POV only; never
    /// affects projection).  Enables the view override.
    void set_view_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov);

    /// Clear the viewport override so render() follows the production camera again.
    void clear_view_camera();

    /// Operator OVERRIDE/freeze: when locked, bound trackers stop driving the
    /// production camera (the operator can still set it manually).
    void set_camera_locked(bool locked);
    bool is_camera_locked() const;

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

    /// Set the eye-point model for a screen: eye_mode 0=CAMERA (follow production
    /// camera), 1=FIXED (use the supplied design-eye position).
    void set_screen_eye_mode(const std::string& name, int eye_mode, float x, float y, float z);

    /// Set the vertical arc (degrees) of a doubly-curved screen (0 = cylinder).
    void set_screen_arc_v(const std::string& name, float arc_v_deg);

    /// Enable/disable ICVFX inner-frustum (in-camera VFX) for a screen.  When
    /// on, auto-projection also computes the camera-eye inner frustum + mask.
    void set_screen_icvfx(const std::string& name, bool enable);

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
