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
#include <core/input/input_event.h>

#include <common/memory.h>

#include <functional>
#include <memory>
#include <string>

namespace caspar { namespace accelerator { namespace ogl {

class device;
class texture;

/// Callback invoked when auto-projection recomputes a screen's frustum.
/// Receives the full computed screen_projection (angles + fov + curve geometry).
/// The struct itself is `core::screen_projection`, re-exported by `previz_scene.h`.
using projection_apply_fn =
    std::function<void(int channel, const screen_projection&)>;


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

    /// ONE CAMERA, READ-MODIFY-WRITE UNDER THE SCENE LOCK: `edit` sees the current camera and
    /// returns false to leave it alone. `view` selects the viewport camera (which also enables the
    /// view override). For a writer that changes one field -- the HTTP stage route and the
    /// timeline's stage-field writer both do -- so neither re-sends the other's stale components.
    bool update_camera(bool                                          view,
                       const std::function<bool(core::previz_camera&)>& edit,
                       core::previz_camera*                          before = nullptr,
                       core::previz_camera*                          after  = nullptr);

    /// Set an independent viewport/navigation camera (render POV only; never
    /// affects projection).  Enables the view override.
    void set_view_camera(float x, float y, float z, float yaw, float pitch, float roll, float fov);

    /// Move or turn a VENUE MESH. False if no non-screen mesh carries that name.
    ///
    /// An offset ON TOP of the baked vertices rather than a replacement for them: the loaders
    /// apply a glTF node's own transform as the file is read, so a freshly loaded venue sits
    /// where the model says and this is where a correction goes.
    bool set_mesh_pose(const std::string& name, const float pos[3], const float rot[3]);
    bool mesh_pose(const std::string& name, float pos[3], float rot[3]) const;

    /// Gesture snapping: the step a DRAG rounds to, in metres and in degrees. 0 disables.
    void set_snap(double metres, double degrees);

    /// Frame a target in the viewport -- a named screen or mesh, else the selection, else the
    /// whole stage. False for a name that is neither, and for an empty stage.
    bool frame_view(const std::string& target);

    /// One of the canonical viewport orientations (TOP, FRONT, SIDE), then frame the stage.
    /// False for a word that is none of them.
    bool set_view_orientation(const std::string& name);

    /// Clear the viewport override so render() follows the production camera again.
    void clear_view_camera();

    /// Operator OVERRIDE/freeze: when locked, bound trackers stop driving the
    /// production camera (the operator can still set it manually).
    void set_camera_locked(bool locked);
    bool is_camera_locked() const;

    /// Orbit, pick and drag, from one pointer or keyboard event.
    ///
    /// Returns true if the event was consumed. False when no scene is loaded, so the channel
    /// passes the event on to its layers instead -- a previz-less channel behaves exactly as
    /// it did before this existed.
    ///
    /// `aspect` is the viewport's width/height, which the caller knows and this class does
    /// not: the renderer draws into whatever target it is handed.
    ///
    /// Every gesture goes through the ordinary mutators (`set_view_camera`,
    /// `set_screen_position`), never at the scene directly, because those re-apply the mesh
    /// transform and call `update_projections()`. A drag that wrote `screen_meta` itself would
    /// move the wireframe and leave the projection behind -- 202 and no picture, the shape of
    /// defect this fork keeps finding.
    bool input(const core::input_event& event, double aspect);

    /// Toggle visibility of a named mesh.
    void set_mesh_visible(const std::string& mesh_name, bool visible);

    /// Display toggles.
    void set_grid(bool on);
    void set_wireframe(bool on);
    void set_gizmo(bool on);

    /// Draw the PRODUCTION camera in the viewport: a small body and its frustum, out to the
    /// far edge of whatever it is pointing at. Distinct from the viewport camera, which is the
    /// eye this is drawn FROM -- so an operator orbiting the stage watches the shot rather
    /// than becoming it.
    void set_show_camera(bool on);

    /// Camera presets.
    void save_camera_preset(const std::string& name);
    /// False when there is no preset by that name -- the command answers 404 rather than 202.
    bool recall_camera_preset(const std::string& name);
    std::vector<std::string> list_camera_presets() const;

    /// Procedural screen creation.
    void add_screen_flat(const std::string& name, float width_m, float height_m);
    void add_screen_curved(const std::string& name, float width_m, float height_m, float radius_m, float arc_deg);
    void set_screen_position(const std::string& name, float x, float y, float z);
    void set_screen_rotation(const std::string& name, float yaw, float pitch, float roll);
    void set_screen_resolution(const std::string& name, int width_px, int height_px);

    /// Resize a screen IN PLACE, keeping everything else it carries.
    ///
    /// The hole this fills: until 2026-09-18 `width_m`, `height_m` and `arc_deg` were writable
    /// only by `add_screen_flat` / `add_screen_curved`, which build a FRESH `screen_meta` -- so
    /// correcting a wall from 10 m to 10.2 m meant re-adding it and losing its position,
    /// rotation, channel, eye mode, ICVFX, vertical arc and resolution. Every other property
    /// had a mutator; these two did not, and the control API's write path refused them saying
    /// so. The mesh is regenerated and the projections recomputed, as every other mutator does.
    void set_screen_size(const std::string& name, float width_m, float height_m);

    /// The horizontal arc. 0 makes the screen FLAT (radius 0); non-zero re-derives the radius
    /// from the current width by `width / 2 / sin(arc / 2)`, the same rule `ADD ... CURVED`
    /// uses, so the two routes cannot disagree about what a curved screen is.
    void set_screen_arc(const std::string& name, float arc_deg);
    void set_screen_channel(const std::string& name, int channel);

    /// Set the eye-point model for a screen: eye_mode 0=CAMERA (follow production
    /// camera), 1=FIXED (use the supplied design-eye position).
    void set_screen_eye_mode(const std::string& name, int eye_mode, float x, float y, float z);

    /// Set the vertical arc (degrees) of a doubly-curved screen (0 = cylinder).
    void set_screen_arc_v(const std::string& name, float arc_v_deg);

    /// Enable/disable ICVFX inner-frustum (in-camera VFX) for a screen.  When
    /// on, auto-projection also computes the camera-eye inner frustum + mask.
    /// LED panel size in PIXELS; 0 0 clears it. With the screen's resolution this derives how
    /// many panels the wall is, and the renderer draws their boundaries.
    /// How `select_screen` treats what is already selected.
    enum class select_mode
    {
        replace, ///< this one only
        add,     ///< add it, and make it the active one
        toggle,  ///< in if out, out if in
        none,    ///< clear
        all      ///< every screen
    };
    /// Select a screen. False when the name is not a screen -- `none` and `all` ignore it.
    bool select_screen(const std::string& name, select_mode mode);
    void set_screen_tile(const std::string& name, int w_px, int h_px);
    void set_screen_icvfx(const std::string& name, bool enable);
    /// Parent a screen to another, or to the stage with an empty name.
    ///
    /// Returns FALSE rather than throwing when the screen or the proposed parent does not exist,
    /// or when the attachment would close a cycle -- the three cases the command turns into one
    /// 400. The child's own position and rotation are left alone: they become LOCAL, which is
    /// what makes a parented screen jump when it is attached.
    bool set_screen_parent(const std::string& name, const std::string& parent);
    /// The same, for a venue mesh. Screens and meshes share one parent namespace.
    bool set_mesh_parent(const std::string& name, const std::string& parent);

    void remove_screen(const std::string& name);

    /// Copy a screen, placed EDGE TO EDGE with the original along its own width axis.
    ///
    /// False if the source does not exist or the new name is already taken. Note what it copies:
    /// EVERYTHING except the name and the position, the channel included -- see the definition
    /// for why that is the right default and not an oversight.
    bool clone_screen(const std::string& source, const std::string& name);
    std::vector<std::string> list_screens() const;

    /// Render the 3D scene to the given target texture.
    void render(std::shared_ptr<texture>&         target,
                const channel_texture_store&       tex_store,
                int                                width,
                int                                height);

    /// Get a snapshot of the current scene (thread-safe copy).
    ///
    /// DEEP-COPIES EVERY MESH, including its full vertex vector, under the scene lock. Fine for
    /// the AMCP queries and the layout save that use it; unusable on the channel tick. Use
    /// `stage_snapshot()` for anything per-frame.
    previz_scene scene() const;

    /// The narrow view: screens, the two cameras and the flags, and no geometry at all.
    ///
    /// This is what the per-tick publication reads and diffs. It exists because `scene()` above
    /// cannot be called 50 times a second, and the six floats a publisher wants are a rounding
    /// error next to the vertex data it would copy to get them.
    core::stage_snapshot stage_snapshot() const;

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
