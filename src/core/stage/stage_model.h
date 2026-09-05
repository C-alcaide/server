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

#include <map>
#include <string>

/// The 3D stage's DESCRIPTION, with none of its rendering.
///
/// These three structs used to live inside the OpenGL accelerator -- `screen_meta` and
/// `previz_camera` in `previz_scene.h`, `screen_projection` in `previz_renderer.h` -- which had
/// two costs. Nothing in `core` could name a screen, so the stage was addressable only through
/// thirteen bespoke AMCP verbs and published nothing to `monitor::state`. And the projection
/// maths that consumes them could not be tested without a GL device, although it does no GL.
///
/// They are plain data and were already plain data; the move is a change of namespace and
/// header, not of layout. `accelerator/ogl/image/previz_scene.h` re-exports all three into
/// `accelerator::ogl`, so every existing use of `ogl::screen_meta` still compiles.
///
/// What deliberately did NOT move: `previz_vertex`, `previz_mesh` (it carries GL `vao`/`vbo`
/// handles), the glTF and OBJ loaders, `channel_texture_store` and `previz_renderer` itself.
/// The line is drawn at "does this touch the GPU", and these do.
namespace caspar { namespace core {

/// A camera in the previz stage.
///
/// There are two of them per scene and they are not interchangeable: the PRODUCTION camera is
/// what `compute_frustum` reads, and the VIEW camera is the operator's viewport. Orbiting the
/// viewport must never move a projection, which is why the split exists.
struct previz_camera
{
    float x = 0.0f, y = 1.5f, z = 5.0f;          // position (metres)
    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f; // rotation (degrees)
    float fov       = 60.0f;                     // vertical field of view (degrees)
    float near_clip = 0.1f;
    float far_clip  = 100.0f;
};

/// One screen in the stage: where it is, how big it is, how it curves, and which channel feeds
/// it.
///
/// `radius_m` is DERIVED for a curved screen -- `add_screen_curved` recomputes it as
/// `width / 2 / sin(arc / 2)` and uses a supplied value only when that sine is degenerate. An
/// operator who sets it is almost never storing the value they typed.
struct screen_meta
{
    std::string name;
    float       width_m   = 1.0f;
    float       height_m  = 1.0f;
    float       radius_m  = 0.0f; // 0 = flat
    float       arc_deg   = 0.0f;
    float       arc_v_deg = 0.0f; // vertical arc (0 = single-curved cylinder)
    float       pos_x = 0.0f, pos_y = 0.0f, pos_z = 0.0f;
    float       rot_yaw = 0.0f, rot_pitch = 0.0f, rot_roll = 0.0f;
    int         res_w   = 0; // 0 = not set (use channel default)
    int         res_h   = 0;
    int         channel = -1; // mapped channel (-1 = unmapped)
    // Eye-point model for curve compensation & FOV:
    //   eye_mode 0 = CAMERA → eye follows the production virtual camera (in-camera VFX)
    //   eye_mode 1 = FIXED  → eye sits at a fixed audience design position (design_eye)
    int   eye_mode = 0;
    float design_eye_x = 0.0f, design_eye_y = 1.5f, design_eye_z = 3.0f;
    // ICVFX inner/outer frustum: when true, auto-projection computes a
    // camera-eye inner frustum + feathered camera-frustum mask for this screen.
    bool icvfx_enable = false;
};

/// What `compute_frustum` produces for one screen: the orientation and field of view a channel
/// must render at to fill that screen from the production camera's eye, plus the curve
/// compensation and ICVFX geometry the warp shader needs.
struct screen_projection
{
    float yaw_deg   = 0.0f;
    float pitch_deg = 0.0f;
    float roll_deg  = 0.0f;
    float fov_deg   = 60.0f;
    // Derived curved-screen compensation geometry.
    int   curve_type       = 0;    // 0=flat,1=cylinder,2=sphere,3=fisheye
    float screen_arc_deg   = 0.0f; // horizontal arc subtended by the screen
    float screen_arc_v_deg = 0.0f; // vertical arc (0 = cylinder)
    float eye_distance     = 1.0f; // viewer distance / screen radius (k)
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

/// The stage-level flags that belong to the scene rather than to any one screen or camera.
///
/// BRIEF SS5: "diagnostic overlays are parameters on the object, not a debug menu" -- so the grid,
/// the wireframe and the gizmo are fields here rather than three more bespoke verbs.
struct previz_flags
{
    bool active            = false;
    bool auto_projection   = false;
    bool show_grid         = true;
    bool show_wireframe    = false;
    bool show_gizmo        = true; // written by PREVIZ GIZMO and read by nothing -- see stage.md
    bool camera_locked     = false;
    bool has_view_override = false;
};

/// Everything about the stage that is worth publishing every tick, and nothing that is not.
///
/// Deliberately NOT `previz_scene`: that carries every mesh with its full vertex vector, and
/// `previz_renderer::scene()` deep-copies the lot under the scene lock. Taking that 50 times a
/// second to read six floats would be absurd. This is the narrow view -- screens, the two
/// cameras and the flags -- and it is what the cache-compare in the publication path diffs.
struct stage_snapshot
{
    previz_flags                       flags;
    previz_camera                      camera;      // the PRODUCTION camera: what compute_frustum reads
    previz_camera                      view_camera; // the operator's viewport
    std::map<std::string, screen_meta> screens;
    std::string                        scene_path;
};

// Equality is deliberately NOT defined here. Comparing these structs member by member would be a
// fourth hand-written list of a screen's properties, and a field added to the table but missed in
// it would make the publication path stop noticing that field had changed -- silently, and with
// exactly the shape of the `image_transform` composition-allowlist trap. `same_stage()` in
// `stage_fields.h` compares through the REGISTRY instead, so a new row is part of the comparison
// the moment it is declared.

}} // namespace caspar::core
