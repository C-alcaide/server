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

#include "stage_math.h"

#include <algorithm>
#include <cmath>

// Moved out of `accelerator/ogl/image/previz_renderer.cpp`'s anonymous namespace VERBATIM --
// not a line of the arithmetic changed, which is what makes `previz-picture` a real check on
// the move rather than a check on a rewrite.
namespace caspar { namespace core {

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

    // Eye point: follow the production camera (in-camera VFX) or sit at a fixed
    // audience design position, per the screen's eye_mode.
    float eye_x = cam.x, eye_y = cam.y, eye_z = cam.z;
    if (meta.eye_mode == 1) {
        eye_x = meta.design_eye_x;
        eye_y = meta.design_eye_y;
        eye_z = meta.design_eye_z;
    }

    float dx = cx - eye_x;
    float dy = cy - eye_y;
    float dz = cz - eye_z;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < 1e-6f)
        return {}; // eye at screen center — degenerate

    // Perpendicular distance from eye to screen plane
    float perp_dist = std::abs(dx * snx + dy * sny + dz * snz);
    if (perp_dist < 1e-6f)
        perp_dist = dist; // eye in screen plane — use total distance

    // Vertical FOV: angular extent of screen height from the eye
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
    // ── Curved-screen compensation geometry ───────────────────────
    // Derive the physical curve parameters the warp shader needs directly
    // from the screen geometry the previz already knows about.
    //   curve_type : cylinder when single-curved, sphere when doubly-curved,
    //                flat when the panel has no radius/arc.
    //   eye_distance (k) = viewer distance / screen radius (Dv/R).  For a flat
    //                panel (radius_m <= 0) there is no curvature, so k is left
    //                at the neutral value and the type forced flat to avoid a
    //                divide-by-zero.
    int   curve_type   = 0;
    float arc_h        = 0.0f;
    float arc_v        = 0.0f;
    float eye_distance = 1.0f;
    if (meta.radius_m > 1e-4f && meta.arc_deg > 1e-4f) {
        arc_h = meta.arc_deg;
        arc_v = (meta.arc_v_deg > 1e-4f) ? meta.arc_v_deg : 0.0f;
        curve_type   = (arc_v > 0.0f) ? 2 /*sphere*/ : 1 /*cylinder*/;
        eye_distance = std::max(0.05f, std::min(100.0f, perp_dist / meta.radius_m));
    }

    screen_projection result{proj_yaw, proj_pitch, proj_roll, fov_v, curve_type, arc_h, arc_v, eye_distance};

    // ── ICVFX inner/outer frustum ──────────────────────────────────
    // Outer projection (above) stays as the design-eye / LED-volume slice.  The
    // inner frustum re-samples the source from the tracked camera's viewpoint and
    // a feathered quad mask marks where the camera looks on the screen.
    if (meta.icvfx_enable) {
        result.icvfx_enable = true;

        // Screen right axis = rotation column 0; up & normal already in scope.
        float rgt_x = rot.m[0], rgt_y = rot.m[1], rgt_z = rot.m[2];

        // Inner view direction = from screen centre toward the camera; this
        // off-axis "normal" shifts the 360 sampling to give the on-camera parallax.
        float inx = cam.x - cx, iny = cam.y - cy, inz = cam.z - cz;
        float in_horiz    = std::sqrt(inx * inx + inz * inz);
        float inner_yaw   = std::atan2(inx, inz) * 180.0f / static_cast<float>(M_PI);
        float inner_pitch = std::atan2(-iny, in_horiz) * 180.0f / static_cast<float>(M_PI);

        // Inner FOV from the camera's perpendicular distance to the screen.
        float icdx = cx - cam.x, icdy = cy - cam.y, icdz = cz - cam.z;
        float icdist = std::sqrt(icdx * icdx + icdy * icdy + icdz * icdz);
        float iperp  = std::abs(icdx * snx + icdy * sny + icdz * snz);
        if (iperp < 1e-6f)
            iperp = (icdist > 1e-6f) ? icdist : 1.0f;
        float inner_fov = 2.0f * std::atan2(meta.height_m * 0.5f, iperp) * 180.0f / static_cast<float>(M_PI);
        inner_fov       = std::max(1.0f, std::min(170.0f, inner_fov));

        result.inner_yaw_deg   = inner_yaw;
        result.inner_pitch_deg = inner_pitch;
        result.inner_roll_deg  = proj_roll;  // screen orientation defines "up"
        result.inner_fov_deg   = inner_fov;
        float inner_k = 1.0f;
        if (curve_type != 0 && meta.radius_m > 1e-4f)
            inner_k = std::max(0.05f, std::min(100.0f, iperp / meta.radius_m));
        result.inner_eye_distance = inner_k;

        // ── Camera-frustum mask quad (content NDC) ─────────────────
        // Camera world basis: forward = R*(0,0,-1) = -col2, right = col0, up = col1.
        auto  cam_rot = mat4::rotate_y(cam.yaw) * mat4::rotate_x(cam.pitch) * mat4::rotate_z(cam.roll);
        float crx = cam_rot.m[0],  cry = cam_rot.m[1],  crz = cam_rot.m[2];   // right
        float cux = cam_rot.m[4],  cuy = cam_rot.m[5],  cuz = cam_rot.m[6];   // up
        float cfx = -cam_rot.m[8], cfy = -cam_rot.m[9], cfz = -cam_rot.m[10]; // forward

        // Frustum half-extents (vertical FOV from camera; aspect = screen aspect).
        float cam_aspect = (meta.height_m > 1e-6f) ? (meta.width_m / meta.height_m) : 1.77778f;
        float half_v     = std::tan(cam.fov * 0.5f * static_cast<float>(M_PI) / 180.0f);
        float half_h     = half_v * cam_aspect;

        // Intersect the 4 corner rays with the screen plane (point = centre,
        // normal = (snx,sny,snz)) and convert to content NDC.
        const float sgn_x[4] = {-1.0f, 1.0f, 1.0f, -1.0f}; // UL, UR, LR, LL
        const float sgn_y[4] = { 1.0f, 1.0f, -1.0f, -1.0f};
        bool        mask_valid = true;
        for (int i = 0; i < 4 && mask_valid; ++i) {
            float dxr = cfx + sgn_x[i] * half_h * crx + sgn_y[i] * half_v * cux;
            float dyr = cfy + sgn_x[i] * half_h * cry + sgn_y[i] * half_v * cuy;
            float dzr = cfz + sgn_x[i] * half_h * crz + sgn_y[i] * half_v * cuz;
            float denom = dxr * snx + dyr * sny + dzr * snz;
            if (std::abs(denom) < 1e-6f) { mask_valid = false; break; }
            float t = ((cx - cam.x) * snx + (cy - cam.y) * sny + (cz - cam.z) * snz) / denom;
            if (t <= 0.0f) { mask_valid = false; break; } // screen behind camera
            float px = cam.x + t * dxr, py = cam.y + t * dyr, pz = cam.z + t * dzr;
            float lx = (px - meta.pos_x) * rgt_x + (py - meta.pos_y) * rgt_y + (pz - meta.pos_z) * rgt_z;
            float ly = (px - meta.pos_x) * sux   + (py - meta.pos_y) * suy   + (pz - meta.pos_z) * suz;
            result.icvfx_q[i * 2 + 0] = (meta.width_m  > 1e-6f) ? (2.0f * lx / meta.width_m)        : 0.0f;
            result.icvfx_q[i * 2 + 1] = (meta.height_m > 1e-6f) ? (1.0f - 2.0f * ly / meta.height_m) : 0.0f;
        }
        if (!mask_valid) {
            // Camera not looking at the screen — collapse to a zero-area quad so
            // the shader renders pure outer with no inner region.
            for (int i = 0; i < 8; ++i)
                result.icvfx_q[i] = 0.0f;
        }
        result.icvfx_feather   = 0.05f;
        result.icvfx_outer_dim = 1.0f;
    }

    return result;
}

std::optional<pick> compute_pick(const previz_camera&                     view,
                                 double                                    aspect,
                                 double                                    sx,
                                 double                                    sy,
                                 const std::map<std::string, screen_meta>& screens)
{
    // The camera's world basis, exactly as `compute_frustum` builds it for the ICVFX quad:
    // Ry(yaw) * Rx(pitch) * Rz(roll), right = column 0, up = column 1, forward = -column 2.
    const auto cam_rot = mat4::rotate_y(view.yaw) * mat4::rotate_x(view.pitch) * mat4::rotate_z(view.roll);

    const double rgt[3] = {cam_rot.m[0], cam_rot.m[1], cam_rot.m[2]};
    const double up[3]  = {cam_rot.m[4], cam_rot.m[5], cam_rot.m[6]};
    const double fwd[3] = {-cam_rot.m[8], -cam_rot.m[9], -cam_rot.m[10]};

    const double half_v = std::tan(view.fov * 0.5 * M_PI / 180.0);
    const double xc     = (2.0 * sx - 1.0) * aspect * half_v;
    // (1 - 2*sy), NOT (2*sy - 1): the renderer negates proj.m[5] so the FBO's bottom-up render
    // arrives top-down, which puts sy = 0 at +Y in camera space. See the header.
    const double yc = (1.0 - 2.0 * sy) * half_v;

    double dir[3];
    for (int i = 0; i < 3; ++i)
        dir[i] = fwd[i] + rgt[i] * xc + up[i] * yc;

    const double eye[3] = {view.x, view.y, view.z};

    std::optional<pick> best;

    for (const auto& [name, meta] : screens) {
        // The screen's own basis, same Ry*Rx*Rz order `apply_screen_transform` and
        // `compute_frustum` use. Normal is +Z un-rotated, i.e. column 2.
        const auto rot = mat4::rotate_y(meta.rot_yaw) * mat4::rotate_x(meta.rot_pitch) *
                         mat4::rotate_z(meta.rot_roll);

        const double s_rgt[3] = {rot.m[0], rot.m[1], rot.m[2]};
        const double s_up[3]  = {rot.m[4], rot.m[5], rot.m[6]};
        const double s_nrm[3] = {rot.m[8], rot.m[9], rot.m[10]};

        const double denom = dir[0] * s_nrm[0] + dir[1] * s_nrm[1] + dir[2] * s_nrm[2];
        if (std::abs(denom) < 1e-9)
            continue; // the ray runs along the plane

        const double to_plane[3] = {meta.pos_x - eye[0], meta.pos_y - eye[1], meta.pos_z - eye[2]};
        const double t = (to_plane[0] * s_nrm[0] + to_plane[1] * s_nrm[1] + to_plane[2] * s_nrm[2]) / denom;
        if (t <= 0.0)
            continue; // behind the eye

        const double hit[3] = {eye[0] + dir[0] * t, eye[1] + dir[1] * t, eye[2] + dir[2] * t};
        const double loc[3] = {hit[0] - meta.pos_x, hit[1] - meta.pos_y, hit[2] - meta.pos_z};

        const double lx = loc[0] * s_rgt[0] + loc[1] * s_rgt[1] + loc[2] * s_rgt[2];
        const double ly = loc[0] * s_up[0] + loc[1] * s_up[1] + loc[2] * s_up[2];

        // The quad: origin is centre-BOTTOM, so x spans +/- width/2 and y spans 0..height.
        if (std::abs(lx) > meta.width_m * 0.5 || ly < 0.0 || ly > meta.height_m)
            continue;

        // `dir` is not unit length -- it is forward plus the off-axis terms -- so `t` is in
        // units of that vector, not metres. Scale it once, here, so `pick::distance` means what
        // it says and comparing two hits compares real distances.
        const double dir_len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        const double dist    = t * dir_len;

        if (!best || dist < best->distance)
            best = pick{name, dist, lx, ly};
    }

    return best;
}

}} // namespace caspar::core
