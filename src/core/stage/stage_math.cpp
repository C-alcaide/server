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

#include <boost/algorithm/string.hpp>

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

std::vector<std::pair<float, float>> clip_to_unit_rect(const float* pts, int count)
{
    std::vector<std::pair<float, float>> poly;
    poly.reserve(static_cast<std::size_t>(count) + 4);
    for (int i = 0; i < count; ++i)
        poly.emplace_back(pts[i * 2 + 0], pts[i * 2 + 1]);

    // Four half-planes: x >= -1, x <= 1, y >= -1, y <= 1. `inside` and the intersection
    // parameter are the only things that differ between them, so the loop body is written once.
    for (int edge = 0; edge < 4 && !poly.empty(); ++edge) {
        const auto inside = [&](const std::pair<float, float>& p) {
            switch (edge) {
                case 0: return p.first >= -1.0f;
                case 1: return p.first <= 1.0f;
                case 2: return p.second >= -1.0f;
                default: return p.second <= 1.0f;
            }
        };
        const auto cross = [&](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            const float bound = (edge == 0 || edge == 2) ? -1.0f : 1.0f;
            if (edge < 2) {
                const float t = (bound - a.first) / (b.first - a.first);
                return std::make_pair(bound, a.second + t * (b.second - a.second));
            }
            const float t = (bound - a.second) / (b.second - a.second);
            return std::make_pair(a.first + t * (b.first - a.first), bound);
        };

        std::vector<std::pair<float, float>> out;
        out.reserve(poly.size() + 1);
        for (std::size_t i = 0; i < poly.size(); ++i) {
            const auto& cur  = poly[i];
            const auto& prev = poly[(i + poly.size() - 1) % poly.size()];
            const bool  ci   = inside(cur);
            const bool  pi   = inside(prev);
            if (ci) {
                if (!pi)
                    out.push_back(cross(prev, cur));
                out.push_back(cur);
            } else if (pi) {
                out.push_back(cross(prev, cur));
            }
        }
        poly.swap(out);
    }
    return poly;
}

std::optional<std::pair<double, double>> compute_plane_point(const previz_camera& view,
                                                             double               aspect,
                                                             double               sx,
                                                             double               sy,
                                                             const screen_meta&   meta)
{
    const auto   cam_rot = mat4::rotate_y(view.yaw) * mat4::rotate_x(view.pitch) * mat4::rotate_z(view.roll);
    const double rgt[3]  = {cam_rot.m[0], cam_rot.m[1], cam_rot.m[2]};
    const double up[3]   = {cam_rot.m[4], cam_rot.m[5], cam_rot.m[6]};
    const double fwd[3]  = {-cam_rot.m[8], -cam_rot.m[9], -cam_rot.m[10]};
    const double half_v  = std::tan(view.fov * 0.5 * M_PI / 180.0);
    const double xc      = (2.0 * sx - 1.0) * aspect * half_v;
    const double yc      = (1.0 - 2.0 * sy) * half_v;
    double       dir[3];
    for (int i = 0; i < 3; ++i)
        dir[i] = fwd[i] + rgt[i] * xc + up[i] * yc;

    // The screen's own basis, as `compute_pick` builds it -- same rotation order, same columns.
    const auto   rot     = mat4::rotate_y(meta.rot_yaw) * mat4::rotate_x(meta.rot_pitch) *
                     mat4::rotate_z(meta.rot_roll);
    const double s_rgt[3] = {rot.m[0], rot.m[1], rot.m[2]};
    const double s_up[3]  = {rot.m[4], rot.m[5], rot.m[6]};
    const double s_nrm[3] = {rot.m[8], rot.m[9], rot.m[10]};

    double denom = 0.0;
    for (int i = 0; i < 3; ++i)
        denom += dir[i] * s_nrm[i];
    if (std::abs(denom) < 1e-9)
        return std::nullopt;

    const double origin[3] = {meta.pos_x, meta.pos_y, meta.pos_z};
    double       num       = 0.0;
    for (int i = 0; i < 3; ++i)
        num += (origin[i] - (i == 0 ? view.x : i == 1 ? view.y : view.z)) * s_nrm[i];
    const double t = num / denom;
    if (t <= 0.0)
        return std::nullopt;

    double lx = 0.0, ly = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double eye = i == 0 ? view.x : i == 1 ? view.y : view.z;
        const double d   = eye + dir[i] * t - origin[i];
        lx += d * s_rgt[i];
        ly += d * s_up[i];
    }
    return std::make_pair(lx, ly);
}

double ring_radius(const screen_meta& meta)
{
    // OUTSIDE THE WHOLE SURFACE, not merely outside its width -- and the distance that matters
    // is to a TOP CORNER, because every rotation here turns about the origin, which is
    // centre-BOTTOM.
    //
    // This was `max(w, h) * 0.6` while the yaw ring was the only one, and that was fine: the yaw
    // ring lies in the floor, which a wall standing on it does not cover. The ROLL ring lies in
    // the screen's OWN PLANE. At the old radius it ran straight across the picture the operator
    // is judging -- and, because the handles are tested before the pick, it swallowed the click
    // that moves a screen: the centre of a 2 x 2 m wall is 1.0 m from its origin, and the old
    // band was 1.2 +- 0.216. Found by a battery arm that had been passing for a day.
    //
    // The inner edge is at `radius * (1 - ring_band_fraction)`, so clearing the corner handles
    // needs `radius > corner / (1 - f)`; the extra 10% is the visible gap between the outermost
    // handle and the ring.
    const double corner = std::hypot(static_cast<double>(meta.width_m) * 0.5,
                                     static_cast<double>(meta.height_m));
    return std::max(0.6, corner * 1.1 / (1.0 - ring_band_fraction));
}

namespace {

/// Apply a `mat4`'s rotation part to a 3-vector. Column-major, so column `c` starts at `m[c*4]`.
void rot_apply(const mat4& m, const double v[3], double out[3])
{
    for (int r = 0; r < 3; ++r)
        out[r] = static_cast<double>(m.m[0 * 4 + r]) * v[0] + static_cast<double>(m.m[1 * 4 + r]) * v[1] +
                 static_cast<double>(m.m[2 * 4 + r]) * v[2];
}

double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

} // namespace

ring_basis ring_frame(const screen_meta& meta, ring_axis axis)
{
    // The PARENT frame is what this Euler component rotates WITHIN, and it is the whole reason
    // the three rings are not one function with a field selector. The stage composes
    // Ry(yaw) * Rx(pitch) * Rz(roll), so yaw turns in the world, pitch turns inside the yaw, and
    // roll turns inside both. A pitch ring drawn in the world's XZ plane is right only while yaw
    // is zero -- which is the state a fresh stage is in, and therefore the state a check written
    // without thinking about this would be written in.
    mat4   parent = mat4::identity();
    double n_local[3]{};
    double base[3]{};
    mat4   quarter = mat4::identity();
    switch (axis) {
        case ring_axis::yaw:
            n_local[1] = 1.0;
            base[2]    = 1.0; // a screen at yaw 0 faces local +Z
            quarter    = mat4::rotate_y(90.0f);
            break;
        case ring_axis::pitch:
            parent     = mat4::rotate_y(meta.rot_yaw);
            n_local[0] = 1.0;
            base[2]    = 1.0;
            quarter    = mat4::rotate_x(90.0f);
            break;
        case ring_axis::roll:
            parent     = mat4::rotate_y(meta.rot_yaw) * mat4::rotate_x(meta.rot_pitch);
            n_local[2] = 1.0;
            base[0]    = 1.0; // roll is measured from the screen's own +X, its width axis
            quarter    = mat4::rotate_z(90.0f);
            break;
    }

    // `b` is where the base direction ENDS UP at +90 degrees of this component, taken from the
    // matrix itself rather than written out. Writing it by hand is where a rotation handle picks
    // up the sign error that makes it turn the wrong way -- and a handle that turns the wrong way
    // is worse than no handle, because the operator compensates and then meets a screen that was
    // right all along.
    double q_local[3];
    rot_apply(quarter, base, q_local);

    ring_basis rb;
    rot_apply(parent, n_local, rb.n);
    rot_apply(parent, base, rb.a);
    rot_apply(parent, q_local, rb.b);
    return rb;
}

std::optional<ring_hit> compute_ring_hit(const previz_camera& view,
                                         double               aspect,
                                         double               sx,
                                         double               sy,
                                         const screen_meta&   meta,
                                         double               band,
                                         ring_axis            axis)
{
    // The same ray `compute_pick` builds, and the comment there explains the y flip.
    const auto   cam_rot = mat4::rotate_y(view.yaw) * mat4::rotate_x(view.pitch) * mat4::rotate_z(view.roll);
    const double rgt[3]  = {cam_rot.m[0], cam_rot.m[1], cam_rot.m[2]};
    const double up[3]   = {cam_rot.m[4], cam_rot.m[5], cam_rot.m[6]};
    const double fwd[3]  = {-cam_rot.m[8], -cam_rot.m[9], -cam_rot.m[10]};
    const double half_v  = std::tan(view.fov * 0.5 * M_PI / 180.0);
    const double xc      = (2.0 * sx - 1.0) * aspect * half_v;
    const double yc      = (1.0 - 2.0 * sy) * half_v;
    double       dir[3];
    for (int i = 0; i < 3; ++i)
        dir[i] = fwd[i] + rgt[i] * xc + up[i] * yc;

    const ring_basis rb = ring_frame(meta, axis);

    // A ring seen EDGE-ON is refused, and that is a feature rather than a guard against dividing
    // by zero. As the view approaches a ring's plane, one pixel of pointer movement sweeps an
    // unbounded distance across it, so the angle the handle reports stops tracking the hand
    // holding it. Every 3D tool drops a gizmo ring at grazing incidence for this reason.
    // 0.09 is about five degrees.
    const double dlen = std::sqrt(dot3(dir, dir));
    const double dn   = dot3(dir, rb.n);
    if (dlen < 1e-12 || std::abs(dn) / dlen < 0.09)
        return std::nullopt;

    // Centred on the screen's ORIGIN -- centre-bottom, the point every rotation in this model
    // turns about, so the handle turns what it looks like it turns.
    const double o[3] = {view.x - meta.pos_x, view.y - meta.pos_y, view.z - meta.pos_z};
    const double t    = -dot3(o, rb.n) / dn;
    if (t <= 0.0)
        return std::nullopt;

    double p[3];
    for (int i = 0; i < 3; ++i)
        p[i] = o[i] + dir[i] * t;

    const double pa = dot3(p, rb.a);
    const double pb = dot3(p, rb.b);
    const double r  = std::sqrt(pa * pa + pb * pb);
    const double rr = ring_radius(meta);
    if (std::abs(r - rr) > band)
        return std::nullopt;

    // Measured against the ring's OWN basis, so the angle is in the same sense as the field it
    // drives. For yaw that works out to atan2(x, z), because `mat4::rotate_y` maps local +Z
    // towards +X as the angle grows; writing that as atan2(z, x) would be 90 degrees out and
    // mirrored.
    ring_hit out;
    out.angle_deg    = std::atan2(pb, pa) * 180.0 / M_PI;
    out.radius       = rr;
    out.axis         = axis;
    out.radial_error = std::abs(r - rr);
    return out;
}

std::optional<ring_hit> compute_ring_grab(const previz_camera& view,
                                          double               aspect,
                                          double               sx,
                                          double               sy,
                                          const screen_meta&   meta,
                                          double               band)
{
    std::optional<ring_hit> best;
    for (const auto axis : {ring_axis::yaw, ring_axis::pitch, ring_axis::roll}) {
        const auto h = compute_ring_hit(view, aspect, sx, sy, meta, band, axis);
        if (h && (!best || h->radial_error < best->radial_error))
            best = h;
    }
    return best;
}

previz_camera frame_view(const previz_camera& view, double aspect, const world_box& box)
{
    previz_camera out = view;
    if (!box.valid)
        return out; // nothing to frame: leave the operator where they were, rather than at 0,0,0

    double c[3];
    double diag = 0.0;
    for (int i = 0; i < 3; ++i) {
        c[i] = (box.lo[i] + box.hi[i]) * 0.5;
        const double e = box.hi[i] - box.lo[i];
        diag += e * e;
    }
    // A floor of 25 cm, because a stage contains DEGENERATE boxes as a matter of course: a flat
    // screen seen edge-on has zero thickness, and a single point has no size at all. Without it
    // the camera lands exactly on the thing it was asked to show.
    const double radius = std::max(0.25, 0.5 * std::sqrt(diag));

    // The tighter of the two half-angles, so the sphere fits the NARROW axis of the frame. `sin`
    // rather than `tan`: the limit is where the frustum's side plane is TANGENT to the sphere,
    // and `tan` would use the distance to the sphere's centre as though it were the edge --
    // framing a little too tight, by more the wider the lens.
    const double half_v = view.fov * 0.5 * M_PI / 180.0;
    const double half_h = std::atan(std::tan(half_v) * std::max(aspect, 1e-6));
    const double half   = std::min(half_v, half_h);
    const double d      = radius / std::max(std::sin(half), 1e-6) * 1.1; // 10% air around it

    const auto   rot    = mat4::rotate_y(view.yaw) * mat4::rotate_x(view.pitch) * mat4::rotate_z(view.roll);
    const double fwd[3] = {-rot.m[8], -rot.m[9], -rot.m[10]};

    out.x = static_cast<float>(c[0] - fwd[0] * d);
    out.y = static_cast<float>(c[1] - fwd[1] * d);
    out.z = static_cast<float>(c[2] - fwd[2] * d);
    return out;
}

std::optional<std::array<float, 3>> view_orientation(const std::string& name)
{
    // yaw, pitch, roll. FRONT is the identity, which is where a screen's own +Z faces, so the
    // default stage is seen face-on. TOP looks down; SIDE stands on +X looking towards -X.
    if (boost::iequals(name, "FRONT"))
        return std::array<float, 3>{0.0f, 0.0f, 0.0f};
    if (boost::iequals(name, "TOP"))
        return std::array<float, 3>{0.0f, -90.0f, 0.0f};
    if (boost::iequals(name, "SIDE"))
        return std::array<float, 3>{90.0f, 0.0f, 0.0f};
    return std::nullopt;
}

std::optional<double>
ray_triangle(const double eye[3], const double dir[3], const float* a, const float* b, const float* c)
{
    const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};

    const double p[3] = {dir[1] * e2[2] - dir[2] * e2[1],
                         dir[2] * e2[0] - dir[0] * e2[2],
                         dir[0] * e2[1] - dir[1] * e2[0]};
    const double det  = dot3(e1, p);
    // No back-face rejection -- see the header. Only a ray PARALLEL to the triangle is refused,
    // and the epsilon is relative to nothing because `dir` is normalised by the caller.
    if (std::abs(det) < 1e-12)
        return std::nullopt;

    const double inv = 1.0 / det;
    const double tv[3] = {eye[0] - a[0], eye[1] - a[1], eye[2] - a[2]};
    const double u     = dot3(tv, p) * inv;
    if (u < 0.0 || u > 1.0)
        return std::nullopt;

    const double q[3] = {tv[1] * e1[2] - tv[2] * e1[1],
                         tv[2] * e1[0] - tv[0] * e1[2],
                         tv[0] * e1[1] - tv[1] * e1[0]};
    const double v    = dot3(dir, q) * inv;
    if (v < 0.0 || u + v > 1.0)
        return std::nullopt;

    const double t = dot3(e2, q) * inv;
    if (t <= 0.0)
        return std::nullopt; // behind the eye
    return t;
}

std::optional<mesh_hit> compute_mesh_pick(const previz_camera&          view,
                                          double                        aspect,
                                          double                        sx,
                                          double                        sy,
                                          const std::vector<pick_mesh>& meshes)
{
    const auto   cam_rot = mat4::rotate_y(view.yaw) * mat4::rotate_x(view.pitch) * mat4::rotate_z(view.roll);
    const double rgt[3]  = {cam_rot.m[0], cam_rot.m[1], cam_rot.m[2]};
    const double up[3]   = {cam_rot.m[4], cam_rot.m[5], cam_rot.m[6]};
    const double fwd[3]  = {-cam_rot.m[8], -cam_rot.m[9], -cam_rot.m[10]};
    const double half_v  = std::tan(view.fov * 0.5 * M_PI / 180.0);
    const double xc      = (2.0 * sx - 1.0) * aspect * half_v;
    const double yc      = (1.0 - 2.0 * sy) * half_v;

    double dir[3];
    for (int i = 0; i < 3; ++i)
        dir[i] = fwd[i] + rgt[i] * xc + up[i] * yc;

    // NORMALISED here, unlike in `compute_pick`, which scales its `t` at the end instead. The
    // difference matters: this function compares thousands of `t` values against each other
    // before it returns one, and doing that in units of an un-normalised direction would be
    // correct only because every comparison shares the same scale -- true today and a trap for
    // anyone who later compares one of these against a `pick::distance`, which IS in metres.
    const double dlen = std::sqrt(dot3(dir, dir));
    if (dlen < 1e-12)
        return std::nullopt;
    for (auto& d : dir)
        d /= dlen;

    const double eye[3] = {view.x, view.y, view.z};

    std::optional<mesh_hit> best;
    for (const auto& m : meshes) {
        if (!m.xyz || !m.name || m.vertex_count < 3)
            continue;
        const std::size_t tris = m.vertex_count / 3;
        for (std::size_t i = 0; i < tris; ++i) {
            const float* a = m.xyz + (i * 3 + 0) * m.stride_floats;
            const float* b = m.xyz + (i * 3 + 1) * m.stride_floats;
            const float* c = m.xyz + (i * 3 + 2) * m.stride_floats;
            const auto   t = ray_triangle(eye, dir, a, b, c);
            if (!t || (best && *t >= best->distance))
                continue;
            best = mesh_hit{*m.name, *t, eye[0] + dir[0] * *t, eye[1] + dir[1] * *t, eye[2] + dir[2] * *t};
        }
    }
    return best;
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
