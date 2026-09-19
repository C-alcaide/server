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

#include "stage_model.h"

#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// The stage's projection maths. Pure functions over plain structs -- no GL, no device, no I/O,
/// no global state.
///
/// It was already written that way, in an anonymous namespace inside `previz_renderer.cpp`, and
/// that is the whole reason this move is cheap. What it was not was TESTABLE: reaching
/// `compute_frustum` meant constructing a renderer, which means a `device`, which means a GL
/// context and a display. `78-client-test-plan.md` §4 gates this surface at 1 LSB on exactly the
/// pure-function tier that placement forfeited.
namespace caspar { namespace core {

/// Column-major 4x4, the layout OpenGL takes: `m[column * 4 + row]`.
///
/// Kept here rather than replaced with a library type: `compute_frustum` reads rotation COLUMNS
/// as basis vectors by index (`m[8],m[9],m[10]` is the screen normal), so the storage order is
/// load-bearing and a swap to a row-major type would silently transpose every axis.
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

/// The projection one screen needs, given where the production camera is.
///
/// Deliberately takes the camera by value-shaped reference rather than reading a scene: it is
/// called both from the auto-projection sweep and from the per-screen query, and neither should
/// be able to change it.
///
/// Two documented degenerate returns, both of which a caller sees as a default-constructed or
/// fallback projection rather than as an error:
///   * eye within 1e-6 m of the screen centre → a default `screen_projection`;
///   * eye in the screen PLANE (perpendicular distance ~0) → the total distance is substituted,
///     so the field of view stays finite.
screen_projection compute_frustum(const previz_camera& cam, const screen_meta& meta);

/// Which screen is under a point on the rendered viewport, and where on it.
///
/// `sx`/`sy` are normalised 0..1 in the DELIVERED IMAGE, top-left origin -- the same
/// convention as `core::input_event`. `aspect` is the viewport's width/height.
///
/// `local_x`/`local_y` are in the screen's own axes, in metres, with the origin where the
/// screen's own origin is: CENTRE-BOTTOM. So a hit dead centre reports `(0, height/2)`.
struct pick
{
    std::string name;
    double      distance = 0.0; //< metres from the eye along the ray
    double      local_x  = 0.0;
    double      local_y  = 0.0;
};

/// Ray-cast a viewport point against every screen and return the nearest hit.
///
/// PURE, like `compute_frustum` beside it, and for the same reason: it is the gate on every
/// interactive gesture -- click-to-select, drag-a-screen, a grabbable gizmo -- and none of that
/// is testable if reaching it needs a GL context. `78-client-test-plan.md` SS4 calls this the T1
/// tier.
///
/// The ray is built the way the renderer builds its projection, which is not the textbook form:
/// `previz_renderer` negates `proj.m[5]` to flip Y, because an FBO renders bottom-up and every
/// consumer downstream expects top-left origin. So a point at `sy = 0` -- the TOP of the
/// delivered image -- is +Y in camera space, and the mapping is `yc = (1 - 2*sy) * half_v`
/// rather than `(2*sy - 1)`. Getting that backwards picks the mirror-image screen, which looks
/// plausible on a symmetric stage and is why the self-test uses an asymmetric one.
///
/// The direction is then `forward + right * xc + up * yc`, the identical construction
/// `compute_frustum` uses for the ICVFX mask quad -- so the two cannot disagree about which way
/// the camera is looking.
///
/// A CURVED SCREEN IS TESTED AGAINST ITS CHORD PLANE, not its surface. That is the same
/// approximation `casparcg-360-client`'s `frustum_check.py` documents for the same reason, and
/// it means a pick near the edge of a deeply curved panel can be off by the sagitta. Named
/// here rather than discovered later.
/// Clip a convex polygon in NDC to the unit rectangle -- Sutherland-Hodgman, four half-planes.
///
/// `compute_frustum`'s `icvfx_q` is where the camera's frustum meets a screen's PLANE, and that
/// is unbounded: a camera whose frustum is wider than the wall produces corners well outside
/// +-1, which is correct geometry and useless as a picture. The mask shader clamps in NDC as a
/// matter of course; anything DRAWING the footprint has to clip, or it draws a quad running off
/// into the distance -- measured 2026-09-19, a 42-degree lens on a 6 m wall at 5.5 m.
///
/// Per-corner clamping is not the same thing and is wrong for a rotated quad: it drags corners
/// along the boundary and changes the shape. Clipping adds vertices instead, which is why the
/// result is up to eight points rather than four.
///
/// Input is x0,y0,x1,y1,... Empty when the polygon misses the rectangle entirely.
std::vector<std::pair<float, float>> clip_to_unit_rect(const float* pts, int count);

/// Where the pointer meets a screen's PLANE, in the screen's own axes -- with NO bounds test.
///
/// `compute_pick` answers "is the pointer on this screen", which means it rejects everything
/// outside the quad. That is right for selecting and wrong for RESIZING: a corner handle is
/// dragged OUTWARD, so the pointer spends the whole gesture off the screen it is resizing, and
/// a bounds-checked re-pick returns nothing exactly when it is needed. Measured 2026-09-19 --
/// the first resize handle grabbed correctly and then did nothing at all, because every frame
/// of the drag fell outside the old quad.
///
/// Returns local (x, y) in metres: x from the centre, y from the BOTTOM, the same axes
/// `pick::local_x` / `local_y` report. Empty when the ray is parallel to the plane or meets it
/// behind the eye.
std::optional<std::pair<double, double>> compute_plane_point(const previz_camera& view,
                                                             double               aspect,
                                                             double               sx,
                                                             double               sy,
                                                             const screen_meta&   meta);

/// THE ROTATION RING, as a pure question: where does the pointer meet the horizontal plane a
/// screen stands on, and is that on the screen's yaw ring?
///
/// Separate from `compute_pick` and deliberately so. A pick asks "which screen is under the
/// pointer" and answers with a surface; this asks "is the pointer on the HANDLE", which is a
/// different plane (horizontal, through the screen's base) and a different acceptance (an
/// annulus, not a rectangle). Folding them together would make the handle a property of the
/// screen's surface, and the handle is deliberately reachable where the screen is not -- that
/// is what a handle is for.
///
/// Returns the angle in DEGREES measured the same way `rot_yaw` is, so a drag is
/// `yaw0 + (angle_now - angle_grab)` with no conversion in between and no chance of the two
/// conventions drifting. Empty when the ray runs parallel to the plane, meets it behind the
/// eye, or lands outside the ring's band.
struct ring_hit
{
    double angle_deg = 0.0; ///< where on the ring, in the same sense as `screen_meta::rot_yaw`
    double radius    = 0.0; ///< the ring's own radius, in metres
};

/// `band` is the half-width of the annulus that counts as a hit, in metres. A handle needs a
/// tolerance an operator can actually hit with a mouse; the caller sizes it from the ring.
std::optional<ring_hit> compute_ring_hit(const previz_camera& view,
                                         double               aspect,
                                         double               sx,
                                         double               sy,
                                         const screen_meta&   meta,
                                         double               band);

/// The ring's radius for a screen: big enough to clear the surface, small enough to stay in
/// frame. One function so the renderer draws exactly what the hit-test accepts -- two copies of
/// this number is a handle you can see and cannot grab.
double ring_radius(const screen_meta& meta);

std::optional<pick> compute_pick(const previz_camera&                      view,
                                 double                                     aspect,
                                 double                                     sx,
                                 double                                     sy,
                                 const std::map<std::string, screen_meta>& screens);

}} // namespace caspar::core
