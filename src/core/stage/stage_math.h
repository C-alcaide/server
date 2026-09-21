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

#include <algorithm>
#include <array>
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

/// How much of a frustum quad lands ON the screen: the clipped area over the whole area, 0..1.
///
/// THE OPERATOR'S QUESTION, and the one previz could not answer. `icvfx_q` is where the camera's
/// frustum meets this screen's PLANE, which is unbounded -- so a lens wider than the wall, or a
/// camera panned past its edge, puts corners outside the raster and part of the shot is simply
/// not on LED. 1.0 is entirely on the wall; 0.0 is entirely off it; 0.6 means two fifths of the
/// frame is looking at whatever is behind the screen.
///
/// Derived from `clip_to_unit_rect` rather than from a second intersection, for the reason the
/// footprint overlay gives: the quad the SHADER masks with is the only one whose answer an
/// operator can act on, and a second implementation is free to drift from it.
///
/// Returns 1.0 for a degenerate quad (zero area), because "no frustum" is not "off the wall" --
/// reporting 0 there would raise an alarm on a screen that simply has ICVFX switched off.
double quad_coverage(const float q[8]);

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
/// Which Euler component a rotation ring drives.
///
/// The three are NOT interchangeable, because the stage composes `Ry(yaw) * Rx(pitch) *
/// Rz(roll)`: yaw turns about WORLD +Y, pitch about the yaw-turned +X, and roll about the
/// yaw-then-pitch-turned +Z. So each ring lies in a different plane, and the plane of the
/// second and third depends on the value of the ones outside them. A single ring re-used for
/// all three by swapping which field it writes would be correct only while the other two are
/// zero -- which is exactly the state a stage is in before anyone has used it, and so exactly
/// the state a check would be written in.
enum class ring_axis
{
    yaw = 0,
    pitch,
    roll
};

struct ring_hit
{
    double    angle_deg    = 0.0; ///< where on the ring, in the same sense as the axis's own field
    double    radius       = 0.0; ///< the ring's own radius, in metres
    ring_axis axis         = ring_axis::yaw; ///< which ring, once `compute_ring_grab` has chosen
    double    radial_error = 0.0; ///< |hit radius - ring radius|, which is how that choice is made
};

/// A ring's PLANE, as a normal and the two in-plane axes its angle is measured against: `a` is
/// where the axis's own field reads zero, and `b` is where it reads +90 degrees.
///
/// Public for the same reason `ring_radius` is: the renderer draws the circle and the hit-test
/// accepts it, and two derivations of one plane is a handle you can see and cannot grab. The
/// basis is derived by APPLYING `mat4::rotate_*` rather than by writing out sines -- so a sign
/// convention in the matrix and a sign convention here cannot disagree, because there is only
/// one of them.
struct ring_basis
{
    double n[3]{}; ///< the axis the ring turns about, in world space
    double a[3]{}; ///< in-plane, at the field's zero
    double b[3]{}; ///< in-plane, at the field's +90 degrees
};

ring_basis ring_frame(const screen_meta& meta, ring_axis axis);

/// `band` is the half-width of the annulus that counts as a hit, in metres. A handle needs a
/// tolerance an operator can actually hit with a mouse; the caller sizes it from the ring.
std::optional<ring_hit> compute_ring_hit(const previz_camera& view,
                                         double               aspect,
                                         double               sx,
                                         double               sy,
                                         const screen_meta&   meta,
                                         double               band,
                                         ring_axis            axis);

/// The ring the pointer is actually on, of the three -- NEAREST rather than first-match.
///
/// Three great circles of one sphere cross at six points, so "the first that accepts" would hand
/// those six to whichever axis the loop happened to test first. Nearest is the only answer that
/// does not depend on an ordering the operator cannot see.
std::optional<ring_hit> compute_ring_grab(const previz_camera& view,
                                          double               aspect,
                                          double               sx,
                                          double               sy,
                                          const screen_meta&   meta,
                                          double               band);

/// How wide the grabbable annulus is, as a fraction of the ring's radius. Shared because
/// `ring_radius` is DERIVED from it: the ring has to sit far enough out that its inner edge
/// clears the screen's own handles, and that distance is a function of how wide the band is.
constexpr double ring_band_fraction = 0.18;

/// The ring's radius for a screen: outside everything else the screen draws, small enough to
/// stay in frame. One function so the renderer draws exactly what the hit-test accepts -- two
/// copies of this number is a handle you can see and cannot grab.
double ring_radius(const screen_meta& meta);

/// Round `value` to the nearest multiple of `step`. A step of zero or less is OFF and returns
/// the value untouched.
///
/// The zero case is the whole reason this is a function rather than one line at each call site:
/// `std::round(v / 0.0) * 0.0` is NaN, and a NaN position does not throw, does not log, and
/// makes a screen disappear from the viewport with the tree still reporting a number. "Off" has
/// to be a value the caller can hold, because that is what an operator with snapping disabled
/// has, and it must travel through the same code as any other step.
double snap_to(double value, double step);

/// An axis-aligned box in world space, grown a point at a time.
///
/// `valid` rather than a sentinel extent: an EMPTY box and a box around a single point are
/// different things, and a caller framing a stage with nothing on it must be able to tell them
/// apart rather than fly somewhere derived from infinities.
struct world_box
{
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
    bool   valid = false;

    void add(double x, double y, double z)
    {
        const double p[3] = {x, y, z};
        for (int i = 0; i < 3; ++i) {
            lo[i] = valid ? std::min(lo[i], p[i]) : p[i];
            hi[i] = valid ? std::max(hi[i], p[i]) : p[i];
        }
        valid = true;
    }
};

/// Move a camera along ITS OWN view direction until `box` fits the frame, keeping its rotation.
///
/// Keeping the rotation is the decision. "Frame the selection" in every 3D application moves the
/// eye and leaves the angle alone, because the operator chose that angle; a version that also
/// pointed the camera somewhere canonical would be two commands welded together, and the second
/// of them is `VIEW TOP` / `FRONT` / `SIDE`.
///
/// The distance comes from the box's bounding SPHERE, so it does not change as the camera orbits
/// -- a framing derived from the projected extent would breathe as the box turned, and a stage is
/// full of long thin walls where that is most of a metre.
previz_camera frame_view(const previz_camera& view, double aspect, const world_box& box);

/// The canonical viewport orientations, as (yaw, pitch, roll). `std::nullopt` for a name that is
/// not one -- which is the whole reason this returns an optional rather than defaulting to FRONT:
/// a typo that silently selected an orientation is this tree's most repeated defect.
std::optional<std::array<float, 3>> view_orientation(const std::string& name);

/// One candidate surface for `compute_mesh_pick`, read IN PLACE.
///
/// A venue model is tens of thousands of triangles and the renderer already holds them
/// interleaved with normals and UVs; copying them out per click would cost more than the pick.
/// So this is a view: a pointer at the first position, how many vertices follow, and how far
/// apart they are in floats. `vertex_count` is a multiple of three -- an expanded triangle list,
/// which is what both loaders produce and what the draw call consumes.
struct pick_mesh
{
    const std::string* name          = nullptr;
    const float*       xyz           = nullptr;
    std::size_t        vertex_count  = 0;
    std::size_t        stride_floats = 3;
    /// The mesh's own pose, applied to its vertices before they are seen. Yaw/pitch/roll in
    /// DEGREES, composed `Ry*Rx*Rz` like everything else on this stage.
    ///
    /// The RAY is transformed into mesh space rather than the triangles into world space, which
    /// is the difference between six multiplies per pick and six per TRIANGLE. Both give the
    /// same answer here only because the pose is rigid -- rotation and translation, no scale --
    /// so distances are preserved and `t` means the same on both sides.
    double             pos[3] = {0.0, 0.0, 0.0};
    double             rot[3] = {0.0, 0.0, 0.0};
};

struct mesh_hit
{
    std::string name;
    double      distance = 0.0;              ///< metres from the eye
    double      x = 0.0, y = 0.0, z = 0.0;   ///< where, in world space
};

/// Moller-Trumbore, TWO-SIDED, returning the ray parameter in units of `dir`.
///
/// Two-sided deliberately: a venue model is a room, and an operator looking around from inside it
/// meets every wall from behind. Back-face culling is the natural-looking choice that makes the
/// walls of a volume unclickable from the only place anyone stands.
std::optional<double>
ray_triangle(const double eye[3], const double dir[3], const float* a, const float* b, const float* c);

/// The nearest triangle under a viewport point, across every candidate mesh.
///
/// Separate from `compute_pick` rather than folded into it, because the two answer different
/// questions: a SCREEN is a rectangle with local coordinates a gesture can drag by, and a venue
/// mesh is geometry with a name. The caller compares the two distances -- see the renderer --
/// which is the only way a wall standing in front of a screen can occlude it.
std::optional<mesh_hit> compute_mesh_pick(const previz_camera&         view,
                                          double                       aspect,
                                          double                       sx,
                                          double                       sy,
                                          const std::vector<pick_mesh>& meshes);

std::optional<pick> compute_pick(const previz_camera&                      view,
                                 double                                     aspect,
                                 double                                     sx,
                                 double                                     sy,
                                 const std::map<std::string, screen_meta>& screens);

// ── Hierarchy ───────────────────────────────────────────────────────────────────────────
//
// A screen may be parented to another, so that moving the parent moves the child with it: a
// wall on a truss, a set of panels that travel as one piece. Every stage visualiser with a
// hierarchy stores the child's pose RELATIVE to its parent and derives the world pose, and
// this does the same -- see `screen_meta::parent`.
//
// THE COMPOSITION IS DONE IN MATRICES AND HANDED BACK AS EULER ANGLES, because the rest of
// this file is written in the `Ry(yaw) * Rx(pitch) * Rz(roll)` convention and forty-odd call
// sites read `rot_yaw` directly. Resolving once, at the point the scene is assembled, is what
// lets all of them stay as they are -- the alternative is teaching every reader the hierarchy,
// which is this tree's seven-readers trap at a larger scale.

/// The composed rotation of `parent` applied to `child`, back in the Ry*Rx*Rz convention.
///
/// Degrees in and out. The decomposition is exact for the composition above it: from
/// `R = Ry(a)Rx(b)Rz(c)` the elements give `b = asin(-R12)`, `c = atan2(R10, R11)` and
/// `a = atan2(R02, R22)`, and at `cos(b) == 0` -- the gimbal case, pitch straight up or down --
/// roll and yaw are not separable, so roll is taken as 0 and yaw carries the whole turn.
void compose_rotation(const float parent_ypr[3], const float child_ypr[3], float out_ypr[3]);

/// `child` expressed in the stage's frame, given its parent's WORLD pose.
///
/// The child's local offset is rotated by the parent's rotation before being added, which is
/// what makes a parented screen ORBIT its parent rather than slide beside it.
void compose_pose(const float parent_pos[3],
                  const float parent_ypr[3],
                  const float child_pos[3],
                  const float child_ypr[3],
                  float       out_pos[3],
                  float       out_ypr[3]);

/// Every screen with its pose resolved into the stage's frame.
///
/// A CYCLE IS BROKEN RATHER THAN HUNG: a chain is followed at most `screens.size()` deep, so a
/// loop that survived the command's own check -- or one loaded from a hand-edited scene file --
/// costs a wrong position and not a frozen render thread. The command refuses to create one; this
/// is the second line, because a renderer is not the place to discover a malformed document.
std::map<std::string, screen_meta> resolve_world_poses(const std::map<std::string, screen_meta>& screens);

/// ONE screen with its pose resolved into the stage's frame -- the primitive the map version is
/// built on, and the one every call site uses.
///
/// Walking a single chain is cheap (a screen has one parent, and a rig is two or three deep), so
/// this is called per draw rather than cached. A cache would need every mutation of the screen
/// map to invalidate it, and finding all of those is the audit that this design exists to avoid.
/// Returns the screen unchanged when it has no parent, and when the name is unknown.
screen_meta resolve_world_pose(const std::map<std::string, screen_meta>& screens,
                               const std::string&                        name);

/// Would making `child` a child of `parent` close a loop? Names, not poses.
///
/// True when `parent` is `child` itself or already descends from it. The command answers 400 on
/// this rather than accepting a document whose world poses cannot be resolved.
bool parent_would_cycle(const std::map<std::string, screen_meta>& screens,
                        const std::string&                        child,
                        const std::string&                        parent);

}} // namespace caspar::core
