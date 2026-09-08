/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "stage_math_self_test.h"

#include "stage_math.h"

#include <common/log.h>
#include <common/utf.h>

#include <cmath>
#include <cstdio>
#include <map>
#include <string>

namespace caspar { namespace core { namespace fields {

namespace {

constexpr double PI = 3.14159265358979323846;

double deg(double r) { return r * 180.0 / PI; }
double rad(double d) { return d * PI / 180.0; }

struct vec3
{
    double x = 0, y = 0, z = 0;
};

vec3 operator+(const vec3& a, const vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
vec3 operator-(const vec3& a, const vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
vec3 operator*(const vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double len(const vec3& a) { return std::sqrt(dot(a, a)); }

/// The camera's world basis, written as CLOSED-FORM TRIGONOMETRY rather than as a product of
/// `mat4` rotations.
///
/// That is the entire point of this function. `compute_frustum` builds the same basis by
/// multiplying three `mat4`s and reading columns out by index, so re-using `mat4` here would
/// compare the implementation against itself and pass for any consistent-but-wrong rotation
/// order. These expansions of Ry(yaw)*Rx(pitch)*Rz(roll) are a second derivation, and they are
/// deliberately the SAME expressions `casparcg-360-client`'s `frustum_check.py` uses -- a
/// pure-numpy, Qt-free implementation of this geometry that was written independently. Agreeing
/// with it here is what makes the client's local previz and the server's mapping one description
/// by construction rather than by discipline.
void camera_basis(const previz_camera& cam, vec3& right, vec3& up, vec3& forward)
{
    const double cy = std::cos(rad(cam.yaw)), sy = std::sin(rad(cam.yaw));
    const double cp = std::cos(rad(cam.pitch)), sp = std::sin(rad(cam.pitch));
    const double cr = std::cos(rad(cam.roll)), sr = std::sin(rad(cam.roll));

    right   = {cy * cr + sy * sp * sr, cp * sr, -sy * cr + cy * sp * sr};
    up      = {-cy * sr + sy * sp * cr, cp * cr, sy * sr + cy * sp * cr};
    forward = vec3{sy * cp, -sp, cy * cp} * -1.0;
}

/// The screen's basis, likewise closed-form: columns of Ry(yaw)*Rx(pitch)*Rz(roll).
void screen_basis(const screen_meta& s, vec3& right, vec3& up, vec3& normal)
{
    const previz_camera as_angles{0, 0, 0, s.rot_yaw, s.rot_pitch, s.rot_roll, 0, 0, 0};
    vec3                f;
    camera_basis(as_angles, right, up, f);
    normal = f * -1.0; // forward is -col2; the screen normal IS +col2
}

vec3 screen_centre(const screen_meta& s)
{
    vec3 r, u, n;
    screen_basis(s, r, u, n);
    return vec3{s.pos_x, s.pos_y, s.pos_z} + u * (s.height_m * 0.5);
}

} // namespace

stage_math_report stage_math_self_test()
{
    stage_math_report rep;

    // Angles in degrees, NDC in [-1,1]. Both are float computations in the implementation, so
    // the tolerance is set by float precision over the magnitudes involved rather than by any
    // display quantity -- 1e-3 deg is roughly four orders tighter than 1 LSB of an 8-bit
    // encoding of a 180 deg range, so a real convention error cannot hide under it.
    const double ANG_TOL = 1e-3;
    const double NDC_TOL = 1e-4;

    auto check = [&](const char* name, double got, double want, double tol) {
        ++rep.checks;
        if (std::isfinite(got) && std::abs(got - want) <= tol)
            return;
        char buf[192];
        std::snprintf(buf, sizeof(buf), "%s: got %.6f want %.6f (tol %.0e)", name, got, want, tol);
        rep.failures.emplace_back(buf);
    };

    // A 2 m x 2 m flat screen at the origin, facing +Z. Origin is centre-BOTTOM, so its centre
    // is at (0, 1, 0).
    auto flat = [] {
        screen_meta s;
        s.name     = "s";
        s.width_m  = 2.0f;
        s.height_m = 2.0f;
        return s;
    };

    // ---- 1. A screen facing the camera projects at yaw 0 ------------------------------
    {
        const previz_camera cam{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, flat());
        check("front/yaw", p.yaw_deg, 0.0, ANG_TOL);
        check("front/pitch", p.pitch_deg, 0.0, ANG_TOL);
        check("front/roll", p.roll_deg, 0.0, ANG_TOL);

        // FOV is the angular height of the screen from the eye, independently: the eye is 5 m
        // from a centre 1 m above the origin, on the normal, so perp == 5.
        check("front/fov", p.fov_deg, deg(2.0 * std::atan2(1.0, 5.0)), ANG_TOL);
    }

    // ---- 2. Rotating the screen about Y rotates the projection with it ----------------
    for (double yaw : {90.0, -90.0, 37.0, 180.0}) {
        auto s     = flat();
        s.rot_yaw  = static_cast<float>(yaw);
        const previz_camera cam{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, s);

        // atan2 wraps at +/-180, so 180 comes back as either sign; compare on the circle.
        double d = p.yaw_deg - yaw;
        while (d > 180.0)
            d -= 360.0;
        while (d < -180.0)
            d += 360.0;
        check("rotated/yaw", d, 0.0, ANG_TOL);
    }

    // ---- 3. Pitching the screen pitches the projection --------------------------------
    {
        auto s      = flat();
        s.rot_pitch = 30.0f;
        const previz_camera cam{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, s);
        check("pitched/pitch", p.pitch_deg, 30.0, ANG_TOL);
    }

    // ---- 4. FOV follows the PERPENDICULAR distance, computed independently -------------
    for (double z : {2.0, 5.0, 12.5}) {
        auto s      = flat();
        s.rot_yaw   = 20.0f; // off-axis, so perpendicular and total distance differ
        const previz_camera cam{1.0f, 1.0f, static_cast<float>(z), 0, 0, 0, 60.0f, 0.1f, 100.0f};

        vec3 sr, su, sn;
        screen_basis(s, sr, su, sn);
        const vec3   d    = screen_centre(s) - vec3{cam.x, cam.y, cam.z};
        const double perp = std::abs(dot(d, sn));
        const double want = std::min(170.0, std::max(1.0, deg(2.0 * std::atan2(1.0, perp))));

        const auto p = compute_frustum(cam, s);
        check("offaxis/fov", p.fov_deg, want, ANG_TOL);
    }

    // ---- 5. Degenerate: the eye AT the screen centre returns the default ---------------
    {
        auto                s = flat();
        const previz_camera cam{0.0f, 1.0f, 0.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, s);
        const screen_projection def{};
        check("degenerate/fov", p.fov_deg, def.fov_deg, 1e-6);
        check("degenerate/yaw", p.yaw_deg, def.yaw_deg, 1e-6);
        check("degenerate/curve", p.curve_type, def.curve_type, 0);
    }

    // ---- 6. Eye IN the screen plane substitutes the total distance ---------------------
    {
        auto s = flat();
        // Screen faces +Z with centre (0,1,0); an eye at (4,1,0) lies in the plane z == 0.
        const previz_camera cam{4.0f, 1.0f, 0.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, s);
        check("inplane/fov", p.fov_deg, deg(2.0 * std::atan2(1.0, 4.0)), ANG_TOL);
    }

    // ---- 7. eye_mode FIXED ignores the camera entirely ---------------------------------
    {
        auto s          = flat();
        s.eye_mode      = 1;
        s.design_eye_x  = 0.0f;
        s.design_eye_y  = 1.0f;
        s.design_eye_z  = 3.0f;

        const previz_camera near_cam{0.0f, 1.0f, 1.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const previz_camera far_cam{0.0f, 1.0f, 40.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const double        want = deg(2.0 * std::atan2(1.0, 3.0));
        check("fixedeye/fov/near", compute_frustum(near_cam, s).fov_deg, want, ANG_TOL);
        check("fixedeye/fov/far", compute_frustum(far_cam, s).fov_deg, want, ANG_TOL);
    }

    // ---- 8. Curve classification and the eye-distance ratio k --------------------------
    {
        const previz_camera cam{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};

        auto flat_s = flat();
        check("curve/flat", compute_frustum(cam, flat_s).curve_type, 0, 0);

        auto cyl      = flat();
        cyl.radius_m  = 4.0f;
        cyl.arc_deg   = 60.0f;
        const auto pc = compute_frustum(cam, cyl);
        check("curve/cylinder", pc.curve_type, 1, 0);
        check("curve/arc_h", pc.screen_arc_deg, 60.0, 1e-4);
        check("curve/k", pc.eye_distance, 5.0 / 4.0, 1e-5);

        auto sph       = cyl;
        sph.arc_v_deg  = 30.0f;
        const auto ps  = compute_frustum(cam, sph);
        check("curve/sphere", ps.curve_type, 2, 0);
        check("curve/arc_v", ps.screen_arc_v_deg, 30.0, 1e-4);
    }

    // ---- 9. ICVFX is off unless asked for, and the quad keeps its default --------------
    {
        const previz_camera cam{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, flat());
        const screen_projection def{};
        ++rep.checks;
        if (p.icvfx_enable)
            rep.failures.emplace_back("icvfx/off: enabled without being asked");
        for (int i = 0; i < 8; ++i)
            check("icvfx/off/quad", p.icvfx_q[i], def.icvfx_q[i], 1e-9);
    }

    // ---- 10. The ICVFX quad, intersected independently ---------------------------------
    //
    // The reference below is the same construction `frustum_check.py` performs, written out
    // in double precision: four corner rays from the closed-form camera basis, each meeting the
    // screen plane, each expressed in the screen's own axes. The implementation reaches the same
    // place through `mat4` products and float arithmetic, so agreement here is agreement between
    // two derivations rather than a tautology.
    {
        struct case_t
        {
            previz_camera cam;
            float         scr_yaw;
        };
        const case_t cases[] = {
            {{0.0f, 1.0f, 5.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f}, 0.0f},
            {{0.5f, 1.2f, 4.0f, 10.0f, -5.0f, 0, 45.0f, 0.1f, 100.0f}, 0.0f},
            {{-1.0f, 1.8f, 6.0f, -12.0f, 3.0f, 7.0f, 70.0f, 0.1f, 100.0f}, 25.0f},
        };

        for (const auto& c : cases) {
            auto s          = flat();
            s.rot_yaw       = c.scr_yaw;
            s.icvfx_enable  = true;
            const auto p    = compute_frustum(c.cam, s);

            ++rep.checks;
            if (!p.icvfx_enable) {
                rep.failures.emplace_back("icvfx/on: asked for and not enabled");
                continue;
            }

            vec3 cr, cu, cf;
            camera_basis(c.cam, cr, cu, cf);
            vec3 sr, su, sn;
            screen_basis(s, sr, su, sn);

            const vec3   eye    = {c.cam.x, c.cam.y, c.cam.z};
            const vec3   centre = screen_centre(s);
            const double aspect = s.width_m / s.height_m;
            const double half_v = std::tan(rad(c.cam.fov) * 0.5);
            const double half_h = half_v * aspect;

            const double sgn_x[4] = {-1.0, 1.0, 1.0, -1.0}; // UL, UR, LR, LL
            const double sgn_y[4] = {1.0, 1.0, -1.0, -1.0};

            for (int i = 0; i < 4; ++i) {
                const vec3   dir   = cf + cr * (sgn_x[i] * half_h) + cu * (sgn_y[i] * half_v);
                const double denom = dot(dir, sn);
                if (std::abs(denom) < 1e-9) {
                    rep.failures.emplace_back("icvfx/on: reference ray parallel to screen");
                    break;
                }
                const double t   = dot(centre - eye, sn) / denom;
                const vec3   hit = eye + dir * t;
                const vec3   loc = hit - vec3{s.pos_x, s.pos_y, s.pos_z};

                check("icvfx/quad/x", p.icvfx_q[i * 2 + 0], 2.0 * dot(loc, sr) / s.width_m, NDC_TOL);
                check("icvfx/quad/y", p.icvfx_q[i * 2 + 1], 1.0 - 2.0 * dot(loc, su) / s.height_m, NDC_TOL);
            }
        }
    }

    // ---- 11. compute_pick: which screen is under a point ------------------------------
    //
    // The gate on every interactive gesture. Deliberately an ASYMMETRIC stage: two screens at
    // different distances, offset to one side, so a mirrored ray or a swapped axis lands
    // somewhere provably wrong instead of on a plausible neighbour.
    {
        std::map<std::string, screen_meta> stage;
        {
            auto near_s   = flat(); // 2 x 2 m, centre (0, 1, 0), faces +Z
            near_s.name   = "near";
            near_s.pos_x  = -1.5f;  // offset LEFT
            stage["near"] = near_s;

            auto far_s     = flat();
            far_s.name     = "far";
            // 8 m wide, and that width is load-bearing rather than arbitrary: the ray through
            // the near screen's centre reaches x = -3 by the time it gets to z = -6, so `far`
            // must span past that or 11c's "nearer wins" has only ONE hit to choose from and
            // cannot fail. It was 4 m first, and the nearest-wins mutation went undetected.
            far_s.width_m  = 8.0f;
            far_s.height_m = 3.0f;
            far_s.pos_z    = -6.0f; // further from the camera
            stage["far"]   = far_s;
        }

        // Eye at +Z looking down -Z, so both screens are in front.
        const previz_camera eye{0.0f, 1.0f, 6.0f, 0, 0, 0, 60.0f, 0.1f, 100.0f};
        const double        aspect = 16.0 / 9.0;

        // Project a world point to (sx, sy) INDEPENDENTLY of compute_pick -- the closed-form
        // basis, then the flipped-Y mapping inverted. If this and compute_pick disagree, one of
        // them has the flip backwards, which is the failure a symmetric fixture cannot see.
        auto project = [&](double wx, double wy, double wz, double& sx, double& sy) {
            vec3 r, u, f;
            camera_basis(eye, r, u, f);
            const vec3   d      = vec3{wx, wy, wz} - vec3{eye.x, eye.y, eye.z};
            const double along  = dot(d, f);
            const double half_v = std::tan(rad(eye.fov) * 0.5);
            sx                  = (dot(d, r) / along / (aspect * half_v) + 1.0) * 0.5;
            sy                  = (1.0 - dot(d, u) / along / half_v) * 0.5;
        };

        // 11a. the centre of the near screen picks it, at its own origin offset
        {
            double sx, sy;
            project(-1.5, 1.0, 0.0, sx, sy);
            const auto p = compute_pick(eye, aspect, sx, sy, stage);
            ++rep.checks;
            if (!p || p->name != "near")
                rep.failures.emplace_back("pick/near: expected 'near', got " +
                                          std::string(p ? p->name : "nothing"));
            if (p) {
                check("pick/near/local_x", p->local_x, 0.0, 1e-3);
                check("pick/near/local_y", p->local_y, 1.0, 1e-3); // centre = height/2
                // The EUCLIDEAN distance from eye to that point, derived rather than typed --
                // the near screen is offset 1.5 m to the left of an eye 6 m away, so the ray is
                // longer than the perpendicular. Writing 6.0 here was the first version of this
                // check and the self-test caught it.
                check("pick/near/distance", p->distance, len(vec3{-1.5, 1.0, 0.0} - vec3{eye.x, eye.y, eye.z}),
                      1e-3);
            }
        }

        // 11b. a point clear of the near screen picks the far one
        {
            double sx, sy;
            project(1.5, 1.5, -6.0, sx, sy);
            const auto p = compute_pick(eye, aspect, sx, sy, stage);
            ++rep.checks;
            if (!p || p->name != "far")
                rep.failures.emplace_back("pick/far: expected 'far', got " +
                                          std::string(p ? p->name : "nothing"));
            if (p)
                check("pick/far/distance", p->distance,
                      len(vec3{1.5, 1.5, -6.0} - vec3{eye.x, eye.y, eye.z}), 1e-3);
        }

        // 11c. NEARER WINS where they overlap -- the ray through the near screen carries on and
        // does hit the far one's surface as well, so this genuinely has two candidates.
        {
            double sx, sy;
            project(-1.5, 1.0, 0.0, sx, sy);

            // Prove the overlap rather than assume it: the same ray must hit `far` when `near`
            // is not there to occlude it. Without this, a fixture change that stops the ray
            // reaching `far` would silently turn the check below into one that cannot fail.
            std::map<std::string, screen_meta> far_only;
            far_only["far"] = stage.at("far");
            ++rep.checks;
            if (!compute_pick(eye, aspect, sx, sy, far_only))
                rep.failures.emplace_back(
                    "pick/nearest/setup: the ray does not reach the far screen, so the "
                    "nearest-wins check below has only one candidate and cannot fail");

            const auto p = compute_pick(eye, aspect, sx, sy, stage);
            ++rep.checks;
            if (!p || p->name != "near")
                rep.failures.emplace_back("pick/nearest: the farther screen won");
        }

        // 11d. empty space hits nothing, rather than the nearest anyway
        {
            const auto p = compute_pick(eye, aspect, 0.02, 0.98, stage);
            ++rep.checks;
            if (p)
                rep.failures.emplace_back("pick/miss: a corner of empty space picked " + p->name);
        }

        // 11e. a screen BEHIND the eye is never picked, however well the ray lines up
        {
            std::map<std::string, screen_meta> behind;
            auto                               b = flat();
            b.name                                = "behind";
            b.pos_z                               = 12.0f;
            behind["behind"]                      = b;
            ++rep.checks;
            if (compute_pick(eye, aspect, 0.5, 0.5, behind))
                rep.failures.emplace_back("pick/behind: picked a screen behind the camera");
        }

        // 11f. THE FLIP. The top of the image must hit ABOVE centre on the screen. This is the
        // check that fails if yc is written the textbook way, and a symmetric stage would not
        // notice because the mirrored point lands on the same screen.
        //
        // The step is DERIVED, not chosen: the near screen is 2 m tall at ~6.2 m through a 60
        // degree viewport, so it spans only ~0.29 of the image height and ~0.14 above centre.
        // The first version stepped 0.15 and landed just off the top edge -- reported as "left
        // the screen", which reads exactly like a broken pick. A quarter of the half-extent is
        // comfortably inside whatever the numbers are.
        {
            double cx, cy, tx, ty;
            project(-1.5, 1.0, 0.0, cx, cy);
            project(-1.5, 2.0, 0.0, tx, ty); // the screen's top edge, projected
            const double step_y = std::abs(cy - ty) * 0.25;

            const auto top = compute_pick(eye, aspect, cx, cy - step_y, stage);
            ++rep.checks;
            if (!top || top->name != "near") {
                rep.failures.emplace_back("pick/flip: a point above centre left the screen");
            } else if (!(top->local_y > 1.0)) {
                char buf[176];
                std::snprintf(buf, sizeof(buf),
                              "pick/flip: sy above centre gave local_y %.4f, expected > 1.0 "
                              "(the renderer negates proj.m[5], so the mapping is 1-2*sy)",
                              top->local_y);
                rep.failures.emplace_back(buf);
            }
        }

        // 11g. and left of centre must hit LEFT on the screen
        {
            double cx, cy, lx2, ly2;
            project(-1.5, 1.0, 0.0, cx, cy);
            project(-2.5, 1.0, 0.0, lx2, ly2); // the screen's left edge, projected
            const double step_x = std::abs(cx - lx2) * 0.25;

            const auto left = compute_pick(eye, aspect, cx - step_x, cy, stage);
            ++rep.checks;
            if (!left || left->name != "near")
                rep.failures.emplace_back("pick/left: a point left of centre left the screen");
            else if (!(left->local_x < 0.0))
                rep.failures.emplace_back("pick/left: sx left of centre gave a positive local_x");
        }

        // 11h. a rotated screen is picked through its own axes
        {
            std::map<std::string, screen_meta> turned;
            auto                               t = flat();
            t.name                                = "turned";
            t.rot_yaw                             = 35.0f;
            turned["turned"]                      = t;
            const auto p = compute_pick(eye, aspect, 0.5, 0.5, turned);
            ++rep.checks;
            if (!p || p->name != "turned")
                rep.failures.emplace_back("pick/rotated: a yawed screen facing the camera was missed");
            else
                check("pick/rotated/local_y", p->local_y, 1.0, 2e-2);
        }

        // 11i. an empty stage picks nothing rather than crashing
        {
            const std::map<std::string, screen_meta> none;
            ++rep.checks;
            if (compute_pick(eye, aspect, 0.5, 0.5, none))
                rep.failures.emplace_back("pick/empty: picked something on an empty stage");
        }
    }

    // ---- 12. A camera looking AWAY collapses the quad to zero area ----------------------
    {
        auto s         = flat();
        s.icvfx_enable = true;
        // Behind the screen and facing further away: every corner ray meets the plane behind
        // the camera, so `t <= 0` for all four.
        const previz_camera cam{0.0f, 1.0f, 5.0f, 180.0f, 0, 0, 60.0f, 0.1f, 100.0f};
        const auto          p = compute_frustum(cam, s);
        for (int i = 0; i < 8; ++i)
            check("icvfx/away/quad", p.icvfx_q[i], 0.0, 1e-9);
    }

    return rep;
}


void run_stage_math_self_test()
{
    const auto rep = stage_math_self_test();

    if (rep.failures.empty()) {
        CASPAR_LOG(info) << L"[core] stage math self-test: " << rep.checks
                         << L" checks, 0 divergences.";
        return;
    }

    // FATAL rather than the warning `run_compose_self_test` logs, and the difference is not
    // stylistic. The generated composition it checks is not on the frame path, so a divergence
    // there breaks nothing that is running. `compute_frustum` IS on the frame path -- every
    // auto-projection recompute calls it -- so a broken property means screens are already
    // being pointed the wrong way, and a warning in a log nobody reads is not the right
    // response to that.
    for (const auto& f : rep.failures)
        CASPAR_LOG(fatal) << L"[core] stage math self-test: " << u16(f);

    CASPAR_LOG(fatal) << L"[core] stage math self-test: " << rep.failures.size() << L" of "
                      << rep.checks << L" checks FAILED. Projection geometry is wrong.";
}

}}} // namespace caspar::core::fields
