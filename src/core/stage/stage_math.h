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

}} // namespace caspar::core
