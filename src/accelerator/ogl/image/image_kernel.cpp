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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */
#include "image_kernel.h"

#include "image_shader.h"

#include "../util/device.h"
#include "../util/shader.h"
#include "../util/texture.h"

#include <common/assert.h>
#include <common/gl/gl_check.h>

#include <core/frame/frame_transform.h>
#include <core/frame/pixel_format.h>

#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <GL/glew.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace caspar::accelerator::ogl {

double get_precision_factor(common::bit_depth depth)
{
    switch (depth) {
        case common::bit_depth::bit8:
            return 1.0;
        case common::bit_depth::bit10:
            return 64.0;
        case common::bit_depth::bit12:
            return 16.0;
        case common::bit_depth::bit16:
            return 1.0;
        default:
            return 1.0;
    }
}

bool is_above_screen(double y) { return y < 0.0; }

bool is_below_screen(double y) { return y > 1.0; }

bool is_left_of_screen(double x) { return x < 0.0; }

bool is_right_of_screen(double x) { return x > 1.0; }

bool is_outside_screen(const std::vector<core::frame_geometry::coord>& coords)
{
    auto x_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_x; });
    auto y_coords =
        coords | boost::adaptors::transformed([](const core::frame_geometry::coord& c) { return c.vertex_y; });

    return boost::algorithm::all_of(x_coords, &is_left_of_screen) ||
           boost::algorithm::all_of(x_coords, &is_right_of_screen) ||
           boost::algorithm::all_of(y_coords, &is_above_screen) || boost::algorithm::all_of(y_coords, &is_below_screen);
}

// Builds a 256-entry 1D LUT from control points using Fritsch-Carlson monotone
// cubic Hermite interpolation. Guarantees no overshoot (safe for color values).
// If fewer than 2 points, returns a linear identity LUT.
static std::array<float, 256> build_curve_lut(const core::curve_channel& cc)
{
    std::array<float, 256> lut;
    if (cc.count < 2) {
        for (int i = 0; i < 256; ++i) lut[i] = i / 255.0f;
        return lut;
    }
    // Copy to a sortable vector of (x, y) pairs
    std::vector<std::pair<double, double>> pts;
    pts.reserve(cc.count);
    for (int i = 0; i < cc.count; ++i)
        pts.push_back({cc.points[i].x, cc.points[i].y});
    std::sort(pts.begin(), pts.end());

    int n = static_cast<int>(pts.size());
    std::vector<double> dx(n - 1), dy(n - 1), delta(n - 1), m(n);
    for (int i = 0; i < n - 1; ++i) {
        dx[i]    = pts[i + 1].first  - pts[i].first;
        dy[i]    = pts[i + 1].second - pts[i].second;
        delta[i] = (dx[i] > 1e-10) ? dy[i] / dx[i] : 0.0;
    }
    // Tangents: endpoint slopes equal adjacent delta, interior = average
    m[0]     = delta[0];
    m[n - 1] = delta[n - 2];
    for (int i = 1; i < n - 1; ++i)
        m[i] = (delta[i - 1] + delta[i]) * 0.5;
    // Fritsch-Carlson monotonicity correction
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(delta[i]) < 1e-10) { m[i] = m[i + 1] = 0.0; continue; }
        double a = m[i]     / delta[i];
        double b = m[i + 1] / delta[i];
        double h = std::sqrt(a * a + b * b);
        if (h > 3.0) { m[i] *= 3.0 / h; m[i + 1] *= 3.0 / h; }
    }
    // Evaluate at 256 uniform positions
    for (int k = 0; k < 256; ++k) {
        double t = k / 255.0;
        if (t <= pts.front().first) { lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, pts.front().second))); continue; }
        if (t >= pts.back().first)  { lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, pts.back().second)));  continue; }
        int seg = 0;
        for (int i = 0; i < n - 2; ++i)
            if (t >= pts[i].first && t < pts[i + 1].first) { seg = i; break; }
        double h_   = dx[seg];
        double t_   = (h_ > 1e-10) ? (t - pts[seg].first) / h_ : 0.0;
        double t2   = t_ * t_;
        double t3   = t2 * t_;
        double h00  = 2*t3 - 3*t2 + 1;
        double h10  = t3  - 2*t2 + t_;
        double h01  = -2*t3 + 3*t2;
        double h11  = t3  - t2;
        double val  = h00 * pts[seg].second  + h10 * h_ * m[seg]
                    + h01 * pts[seg+1].second + h11 * h_ * m[seg+1];
        lut[k] = static_cast<float>(std::max(0.0, std::min(1.0, val)));
    }
    return lut;
}

static const double epsilon = 0.001;

struct image_kernel::impl
{
    spl::shared_ptr<device> ogl_;
    spl::shared_ptr<shader> shader_;
    GLuint                  vao_;
    GLuint                  vbo_;
    GLuint                  curve_lut_tex_id_ = 0;

    explicit impl(const spl::shared_ptr<device>& ogl)
        : ogl_(ogl)
        , shader_(ogl_->dispatch_sync([&] { return get_image_shader(ogl); }))
    {
        ogl_->dispatch_sync([&] {
            GL(glGenVertexArrays(1, &vao_));
            GL(glGenBuffers(1, &vbo_));
        });
    }

    ~impl()
    {
        ogl_->dispatch_sync([&] {
            GL(glDeleteVertexArrays(1, &vao_));
            GL(glDeleteBuffers(1, &vbo_));
            if (curve_lut_tex_id_)
                GL(glDeleteTextures(1, &curve_lut_tex_id_));
        });
    }

    void draw(draw_params params)
    {
        CASPAR_ASSERT(params.pix_desc.planes.size() == params.textures.size());

        if (params.textures.empty() || !params.background) {
            return;
        }

        if (params.transforms.image_transform.opacity < epsilon) {
            return;
        }

        auto coords = params.geometry.data();
        if (coords.empty()) {
            return;
        }

        auto transforms = params.transforms;

        auto const first_plane = params.pix_desc.planes.at(0);
        if (params.geometry.mode() != core::frame_geometry::scale_mode::stretch && first_plane.width > 0 &&
            first_plane.height > 0) {
            auto width_scale  = static_cast<double>(params.target_width) / static_cast<double>(first_plane.width);
            auto height_scale = static_cast<double>(params.target_height) / static_cast<double>(first_plane.height);

            core::image_transform transform;
            double                target_scale;
            switch (params.geometry.mode()) {
                case core::frame_geometry::scale_mode::fit:
                    target_scale = std::min(width_scale, height_scale);

                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::fill:
                    target_scale = std::max(width_scale, height_scale);
                    transform.fill_scale[0] *= target_scale / width_scale;
                    transform.fill_scale[1] *= target_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::original:
                    transform.fill_scale[0] /= width_scale;
                    transform.fill_scale[1] /= height_scale;
                    break;

                case core::frame_geometry::scale_mode::hfill:
                    transform.fill_scale[1] *= width_scale / height_scale;
                    break;

                case core::frame_geometry::scale_mode::vfill:
                    transform.fill_scale[0] *= height_scale / width_scale;
                    break;

                default:;
            }

            transforms = transforms.combine_transform(transform, params.aspect_ratio);
        }

        coords = transforms.transform_coords(coords);

        // Skip drawing if all the coordinates will be outside the screen.
        if (coords.size() < 3 || is_outside_screen(coords)) {
            return;
        }

        double precision_factor[4] = {1, 1, 1, 1};

        // Bind textures

        for (int n = 0; n < params.textures.size(); ++n) {
            params.textures[n]->bind(n);
            precision_factor[n] = get_precision_factor(params.textures[n]->depth());
        }

        if (params.local_key) {
            params.local_key->bind(static_cast<int>(texture_id::local_key));
        }

        if (params.layer_key) {
            params.layer_key->bind(static_cast<int>(texture_id::layer_key));
        }

        const auto is_hd       = params.pix_desc.planes.at(0).height > 700;
        const auto color_space = is_hd ? params.pix_desc.color_space : core::color_space::bt601;

        const float color_matrices[3][9] = {
            {1.0, 0.0, 1.402, 1.0, -0.344, -0.509, 1.0, 1.772, 0.0},                          // bt.601
            {1.0, 0.0, 1.5748, 1.0, -0.1873, -0.4681, 1.0, 1.8556, 0.0},                      // bt.709
            {1.0, 0.0, 1.4746, 1.0, -0.16455312684366, -0.57135312684366, 1.0, 1.8814, 0.0}}; // bt.2020
        const auto color_matrix = color_matrices[static_cast<int>(color_space)];

        const float luma_coefficients[3][3] = {{0.299, 0.587, 0.114},     // bt.601
                                               {0.2126, 0.7152, 0.0722},  // bt.709
                                               {0.2627, 0.6780, 0.0593}}; // bt.2020
        const auto  luma_coeff              = luma_coefficients[static_cast<int>(color_space)];

        // Setup shader
        shader_->use();

        shader_->set("is_straight_alpha", params.pix_desc.is_straight_alpha);
        shader_->set("plane[0]", texture_id::plane0);
        shader_->set("plane[1]", texture_id::plane1);
        shader_->set("plane[2]", texture_id::plane2);
        shader_->set("plane[3]", texture_id::plane3);
        shader_->set("precision_factor[0]", precision_factor[0]);
        shader_->set("precision_factor[1]", precision_factor[1]);
        shader_->set("precision_factor[2]", precision_factor[2]);
        shader_->set("precision_factor[3]", precision_factor[3]);
        shader_->set("local_key", texture_id::local_key);
        shader_->set("layer_key", texture_id::layer_key);
        shader_->set_matrix3("color_matrix", color_matrix);
        shader_->set("luma_coeff", luma_coeff[0], luma_coeff[1], luma_coeff[2]);
        shader_->set("has_local_key", static_cast<bool>(params.local_key));
        shader_->set("has_layer_key", static_cast<bool>(params.layer_key));
        shader_->set("pixel_format", params.pix_desc.format);
        shader_->set("opacity", transforms.image_transform.is_key ? 1.0 : transforms.image_transform.opacity);

        if (transforms.image_transform.chroma.enable) {
            shader_->set("chroma", true);
            shader_->set("chroma_show_mask", transforms.image_transform.chroma.show_mask);
            shader_->set("chroma_target_hue", transforms.image_transform.chroma.target_hue / 360.0);
            shader_->set("chroma_hue_width", transforms.image_transform.chroma.hue_width);
            shader_->set("chroma_min_saturation", transforms.image_transform.chroma.min_saturation);
            shader_->set("chroma_min_brightness", transforms.image_transform.chroma.min_brightness);
            shader_->set("chroma_softness", 1.0 + transforms.image_transform.chroma.softness);
            shader_->set("chroma_spill_suppress", transforms.image_transform.chroma.spill_suppress / 360.0);
            shader_->set("chroma_spill_suppress_saturation",
                         transforms.image_transform.chroma.spill_suppress_saturation);
        } else {
            shader_->set("chroma", false);
        }

        if (transforms.image_transform.projection.enable) {
            shader_->set("is_360", true);
            shader_->set("view_yaw",      static_cast<float>(transforms.image_transform.projection.yaw));
            shader_->set("view_pitch",    static_cast<float>(transforms.image_transform.projection.pitch));
            shader_->set("view_roll",     static_cast<float>(transforms.image_transform.projection.roll));
            shader_->set("view_fov",      static_cast<float>(transforms.image_transform.projection.fov));
            shader_->set("view_offset_x", static_cast<float>(transforms.image_transform.projection.offset_x));
            shader_->set("view_offset_y", static_cast<float>(transforms.image_transform.projection.offset_y));
            shader_->set("aspect_ratio",  static_cast<float>(params.aspect_ratio));
        } else {
            shader_->set("is_360", false);
        }

        // Color grading: ACES-based gamut/transfer/tonemapping pipeline
        // Gamut index: 0=bt709, 1=bt2020, 2=dcip3_d65, 3=aces_ap0, 4=aces_ap1(acescg), 5=arri_wg3, 6=sgamut3_cine
        // Transfer:    0=linear, 1=srgb, 2=rec709, 3=pq, 4=hlg, 5=logc3, 6=slog3
        // Tonemapping: 0=none, 1=reinhard, 2=aces_filmic, 3=aces_rrt
        static const float k_to_working[7][9] = {
            // bt709 -> ACEScg (AP1) -- from ACES OCIO config
            {0.6131516f,  0.3395148f,  0.0472947f,  0.0701011f,  0.9162792f,  0.0136197f,  0.0206177f,  0.1095763f,  0.8698060f},
            // bt2020 -> ACEScg
            {0.7951281f,  0.1643585f,  0.0405134f,  0.0234399f,  0.9415642f,  0.0349959f,  0.0036186f,  0.0613513f,  0.9350301f},
            // dcip3 d65 -> ACEScg
            {0.8224549f,  0.1774521f, -0.0000070f,  0.0332021f,  0.9618927f,  0.0049052f,  0.0170512f,  0.0723025f,  0.9106463f},
            // aces_ap0 -> ACEScg (AP1)
            {1.4514393f, -0.2362486f, -0.0153674f, -0.3194787f,  1.1765407f, -0.0083099f, -0.0153694f,  0.0183597f,  1.0023859f},
            // aces_ap1 identity
            {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
            // arri wide gamut 3 -> ACEScg
            {0.6954522f,  0.1446577f,  0.1598901f,  0.0439823f,  0.8591788f,  0.0968389f, -0.0055023f,  0.0040678f,  1.0014345f},
            // sony sgamut3.cine -> ACEScg
            {0.7112957f,  0.1903613f,  0.0983436f,  0.0406952f,  0.8550396f,  0.1042651f, -0.0025079f,  0.0085993f,  0.9939086f}
        };
        static const float k_to_output[7][9] = {
            // ACEScg -> bt709
            { 1.7050585f, -0.6217876f, -0.0832709f, -0.1302597f,  1.1407927f, -0.0105330f, -0.0240003f, -0.1289711f,  1.1529714f},
            // ACEScg -> bt2020
            { 1.2746843f, -0.2692490f, -0.0054353f, -0.0293524f,  1.0763680f, -0.0470156f, -0.0160993f, -0.0606079f,  1.0767072f},
            // ACEScg -> dcip3 d65
            { 1.2239840f, -0.2239840f,  0.0000000f, -0.0421197f,  1.0421197f,  0.0000000f, -0.0196576f, -0.0787093f,  1.0983669f},
            // ACEScg -> aces_ap0
            { 0.6954522f,  0.1446577f,  0.1598901f,  0.0439823f,  0.8591788f,  0.0968389f, -0.0055023f,  0.0040678f,  1.0014345f},
            // ACEScg identity (ap1)
            { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
            // ACEScg -> arri wide gamut 3 (approximate inverse)
            { 1.4514393f, -0.2362486f, -0.0153674f, -0.3194787f,  1.1765407f, -0.0083099f, -0.0153694f,  0.0183597f,  1.0023859f},
            // ACEScg -> sgamut3.cine (approximate inverse)
            { 1.4087253f, -0.3132873f, -0.0954380f, -0.0667453f,  1.1571781f, -0.0904328f, -0.0057800f, -0.0066708f,  1.0124508f}
        };
        const auto& cg = transforms.image_transform.color_grade;
        if (cg.enable) {
            int ig = std::min(std::max(cg.input_gamut,  0), 6);
            int og = std::min(std::max(cg.output_gamut, 0), 6);
            shader_->set("color_grading",     true);
            shader_->set("input_transfer",    cg.input_transfer);
            shader_->set("output_transfer",   cg.output_transfer);
            shader_->set("tone_mapping_op",   cg.tone_mapping);
            shader_->set("exposure",          cg.exposure);
            shader_->set_matrix3("input_to_working",  k_to_working[ig]);
            shader_->set_matrix3("working_to_output", k_to_output[og]);
        } else {
            shader_->set("color_grading", false);
        }

        // Setup blend_func

        if (transforms.image_transform.is_key) {
            params.blend_mode = core::blend_mode::normal;
        }

        params.background->bind(static_cast<int>(texture_id::background));
        shader_->set("background", texture_id::background);
        shader_->set("blend_mode", params.blend_mode);
        shader_->set("keyer", params.keyer);

        // Setup image-adjustments
        shader_->set("invert",  transforms.image_transform.invert);
        shader_->set("flip_h",  transforms.image_transform.flip_h);
        shader_->set("flip_v",  transforms.image_transform.flip_v);

        if (transforms.image_transform.levels.min_input > epsilon ||
            transforms.image_transform.levels.max_input < 1.0 - epsilon ||
            transforms.image_transform.levels.min_output > epsilon ||
            transforms.image_transform.levels.max_output < 1.0 - epsilon ||
            std::abs(transforms.image_transform.levels.gamma - 1.0) > epsilon) {
            shader_->set("levels", true);
            shader_->set("min_input", transforms.image_transform.levels.min_input);
            shader_->set("max_input", transforms.image_transform.levels.max_input);
            shader_->set("min_output", transforms.image_transform.levels.min_output);
            shader_->set("max_output", transforms.image_transform.levels.max_output);
            shader_->set("gamma", transforms.image_transform.levels.gamma);
        } else {
            shader_->set("levels", false);
        }

        if (std::abs(transforms.image_transform.brightness - 1.0) > epsilon ||
            std::abs(transforms.image_transform.saturation - 1.0) > epsilon ||
            std::abs(transforms.image_transform.contrast - 1.0) > epsilon) {
            shader_->set("csb", true);

            shader_->set("brt", transforms.image_transform.brightness);
            shader_->set("sat", transforms.image_transform.saturation);
            shader_->set("con", transforms.image_transform.contrast);
        } else {
            shader_->set("csb", false);
        }

        // White balance
        if (std::abs(transforms.image_transform.temperature) > epsilon ||
            std::abs(transforms.image_transform.tint) > epsilon) {
            shader_->set("white_balance",  true);
            shader_->set("wb_temperature", static_cast<float>(transforms.image_transform.temperature));
            shader_->set("wb_tint",        static_cast<float>(transforms.image_transform.tint));
        } else {
            shader_->set("white_balance", false);
        }

        // Lift / Midtone / Gain (per-channel 3-way color corrector)
        {
            const auto& lift    = transforms.image_transform.lift;
            const auto& midtone = transforms.image_transform.midtone;
            const auto& gain    = transforms.image_transform.gain;
            bool lmg_active =
                std::abs(lift[0]) > epsilon || std::abs(lift[1]) > epsilon || std::abs(lift[2]) > epsilon ||
                std::abs(midtone[0] - 1.0) > epsilon || std::abs(midtone[1] - 1.0) > epsilon || std::abs(midtone[2] - 1.0) > epsilon ||
                std::abs(gain[0]   - 1.0) > epsilon || std::abs(gain[1]   - 1.0) > epsilon || std::abs(gain[2]   - 1.0) > epsilon;
            if (lmg_active) {
                shader_->set("lmg_enable",  true);
                // Shader internal: color.r=Blue_displayed, color.b=Red_displayed
                // Upload as [B_param, G_param, R_param] so .r affects displayed Blue, .b affects displayed Red
                shader_->set("lmg_lift",    lift[2],    lift[1],    lift[0]);
                shader_->set("lmg_midtone", midtone[2], midtone[1], midtone[0]);
                shader_->set("lmg_gain",    gain[2],    gain[1],    gain[0]);
            } else {
                shader_->set("lmg_enable", false);
            }
        }

        // Hue shift
        if (std::abs(transforms.image_transform.hue_shift) > epsilon) {
            shader_->set("hue_shift_enable",  true);
            shader_->set("hue_shift_degrees", static_cast<float>(transforms.image_transform.hue_shift));
        } else {
            shader_->set("hue_shift_enable", false);
        }

        // Tonal balance (shadows / highlights)
        if (std::abs(transforms.image_transform.shadows) > epsilon ||
            std::abs(transforms.image_transform.highlights) > epsilon) {
            shader_->set("tonebalance_enable", true);
            shader_->set("tb_shadows",    static_cast<float>(transforms.image_transform.shadows));
            shader_->set("tb_highlights", static_cast<float>(transforms.image_transform.highlights));
        } else {
            shader_->set("tonebalance_enable", false);
        }

        // Per-channel RGB Levels
        {
            const auto& rl = transforms.image_transform.per_channel_levels;
            if (rl.enable) {
                shader_->set("rgb_levels_enable", true);
                // Shader [0] -> color.r = Blue_displayed, [2] -> color.b = Red_displayed
                // Map user's R channel -> index [2], B channel -> index [0]
                shader_->set("rgb_levels_min_input[0]",  static_cast<float>(rl.b.min_input));
                shader_->set("rgb_levels_min_input[1]",  static_cast<float>(rl.g.min_input));
                shader_->set("rgb_levels_min_input[2]",  static_cast<float>(rl.r.min_input));
                shader_->set("rgb_levels_max_input[0]",  static_cast<float>(rl.b.max_input));
                shader_->set("rgb_levels_max_input[1]",  static_cast<float>(rl.g.max_input));
                shader_->set("rgb_levels_max_input[2]",  static_cast<float>(rl.r.max_input));
                shader_->set("rgb_levels_gamma[0]",      static_cast<float>(rl.b.gamma));
                shader_->set("rgb_levels_gamma[1]",      static_cast<float>(rl.g.gamma));
                shader_->set("rgb_levels_gamma[2]",      static_cast<float>(rl.r.gamma));
                shader_->set("rgb_levels_min_output[0]", static_cast<float>(rl.b.min_output));
                shader_->set("rgb_levels_min_output[1]", static_cast<float>(rl.g.min_output));
                shader_->set("rgb_levels_min_output[2]", static_cast<float>(rl.r.min_output));
                shader_->set("rgb_levels_max_output[0]", static_cast<float>(rl.b.max_output));
                shader_->set("rgb_levels_max_output[1]", static_cast<float>(rl.g.max_output));
                shader_->set("rgb_levels_max_output[2]", static_cast<float>(rl.r.max_output));
            } else {
                shader_->set("rgb_levels_enable", false);
            }
        }

        // Tone Curves: build LUTs on CPU, pack into RGBA32F 256x1 texture
        {
            const auto& cv = transforms.image_transform.curves;
            if (cv.enable) {
                auto lut_r = build_curve_lut(cv.red);
                auto lut_g = build_curve_lut(cv.green);
                auto lut_b = build_curve_lut(cv.blue);
                auto lut_m = build_curve_lut(cv.master);

                // Pack 4 LUTs into interleaved RGBA data
                // Channel 0 (.r) -> color.r = Blue_displayed -> store user's Blue LUT
                // Channel 2 (.b) -> color.b = Red_displayed  -> store user's Red LUT
                std::vector<float> rgba_data(256 * 4);
                for (int i = 0; i < 256; ++i) {
                    rgba_data[i * 4 + 0] = lut_b[i];  // .r slot = Blue displayed
                    rgba_data[i * 4 + 1] = lut_g[i];  // .g slot = Green (unchanged)
                    rgba_data[i * 4 + 2] = lut_r[i];  // .b slot = Red displayed
                    rgba_data[i * 4 + 3] = lut_m[i];  // .a slot = master (unchanged)
                }

                // Create texture on first use
                if (!curve_lut_tex_id_) {
                    GL(glCreateTextures(GL_TEXTURE_2D, 1, &curve_lut_tex_id_));
                    GL(glTextureStorage2D(curve_lut_tex_id_, 1, GL_RGBA32F, 256, 1));
                    GL(glTextureParameteri(curve_lut_tex_id_, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
                    GL(glTextureParameteri(curve_lut_tex_id_, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
                    GL(glTextureParameteri(curve_lut_tex_id_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                    GL(glTextureParameteri(curve_lut_tex_id_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                }

                GL(glTextureSubImage2D(curve_lut_tex_id_, 0, 0, 0, 256, 1, GL_RGBA, GL_FLOAT, rgba_data.data()));
                GL(glBindTextureUnit(static_cast<int>(texture_id::curve_lut_tex), curve_lut_tex_id_));
                shader_->set("curves_enable", true);
                shader_->set("curve_lut_tex", static_cast<int>(texture_id::curve_lut_tex));
            } else {
                shader_->set("curves_enable", false);
            }
        }

        // Setup drawing area

        GL(glViewport(0, 0, params.background->width(), params.background->height()));
        glDisable(GL_DEPTH_TEST);

        // Set render target
        params.background->attach();

        // Draw
        GL(glBindVertexArray(vao_));
        GL(glBindBuffer(GL_ARRAY_BUFFER, vbo_));

        GL(glBufferData(GL_ARRAY_BUFFER,
                        static_cast<GLsizeiptr>(sizeof(core::frame_geometry::coord)) * coords.size(),
                        coords.data(),
                        GL_STATIC_DRAW));

        auto stride = static_cast<GLsizei>(sizeof(core::frame_geometry::coord));

        auto vtx_loc = shader_->get_attrib_location("Position");
        auto tex_loc = shader_->get_attrib_location("TexCoordIn");

        GL(glEnableVertexAttribArray(vtx_loc));
        GL(glEnableVertexAttribArray(tex_loc));

        GL(glVertexAttribPointer(vtx_loc, 2, GL_DOUBLE, GL_FALSE, stride, nullptr));
        GL(glVertexAttribPointer(tex_loc, 4, GL_DOUBLE, GL_FALSE, stride, (GLvoid*)(2 * sizeof(GLdouble))));

        GL(glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<GLsizei>(coords.size())));
        GL(glTextureBarrier());

        GL(glDisableVertexAttribArray(vtx_loc));
        GL(glDisableVertexAttribArray(tex_loc));

        GL(glBindVertexArray(0));
        GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

        // Cleanup
        GL(glDisable(GL_SCISSOR_TEST));
        GL(glDisable(GL_BLEND));
    }
};

image_kernel::image_kernel(const spl::shared_ptr<device>& ogl)
    : impl_(new impl(ogl))
{
}
image_kernel::~image_kernel() {}
void image_kernel::draw(const draw_params& params) { impl_->draw(params); }

} // namespace caspar::accelerator::ogl
