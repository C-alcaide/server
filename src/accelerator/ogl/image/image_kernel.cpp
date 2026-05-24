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
    GLuint                  lut3d_tex_id_     = 0;
    const core::lut3d_data* lut3d_data_ptr_   = nullptr;  // tracks which data is uploaded
    GLuint                  hue_curve_tex_id_ = 0;
    int                     frame_counter_    = 0;

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
            if (lut3d_tex_id_)
                GL(glDeleteTextures(1, &lut3d_tex_id_));
            if (hue_curve_tex_id_)
                GL(glDeleteTextures(1, &hue_curve_tex_id_));
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
            shader_->set("frustum_h",     static_cast<float>(transforms.image_transform.projection.frustum_h));
            shader_->set("frustum_v",     static_cast<float>(transforms.image_transform.projection.frustum_v));
            shader_->set("lens_k1",       static_cast<float>(transforms.image_transform.projection.lens_k1));
            shader_->set("lens_k2",       static_cast<float>(transforms.image_transform.projection.lens_k2));
            shader_->set("lens_k3",       static_cast<float>(transforms.image_transform.projection.lens_k3));
            shader_->set("aspect_ratio",  static_cast<float>(params.aspect_ratio));
        } else {
            shader_->set("is_360", false);
        }

        // Curved screen compensation — dispatched independently of 360 mode
        shader_->set("is_curved",         transforms.image_transform.projection.curve_enable);
        shader_->set("screen_curve_type", static_cast<int>(transforms.image_transform.projection.curve_type));
        shader_->set("screen_arc",        static_cast<float>(transforms.image_transform.projection.screen_arc));

        // Soft-edge blending
        shader_->set("edge_blend_left",   static_cast<float>(transforms.image_transform.projection.edge_blend_left));
        shader_->set("edge_blend_right",  static_cast<float>(transforms.image_transform.projection.edge_blend_right));
        shader_->set("edge_blend_top",    static_cast<float>(transforms.image_transform.projection.edge_blend_top));
        shader_->set("edge_blend_bottom", static_cast<float>(transforms.image_transform.projection.edge_blend_bottom));
        shader_->set("edge_blend_gamma",  static_cast<float>(transforms.image_transform.projection.edge_blend_gamma));

        if (transforms.image_transform.blur.enable) {
            shader_->set("blur_enable", true);
            shader_->set("blur_radius", static_cast<float>(transforms.image_transform.blur.radius));
            shader_->set("blur_type",   static_cast<int>(transforms.image_transform.blur.type));
            shader_->set("blur_angle",  static_cast<float>(transforms.image_transform.blur.angle));
            shader_->set("blur_center", static_cast<float>(transforms.image_transform.blur.center[0]),
                         static_cast<float>(transforms.image_transform.blur.center[1]));
            shader_->set("blur_tilt",   static_cast<float>(transforms.image_transform.blur.tilt_y),
                         static_cast<float>(transforms.image_transform.blur.tilt_h));
            shader_->set("target_size", static_cast<float>(params.target_width), static_cast<float>(params.target_height));
        } else {
            shader_->set("blur_enable", false);
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
            // aces_ap0 -> ACEScg (AP1)  [ACES CTL reference]
            {1.4514393f, -0.2365107f, -0.2149286f, -0.0765538f,  1.1762297f, -0.0996759f,  0.0083161f, -0.0060324f,  0.9977163f},
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
            // ACEScg -> aces_ap0  [inverse of AP0->AP1]
            {0.6954522f,  0.1406787f,  0.1638691f,  0.0447946f,  0.8596711f,  0.0955343f, -0.0055259f,  0.0040252f,  1.0015007f},
            // ACEScg identity (ap1)
            { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
            // ACEScg -> arri wide gamut 3 (computed inverse)
            { 1.4516608f, -0.2434265f, -0.2082343f, -0.0752455f,  1.1770530f, -0.1018075f,  0.0082817f, -0.0061186f,  0.9978370f},
            // ACEScg -> sgamut3.cine (computed inverse)
            { 1.4235761f, -0.3158537f, -0.1077233f, -0.0682645f,  1.1859178f, -0.1176531f,  0.0041827f, -0.0110575f,  1.0068749f}
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

            // BT.2408 luminance adaptation: scale linear light when crossing
            // HDR/SDR domains.  Each transfer defines a "peak" in cd/m²:
            //   SDR/sRGB/rec709 = 100, HLG = 1000, PQ = 10000
            // Scale = source_peak / target_peak
            auto get_peak = [](int transfer) -> float {
                switch (transfer) {
                    case 3: return 10000.0f; // PQ
                    case 4: return 1000.0f;  // HLG
                    default: return 100.0f;  // SDR, sRGB, rec709, linear
                }
            };
            float src_peak = get_peak(cg.input_transfer);
            float tgt_peak = get_peak(cg.output_transfer);
            shader_->set("luminance_scale", src_peak / tgt_peak);
        } else if (params.auto_color_convert &&
                   (params.pix_desc.color_space != params.target_color_space ||
                    params.pix_desc.color_transfer != params.target_color_transfer)) {
            static int convert_count = 0;
            convert_count++;
            if (convert_count <= 3 || convert_count % 100 == 0) {
                CASPAR_LOG(trace) << L"[ogl_kernel] auto_color_convert frame #" << convert_count
                    << L" src_cs=" << static_cast<int>(params.pix_desc.color_space)
                    << L" src_ct=" << static_cast<int>(params.pix_desc.color_transfer)
                    << L" tgt_cs=" << static_cast<int>(params.target_color_space)
                    << L" tgt_ct=" << static_cast<int>(params.target_color_transfer)
                    << L" fmt=" << static_cast<int>(params.pix_desc.format);
            }
            // Auto color conversion: source differs from channel output.
            // Map core enums to shader indices.
            auto gamut_index = [](core::color_space cs) -> int {
                switch (cs) {
                    case core::color_space::bt2020: return 1;
                    default:                       return 0; // bt601/bt709 → shader index 0 (bt709)
                }
            };
            auto transfer_index = [](core::color_transfer ct) -> int {
                switch (ct) {
                    case core::color_transfer::pq:  return 3;
                    case core::color_transfer::hlg: return 4;
                    default:                        return 2; // sdr → rec709
                }
            };
            int ig = gamut_index(params.pix_desc.color_space);
            int og = gamut_index(params.target_color_space);
            // Skip if the mapped indices are identical (e.g. bt601 source on bt709 channel)
            if (ig == og && params.pix_desc.color_transfer == params.target_color_transfer) {
                shader_->set("color_grading", false);
            } else {
                int it = transfer_index(params.pix_desc.color_transfer);
                int ot = transfer_index(params.target_color_transfer);
                // Apply tonemapping when going from HDR to SDR.
                // Use Stephen Hill ACES approximation (op=3) — the full spline
                // path (op=4) has a c9 input domain bug that clips everything.
                int tm = (it >= 3 && ot <= 2) ? 3 : 0; // 3 = ACES_RRT (Hill), 0 = NONE
                shader_->set("color_grading",     true);
                shader_->set("input_transfer",    it);
                shader_->set("output_transfer",   ot);
                shader_->set("tone_mapping_op",   tm);
                shader_->set("exposure",          1.0f);

                // Direct gamut matrices for auto conversion (ITU-R BT.2087).
                // Unlike the color grading path (which routes through ACEScg/AP1
                // working space for perceptual benefits), auto conversion uses
                // standard direct matrices — both BT.709 and BT.2020 share the
                // D65 white point, so no chromatic adaptation is needed.
                static const float k_direct[2][2][9] = {
                    { // from bt709
                        {1,0,0, 0,1,0, 0,0,1}, // → bt709 (identity)
                        {0.6274039f,0.3292830f,0.0433131f, 0.0690972f,0.9195404f,0.0113623f, 0.0163914f,0.0880133f,0.8955953f}, // → bt2020
                    },
                    { // from bt2020
                        {1.6604910f,-0.5876411f,-0.0728499f, -0.1245505f,1.1328999f,-0.0083494f, -0.0181508f,-0.1005789f,1.1187297f}, // → bt709
                        {1,0,0, 0,1,0, 0,0,1}, // → bt2020 (identity)
                    },
                };
                static const float k_identity[9] = {1,0,0, 0,1,0, 0,0,1};
                shader_->set_matrix3("input_to_working",  k_direct[ig][og]);
                shader_->set_matrix3("working_to_output", k_identity);
                static bool logged_matrices = false;
                if (!logged_matrices) {
                    CASPAR_LOG(trace) << L"[ogl_kernel] GAMUT: ig=" << ig << L" og=" << og
                        << L" it=" << it << L" ot=" << ot << L" tm=" << tm;
                    CASPAR_LOG(trace) << L"[ogl_kernel] input_to_working[0..2]="
                        << k_to_working[ig][0] << L"," << k_to_working[ig][1] << L"," << k_to_working[ig][2];
                    CASPAR_LOG(trace) << L"[ogl_kernel] working_to_output[0..2]="
                        << k_to_output[og][0] << L"," << k_to_output[og][1] << L"," << k_to_output[og][2];
                    logged_matrices = true;
                }

                // BT.2408 luminance adaptation for auto conversion path
                auto get_peak = [](int transfer) -> float {
                    switch (transfer) {
                        case 3: return 10000.0f; // PQ
                        case 4: return 1000.0f;  // HLG
                        default: return 100.0f;  // SDR, sRGB, rec709, linear
                    }
                };
                float src_peak = get_peak(it);
                float tgt_peak = get_peak(ot);
                shader_->set("luminance_scale", src_peak / tgt_peak);
            }
        } else {
            static int no_convert_count = 0;
            no_convert_count++;
            if (no_convert_count <= 5 || no_convert_count % 100 == 0) {
                CASPAR_LOG(trace) << L"[ogl_kernel] NO_CONVERT frame #" << no_convert_count
                    << L" auto=" << params.auto_color_convert
                    << L" src_cs=" << static_cast<int>(params.pix_desc.color_space)
                    << L" src_ct=" << static_cast<int>(params.pix_desc.color_transfer)
                    << L" tgt_cs=" << static_cast<int>(params.target_color_space)
                    << L" tgt_ct=" << static_cast<int>(params.target_color_transfer)
                    << L" fmt=" << static_cast<int>(params.pix_desc.format);
            }
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

        // Linear saturation (scene-referred)
        if (std::abs(transforms.image_transform.linear_saturation - 1.0) > epsilon) {
            shader_->set("linear_sat_enable", true);
            shader_->set("linear_sat_value",  static_cast<float>(transforms.image_transform.linear_saturation));
        } else {
            shader_->set("linear_sat_enable", false);
        }

        // ASC CDL (Slope/Offset/Power)
        {
            const auto& s = transforms.image_transform.cdl_slope;
            const auto& o = transforms.image_transform.cdl_offset;
            const auto& p = transforms.image_transform.cdl_power;
            double      cs = transforms.image_transform.cdl_saturation;
            bool cdl_active =
                std::abs(s[0] - 1.0) > epsilon || std::abs(s[1] - 1.0) > epsilon || std::abs(s[2] - 1.0) > epsilon ||
                std::abs(o[0]) > epsilon       || std::abs(o[1]) > epsilon       || std::abs(o[2]) > epsilon       ||
                std::abs(p[0] - 1.0) > epsilon || std::abs(p[1] - 1.0) > epsilon || std::abs(p[2] - 1.0) > epsilon ||
                std::abs(cs - 1.0) > epsilon;
            if (cdl_active) {
                shader_->set("cdl_enable", true);
                // Swap R<->B for BGRA convention: user R=[0], shader .b=Red -> index [2]
                shader_->set("cdl_slope",      s[2], s[1], s[0]);
                shader_->set("cdl_offset",     o[2], o[1], o[0]);
                shader_->set("cdl_power",      p[2], p[1], p[0]);
                shader_->set("cdl_saturation", static_cast<float>(cs));
            } else {
                shader_->set("cdl_enable", false);
            }
        }

        // Split toning
        {
            const auto& sc = transforms.image_transform.split_shadow_color;
            const auto& hc = transforms.image_transform.split_highlight_color;
            bool split_active =
                std::abs(sc[0]) > epsilon || std::abs(sc[1]) > epsilon || std::abs(sc[2]) > epsilon ||
                std::abs(hc[0]) > epsilon || std::abs(hc[1]) > epsilon || std::abs(hc[2]) > epsilon;
            if (split_active) {
                shader_->set("split_tone_enable", true);
                // Swap R<->B for BGRA convention
                shader_->set("split_shadow_color",    sc[2], sc[1], sc[0]);
                shader_->set("split_highlight_color", hc[2], hc[1], hc[0]);
                shader_->set("split_balance", static_cast<float>(transforms.image_transform.split_balance));
            } else {
                shader_->set("split_tone_enable", false);
            }
        }

        // Gamut compression (ACES 1.3 Reference Gamut Compress)
        if (transforms.image_transform.gamut_compress) {
            shader_->set("gamut_compress_enable", true);
            // BGRA order: .r=Blue(yellow), .g=Green(magenta), .b=Red(cyan)
            shader_->set("gc_limit",
                         static_cast<float>(transforms.image_transform.gc_yellow),
                         static_cast<float>(transforms.image_transform.gc_magenta),
                         static_cast<float>(transforms.image_transform.gc_cyan));
        } else {
            shader_->set("gamut_compress_enable", false);
        }

        // 3D LUT
        {
            const auto& lut = transforms.image_transform.lut3d;
            if (lut && lut->size > 0 && !lut->data.empty()) {
                // Re-upload if the data pointer changed (new LUT loaded)
                if (lut.get() != lut3d_data_ptr_) {
                    if (lut3d_tex_id_)
                        GL(glDeleteTextures(1, &lut3d_tex_id_));
                    GL(glCreateTextures(GL_TEXTURE_3D, 1, &lut3d_tex_id_));
                    GL(glTextureStorage3D(lut3d_tex_id_, 1, GL_RGB32F, lut->size, lut->size, lut->size));
                    GL(glTextureParameteri(lut3d_tex_id_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                    GL(glTextureParameteri(lut3d_tex_id_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                    GL(glTextureParameteri(lut3d_tex_id_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                    GL(glTextureParameteri(lut3d_tex_id_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                    GL(glTextureParameteri(lut3d_tex_id_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
                    GL(glTextureSubImage3D(lut3d_tex_id_, 0, 0, 0, 0,
                                           lut->size, lut->size, lut->size,
                                           GL_RGB, GL_FLOAT, lut->data.data()));
                    lut3d_data_ptr_ = lut.get();
                }
                GL(glBindTextureUnit(static_cast<int>(texture_id::lut3d_tex), lut3d_tex_id_));
                shader_->set("lut3d_enable", true);
                shader_->set("lut3d_tex", static_cast<int>(texture_id::lut3d_tex));
                shader_->set("lut3d_strength", transforms.image_transform.lut3d_strength);
            } else {
                shader_->set("lut3d_enable", false);
                if (lut3d_tex_id_ && !lut) {
                    lut3d_data_ptr_ = nullptr;
                }
            }
        }

        // Hue-vs-Hue / Hue-vs-Sat curves
        {
            const auto& hc = transforms.image_transform.hue_curves;
            if (hc && !hc->data.empty()) {
                if (!hue_curve_tex_id_) {
                    GL(glCreateTextures(GL_TEXTURE_2D, 1, &hue_curve_tex_id_));
                    GL(glTextureStorage2D(hue_curve_tex_id_, 1, GL_RGBA32F, 256, 1));
                    GL(glTextureParameteri(hue_curve_tex_id_, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                    GL(glTextureParameteri(hue_curve_tex_id_, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                    GL(glTextureParameteri(hue_curve_tex_id_, GL_TEXTURE_WRAP_S, GL_REPEAT));
                    GL(glTextureParameteri(hue_curve_tex_id_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                }
                GL(glTextureSubImage2D(hue_curve_tex_id_, 0, 0, 0, 256, 1, GL_RGBA, GL_FLOAT, hc->data.data()));
                GL(glBindTextureUnit(static_cast<int>(texture_id::hue_curve_tex), hue_curve_tex_id_));
                shader_->set("hue_curve_enable", true);
                shader_->set("hue_curve_tex", static_cast<int>(texture_id::hue_curve_tex));
            } else {
                shader_->set("hue_curve_enable", false);
            }
        }

        // Sharpening
        if (std::abs(transforms.image_transform.sharpen_amount) > epsilon) {
            shader_->set("sharpen_enable", true);
            shader_->set("sharpen_amount", static_cast<float>(transforms.image_transform.sharpen_amount));
            shader_->set("sharpen_radius", static_cast<float>(transforms.image_transform.sharpen_radius));
        } else {
            shader_->set("sharpen_enable", false);
        }

        // Film grain
        if (std::abs(transforms.image_transform.grain_intensity) > epsilon) {
            shader_->set("grain_enable",    true);
            shader_->set("grain_intensity", static_cast<float>(transforms.image_transform.grain_intensity));
            shader_->set("grain_size",      static_cast<float>(transforms.image_transform.grain_size));
            shader_->set("grain_frame",     frame_counter_++);
        } else {
            shader_->set("grain_enable", false);
        }

        // Secondary qualifier
        if (transforms.image_transform.qualifier_enable) {
            shader_->set("qualifier_enable", true);
            shader_->set("qual_target_hue",  static_cast<float>(transforms.image_transform.qual_target_hue));
            shader_->set("qual_hue_width",   static_cast<float>(transforms.image_transform.qual_hue_width));
            shader_->set("qual_min_sat",     static_cast<float>(transforms.image_transform.qual_min_sat));
            shader_->set("qual_max_sat",     static_cast<float>(transforms.image_transform.qual_max_sat));
            shader_->set("qual_min_lum",     static_cast<float>(transforms.image_transform.qual_min_lum));
            shader_->set("qual_max_lum",     static_cast<float>(transforms.image_transform.qual_max_lum));
            shader_->set("qual_softness",    static_cast<float>(transforms.image_transform.qual_softness));
            shader_->set("qual_exposure",    static_cast<float>(transforms.image_transform.qual_exposure));
            shader_->set("qual_sat_offset",  static_cast<float>(transforms.image_transform.qual_sat_offset));
            shader_->set("qual_hue_offset",  static_cast<float>(transforms.image_transform.qual_hue_offset));
        } else {
            shader_->set("qualifier_enable", false);
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

        // Shape overlay
        {
            const auto& sh = transforms.image_transform.shape;
            if (sh.enable) {
                shader_->set("shape_enable",      true);
                shader_->set("shape_type",        static_cast<int>(sh.type));
                shader_->set("shape_fill_type",   static_cast<int>(sh.fill_type));
                shader_->set("shape_center",      sh.center[0], sh.center[1]);
                shader_->set("shape_size",        sh.size[0],   sh.size[1]);
                shader_->set("shape_corner_radius", sh.corner_radius);
                shader_->set("shape_softness",    sh.edge_softness);
                shader_->set("shape_color1",      sh.color1[0], sh.color1[1], sh.color1[2], sh.color1[3]);
                shader_->set("shape_color2",      sh.color2[0], sh.color2[1], sh.color2[2], sh.color2[3]);
                shader_->set("shape_gradient_angle",    sh.gradient_angle);
                shader_->set("shape_gradient_center",   sh.gradient_center[0],
                                                        sh.gradient_center[1]);
                shader_->set("shape_stroke_enable", sh.stroke_enable);
                shader_->set("shape_stroke_width",  sh.stroke_width);
                shader_->set("shape_stroke_color",  sh.stroke_color[0], sh.stroke_color[1],
                                                    sh.stroke_color[2], sh.stroke_color[3]);
            } else {
                shader_->set("shape_enable", false);
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

        GLenum draw_mode = (params.geometry.type() == core::frame_geometry::geometry_type::mesh) ? GL_TRIANGLES
                                                                                                 : GL_TRIANGLE_FAN;
        GL(glDrawArrays(draw_mode, 0, static_cast<GLsizei>(coords.size())));
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
