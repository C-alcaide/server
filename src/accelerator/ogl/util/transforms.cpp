
#include "transforms.h"

#include <array>

// CASPAR_THROW_EXCEPTION in run_node_uv_self_test, which is FATAL rather than a warning.
#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>
#include <core/frame/transform_fields.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace caspar::accelerator::ogl {

draw_crop_region::draw_crop_region(double left, double top, double right, double bottom)
{
    // upper left
    coords[0]    = t_point(3);
    coords[0](0) = left;
    coords[0](1) = top;
    coords[0](2) = 1;

    // upper right
    coords[1]    = t_point(3);
    coords[1](0) = right;
    coords[1](1) = top;
    coords[1](2) = 1;

    // lower right
    coords[2]    = t_point(3);
    coords[2](0) = right;
    coords[2](1) = bottom;
    coords[2](2) = 1;

    // lower left
    coords[3]    = t_point(3);
    coords[3](0) = left;
    coords[3](1) = bottom;
    coords[3](2) = 1;
}

void draw_crop_region::apply_transform(const caspar::accelerator::ogl::t_matrix& matrix)
{
    coords[0] = coords[0] * matrix;
    coords[1] = coords[1] * matrix;
    coords[2] = coords[2] * matrix;
    coords[3] = coords[3] * matrix;
}

namespace {
constexpr const wchar_t* BACKEND_NAME = L"opengl";
}

void apply_transform_colour_values(core::image_transform& self, const core::image_transform& other)
{
    // Note: this intentionally does not affect any geometry-related fields, they follow a separate flow

    self.opacity *= other.opacity;
    self.brightness *= other.brightness;
    self.contrast *= other.contrast;
    self.saturation *= other.saturation;

    self.levels.min_input  = std::max(self.levels.min_input, other.levels.min_input);
    self.levels.max_input  = std::min(self.levels.max_input, other.levels.max_input);
    self.levels.min_output = std::max(self.levels.min_output, other.levels.min_output);
    self.levels.max_output = std::min(self.levels.max_output, other.levels.max_output);
    self.levels.gamma *= other.levels.gamma;
    self.chroma.enable |= other.chroma.enable;
    self.chroma.show_mask |= other.chroma.show_mask;
    self.chroma.target_hue     = std::max(other.chroma.target_hue, self.chroma.target_hue);
    self.chroma.min_saturation = std::max(other.chroma.min_saturation, self.chroma.min_saturation);
    self.chroma.min_brightness = std::max(other.chroma.min_brightness, self.chroma.min_brightness);
    self.chroma.hue_width      = std::max(other.chroma.hue_width, self.chroma.hue_width);
    self.chroma.softness       = std::max(other.chroma.softness, self.chroma.softness);
    self.chroma.spill_suppress = std::max(other.chroma.spill_suppress, self.chroma.spill_suppress);
    self.chroma.spill_suppress_saturation =
        std::min(other.chroma.spill_suppress_saturation, self.chroma.spill_suppress_saturation);
    self.is_key |= other.is_key;
    self.invert |= other.invert;
    self.flip_h ^= other.flip_h;
    self.flip_v ^= other.flip_v;
    self.is_mix |= other.is_mix;
    self.blend_mode = std::max(self.blend_mode, other.blend_mode);
    self.layer_depth += other.layer_depth;
    if (other.projection.enable) {
        // 360 virtual-camera view params (does not touch destination compensation)
        self.projection.enable      = true;
        self.projection.yaw         = other.projection.yaw;
        self.projection.pitch       = other.projection.pitch;
        self.projection.roll        = other.projection.roll;
        self.projection.fov         = other.projection.fov;
        self.projection.offset_x    = other.projection.offset_x;
        self.projection.offset_y    = other.projection.offset_y;
        self.projection.frustum_h   = other.projection.frustum_h;
        self.projection.frustum_v   = other.projection.frustum_v;
        self.projection.lens_k1     = other.projection.lens_k1;
        self.projection.lens_k2     = other.projection.lens_k2;
        self.projection.lens_k3     = other.projection.lens_k3;
        self.projection.lens_p1     = other.projection.lens_p1;
        self.projection.lens_p2     = other.projection.lens_p2;
        self.projection.source_lens = other.projection.source_lens;
    }
    // Curved screen compensation merges independently of 360 mode
    if (other.projection.curve_enable) {
        self.projection.curve_type   = other.projection.curve_type;
        self.projection.screen_arc   = other.projection.screen_arc;
        self.projection.screen_arc_v = other.projection.screen_arc_v;
        self.projection.eye_distance = other.projection.eye_distance;
        self.projection.curve_auto   = other.projection.curve_auto;
    }
    self.projection.curve_enable |= other.projection.curve_enable;
    // Edge blending merges independently as well
    if (other.projection.edge_blend_left > 0.0 || other.projection.edge_blend_right > 0.0 ||
        other.projection.edge_blend_top > 0.0 || other.projection.edge_blend_bottom > 0.0) {
        self.projection.edge_blend_left   = other.projection.edge_blend_left;
        self.projection.edge_blend_right  = other.projection.edge_blend_right;
        self.projection.edge_blend_top    = other.projection.edge_blend_top;
        self.projection.edge_blend_bottom = other.projection.edge_blend_bottom;
        self.projection.edge_blend_gamma  = other.projection.edge_blend_gamma;
    }
    // ICVFX inner/outer frustum merges independently of 360/curve modes
    if (other.projection.icvfx_enable) {
        self.projection.icvfx_enable       = true;
        self.projection.icvfx_auto         = other.projection.icvfx_auto;
        self.projection.inner_yaw          = other.projection.inner_yaw;
        self.projection.inner_pitch        = other.projection.inner_pitch;
        self.projection.inner_roll         = other.projection.inner_roll;
        self.projection.inner_fov          = other.projection.inner_fov;
        self.projection.inner_eye_distance = other.projection.inner_eye_distance;
        self.projection.inner_offset_x     = other.projection.inner_offset_x;
        self.projection.inner_offset_y     = other.projection.inner_offset_y;
        self.projection.icvfx_q0x          = other.projection.icvfx_q0x;
        self.projection.icvfx_q0y          = other.projection.icvfx_q0y;
        self.projection.icvfx_q1x          = other.projection.icvfx_q1x;
        self.projection.icvfx_q1y          = other.projection.icvfx_q1y;
        self.projection.icvfx_q2x          = other.projection.icvfx_q2x;
        self.projection.icvfx_q2y          = other.projection.icvfx_q2y;
        self.projection.icvfx_q3x          = other.projection.icvfx_q3x;
        self.projection.icvfx_q3y          = other.projection.icvfx_q3y;
        self.projection.icvfx_feather      = other.projection.icvfx_feather;
        self.projection.icvfx_outer_dim    = other.projection.icvfx_outer_dim;
        self.projection.icvfx_inner_dim    = other.projection.icvfx_inner_dim;
        self.projection.icvfx_inner_gain_r = other.projection.icvfx_inner_gain_r;
        self.projection.icvfx_inner_gain_g = other.projection.icvfx_inner_gain_g;
        self.projection.icvfx_inner_gain_b = other.projection.icvfx_inner_gain_b;
        self.projection.icvfx_outer_gain_r = other.projection.icvfx_outer_gain_r;
        self.projection.icvfx_outer_gain_g = other.projection.icvfx_outer_gain_g;
        self.projection.icvfx_outer_gain_b = other.projection.icvfx_outer_gain_b;
    }
    if (other.color_grade.enable) {
        self.color_grade = other.color_grade;
    }
    // Easy to miss, and it fails silently: this merge is explicit field by field, so a new
    // image_transform member that is not listed here simply never reaches the kernel. The
    // symptom is a command that reports 202 and changes nothing.
    if (other.ocio.enable) {
        self.ocio = other.ocio;
    }
    if (other.blur.enable) {
        self.blur = other.blur;
    }
    if (other.shape.enable) {
        self.shape = other.shape;
    }

    // Every combined grading value below is clamped back into the range its MIXER command
    // accepts (core::grade_limits, the same table the commands validate against). Two
    // layers at the edge of legal would otherwise reach a value no single command could
    // set: lift 0.9 over lift 0.9 is 1.8, and a midtone of 0.2 under another 0.2 is an
    // exponent of 25. Clamping here rather than in the shader keeps the value the kernel
    // reports and the value it renders the same thing.
    //
    // The bounds are wide enough that ordinary stacking is untouched -- gain 2 over gain 2
    // is 4, well inside [0, 10]. hue_shift is the exception and is wrapped instead, below,
    // because rotation is periodic. Both mixers must carry this identically.
    namespace lim = core::grade_limits;

    // White balance: additive combination
    self.temperature = lim::temperature.clamp(self.temperature + other.temperature);
    self.tint        = lim::tint.clamp(self.tint + other.tint);

    // Lift/Midtone/Gain: additive lift, multiplicative midtone+gain
    for (int i = 0; i < 3; ++i) {
        self.lift[i]    = lim::lift.clamp(self.lift[i] + other.lift[i]);
        self.midtone[i] = lim::midtone.clamp(self.midtone[i] * other.midtone[i]);
        self.gain[i]    = lim::gain.clamp(self.gain[i] * other.gain[i]);
    }

    // Hue shift: additive, then wrapped into [-180, 180].
    //
    // Wrapped, not clamped: 200 degrees of rotation is -160, not 180. The shader rotates
    // with fract(), so rotation is already periodic and this does not change what is
    // rendered -- what it does is bound the accumulation. Stacking layer, channel and
    // tweened transforms can otherwise walk hue_shift arbitrarily far from zero, which
    // costs precision in fract() and makes the "is this effect active" test dishonest:
    // an accumulated 360 is exactly identity but reads as active and pays for the branch.
    self.hue_shift = std::remainder(self.hue_shift + other.hue_shift, 360.0);

    // Tone balance: additive
    self.shadows    = lim::tone.clamp(self.shadows + other.shadows);
    self.highlights = lim::tone.clamp(self.highlights + other.highlights);

    // Linear saturation: multiplicative (default 1)
    self.linear_saturation = lim::cdl_saturation.clamp(self.linear_saturation * other.linear_saturation);

    // ASC CDL
    for (int i = 0; i < 3; ++i) {
        self.cdl_slope[i]  = lim::cdl_slope.clamp(self.cdl_slope[i] * other.cdl_slope[i]);
        self.cdl_offset[i] = lim::cdl_offset.clamp(self.cdl_offset[i] + other.cdl_offset[i]);
        self.cdl_power[i]  = lim::cdl_power.clamp(self.cdl_power[i] * other.cdl_power[i]);
    }
    self.cdl_saturation = lim::cdl_saturation.clamp(self.cdl_saturation * other.cdl_saturation);

    // Split toning: colours add, and the balance of whichever transform actually has
    // split toning active wins.
    //
    // split_balance is a crossover position in luma, not a strength, so no arithmetic
    // composition of two of them means anything: adding 0.5 and 0.5 gives 1.0 (everything
    // is shadow), multiplying gives 0.25. And the shader has one crossover and one colour
    // pair, so two stacked split tones cannot both be represented however they are
    // combined. Summing the colours and keeping a single crossover is the least-lossy
    // approximation available. "Is other active" is the same test the mixer uses to decide
    // whether to enable the effect at all; comparing other.split_balance against its 0.5
    // default instead would misread a tweened 0.4999999 as an explicit setting.
    const auto is_set = [](double v) { return v != 0.0; };
    const bool other_splits =
        std::any_of(other.split_shadow_color.begin(), other.split_shadow_color.end(), is_set) ||
        std::any_of(other.split_highlight_color.begin(), other.split_highlight_color.end(), is_set);
    for (int i = 0; i < 3; ++i) {
        self.split_shadow_color[i] =
            lim::split_color.clamp(self.split_shadow_color[i] + other.split_shadow_color[i]);
        self.split_highlight_color[i] =
            lim::split_color.clamp(self.split_highlight_color[i] + other.split_highlight_color[i]);
    }
    if (other_splits)
        self.split_balance = lim::split_balance.clamp(other.split_balance);

    // Exposure. MULTIPLIES, like opacity at the top of this function: nested transforms
    // each contribute a gain and the composition of two gains is their product.
    //
    // It has to be here at all because this function is an ALLOWLIST -- a field a layer
    // transform sets is dropped unless it is named. That is what made `MIXER EXPOSURE`
    // return 202 and change nothing: the value reached the stage and never reached the
    // kernel, which read the default 1.0 every frame. Both mixers keep their own copy of
    // this list, so a new colour field has to be added twice or the backends diverge.
    self.exposure = lim::exposure.clamp(self.exposure * other.exposure);

    // Gamut compression
    if (other.gamut_compress) {
        self.gamut_compress = true;
        self.gc_cyan        = other.gc_cyan;
        self.gc_magenta     = other.gc_magenta;
        self.gc_yellow      = other.gc_yellow;
    }

    // Per-pixel projection blend mask.
    //
    // Innermost wins, like the LUT and the hue curves below: two masks cannot be composed
    // without resampling one onto the other's raster, and silently picking a resampling
    // rule is worse than picking the layer's own.
    //
    // It has to be named here at all for the reason `exposure` above does: this function is
    // an ALLOWLIST, and a field it does not mention is dropped during composition. Measured
    // 2026-08-13 before the fix -- `MIXER PROJECTION_BLEND_MASK` returned 202, the query
    // read the mask back at its right dimensions, and all four patches rendered
    // byte-identical to no mask at all on both backends. `cli.py blend-mask`.
    if (other.blend_mask) {
        self.blend_mask = other.blend_mask;
    }

    // Grading node chain -- innermost wins, same rule and same reason as the mask above:
    // two graphs cannot be composed without resampling one set of windows onto the other's
    // space. Must be mirrored in the Vulkan copy of this function or the backends diverge in
    // a way no single-backend test can see.
    if (other.node_plan) {
        // BOTH MEMBERS TOGETHER, always. The values are indexed by offsets THIS plan carries,
        // so taking one without the other is an out-of-range read on the frame path -- and a
        // field missing from this list is silently dropped with the command still answering 202,
        // which is what `apply_transform_colour_values`' whole header warns about.
        self.node_plan   = other.node_plan;
        self.node_values = other.node_values;
    }

    // 3D LUT
    if (other.lut3d) {
        self.lut3d          = other.lut3d;
        self.lut3d_strength = other.lut3d_strength;
    }

    // Hue curves
    if (other.hue_curves) {
        self.hue_curves = other.hue_curves;
    }

    // Sharpening
    self.sharpen_amount += other.sharpen_amount;
    if (other.sharpen_radius != 1.0)
        self.sharpen_radius = other.sharpen_radius;

    // Film grain
    self.grain_intensity += other.grain_intensity;
    if (other.grain_size != 1.0)
        self.grain_size = other.grain_size;

    // Secondary qualifier
    if (other.qualifier_enable) {
        self.qualifier_enable = true;
        self.qual_target_hue  = other.qual_target_hue;
        self.qual_hue_width   = other.qual_hue_width;
        self.qual_min_sat     = other.qual_min_sat;
        self.qual_max_sat     = other.qual_max_sat;
        self.qual_min_lum     = other.qual_min_lum;
        self.qual_max_lum     = other.qual_max_lum;
        self.qual_softness    = other.qual_softness;
        self.qual_exposure    = other.qual_exposure;
        self.qual_sat_offset  = other.qual_sat_offset;
        self.qual_hue_offset  = other.qual_hue_offset;
    }

    // Per-channel RGB levels: intersect input range, multiply gamma, intersect output range (per-channel)
    if (other.per_channel_levels.enable) {
        self.per_channel_levels.enable = true;
        auto merge_ch = [](core::rgb_levels_channel& s, const core::rgb_levels_channel& o) {
            s.min_input  = std::max(s.min_input,  o.min_input);
            s.max_input  = std::min(s.max_input,  o.max_input);
            s.gamma     *= o.gamma;
            s.min_output = std::max(s.min_output, o.min_output);
            s.max_output = std::min(s.max_output, o.max_output);
        };
        merge_ch(self.per_channel_levels.r, other.per_channel_levels.r);
        merge_ch(self.per_channel_levels.g, other.per_channel_levels.g);
        merge_ch(self.per_channel_levels.b, other.per_channel_levels.b);
    }

    // Tone curves: if other has curves enabled, override self (curves don't compose additively)
    if (other.curves.enable) {
        self.curves = other.curves;
    }
}

bool is_default_perspective(const core::corners& perspective)
{
    return perspective.ul[0] == 0 && perspective.ul[1] == 0 && perspective.ur[0] == 1 && perspective.ur[1] == 0 &&
           perspective.ll[0] == 0 && perspective.ll[1] == 1 && perspective.lr[0] == 1 && perspective.lr[1] == 1;
}

draw_transforms draw_transforms::combine_transform(const core::image_transform& transform, double aspect_ratio) const
{
    draw_transforms new_transform(image_transform, steps);

    auto transform_before = new_transform.current().vertex_matrix;

    // Get matrix for turning coords in 'transform' into the parent frame.
    auto new_matrix = get_vertex_matrix(transform, aspect_ratio);

    apply_transform_colour_values(new_transform.image_transform, transform);
    new_transform.current().vertex_matrix = new_matrix * new_transform.current().vertex_matrix;

    // Only enable this for some transforms, to avoid applying crops when a draw_frame is just being used to flatten
    // other draw_frames
    if (transform.enable_geometry_modifiers) {
        // Push the new clip before the new transform applied
        draw_crop_region new_clip(transform.clip_translation[0],
                                  transform.clip_translation[1],
                                  transform.clip_translation[0] + transform.clip_scale[0],
                                  transform.clip_translation[1] + transform.clip_scale[1]);
        new_clip.apply_transform(transform_before);
        new_transform.current().crop_regions.push_back(std::move(new_clip));

        if (!is_default_perspective(transform.perspective)) {
            // Split into a new step
            new_transform.steps.emplace_back(transform.perspective,
                                             boost::numeric::ublas::identity_matrix<double>(3, 3));
        }

        // Push the new crop region with the new transform applied
        draw_crop_region new_crop(
            transform.crop.ul[0], transform.crop.ul[1], transform.crop.lr[0], transform.crop.lr[1]);
        new_crop.apply_transform(new_transform.current().vertex_matrix);
        new_transform.current().crop_regions.push_back(std::move(new_crop));
    }

    return std::move(new_transform);
}

void apply_perspective_to_vertex(t_point& vertex, const core::corners& perspective)
{
    const double x = vertex(0);
    const double y = vertex(1);

    // ul: x' =  (1-y) * a + (1 - a * (1-y)) * x
    vertex(0) += (1 - y) * perspective.ul[0] + (1 - perspective.ul[0] + perspective.ul[0] * y) * x - x;
    vertex(1) += (1 - x) * perspective.ul[1] + (1 - perspective.ul[1] + perspective.ul[1] * x) * y - y;

    // ur/ll: x' = x * (a * (1-y) + y)
    vertex(0) += x * (perspective.ur[0] * (1 - y) + y) - x;
    vertex(1) += y * (perspective.ll[1] * (1 - x) + x) - y;

    // ur/ll: x' = y * a + x * (1 - a * y)
    vertex(0) += y * perspective.ll[0] + x * (1 - perspective.ll[0] * y) - x;
    vertex(1) += x * perspective.ur[1] + y * (1 - perspective.ur[1] * x) - y;

    // lr: x' = x * (y * a + (1-y))
    vertex(0) += x * (y * perspective.lr[0] + (1 - y)) - x;
    vertex(1) += y * (x * perspective.lr[1] + (1 - x)) - y;
}

struct wrapped_vertex
{
    explicit wrapped_vertex(const core::frame_geometry::coord& coord)
    {
        vertex(0) = coord.vertex_x;
        vertex(1) = coord.vertex_y;
        vertex(2) = 1;

        texture_x = coord.texture_x;
        texture_y = coord.texture_y;
        texture_r = coord.texture_r;
        texture_q = coord.texture_q;
    }

    explicit wrapped_vertex() { vertex(2) = 1; };

    [[nodiscard]] core::frame_geometry::coord as_geometry() const
    {
        core::frame_geometry::coord res = {vertex(0), vertex(1), texture_x, texture_y};
        res.texture_r                   = texture_r;
        res.texture_q                   = texture_q;
        return res;
    }

    t_point vertex = t_point(3);

    double texture_x = 0.0;
    double texture_y = 0.0;
    double texture_r = 0.0;
    double texture_q = 1.0;
};

static const double epsilon = 0.001;

bool inline point_is_outside_of_line(const t_point& line_1,
                                     const t_point& line_2,
                                     const t_point& vertex,
                                     bool           invert_winding)
{
    // use a cross product to check if the point is outside the crop region
    auto cross = (line_2(0) - line_1(0)) * (vertex(1) - line_1(1)) - (line_2(1) - line_1(1)) * (vertex(0) - line_1(0));
    return invert_winding ? cross > epsilon : cross < -epsilon;
}

// http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
bool get_intersection_with_crop_line(const t_point& crop0,
                                     const t_point& crop1,
                                     const t_point& p0,
                                     const t_point& p1,
                                     t_point&       result)
{
    double s1_x = crop1(0) - crop0(0);
    double s1_y = crop1(1) - crop0(1);
    double s2_x = p1(0) - p0(0);
    double s2_y = p1(1) - p0(1);

    double s = (-s1_y * (crop0(0) - p0(0)) + s1_x * (crop0(1) - p0(1))) / (-s2_x * s1_y + s1_x * s2_y);
    double t = (s2_x * (crop0(1) - p0(1)) - s2_y * (crop0(0) - p0(0))) / (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1) {
        // Collision detected
        result(0) = crop0(0) + t * s1_x;
        result(1) = crop0(1) + t * s1_y;

        return true;
    }

    return false; // No collision
}

double hypotenuse(double x1, double y1, double x2, double y2)
{
    auto x = x2 - x1;
    auto y = y2 - y1;

    return std::sqrt(x * x + y * y);
}

double calc_q(double close_diagonal, double distant_diagonal)
{
    return (close_diagonal + distant_diagonal) / distant_diagonal;
}

void crop_texture_for_vertex(const wrapped_vertex& line_a, const wrapped_vertex& line_b, wrapped_vertex& vertex)
{
    auto delta_point = vertex.vertex - line_a.vertex;
    auto delta_line  = line_b.vertex - line_a.vertex;

    // Calculate the dot product
    auto dot_product      = delta_point(0) * delta_line(0) + delta_point(1) * delta_line(1);
    auto line_len_squared = delta_line(0) * delta_line(0) + delta_line(1) * delta_line(1);

    // Skip if line has no length
    if (line_len_squared == 0) {
        vertex.texture_x = line_a.texture_x;
        vertex.texture_y = line_a.texture_y;
        return;
    }

    auto dist_delta = dot_product / line_len_squared;

    vertex.texture_x = line_a.texture_x + dist_delta * (line_b.texture_x - line_a.texture_x);
    vertex.texture_y = line_a.texture_y + dist_delta * (line_b.texture_y - line_a.texture_y);
    vertex.texture_q = line_a.texture_q + dist_delta * (line_b.texture_q - line_a.texture_q);
}

void fill_texture_q_for_quad(std::vector<wrapped_vertex>& coords)
{
    if (coords.size() != 4)
        return;

    // Based on formula from:
    // http://www.reedbeta.com/blog/2012/05/26/quadrilateral-interpolation-part-1/

    double s1_x = coords[2].vertex(0) - coords[0].vertex(0);
    double s1_y = coords[2].vertex(1) - coords[0].vertex(1);
    double s2_x = coords[3].vertex(0) - coords[1].vertex(0);
    double s2_y = coords[3].vertex(1) - coords[1].vertex(1);

    double s =
        (-s1_y * (coords[0].vertex(0) - coords[1].vertex(0)) + s1_x * (coords[0].vertex(1) - coords[1].vertex(1))) /
        (-s2_x * s1_y + s1_x * s2_y);
    double t =
        (s2_x * (coords[0].vertex(1) - coords[1].vertex(1)) - s2_y * (coords[0].vertex(0) - coords[1].vertex(0))) /
        (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
        // Collision detected
        double diagonal_intersection_x = coords[0].vertex(0) + t * s1_x;
        double diagonal_intersection_y = coords[0].vertex(1) + t * s1_y;

        auto d0 =
            hypotenuse(coords[3].vertex(0), coords[3].vertex(1), diagonal_intersection_x, diagonal_intersection_y);
        auto d1 =
            hypotenuse(coords[2].vertex(0), coords[2].vertex(1), diagonal_intersection_x, diagonal_intersection_y);
        auto d2 =
            hypotenuse(coords[1].vertex(0), coords[1].vertex(1), diagonal_intersection_x, diagonal_intersection_y);
        auto d3 =
            hypotenuse(coords[0].vertex(0), coords[0].vertex(1), diagonal_intersection_x, diagonal_intersection_y);

        auto ulq = calc_q(d3, d1);
        auto urq = calc_q(d2, d0);
        auto lrq = calc_q(d1, d3);
        auto llq = calc_q(d0, d2);

        std::vector<double> q_values = {ulq, urq, lrq, llq};

        int corner = 0;
        for (auto& coord : coords) {
            coord.texture_q = q_values[corner];
            coord.texture_x *= q_values[corner];
            coord.texture_y *= q_values[corner];

            if (++corner == 4)
                corner = 0;
        }
    }
}

void transform_vertex(const draw_transform_step& step, t_point& vertex)
{
    // Apply basic transforms of this step
    vertex = vertex * step.vertex_matrix;

    // Apply perspective. These rely on x and y of the coord, so can't be done as a shared matrix
    apply_perspective_to_vertex(vertex, step.perspective);
}

std::vector<core::frame_geometry::coord>
draw_transforms::transform_coords(const std::vector<core::frame_geometry::coord>& coords) const
{
    // Convert to matrix representations
    std::vector<wrapped_vertex> cropped_coords;
    cropped_coords.reserve(coords.size());

    for (const auto& coord : coords) {
        cropped_coords.emplace_back(coord);
    }

    std::vector<draw_crop_region> transformed_regions;

    // Apply the transforms
    for (int i = (int)steps.size() - 1; i >= 0; i--) {
        for (auto& coord : cropped_coords) {
            transform_vertex(steps[i], coord.vertex);
        }

        // Transform existing regions
        for (auto& region : transformed_regions) {
            for (int l = 0; l < 4; ++l) {
                transform_vertex(steps[i], region.coords[l]);
            }
        }

        // Push new regions
        for (auto& region : steps[i].crop_regions) {
            draw_crop_region new_region = region;
            for (int l = 0; l < 4; ++l) {
                // Only apply perspective for new ones
                apply_perspective_to_vertex(new_region.coords[l], steps[i].perspective);
            }

            transformed_regions.push_back(new_region);
        }
    }

    // Apply the perspective correction
    fill_texture_q_for_quad(cropped_coords);

    // Perform the crop
    for (auto& crop_region : transformed_regions) {
        // Determine the winding order of the crop region
        // Calculate the signed area to determine if the region is clockwise or counter-clockwise
        double signed_area = 0.0;
        for (int l = 0; l < 4; ++l) {
            int next_l = (l + 1) % 4;
            signed_area += (crop_region.coords[next_l](0) - crop_region.coords[l](0)) *
                           (crop_region.coords[next_l](1) + crop_region.coords[l](1));
        }
        // In screen coordinates (Y down), normal clockwise winding gives negative signed area
        // We need to invert the test when the winding is flipped (positive signed area)
        bool invert_winding = signed_area > 0;

        for (int l = 0; l < 4; ++l) {
            // Apply the crop, one edge at a time
            int     to_index   = l == 3 ? 0 : l + 1;
            t_point from_point = crop_region.coords[l];
            t_point to_point   = crop_region.coords[to_index];

            std::unordered_set<size_t> points_outside_of_line;

            // Figure out which points are outside the crop region
            for (size_t j = 0; j < cropped_coords.size(); ++j) {
                bool v = point_is_outside_of_line(from_point, to_point, cropped_coords[j].vertex, invert_winding);
                if (v)
                    points_outside_of_line.insert(j);
            }

            if (points_outside_of_line.empty()) {
                // Line has no effect, skip
                continue;
            } else if (points_outside_of_line.size() == cropped_coords.size()) {
                // All are outside, shape has no geometry
                return {};
            }

            std::vector<wrapped_vertex> new_coords;
            new_coords.reserve(cropped_coords.size() * 2); // Avoid reallocs for complex shapes

            // Iterate through the coords
            for (size_t j = 0; j < cropped_coords.size(); ++j) {
                if (points_outside_of_line.count(j) == 0) {
                    new_coords.push_back(cropped_coords[j]);
                    continue;
                }

                size_t prev_index = j == 0 ? cropped_coords.size() - 1 : j - 1;
                size_t next_index = j == cropped_coords.size() - 1 ? 0 : j + 1;

                bool prev_is_outside = points_outside_of_line.count(prev_index) == 1;
                bool next_is_outside = points_outside_of_line.count(next_index) == 1;

                if (prev_is_outside && next_is_outside) {
                    // Vertex and its edges are completely outside, skip
                    continue;
                }

                if (!prev_is_outside) {
                    // This edge intersects the crop line, calculate the new coordinates
                    wrapped_vertex new_coord;
                    if (get_intersection_with_crop_line(to_point,
                                                        from_point,
                                                        cropped_coords[prev_index].vertex,
                                                        cropped_coords[j].vertex,
                                                        new_coord.vertex)) {
                        crop_texture_for_vertex(cropped_coords[prev_index], cropped_coords[j], new_coord);
                        new_coords.emplace_back(std::move(new_coord));
                    } else {
                        // Geometry error! skip coordinate
                    }
                }

                if (!next_is_outside) {
                    // This edge intersects the crop line, calculate the new coordinates
                    wrapped_vertex new_coord;
                    if (get_intersection_with_crop_line(to_point,
                                                        from_point,
                                                        cropped_coords[j].vertex,
                                                        cropped_coords[next_index].vertex,
                                                        new_coord.vertex)) {
                        crop_texture_for_vertex(cropped_coords[j], cropped_coords[next_index], new_coord);
                        new_coords.emplace_back(std::move(new_coord));
                    } else {
                        // Geometry error! skip coordinate
                    }
                }
            }

            // Polygon is cropped, update state
            cropped_coords = new_coords;
        }

        {
            static const double pixel_epsilon = 0.0001; // less than a pixel at 8k

            // Prune duplicate coords
            std::vector<wrapped_vertex> new_coords;
            new_coords.reserve(cropped_coords.size()); // Avoid reallocs

            for (size_t j = 0; j < cropped_coords.size(); ++j) {
                size_t prev_index = j == 0 ? cropped_coords.size() - 1 : j - 1;

                auto delta = cropped_coords[j].vertex - cropped_coords[prev_index].vertex;
                if (std::abs(delta(0)) > pixel_epsilon || std::abs(delta(1)) > pixel_epsilon) {
                    new_coords.emplace_back(cropped_coords[j]);
                }
            }

            if (new_coords.size() < 3) {
                // Not enough coords to draw anything
                return {};
            }
            cropped_coords = new_coords;
        }
    }

    // Convert back to frame_geometry types
    std::vector<core::frame_geometry::coord> result;
    result.reserve(cropped_coords.size());
    for (auto& coord : cropped_coords) {
        result.push_back(coord.as_geometry());
    }

    return result;
}


void run_compose_self_test()
{
    const auto rep = core::fields::compose_self_test(&apply_transform_colour_values);

    if (rep.divergences == 0) {
        CASPAR_LOG(info) << L"[core] compose self-test: " << BACKEND_NAME << L" " << rep.fields << L" fields, "
                         << rep.iterations << L" iterations, 0 divergences.";
        return;
    }

    std::wstring names;
    for (const auto& n : rep.diverged) {
        if (!names.empty())
            names += L", ";
        names += u16(n);
    }

    // A warning rather than a throw. The generated composition is not on the frame path
    // yet, so a divergence here breaks nothing that is running -- but it means the registry
    // and this backend disagree about a rule, and whichever of them a reader trusts is
    // now wrong. It becomes fatal on the commit that makes the mixer CALL the generated
    // version.
    CASPAR_LOG(warning) << L"[core] compose self-test: " << BACKEND_NAME << L" " << rep.divergences << L" of "
                        << rep.iterations << L" iterations diverged, in: " << names;
}


/// The geometry SCALE MODE's contribution to an item's placement, as a further composed
/// transform.
///
/// Extracted from both kernels' `draw()`, which is where it lived, because the node path needs
/// the same answer: a source-space mask is evaluated through the inverse of the item's full
/// placement, and the scale mode is part of that placement. Two hand-written copies of this
/// switch would diverge exactly the way `apply_transform_colour_values` already warns about.
///
/// `stretch` and a zero-sized plane both return the input unchanged, which is why the caller
/// does not have to test for them.
draw_transforms apply_geometry_scale_mode(const draw_transforms&      transforms,
                                          const core::frame_geometry& geometry,
                                          int                         target_width,
                                          int                         target_height,
                                          int                         plane_width,
                                          int                         plane_height,
                                          double                      aspect_ratio)
{
    if (geometry.mode() == core::frame_geometry::scale_mode::stretch || plane_width <= 0 || plane_height <= 0)
        return transforms;

    const auto width_scale  = static_cast<double>(target_width) / static_cast<double>(plane_width);
    const auto height_scale = static_cast<double>(target_height) / static_cast<double>(plane_height);

    core::image_transform transform;
    double                target_scale;
    switch (geometry.mode()) {
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

    return transforms.combine_transform(transform, aspect_ratio);
}

/// FRAME UV -> the item's own 0..1 source UV, as a 3x3 in the row-vector convention
/// `transform_coords` uses.
///
/// `transform_coords` walks the steps BACK TO FRONT -- `vertex = vertex * steps[i].vertex_matrix`
/// for i from last to first -- so a vertex placed by the chain equals `v * M[n-1] * ... * M[0]`,
/// and THAT product is what has to be inverted. The default geometry's vertex and texture
/// coordinates are the same 0..1 quad (`frame_geometry::get_default`), so the forward product
/// maps source UV to frame UV and its inverse is exactly what a node pass masking in source
/// space needs.
///
/// Returns FALSE, and the caller must then fall back to frame space and say so, in two cases:
///
///   * any step carries a non-default `perspective`. A corner pin is not a 3x3 by construction --
///     `apply_perspective_to_vertex` reads x and y of the coord, which is why `transform_vertex`
///     applies it per step instead of folding it into the matrix. Approximating it would put the
///     mask somewhere plausible and wrong, which is worse than putting it in frame space and
///     saying so.
///   * the placement is singular. `MIXER FILL x y 0 1` is how a layer is hidden without being
///     cleared, and it reaches here as a determinant of zero.
///
/// WHAT `out` HOLDS IS DEFINED BY HOW THE SHADER USES IT, not by the maths that produced it:
/// three ROWS of a matrix R for which `R * vec3(uv, 1)` is the item's uv, so both shaders write
/// exactly that and neither has to know which way `transform_coords` composes. `inv` is in the
/// row-vector convention, so R is its transpose -- `out[r * 3 + c] = inv(c, r)`.
///
/// THE CONVENTION IS NOT ASSERTED HERE, IT IS MEASURED, and in the same form: `node_uv_self_test`
/// runs the default quad through `transform_coords` and checks that `R * vec3(placed_vertex, 1)`
/// returns each vertex's own texture coordinate. A row/column slip produces a matrix that looks
/// right and masks the wrong region, and nothing else in the build would notice.
bool source_uv_inverse(const draw_transforms& transforms, std::array<float, 9>& out)
{
    // `is_default_perspective` rather than a hand-rolled comparison: it is the same predicate
    // `combine_transform` uses to decide whether a perspective needs its own STEP, so the two
    // cannot disagree about what counts as a corner pin.
    for (const auto& step : transforms.steps)
        if (!is_default_perspective(step.perspective))
            return false;

    t_matrix composed = boost::numeric::ublas::identity_matrix<double>(3, 3);
    for (int i = static_cast<int>(transforms.steps.size()) - 1; i >= 0; --i)
        composed = composed * transforms.steps[i].vertex_matrix;

    t_matrix inv;
    if (!invert_3x3(composed, inv))
        return false;

    // TRANSPOSED on the way out, so `out` is the row form both shaders multiply a column vector
    // by. See the note above -- this one line is the whole convention.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = static_cast<float>(inv(c, r));
    return true;
}


void run_node_uv_self_test()
{
    // THE CONVENTION, MEASURED RATHER THAN ASSERTED.
    //
    // `source_uv_inverse` hands back three rows of a matrix R for which `R * vec3(uv, 1)` is the
    // item's own uv. Whether that is the transpose of the composed placement or the placement
    // itself depends on which way `transform_coords` multiplies, and getting it backwards
    // produces a matrix that looks entirely reasonable and masks the wrong region of the frame.
    // Nothing else in the build would notice: it compiles, it runs, and only a picture under a
    // non-default `MIXER FILL` disagrees.
    //
    // So the check is the round trip that defines the quantity: place the default quad through
    // the real `transform_coords`, then take each PLACED vertex back through R and require its
    // own TEXTURE coordinate. The default quad's vertex and texture coordinates are the same
    // 0..1 pair (`frame_geometry::get_default`), which is what makes the item's uv well defined
    // in the first place.
    //
    // Cases chosen so a single mistake cannot pass all of them: identity (passes under almost
    // any error), a pure translation (catches a dropped or negated offset), a non-uniform scale
    // (catches a row/column swap, which a uniform scale is invariant under), an off-centre
    // anchor with rotation (catches an order-of-composition error, which translation alone is
    // invariant under), and a corner pin (must be REFUSED, not approximated).
    struct arm
    {
        const wchar_t*        name;
        core::image_transform t;
        bool                  expect_affine;
    };

    std::vector<arm> arms;
    {
        arm a{L"identity", core::image_transform(), true};
        arms.push_back(a);
    }
    {
        arm a{L"translation", core::image_transform(), true};
        a.t.fill_translation = {0.25, -0.125};
        arms.push_back(a);
    }
    {
        // NON-UNIFORM on purpose: a square scale is invariant under a row/column swap, which is
        // exactly the error this whole self-test exists to catch.
        arm a{L"non-uniform scale + offset", core::image_transform(), true};
        a.t.fill_scale       = {0.5, 1.0};
        a.t.fill_translation = {0.25, 0.0};
        arms.push_back(a);
    }
    {
        arm a{L"anchored rotation", core::image_transform(), true};
        a.t.anchor           = {0.5, 0.5};
        a.t.angle            = 0.4;
        a.t.fill_scale       = {0.8, 0.6};
        a.t.fill_translation = {0.1, 0.2};
        arms.push_back(a);
    }
    {
        arm a{L"corner pin -- must be refused", core::image_transform(), false};
        // BOTH, and the flag is the half that is easy to miss: `combine_transform` only splits a
        // perspective into its own step when `enable_geometry_modifiers` is set, so without this
        // line the corner pin is silently dropped, the placement stays affine, and the arm would
        // have asserted a refusal that never had anything to refuse.
        a.t.enable_geometry_modifiers = true;
        a.t.perspective.ur            = {0.9, 0.05};
        arms.push_back(a);
    }

    int failures = 0;
    for (const auto& a : arms) {
        const auto placement = draw_transforms().combine_transform(a.t, 1.0);

        std::array<float, 9> r{};
        const bool           ok = source_uv_inverse(placement, r);

        if (ok != a.expect_affine) {
            CASPAR_LOG(error) << L"[core] node-uv self-test: " << BACKEND_NAME << L" " << a.name
                              << L" -- source_uv_inverse returned " << (ok ? L"true" : L"false")
                              << L", expected " << (a.expect_affine ? L"true" : L"false");
            ++failures;
            continue;
        }
        if (!ok)
            continue;

        const auto placed = placement.transform_coords(core::frame_geometry::get_default().data());
        for (const auto& c : placed) {
            // `R * vec3(vertex, 1)`, spelled exactly as both shaders spell it.
            const double x = r[0] * c.vertex_x + r[1] * c.vertex_y + r[2];
            const double y = r[3] * c.vertex_x + r[4] * c.vertex_y + r[5];
            const double w = r[6] * c.vertex_x + r[7] * c.vertex_y + r[8];
            if (std::abs(w) < 1e-9)
                continue;

            // 1e-4 of a normalised coordinate: a fifth of a pixel at 1080p, and four orders
            // coarser than double precision needs -- because the uniform is a `float` and this
            // check has to be about the CONVENTION rather than about rounding.
            // NORMALISED BY `texture_q`, and the self-test found that for me on its first run.
            // `transform_coords` ends with `fill_texture_q_for_quad`, which multiplies
            // `texture_x`/`texture_y` by a per-corner perspective factor and stores it in
            // `texture_q` -- and for a quad with NO perspective that factor is 2, not 1
            // (`calc_q` is (close + distant) / distant, and the two diagonals are equal). So the
            // coordinates coming out of here are HOMOGENEOUS, the shader divides by `texture_q`
            // when it samples, and comparing against the raw `texture_x` expects 2 where the
            // answer is 1.
            //
            // All four affine arms failed by exactly that factor on the first run, which is how
            // a wrong EXPECTATION was told apart from a wrong matrix: a row/column slip would
            // have produced four DIFFERENT wrong answers, not one consistent factor.
            const double tx = c.texture_q != 0.0 ? c.texture_x / c.texture_q : c.texture_x;
            const double ty = c.texture_q != 0.0 ? c.texture_y / c.texture_q : c.texture_y;
            if (std::abs(x / w - tx) > 1e-4 || std::abs(y / w - ty) > 1e-4) {
                CASPAR_LOG(error) << L"[core] node-uv self-test: " << BACKEND_NAME << L" " << a.name
                                  << L" -- vertex (" << c.vertex_x << L", " << c.vertex_y
                                  << L") came back as source uv (" << (x / w) << L", " << (y / w)
                                  << L"), expected (" << tx << L", " << ty << L")";
                ++failures;
                break;
            }
        }
    }

    if (failures == 0) {
        CASPAR_LOG(info) << L"[core] node-uv self-test: " << BACKEND_NAME << L" " << arms.size()
                         << L" placements, 0 failures.";
        return;
    }

    // FATAL, unlike `run_compose_self_test` above, and the difference is which way the mistake
    // points. A compose divergence is between two tables, only one of which is on the frame
    // path. This matrix IS on the frame path the moment a mask declares `space: source`, and a
    // wrong one puts a grade over the wrong part of the picture with no error anywhere -- which
    // is the class of defect this repository has paid for most often.
    CASPAR_THROW_EXCEPTION(caspar_exception()
                           << msg_info(L"node-uv self-test failed with " + std::to_wstring(failures) +
                                       L" failure(s) -- see the log. A source-space node mask would "
                                       L"grade the wrong region."));
}

} // namespace caspar::accelerator::ogl
