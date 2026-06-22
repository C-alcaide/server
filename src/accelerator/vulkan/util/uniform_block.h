#pragma once

#include <cstdint>

namespace caspar { namespace accelerator { namespace vulkan {

// Aligned to std140 rules for UBO.  Keep field order synchronised with
// fragment_shader.frag  `layout(std140, binding = 2) uniform ParamsBlock`.
// Total size: must stay under 4096 bytes (Vulkan minimum UBO max).
struct alignas(16) uniform_block
{
    // ── Core ────────────────────────────────────────────────────────────
    uint32_t color_space_index   = 0;       // 0
    float    precision_factor[4] = {1.f, 1.f, 1.f, 1.f}; // 4
    int32_t  blend_mode          = 0;       // 20
    int32_t  keyer               = 0;       // 24
    int32_t  pixel_format        = 0;       // 28
    float    opacity             = 1.0f;    // 32

    // ── Levels ──────────────────────────────────────────────────────────
    float min_input  = 0;                   // 36
    float max_input  = 0;                   // 40
    float gamma      = 0;                   // 44
    float min_output = 0;                   // 48
    float max_output = 0;                   // 52

    // ── CSB ─────────────────────────────────────────────────────────────
    float brt = 0;                          // 56
    float sat = 0;                          // 60
    float con = 0;                          // 64

    // ── Chroma ──────────────────────────────────────────────────────────
    float chroma_target_hue                = 0;  // 68
    float chroma_hue_width                 = 0;  // 72
    float chroma_min_saturation            = 0;  // 76
    float chroma_min_brightness            = 0;  // 80
    float chroma_softness                  = 0;  // 84
    float chroma_spill_suppress            = 0;  // 88
    float chroma_spill_suppress_saturation = 0;  // 92

    uint32_t flags = 0;                     // 96

    // ── 360° Projection ─────────────────────────────────────────────────
    float view_yaw       = 0;              // 100
    float view_pitch     = 0;              // 104
    float view_roll      = 0;              // 108
    float view_fov       = 1.5707963f;     // 112  (PI/2 = 90°)
    float aspect_ratio   = 1.777778f;      // 116
    float view_offset_x  = 0;             // 120
    float view_offset_y  = 0;             // 124
    float frustum_h      = 0;             // 128
    float frustum_v      = 0;             // 132
    float lens_k1        = 0;             // 136
    float lens_k2        = 0;             // 140
    float lens_k3        = 0;             // 144

    // ── Curved Screen ───────────────────────────────────────────────────
    int32_t screen_curve_type = 0;          // 148
    float   screen_arc        = 0;          // 152

    // ── Edge Blending ───────────────────────────────────────────────────
    float edge_blend_left   = 0;           // 156
    float edge_blend_right  = 0;           // 160
    float edge_blend_top    = 0;           // 164
    float edge_blend_bottom = 0;           // 168
    float edge_blend_gamma  = 2.2f;        // 172

    // ── Color Grading ───────────────────────────────────────────────────
    int32_t input_transfer   = 0;           // 176
    int32_t output_transfer  = 0;           // 180
    int32_t tone_mapping_op  = 0;           // 184
    float   exposure         = 1.0f;        // 188
    // mat3 in std140 = 3 × vec4 (each column padded to 16 bytes)
    float   display_peak_luminance = 1000.0f; // 192
    float   input_to_working[12]  = {      // 196  mat3 as 3×vec4 (std140)
        1,0,0,0,  0,1,0,0,  0,0,1,0
    };
    float   working_to_output[12] = {      // 244  mat3 as 3×vec4 (std140)
        1,0,0,0,  0,1,0,0,  0,0,1,0
    };

    // ── White Balance ───────────────────────────────────────────────────
    float wb_temperature = 0;              // 292
    float wb_tint        = 0;              // 296

    // ── Lift / Midtone / Gain ───────────────────────────────────────────
    float lmg_lift[3]    = {0, 0, 0};     // 300  vec3
    float _pad1 = 0;                       // 312  pad
    float lmg_midtone[3] = {1, 1, 1};     // 316  vec3
    float _pad2 = 0;                       // 328  pad
    float lmg_gain[3]    = {1, 1, 1};     // 332  vec3
    float _pad3 = 0;                       // 344  pad

    // ── Hue Shift ───────────────────────────────────────────────────────
    float hue_shift_degrees = 0;           // 348

    // ── Tonal Balance ───────────────────────────────────────────────────
    float tb_shadows    = 0;               // 352
    float tb_highlights = 0;               // 356

    // ── Linear Saturation ───────────────────────────────────────────────
    float linear_sat_value = 1.0f;         // 360

    // ── ASC CDL ─────────────────────────────────────────────────────────
    float cdl_slope[3]      = {1, 1, 1};   // 364  vec3
    float cdl_saturation    = 1.0f;         // 376
    float cdl_offset[3]     = {0, 0, 0};   // 380  vec3
    float _pad4 = 0;                        // 392  pad
    float cdl_power[3]      = {1, 1, 1};   // 396  vec3
    float _pad5 = 0;                        // 408  pad

    // ── Split Toning ────────────────────────────────────────────────────
    float split_shadow_color[3]    = {0, 0, 0}; // 412  vec3
    float split_balance            = 0.5f;       // 424
    float split_highlight_color[3] = {0, 0, 0}; // 428  vec3
    float _pad6 = 0;                             // 440  pad

    // ── Gamut Compression ───────────────────────────────────────────────
    float gc_limit[3] = {1.147f, 1.264f, 1.312f}; // 444  vec3
    float _pad7 = 0;                                // 456  pad

    // ── 3D LUT ──────────────────────────────────────────────────────────
    float lut3d_strength = 1.0f;           // 460

    // ── Sharpening ──────────────────────────────────────────────────────
    float sharpen_amount = 0;              // 464
    float sharpen_radius = 1.0f;           // 468

    // ── Film Grain ──────────────────────────────────────────────────────
    float   grain_intensity = 0;           // 472
    float   grain_size      = 1.0f;        // 476
    int32_t grain_frame     = 0;           // 480

    // ── Secondary Qualifier ─────────────────────────────────────────────
    float qual_target_hue = 0;             // 484
    float qual_hue_width  = 0;             // 488
    float qual_min_sat    = 0;             // 492
    float qual_max_sat    = 1.0f;          // 496
    float qual_min_lum    = 0;             // 500
    float qual_max_lum    = 1.0f;          // 504
    float qual_softness   = 0;             // 508
    float qual_exposure   = 0;             // 512
    float qual_sat_offset = 0;             // 516
    float qual_hue_offset = 0;             // 520

    // ── Per-Channel RGB Levels ──────────────────────────────────────────
    float rgb_levels_min_input[3]  = {0, 0, 0};   // 524
    float _pad8 = 0;                                // 536
    float rgb_levels_max_input[3]  = {1, 1, 1};   // 540
    float _pad9 = 0;                                // 552
    float rgb_levels_gamma[3]      = {1, 1, 1};   // 556
    float _padA = 0;                                // 568
    float rgb_levels_min_output[3] = {0, 0, 0};   // 572
    float _padB = 0;                                // 584
    float rgb_levels_max_output[3] = {1, 1, 1};   // 588
    float _padC = 0;                                // 600

    // ── Blur ────────────────────────────────────────────────────────────
    float   blur_radius = 0;               // 604
    int32_t blur_type   = 0;               // 608
    float   blur_angle  = 0;               // 612
    float   blur_center[2] = {0.5f, 0.5f}; // 616  vec2
    float   blur_tilt[2]   = {0.5f, 0.3f}; // 624  vec2
    float   target_size[2] = {1920, 1080};  // 632  vec2

    // ── Shape Overlay ───────────────────────────────────────────────────
    int32_t shape_type      = 0;           // 640
    int32_t shape_fill_type = 0;           // 644
    float   shape_center[2]     = {0.5f, 0.5f}; // 648  vec2
    float   shape_size[2]       = {0.5f, 0.5f}; // 656  vec2
    float   shape_corner_radius = 0;        // 664
    float   shape_softness      = 0.003f;   // 668
    float   shape_color1[4]     = {1,1,1,1}; // 672  vec4
    float   shape_color2[4]     = {0,0,0,0}; // 688  vec4
    float   shape_gradient_angle    = 0;     // 704
    float   shape_gradient_center[2] = {0.5f, 0.5f}; // 708  vec2
    float   shape_stroke_width  = 0;        // 716
    float   shape_stroke_color[4] = {1,1,1,1}; // 720  vec4

    // ── Extended flags2 (for features beyond 32 bits of 'flags') ────────
    uint32_t flags2 = 0;                    // 736
    float    eye_distance = 1.0f;           // 740  viewer distance / screen radius (k)
    int32_t  source_lens  = 0;              // 744  0=rectilinear,1=cyl,2=sph,3=fisheye
    float    screen_arc_v = 0;              // 748  vertical screen arc (rad); 0=cylinder,>0=sphere/dome

    // ── ICVFX inner/outer frustum (in-camera VFX) ───────────────────────
    float    inner_yaw      = 0;            // 752
    float    inner_pitch    = 0;            // 756
    float    inner_roll     = 0;            // 760
    float    inner_fov      = 1.5707963f;   // 764
    float    inner_offset_x = 0;            // 768
    float    inner_offset_y = 0;            // 772
    float    icvfx_q0x      = -1.0f;        // 776  camera-frustum mask quad (output NDC)
    float    icvfx_q0y      =  1.0f;        // 780  UL
    float    icvfx_q1x      =  1.0f;        // 784  UR
    float    icvfx_q1y      =  1.0f;        // 788
    float    icvfx_q2x      =  1.0f;        // 792  LR
    float    icvfx_q2y      = -1.0f;        // 796
    float    icvfx_q3x      = -1.0f;        // 800  LL
    float    icvfx_q3y      = -1.0f;        // 804
    float    icvfx_feather  = 0.05f;        // 808  mask edge feather (NDC units)
    float    icvfx_outer_dim = 1.0f;        // 812  outer-region brightness multiplier (0..1)
    // ── Lens tangential distortion (Brown-Conrady, OpenLensIO) ──────────
    // Appended at the end to preserve all existing offsets above.
    float    lens_p1        = 0;            // 816  tangential (decentering) coefficient
    float    lens_p2        = 0;            // 820  tangential (decentering) coefficient
    float    icvfx_inner_dim = 1.0f;        // 824  inner-region brightness multiplier (0..1)
    float    icvfx_inner_gain_r = 1.0f;     // 828  inner-region RGB gain (white-balance / tint)
    float    icvfx_inner_gain_g = 1.0f;     // 832
    float    icvfx_inner_gain_b = 1.0f;     // 836
    float    icvfx_outer_gain_r = 1.0f;     // 840  outer-region RGB gain (white-balance / tint)
    float    icvfx_outer_gain_g = 1.0f;     // 844
    float    icvfx_outer_gain_b = 1.0f;     // 848
    // Total: 852 bytes
};

// Bit flags for `flags` field
enum class shader_flags : uint32_t
{
    none              = 0,
    is_straight_alpha = 1u << 0,
    has_local_key     = 1u << 1,
    has_layer_key     = 1u << 2,
    invert            = 1u << 3,
    levels            = 1u << 4,
    csb               = 1u << 5,
    chroma            = 1u << 6,
    chroma_show_mask  = 1u << 7,
    // CasparVP extensions
    is_360            = 1u << 8,
    is_curved         = 1u << 9,
    color_grading     = 1u << 10,
    flip_h            = 1u << 11,
    flip_v            = 1u << 12,
    white_balance     = 1u << 13,
    lmg_enable        = 1u << 14,
    hue_shift_enable  = 1u << 15,
    tonebalance_enable= 1u << 16,
    linear_sat_enable = 1u << 17,
    cdl_enable        = 1u << 18,
    split_tone_enable = 1u << 19,
    gamut_compress    = 1u << 20,
    lut3d_enable      = 1u << 21,
    hue_curve_enable  = 1u << 22,
    sharpen_enable    = 1u << 23,
    grain_enable      = 1u << 24,
    qualifier_enable  = 1u << 25,
    rgb_levels_enable = 1u << 26,
    curves_enable     = 1u << 27,
    blur_enable       = 1u << 28,
    shape_enable      = 1u << 29,
    shape_stroke      = 1u << 30,
    edge_blend        = 1u << 31,
};

inline shader_flags operator|(shader_flags a, shader_flags b) { return static_cast<shader_flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b)); }

// Bit flags for `flags2` field (extended beyond 32 bits of `flags`)
enum class shader_flags2 : uint32_t
{
    none         = 0,
    output_bgra  = 1u << 0,  // Apply .bgra swizzle on fragment output (8-bit path)
    icvfx_enable = 1u << 1,  // Inner/outer frustum (in-camera VFX) blend active
    blend_mask   = 1u << 2,  // Per-pixel projection blend mask multiply active
};

}}} // namespace caspar::accelerator::vulkan
