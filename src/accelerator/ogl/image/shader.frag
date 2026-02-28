#version 450
in vec4 TexCoord;
in vec4 TexCoord2;
out vec4 fragColor;

uniform sampler2D	background;
uniform sampler2D	plane[4];
uniform sampler2D	local_key;
uniform sampler2D	layer_key;
uniform sampler2D	curve_lut_tex;

uniform bool        is_straight_alpha;

uniform mat3		color_matrix;
uniform vec3		luma_coeff;
uniform bool		has_local_key;
uniform bool		has_layer_key;
uniform int			blend_mode;
uniform int			keyer;
uniform int			pixel_format;

uniform bool        invert;
uniform bool        flip_h;  // mirror left <-> right
uniform bool        flip_v;  // mirror top  <-> bottom
uniform float		opacity;
uniform bool		levels;
uniform float		min_input;
uniform float		max_input;
uniform float		gamma;
uniform float		min_output;
uniform float		max_output;
uniform float	    precision_factor[4];

uniform bool		csb;
uniform float		brt;
uniform float		sat;
uniform float		con;

uniform bool		chroma;
uniform bool		chroma_show_mask;
uniform float		chroma_target_hue;
uniform float		chroma_hue_width;
uniform float		chroma_min_saturation;
uniform float		chroma_min_brightness;
uniform float		chroma_softness;
uniform float		chroma_spill_suppress;
uniform float		chroma_spill_suppress_saturation;

uniform bool  is_360;
uniform float view_yaw;      // in radians
uniform float view_pitch;    // in radians
uniform float view_roll;     // in radians
uniform float view_fov;      // Vertical FOV in radians
uniform float aspect_ratio;  // Screen aspect ratio (width/height)
uniform float view_offset_x; // NDC lens-shift: +1 = pan right
uniform float view_offset_y; // NDC lens-shift: +1 = pan up

// Color Grading (ACES workflow)
uniform bool  color_grading;
uniform int   input_transfer;    // 0=linear,1=srgb,2=rec709,3=pq,4=hlg,5=logc3,6=slog3
uniform int   output_transfer;
uniform mat3  input_to_working;  // input gamut -> ACEScg (AP1)
uniform mat3  working_to_output; // ACEScg (AP1) -> output gamut
uniform int   tone_mapping_op;   // 0=none,1=reinhard,2=aces_filmic,3=aces_rrt
uniform float exposure;          // linear exposure multiplier

// White balance
uniform bool  white_balance;
uniform float wb_temperature;   // -1..+1 (neg=cool/blue, pos=warm/orange)
uniform float wb_tint;          // -1..+1 (neg=magenta, pos=green)

// Lift / Midtone / Gain (3-way color corrector: DaVinci Resolve primary wheels)
uniform bool  lmg_enable;
uniform vec3  lmg_lift;         // shadow offset per channel, default vec3(0)
uniform vec3  lmg_midtone;      // midtone power per channel,  default vec3(1)
uniform vec3  lmg_gain;         // highlight multiplier per channel, default vec3(1)

// Hue shift
uniform bool  hue_shift_enable;
uniform float hue_shift_degrees;  // -180..+180

// Tonal balance (shadows / highlights)
uniform bool  tonebalance_enable;
uniform float tb_shadows;     // -1..+1
uniform float tb_highlights;  // -1..+1

// Per-channel RGB Levels (independent levels per R, G, B channel)
uniform bool  rgb_levels_enable;
uniform float rgb_levels_min_input[3];   // [0]=R [1]=G [2]=B
uniform float rgb_levels_max_input[3];
uniform float rgb_levels_gamma[3];
uniform float rgb_levels_min_output[3];
uniform float rgb_levels_max_output[3];

// Tone Curves: per-channel 1D LUTs (256 entries) built on the CPU from control points
// using Fritsch-Carlson monotone cubic Hermite splines.
uniform bool  curves_enable;
// curve LUT data is in curve_lut_tex (sampler2D 256x1 RGBA32F: r=R-curve, g=G-curve, b=B-curve, a=master-curve)

/*
** Contrast, saturation, brightness
** Code of this function is from TGM's shader pack
** http://irrlicht.sourceforge.net/phpBB2/viewtopic.php?t=21057
*/

vec3 ContrastSaturationBrightness(vec4 color, float brt, float sat, float con)
{
    const float AvgLumR = 0.5;
    const float AvgLumG = 0.5;
    const float AvgLumB = 0.5;

    vec3 LumCoeff = luma_coeff.bgr;

    if (color.a > 0.0)
        color.rgb /= color.a;

    vec3 AvgLumin = vec3(AvgLumR, AvgLumG, AvgLumB);
    vec3 brtColor = color.rgb * brt;
    vec3 intensity = vec3(dot(brtColor, LumCoeff));
    vec3 satColor = mix(intensity, brtColor, sat);
    vec3 conColor = mix(AvgLumin, satColor, con);

    conColor.rgb *= color.a;

    return conColor;
}

/*
** Gamma correction
** Details: http://blog.mouaif.org/2009/01/22/photoshop-gamma-correction-shader/
*/
#define GammaCorrection(color, gamma)								pow(color, vec3(1.0 / gamma))

/*
** Levels control (input (+gamma), output)
** Details: http://blog.mouaif.org/2009/01/28/levels-control-shader/
*/

#define LevelsControlInputRange(color, minInput, maxInput)				min(max(color - vec3(minInput), vec3(0.0)) / (vec3(maxInput) - vec3(minInput)), vec3(1.0))
#define LevelsControlInput(color, minInput, gamma, maxInput)				GammaCorrection(LevelsControlInputRange(color, minInput, maxInput), gamma)
#define LevelsControlOutputRange(color, minOutput, maxOutput) 			mix(vec3(minOutput), vec3(maxOutput), color)
#define LevelsControl(color, minInput, gamma, maxInput, minOutput, maxOutput) 	LevelsControlOutputRange(LevelsControlInput(color, minInput, gamma, maxInput), minOutput, maxOutput)

/*
** Photoshop & misc math
** Blending modes, RGB/HSL/Contrast/Desaturate, levels control
**
** Romain Dura | Romz
** Blog: http://blog.mouaif.org
** Post: http://blog.mouaif.org/?p=94
*/


/*
** Desaturation
*/

vec4 Desaturate(vec3 color, float Desaturation)
{
    vec3 grayXfer = vec3(0.3, 0.59, 0.11);
    vec3 gray = vec3(dot(grayXfer, color));
    return vec4(mix(color, gray, Desaturation), 1.0);
}


/*
** Hue, saturation, luminance
*/

vec3 RGBToHSL(vec3 color)
{
    vec3 hsl;

    float fmin = min(min(color.r, color.g), color.b);
    float fmax = max(max(color.r, color.g), color.b);
    float delta = fmax - fmin;

    hsl.z = (fmax + fmin) / 2.0;

    if (delta == 0.0)
    {
        hsl.x = 0.0;
        hsl.y = 0.0;
    }
    else
    {
        if (hsl.z < 0.5)
            hsl.y = delta / (fmax + fmin);
        else
            hsl.y = delta / (2.0 - fmax - fmin);

        float deltaR = (((fmax - color.r) / 6.0) + (delta / 2.0)) / delta;
        float deltaG = (((fmax - color.g) / 6.0) + (delta / 2.0)) / delta;
        float deltaB = (((fmax - color.b) / 6.0) + (delta / 2.0)) / delta;

        if (color.r == fmax )
            hsl.x = deltaB - deltaG;
        else if (color.g == fmax)
            hsl.x = (1.0 / 3.0) + deltaR - deltaB;
        else if (color.b == fmax)
            hsl.x = (2.0 / 3.0) + deltaG - deltaR;

        if (hsl.x < 0.0)
            hsl.x += 1.0;
        else if (hsl.x > 1.0)
            hsl.x -= 1.0;
    }

    return hsl;
}

float HueToRGB(float f1, float f2, float hue)
{
    if (hue < 0.0)
        hue += 1.0;
    else if (hue > 1.0)
        hue -= 1.0;
    float res;
    if ((6.0 * hue) < 1.0)
        res = f1 + (f2 - f1) * 6.0 * hue;
    else if ((2.0 * hue) < 1.0)
        res = f2;
    else if ((3.0 * hue) < 2.0)
        res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    else
        res = f1;
    return res;
}

vec3 HSLToRGB(vec3 hsl)
{
    vec3 rgb;

    if (hsl.y == 0.0)
        rgb = vec3(hsl.z);
    else
    {
        float f2;

        if (hsl.z < 0.5)
            f2 = hsl.z * (1.0 + hsl.y);
        else
            f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);

        float f1 = 2.0 * hsl.z - f2;

        rgb.r = HueToRGB(f1, f2, hsl.x + (1.0/3.0));
        rgb.g = HueToRGB(f1, f2, hsl.x);
        rgb.b= HueToRGB(f1, f2, hsl.x - (1.0/3.0));
    }

    return rgb;
}




/*
** Float blending modes
** Adapted from here: http://www.nathanm.com/photoshop-blending-math/
** But I modified the HardMix (wrong condition), Overlay, SoftLight, ColorDodge, ColorBurn, VividLight, PinLight (inverted layers) ones to have correct results
*/

#define BlendLinearDodgef 					BlendAddf
#define BlendLinearBurnf 					BlendSubstractf
#define BlendAddf(base, blend) 				min(base + blend, 1.0)
#define BlendSubstractf(base, blend) 		max(base + blend - 1.0, 0.0)
#define BlendLightenf(base, blend) 		max(blend, base)
#define BlendDarkenf(base, blend) 			min(blend, base)
#define BlendLinearLightf(base, blend) 	(blend < 0.5 ? BlendLinearBurnf(base, (2.0 * blend)) : BlendLinearDodgef(base, (2.0 * (blend - 0.5))))
#define BlendScreenf(base, blend) 			(1.0 - ((1.0 - base) * (1.0 - blend)))
#define BlendOverlayf(base, blend) 		(base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend)))
#define BlendSoftLightf(base, blend) 		((blend < 0.5) ? (2.0 * base * blend + base * base * (1.0 - 2.0 * blend)) : (sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)))
#define BlendColorDodgef(base, blend) 		((blend == 1.0) ? blend : min(base / (1.0 - blend), 1.0))
#define BlendColorBurnf(base, blend) 		((blend == 0.0) ? blend : max((1.0 - ((1.0 - base) / blend)), 0.0))
#define BlendVividLightf(base, blend)		((blend < 0.5) ? BlendColorBurnf(base, (2.0 * blend)) : BlendColorDodgef(base, (2.0 * (blend - 0.5))))
#define BlendPinLightf(base, blend) 		((blend < 0.5) ? BlendDarkenf(base, (2.0 * blend)) : BlendLightenf(base, (2.0 *(blend - 0.5))))
#define BlendHardMixf(base, blend) 		((BlendVividLightf(base, blend) < 0.5) ? 0.0 : 1.0)
#define BlendReflectf(base, blend) 		((blend == 1.0) ? blend : min(base * base / (1.0 - blend), 1.0))


/*
** Vector3 blending modes
*/

#define Blend(base, blend, funcf) 			vec3(funcf(base.r, blend.r), funcf(base.g, blend.g), funcf(base.b, blend.b))

#define BlendNormal(base, blend) 			(blend)
#define BlendLighten						BlendLightenf
#define BlendDarken						BlendDarkenf
#define BlendMultiply(base, blend) 		(base * blend)
#define BlendAverage(base, blend) 			((base + blend) / 2.0)
#define BlendAdd(base, blend) 				min(base + blend, vec3(1.0))
#define BlendSubstract(base, blend) 		max(base + blend - vec3(1.0), vec3(0.0))
#define BlendDifference(base, blend) 		abs(base - blend)
#define BlendNegation(base, blend) 		(vec3(1.0) - abs(vec3(1.0) - base - blend))
#define BlendExclusion(base, blend) 		(base + blend - 2.0 * base * blend)
#define BlendScreen(base, blend) 			Blend(base, blend, BlendScreenf)
#define BlendOverlay(base, blend) 			Blend(base, blend, BlendOverlayf)
#define BlendSoftLight(base, blend) 		Blend(base, blend, BlendSoftLightf)
#define BlendHardLight(base, blend) 		BlendOverlay(blend, base)
#define BlendColorDodge(base, blend) 		Blend(base, blend, BlendColorDodgef)
#define BlendColorBurn(base, blend) 		Blend(base, blend, BlendColorBurnf)
#define BlendLinearDodge					BlendAdd
#define BlendLinearBurn					BlendSubstract
#define BlendLinearLight(base, blend) 		Blend(base, blend, BlendLinearLightf)
#define BlendVividLight(base, blend) 		Blend(base, blend, BlendVividLightf)
#define BlendPinLight(base, blend) 		Blend(base, blend, BlendPinLightf)
#define BlendHardMix(base, blend) 			Blend(base, blend, BlendHardMixf)
#define BlendReflect(base, blend) 			Blend(base, blend, BlendReflectf)
#define BlendGlow(base, blend) 			BlendReflect(blend, base)
#define BlendPhoenix(base, blend) 			(min(base, blend) - max(base, blend) + vec3(1.0))
#define BlendOpacity(base, blend, F, O) 	(F(base, blend) * O + blend * (1.0 - O))


vec3 BlendHue(vec3 base, vec3 blend)
{
    vec3 baseHSL = RGBToHSL(base);
    return HSLToRGB(vec3(RGBToHSL(blend).r, baseHSL.g, baseHSL.b));
}

vec3 BlendSaturation(vec3 base, vec3 blend)
{
    vec3 baseHSL = RGBToHSL(base);
    return HSLToRGB(vec3(baseHSL.r, RGBToHSL(blend).g, baseHSL.b));
}

vec3 BlendColor(vec3 base, vec3 blend)
{
    vec3 blendHSL = RGBToHSL(blend);
    return HSLToRGB(vec3(blendHSL.r, blendHSL.g, RGBToHSL(base).b));
}

vec3 BlendLuminosity(vec3 base, vec3 blend)
{
    vec3 baseHSL = RGBToHSL(base);
    return HSLToRGB(vec3(baseHSL.r, baseHSL.g, RGBToHSL(blend).b));
}

// Chroma keying
// Author: Tim Eves <timseves@googlemail.com>
//
// This implements the Chroma key algorithm described in the paper:
//      'Software Chroma Keying in an Imersive Virtual Environment'
//      by F. van den Bergh & V. Lalioti
// but as a pixel shader algorithm.
//
vec4  grey_xfer  = vec4(luma_coeff, 0);

// This allows us to implement the paper's alphaMap curve in software
// rather than a largeish array
float alpha_map(float d)
{
    return 1.0 - smoothstep(1.0, chroma_softness, d);
}

// http://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// From the same page
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float AngleDiff(float angle1, float angle2)
{
    return 0.5 - abs(abs(angle1 - angle2) - 0.5);
}

float AngleDiffDirectional(float angle1, float angle2)
{
    float diff = angle1 - angle2;

    return diff < -0.5
            ? diff + 1.0
            : (diff > 0.5 ? diff - 1.0 : diff);
}

float Distance(float actual, float target)
{
    return min(0.0, target - actual);
}

float ColorDistance(vec3 hsv)
{
    float hueDiff					= AngleDiff(hsv.x, chroma_target_hue) * 2;
    float saturationDiff			= Distance(hsv.y, chroma_min_saturation);
    float brightnessDiff			= Distance(hsv.z, chroma_min_brightness);

    float saturationBrightnessScore	= max(brightnessDiff, saturationDiff);
    float hueScore					= hueDiff - chroma_hue_width;

    return -hueScore * saturationBrightnessScore;
}

vec3 supress_spill(vec3 c)
{
    float hue		= c.x;
    float diff		= AngleDiffDirectional(hue, chroma_target_hue);
    float distance	= abs(diff) / chroma_spill_suppress;

    if (distance < 1)
    {
        c.x = diff < 0
                ? chroma_target_hue - chroma_spill_suppress
                : chroma_target_hue + chroma_spill_suppress;
        c.y *= min(1.0, distance + chroma_spill_suppress_saturation);
    }

    return c;
}

// Key on any color
vec4 ChromaOnCustomColor(vec4 c)
{
    vec3 hsv		= rgb2hsv(c.rgb);
    float distance	= ColorDistance(hsv);
    float d			= distance * -2.0 + 1.0;
    vec4 suppressed	= vec4(hsv2rgb(supress_spill(hsv)), 1.0);
    float alpha		= alpha_map(d);

    suppressed *= alpha;

    return chroma_show_mask ? vec4(suppressed.a, suppressed.a, suppressed.a, 1) : suppressed;
}



// ---- White Balance ----
// temp: -1=cool (blue), +1=warm (orange); tint_val: -1=magenta, +1=green
vec3 apply_white_balance(vec3 c, float temp, float tint_val)
{
    // NOTE: shader internal convention is color.r=Blue_displayed, color.b=Red_displayed
    // (all formats go through a .bgra swizzle; fragColor=color.bgra restores at output)
    // So warm = more Red = more color.b, less Blue = less color.r
    c.b = clamp(c.b + temp     * 0.20, 0.0, 1.0);  // warm -> boost color.b (= Red displayed)
    c.g = clamp(c.g + tint_val * 0.10, 0.0, 1.0);
    c.r = clamp(c.r - temp     * 0.20, 0.0, 1.0);  // warm -> reduce color.r (= Blue displayed)
    return c;
}

// ---- Lift / Midtone / Gain (3-way color corrector) ----
// Mirrors DaVinci Resolve's Lift/Gamma/Gain primary wheels.
// lift:    per-channel shadow offset  (-0.5..+0.5, default 0)
// midtone: per-channel midtone power  (0.1..4.0,   default 1)
// gain:    per-channel highlight mult (0..4.0,      default 1)
vec3 apply_lmg(vec3 c, vec3 lift, vec3 midtone, vec3 gain)
{
    c = clamp(c * gain + lift, 0.0, 1.0);
    c = pow(c, max(vec3(0.01), 1.0 / midtone));
    return c;
}

// ---- Hue Shift ----
vec3 apply_hue_shift(vec3 c, float degrees)
{
    vec3 hsv = rgb2hsv(c);
    hsv.x    = fract(hsv.x + degrees / 360.0);
    return hsv2rgb(hsv);
}

// ---- Tonal Balance (Shadows / Highlights separation) ----
// Mirrors DaVinci Resolve's Shadows and Highlights sliders.
// shadows   > 0 lifts dark areas;    shadows   < 0 crushes them.
// highlights > 0 lifts bright areas; highlights < 0 crushes them.
vec3 apply_tone_balance(vec3 c, float shadows, float highlights)
{
    float lum         = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float shadow_mask = 1.0 - smoothstep(0.0, 0.6, lum);
    float hl_mask     = smoothstep(0.4, 1.0, lum);
    c = clamp(c + vec3(shadows    * 0.5 * shadow_mask), 0.0, 1.0);
    c = clamp(c + vec3(highlights * 0.5 * hl_mask),     0.0, 1.0);
    return c;
}

// ---- Per-channel RGB Levels ----
// Applies independent input range, gamma, and output range per R, G, B channel.
vec3 apply_rgb_levels(vec3 c)
{
    // R
    c.r = clamp((c.r - rgb_levels_min_input[0]) / max(rgb_levels_max_input[0] - rgb_levels_min_input[0], 0.0001), 0.0, 1.0);
    c.r = pow(c.r, 1.0 / max(rgb_levels_gamma[0], 0.01));
    c.r = mix(rgb_levels_min_output[0], rgb_levels_max_output[0], c.r);
    // G
    c.g = clamp((c.g - rgb_levels_min_input[1]) / max(rgb_levels_max_input[1] - rgb_levels_min_input[1], 0.0001), 0.0, 1.0);
    c.g = pow(c.g, 1.0 / max(rgb_levels_gamma[1], 0.01));
    c.g = mix(rgb_levels_min_output[1], rgb_levels_max_output[1], c.g);
    // B
    c.b = clamp((c.b - rgb_levels_min_input[2]) / max(rgb_levels_max_input[2] - rgb_levels_min_input[2], 0.0001), 0.0, 1.0);
    c.b = pow(c.b, 1.0 / max(rgb_levels_gamma[2], 0.01));
    c.b = mix(rgb_levels_min_output[2], rgb_levels_max_output[2], c.b);
    return c;
}

// ---- Tone Curves: bilinear sample of a 256-entry float LUT ----
// ch: 0=R, 1=G, 2=B, 3=Master
float sample_lut_256(float val, int ch)
{
    float s  = clamp(val, 0.0, 1.0) * 255.0;
    int   lo = int(s);
    int   hi = min(lo + 1, 255);
    float f  = fract(s);
    vec4 lo4 = texelFetch(curve_lut_tex, ivec2(lo, 0), 0);
    vec4 hi4 = texelFetch(curve_lut_tex, ivec2(hi, 0), 0);
    vec4 v4  = mix(lo4, hi4, f);
    if (ch == 0) return v4.r;
    if (ch == 1) return v4.g;
    if (ch == 2) return v4.b;
    return v4.a;
}

vec3 apply_curves(vec3 c)
{
    // Per-channel curves first
    c.r = sample_lut_256(c.r, 0);
    c.g = sample_lut_256(c.g, 1);
    c.b = sample_lut_256(c.b, 2);
    // Master curve applied as global tone (same remap to each channel)
    c.r = sample_lut_256(c.r, 3);
    c.g = sample_lut_256(c.g, 3);
    c.b = sample_lut_256(c.b, 3);
    return c;
}

vec3 get_blend_color(vec3 back, vec3 fore)
{
    switch(blend_mode)
    {
    case  0: return BlendNormal(back, fore);
    case  1: return BlendLighten(back, fore);
    case  2: return BlendDarken(back, fore);
    case  3: return BlendMultiply(back, fore);
    case  4: return BlendAverage(back, fore);
    case  5: return BlendAdd(back, fore);
    case  6: return BlendSubstract(back, fore);
    case  7: return BlendDifference(back, fore);
    case  8: return BlendNegation(back, fore);
    case  9: return BlendExclusion(back, fore);
    case 10: return BlendScreen(back, fore);
    case 11: return BlendOverlay(back, fore);
//	case 12: return BlendSoftLight(back, fore);
    case 13: return BlendHardLight(back, fore);
    case 14: return BlendColorDodge(back, fore);
    case 15: return BlendColorBurn(back, fore);
    case 16: return BlendLinearDodge(back, fore);
    case 17: return BlendLinearBurn(back, fore);
    case 18: return BlendLinearLight(back, fore);
    case 19: return BlendVividLight(back, fore);
    case 20: return BlendPinLight(back, fore);
    case 21: return BlendHardMix(back, fore);
    case 22: return BlendReflect(back, fore);
    case 23: return BlendGlow(back, fore);
    case 24: return BlendPhoenix(back, fore);
    case 25: return BlendHue(back, fore);
    case 26: return BlendSaturation(back, fore);
    case 27: return BlendColor(back, fore);
    case 28: return BlendLuminosity(back, fore);
    }
    return BlendNormal(back, fore);
}

vec4 blend(vec4 fore)
{
    vec4 back = texture(background, TexCoord2.st).bgra;
    if(blend_mode != 0)
        fore.rgb = get_blend_color(back.rgb/(back.a+0.0000001), fore.rgb/(fore.a+0.0000001))*fore.a;
    switch(keyer)
    {
        case 1:  return fore + back; // additive
        default: return fore + (1.0-fore.a)*back; // linear
    }
}

vec4 chroma_key(vec4 c)
{
    return ChromaOnCustomColor(c.bgra).bgra;
}

vec4 ycbcra_to_rgba(float Y, float Cb, float Cr, float A)
{
    const float luma_coefficient = 255.0/219.0;
    const float chroma_coefficient = 255.0/224.0;

    vec3 YCbCr = vec3(Y, Cb, Cr) * 255;
    YCbCr -= vec3(16.0, 128.0, 128.0);
    YCbCr *= vec3(luma_coefficient, chroma_coefficient, chroma_coefficient);

    return vec4(color_matrix * YCbCr / 255, A).bgra;
}

// ---- Color Grading: EOTFs (encoded -> linear) ----
float eotf_srgb(float x)   { return x <= 0.04045  ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4); }
float eotf_rec709(float x) { return x < 0.081     ? x / 4.5   : pow((x + 0.099) / 1.099, 1.0 / 0.45); }
float eotf_pq(float x) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float xp = pow(max(x, 0.0), 1.0 / m2);
    return pow(max(xp - c1, 0.0) / (c2 - c3 * xp), 1.0 / m1) * (10000.0 / 100.0);
}
float eotf_hlg(float x) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    return x <= 0.5 ? (x * x) / 3.0 : (exp((x - c) / a) + b) / 12.0;
}
float eotf_logc3(float x) {
    const float a = 5.555556, b = 0.052272, c = 0.247190, d = 0.385537, e = 5.367655, f = 0.092809;
    return x > e * 0.010591 + f ? (pow(10.0, (x - d) / c) - b) / a : (x - f) / e;
}
float eotf_slog3(float x) {
    const float cut = 171.2102946929 / 1023.0;
    return x >= cut ? pow(10.0, (x - 0.410557184750733) / 0.341132524981570) * 0.18 + 0.01
                    : (x - 95.0 / 1023.0) * 0.01 / (cut - 95.0 / 1023.0);
}
vec3 apply_eotf(vec3 rgb, int t) {
    switch (t) {
        case 1: return vec3(eotf_srgb(rgb.r),  eotf_srgb(rgb.g),  eotf_srgb(rgb.b));
        case 2: return vec3(eotf_rec709(rgb.r), eotf_rec709(rgb.g), eotf_rec709(rgb.b));
        case 3: return vec3(eotf_pq(rgb.r),    eotf_pq(rgb.g),    eotf_pq(rgb.b));
        case 4: return vec3(eotf_hlg(rgb.r),   eotf_hlg(rgb.g),   eotf_hlg(rgb.b));
        case 5: return vec3(eotf_logc3(rgb.r), eotf_logc3(rgb.g), eotf_logc3(rgb.b));
        case 6: return vec3(eotf_slog3(rgb.r), eotf_slog3(rgb.g), eotf_slog3(rgb.b));
        default: return rgb;
    }
}
// ---- Color Grading: OETFs (linear -> encoded) ----
float oetf_srgb(float x)   { return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(max(x, 0.0), 1.0 / 2.4) - 0.055; }
float oetf_rec709(float x) { return x < 0.018      ? x * 4.5   : 1.099 * pow(max(x, 0.0), 0.45) - 0.099; }
float oetf_pq(float x) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float xn = pow(clamp(x * 100.0 / 10000.0, 0.0, 1.0), m1);
    return pow((c1 + c2 * xn) / (1.0 + c3 * xn), m2);
}
float oetf_hlg(float x) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    x = max(x, 0.0);
    return x <= 1.0 / 12.0 ? sqrt(3.0 * x) : a * log(12.0 * x - b) + c;
}
vec3 apply_oetf(vec3 rgb, int t) {
    rgb = max(rgb, vec3(0.0));
    switch (t) {
        case 1: return vec3(oetf_srgb(rgb.r),  oetf_srgb(rgb.g),  oetf_srgb(rgb.b));
        case 2: return vec3(oetf_rec709(rgb.r), oetf_rec709(rgb.g), oetf_rec709(rgb.b));
        case 3: return vec3(oetf_pq(rgb.r),    oetf_pq(rgb.g),    oetf_pq(rgb.b));
        case 4: return vec3(oetf_hlg(rgb.r),   oetf_hlg(rgb.g),   oetf_hlg(rgb.b));
        default: return rgb;
    }
}
// ---- Color Grading: Tone Mapping ----
vec3 tonemap_reinhard(vec3 v)    { return v / (v + 1.0); }
vec3 tonemap_aces_filmic(vec3 x) { return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
vec3 tonemap_aces_rrt(vec3 v) {
    v *= 0.6;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.432951) + 0.238081;
    return clamp(a / b, 0.0, 1.0);
}
vec3 apply_tone_mapping(vec3 rgb, int op) {
    switch (op) {
        case 1: return tonemap_reinhard(rgb);
        case 2: return tonemap_aces_filmic(rgb);
        case 3: return tonemap_aces_rrt(rgb);
        default: return rgb;
    }
}

const float PI = 3.14159265359;

vec2 get_equirect_uv(vec2 screen_uv) {
    // 1. Convert Screen UV (0..1) to Normalized Device Coordinates (-1..1)
    vec2 ndc = screen_uv * 2.0 - 1.0;
    // Lens-shift: slide the viewport across the sphere without changing orientation
    ndc -= vec2(view_offset_x, view_offset_y);
    
    // 2. Calculate View Vector (Rectilinear Projection)
    // Assume scale is based on vertical FOV
    float scale = tan(view_fov * 0.5);
    vec3 dir = vec3(ndc.x * scale * aspect_ratio, ndc.y * scale, -1.0);
    dir = normalize(dir);

    // 3. Rotation Matrices
    // Roll (Z)
    float cr = cos(view_roll); float sr = sin(view_roll);
    mat3 rot_z = mat3(cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0);
    
    // Pitch (X)
    float cp = cos(view_pitch); float sp = sin(view_pitch);
    mat3 rot_x = mat3(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
    
    // Yaw (Y)
    float cy = cos(view_yaw); float sy = sin(view_yaw);
    mat3 rot_y = mat3(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);

    // Apply rotations: Yaw * Pitch * Roll * dir
    dir = rot_y * rot_x * rot_z * dir;

    // 4. Convert 3D Vector to Spherical Coordinates (Longitude/Latitude)
    float theta = atan(dir.x, -dir.z); // Longitude (-PI to PI)
    float phi = asin(dir.y);           // Latitude (-PI/2 to PI/2)

    // 5. Map to Equirectangular UV (0..1)
    vec2 uv;
    uv.x = 0.5 + theta / (2.0 * PI);
    uv.y = 0.5 + phi / PI;
    
    return uv;
}

vec4 get_sample(sampler2D sampler, vec2 coords)
{
    return texture(sampler, coords);
}

vec4 get_rgba_color(vec2 uv)
{
    switch(pixel_format)
    {
    case 0:		//gray
        return vec4(get_sample(plane[0], uv).rrr * precision_factor[0], 1.0);
    case 1:		//bgra,
        return get_sample(plane[0], uv).bgra * precision_factor[0];
    case 2:		//rgba,
        return get_sample(plane[0], uv).rgba * precision_factor[0];
    case 3:		//argb,
        return get_sample(plane[0], uv).argb * precision_factor[0];
    case 4:		//abgr,
        return get_sample(plane[0], uv).gbar * precision_factor[0];
    case 5:		//ycbcr,
        {
            float y  = get_sample(plane[0], uv).r * precision_factor[0];
            float cb = get_sample(plane[1], uv).r * precision_factor[1];
            float cr = get_sample(plane[2], uv).r * precision_factor[2];
            return ycbcra_to_rgba(y, cb, cr, 1.0);
        }
    case 6:		//ycbcra
        {
            float y  = get_sample(plane[0], uv).r * precision_factor[0];
            float cb = get_sample(plane[1], uv).r * precision_factor[1];
            float cr = get_sample(plane[2], uv).r * precision_factor[2];
            float a  = get_sample(plane[3], uv).r * precision_factor[3];
            return ycbcra_to_rgba(y, cb, cr, a);
        }
    case 7:		//luma
        {
            vec3 y3 = get_sample(plane[0], uv).rrr * precision_factor[0];
            return vec4((y3-0.065)/0.859, 1.0);
        }
    case 8:		//bgr,
        return vec4(get_sample(plane[0], uv).bgr * precision_factor[0], 1.0);
    case 9:		//rgb,
        return vec4(get_sample(plane[0], uv).rgb * precision_factor[0], 1.0);
	case 10:	// uyvy
		{
			float y = get_sample(plane[0], uv).g * precision_factor[0];
			float cb = get_sample(plane[1], uv).b * precision_factor[1];
			float cr = get_sample(plane[1], uv).r * precision_factor[1];
			return ycbcra_to_rgba(y, cb, cr, 1.0);
		}
    case 11:    // gbrp
        {
            float g  = get_sample(plane[0], uv).r * precision_factor[0];
            float b = get_sample(plane[1], uv).r * precision_factor[1];
            float r = get_sample(plane[2], uv).r * precision_factor[2];
			return vec4(b, g, r, 1.0);
        }
    case 12:    // gbrap
        {
            float g  = get_sample(plane[0], uv).r * precision_factor[0];
            float b = get_sample(plane[1], uv).r * precision_factor[1];
            float r = get_sample(plane[2], uv).r * precision_factor[2];
            float a  = get_sample(plane[3], uv).r * precision_factor[3];
			return vec4(b, g, r, a);
        }
    }
    return vec4(0.0, 0.0, 0.0, 0.0);
}

void main()
{
    vec2 uv = TexCoord.st / TexCoord.q;
    if (is_360) {
        uv = get_equirect_uv(uv);
    }
    if (flip_h) uv.x = 1.0 - uv.x;
    if (flip_v) uv.y = 1.0 - uv.y;
    vec4 color = get_rgba_color(uv);
    if (color_grading) {
        vec3 rgb  = color.rgb;
        rgb       = apply_eotf(rgb, input_transfer);
        rgb       = input_to_working * rgb;
        rgb      *= exposure;
        rgb       = apply_tone_mapping(rgb, tone_mapping_op);
        rgb       = working_to_output * rgb;
        rgb       = apply_oetf(rgb, output_transfer);
        color.rgb = rgb;
    }
    if (is_straight_alpha)
        color.rgb *= color.a;
    if (chroma)
        color = chroma_key(color);
    if(levels)
        color.rgb = LevelsControl(color.rgb, min_input, gamma, max_input, min_output, max_output);
    if(csb)
        color.rgb = ContrastSaturationBrightness(color, brt, sat, con);
    if (white_balance)
        color.rgb = apply_white_balance(color.rgb, wb_temperature, wb_tint);
    if (lmg_enable)
        color.rgb = apply_lmg(color.rgb, lmg_lift, lmg_midtone, lmg_gain);
    if (hue_shift_enable)
        color.rgb = apply_hue_shift(color.rgb, hue_shift_degrees);
    if (tonebalance_enable)
        color.rgb = apply_tone_balance(color.rgb, tb_shadows, tb_highlights);
    if (rgb_levels_enable)
        color.rgb = apply_rgb_levels(color.rgb);
    if (curves_enable)
        color.rgb = apply_curves(color.rgb);
    if(has_local_key)
        color *= texture(local_key, TexCoord2.st).r;
    if(has_layer_key)
        color *= texture(layer_key, TexCoord2.st).r;
    color *= opacity;
    if (invert)
        color = 1.0 - color;
    if (blend_mode >= 0)
        color = blend(color);
    fragColor = color.bgra;
}
