#version 450

#define COLOUR_SPACE_RGB                0
#define COLOUR_SPACE_DATAVIDEO_FULL     1
#define COLOUR_SPACE_DATAVIDEO_LIMITED  2

#define RANGE_16        (16.0f / 256.0f)
#define RANGE_235       (235.0f / 256.0f)
#define RANGE_HALF      (0.5f / 256.0f)
#define RANGE_LIMITED   ((235.0f - 16.0f) / 256.0f)

// Channel transfer function identifiers
#define TRANSFER_SDR   2
#define TRANSFER_PQ    3
#define TRANSFER_HLG   4

uniform sampler2D background;

in vec4 TexCoord;
out vec4 fragColor;

uniform bool key_only;
uniform int colour_space;
uniform int window_width;
uniform int tone_map_op;
uniform float display_peak_luminance;
uniform int channel_transfer;

// rgb=0~255, y=16~235, uv=16~240
mat3 rgb2yuv_709 = mat3(0.183f, -0.101f, 0.439f,  0.614f, -0.338f, -0.399f, 0.062f, 0.439f,-0.040f);

vec4 dtv_color(vec4 color)
{
    color.stp = rgb2yuv_709 * clamp(color.rgb  / (color.a + 0.0000001f), 0.0f, 1.0f);
    return color;
}

// --- EOTF functions (decode transfer to linear) ---

vec3 eotf_rec709(vec3 v)
{
    // Inverse of rec709 OETF
    return mix(v / 4.5f, pow((v + 0.099f) / 1.099f, vec3(1.0f / 0.45f)), step(vec3(0.081f), v));
}

vec3 eotf_hlg(vec3 v)
{
    // BT.2100 HLG OETF inverse
    const float a = 0.17883277f;
    const float b = 0.28466892f; // 1 - 4*a
    const float c = 0.55991073f; // 0.5 - a*ln(4*a)
    return mix(v * v / 3.0f, (exp((v - c) / a) + b) / 12.0f, step(vec3(0.5f), v));
}

vec3 eotf_pq(vec3 v)
{
    // BT.2100 PQ EOTF (ST 2084)
    const float m1 = 0.1593017578125f;
    const float m2 = 78.84375f;
    const float c1 = 0.8359375f;
    const float c2 = 18.8515625f;
    const float c3 = 18.6875f;
    vec3 p = pow(max(v, vec3(0.0f)), vec3(1.0f / m2));
    return pow(max(p - c1, vec3(0.0f)) / (c2 - c3 * p), vec3(1.0f / m1));
}

// --- Tone-map operators ---

vec3 tonemap_hlg_ootf(vec3 v, float npl)
{
    // BT.2100 HLG OOTF: scene-linear → display-linear
    float gamma = 1.2f * pow(1.111f, log2(npl / 1000.0f));
    float Ys = dot(v, vec3(0.2627f, 0.6780f, 0.0593f));
    return v * pow(max(Ys, 1e-6f), gamma - 1.0f);
}

vec3 tonemap_reinhard(vec3 v)
{
    return v / (v + vec3(1.0f));
}

vec3 tonemap_aces_filmic(vec3 v)
{
    // Simple ACES filmic approximation (Narkowicz 2015)
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0f, 1.0f);
}

vec3 tonemap_aces_rrt(vec3 v)
{
    // RRT+ODT approximation
    const mat3 input_mat = mat3(
        0.59719f, 0.07600f, 0.02840f,
        0.35458f, 0.90834f, 0.13383f,
        0.04823f, 0.01566f, 0.83777f);
    const mat3 output_mat = mat3(
         1.60475f, -0.10208f, -0.00327f,
        -0.53108f,  1.10813f, -0.07276f,
        -0.07367f, -0.00605f,  1.07602f);
    v = input_mat * v;
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    v = a / b;
    return clamp(output_mat * v, 0.0f, 1.0f);
}

// --- OETF (re-encode for display) ---

vec3 oetf_rec709(vec3 v)
{
    return mix(v * 4.5f, 1.099f * pow(v, vec3(0.45f)) - 0.099f, step(vec3(0.018f), v));
}

// --- Display transform pipeline ---

vec3 apply_display_tone_map(vec3 color)
{
    // Step 1: Decode from channel transfer to linear
    vec3 linear_color;
    if (channel_transfer == TRANSFER_HLG)
        linear_color = eotf_hlg(color);
    else if (channel_transfer == TRANSFER_PQ)
        linear_color = eotf_pq(color);
    else
        linear_color = eotf_rec709(color);

    // Step 2: Apply tone-mapping operator
    vec3 mapped;
    if (tone_map_op == 7)
        mapped = tonemap_hlg_ootf(linear_color, display_peak_luminance);
    else if (tone_map_op == 1)
        mapped = tonemap_reinhard(linear_color);
    else if (tone_map_op == 2)
        mapped = tonemap_aces_filmic(linear_color);
    else if (tone_map_op == 3)
        mapped = tonemap_aces_rrt(linear_color);
    else
        mapped = linear_color;

    // Step 3: Re-encode for display (rec709 gamma)
    return oetf_rec709(clamp(mapped, 0.0f, 1.0f));
}

void main()
{
    vec4 color = texture(background, TexCoord.xy);
    if (key_only)
        color = vec4(color.aaa, 1);

    else if (colour_space == COLOUR_SPACE_DATAVIDEO_FULL) {  // Full range 0-255
        if (tone_map_op > 0)
            color.rgb = apply_display_tone_map(color.rgb / (color.a + 0.0000001f)) * color.a;
        color = dtv_color(color);
        float x_coord = TexCoord.x * window_width * 0.5f;
        bool isEvenPixel = round(x_coord) - x_coord < 0.0f ;
        vec4 color2 = dtv_color(texture(background, TexCoord.xy + (isEvenPixel ? vec2(1.0f / window_width, 0.0f) : vec2(-1.0f / window_width, 0.0f))));
        color.s = clamp((color.s * RANGE_LIMITED) + RANGE_16 + RANGE_HALF, RANGE_16, RANGE_235);
        color.t = clamp(((isEvenPixel ? color.t + color2.t : color.p + color2.p) * RANGE_LIMITED * 0.5f) + RANGE_HALF + 0.5f, RANGE_16, RANGE_235);
        color.p = clamp((color.w * RANGE_LIMITED) + RANGE_16 + RANGE_HALF, RANGE_16, RANGE_235);

    } else if (colour_space == COLOUR_SPACE_DATAVIDEO_LIMITED) {  // Limited range 16-235
        if (tone_map_op > 0)
            color.rgb = apply_display_tone_map(color.rgb / (color.a + 0.0000001f)) * color.a;
        color = dtv_color(color);
        float x_coord = TexCoord.x * window_width * 0.5f;
        bool isEvenPixel = round(x_coord) - x_coord < 0.0f ;
        vec4 color2 = dtv_color(texture(background, TexCoord.xy + (isEvenPixel ? vec2(1.0f / window_width, 0.0f) : vec2(-1.0f / window_width, 0.0f))));
        color.s = clamp(color.s + RANGE_HALF, 0.0f, 1.0f);
        color.t = clamp(((isEvenPixel ? color.t + color2.t : color.p + color2.p) * 0.5f) + RANGE_HALF + 0.5f, 0.0f, 1.0f);
        color.p = clamp(color.w + RANGE_HALF, 0.0f, 1.0f);
    } else {
        // Standard RGB output — apply tone mapping
        if (tone_map_op > 0)
            color.rgb = apply_display_tone_map(color.rgb / (color.a + 0.0000001f)) * color.a;
    }
    fragColor = color;
}
