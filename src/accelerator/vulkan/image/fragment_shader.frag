#version 450

#extension GL_ARB_fragment_shader_interlock : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec4 TexCoord;
layout(location = 1) in vec4 TexCoord2;

layout(location = 0) out vec4 fragColor;

// ── Descriptor bindings ─────────────────────────────────────────────────
layout(binding = 0) uniform sampler2D textures[8];
layout(binding = 1, input_attachment_index = 0) uniform subpassInput background;
layout(scalar, binding = 2) uniform ParamsBlock {
    uint  color_space_index;
    float precision_factor[4];
    int   blend_mode;
    int   keyer;
    int   pixel_format;
    float opacity;
    float min_input, max_input, gamma_val, min_output, max_output;
    float brt, sat, con;
    float chroma_target_hue, chroma_hue_width, chroma_min_saturation;
    float chroma_min_brightness, chroma_softness;
    float chroma_spill_suppress, chroma_spill_suppress_saturation;
    uint  flags;
    float view_yaw, view_pitch, view_roll, view_fov;
    float aspect_ratio, view_offset_x, view_offset_y;
    float frustum_h, frustum_v;
    float lens_k1, lens_k2, lens_k3;
    int   screen_curve_type;
    float screen_arc;
    float edge_blend_left, edge_blend_right, edge_blend_top, edge_blend_bottom;
    float edge_blend_gamma;
    int   input_transfer, output_transfer, tone_mapping_op;
    float exposure;
    float display_peak_luminance;
    vec4 input_to_working_c0, input_to_working_c1, input_to_working_c2;
    vec4 working_to_output_c0, working_to_output_c1, working_to_output_c2;
    float wb_temperature, wb_tint;
    vec4 lmg_lift_pad;
    vec4 lmg_midtone_pad;
    vec4 lmg_gain_pad;
    float hue_shift_degrees;
    float tb_shadows, tb_highlights;
    float linear_sat_value;
    vec4 cdl_slope_sat;
    vec4 cdl_offset_pad;
    vec4 cdl_power_pad;
    vec4 split_shadow_balance;
    vec4 split_highlight_pad;
    vec4 gc_limit_pad;
    float lut3d_strength;
    float sharpen_amount, sharpen_radius;
    float grain_intensity, grain_size;
    int   grain_frame;
    float qual_target_hue, qual_hue_width;
    float qual_min_sat, qual_max_sat;
    float qual_min_lum, qual_max_lum;
    float qual_softness, qual_exposure;
    float qual_sat_offset, qual_hue_offset;
    vec4 rgb_min_input_pad;
    vec4 rgb_max_input_pad;
    vec4 rgb_gamma_pad;
    vec4 rgb_min_output_pad;
    vec4 rgb_max_output_pad;
    float blur_radius;
    int   blur_type;
    float blur_angle;
    vec2  blur_center;
    vec2  blur_tilt;
    vec2  target_size;
    int   shape_type, shape_fill_type;
    vec2  shape_center;
    vec2  shape_size;
    float shape_corner_radius, shape_softness;
    vec4  shape_color1, shape_color2;
    float shape_gradient_angle;
    vec2  shape_gradient_center;
    float shape_stroke_width;
    vec4  shape_stroke_color;
    uint  flags2;
    float _padEnd[3];
};
layout(binding = 3) uniform sampler3D lut3d_tex;
layout(binding = 4) uniform sampler2D hue_curve_tex;
layout(binding = 5) uniform sampler2D curve_lut_tex;

// ── Constants ───────────────────────────────────────────────────────────
const uint PLANE0=0, PLANE1=1, PLANE2=2, PLANE3=3, LOCAL_KEY=4, LAYER_KEY=5;
const uint F_STRAIGHT_ALPHA=1u<<0, F_LOCAL_KEY=1u<<1, F_LAYER_KEY=1u<<2,
           F_INVERT=1u<<3, F_LEVELS=1u<<4, F_CSB=1u<<5, F_CHROMA=1u<<6,
           F_CHROMA_MASK=1u<<7, F_360=1u<<8, F_CURVED=1u<<9,
           F_COLOR_GRADING=1u<<10, F_FLIP_H=1u<<11, F_FLIP_V=1u<<12,
           F_WHITE_BALANCE=1u<<13, F_LMG=1u<<14, F_HUE_SHIFT=1u<<15,
           F_TONEBALANCE=1u<<16, F_LINEAR_SAT=1u<<17, F_CDL=1u<<18,
           F_SPLIT_TONE=1u<<19, F_GAMUT_COMPRESS=1u<<20, F_LUT3D=1u<<21,
           F_HUE_CURVE=1u<<22, F_SHARPEN=1u<<23, F_GRAIN=1u<<24,
           F_QUALIFIER=1u<<25, F_RGB_LEVELS=1u<<26, F_CURVES=1u<<27,
           F_BLUR=1u<<28, F_SHAPE=1u<<29, F_SHAPE_STROKE=1u<<30,
           F_EDGE_BLEND=1u<<31;
bool flag(uint f) { return (flags & f) != 0u; }

// Extended flags (flags2)
const uint F2_OUTPUT_BGRA=1u<<0;
bool flag2(uint f) { return (flags2 & f) != 0u; }

const float PI = 3.14159265359;
const mat3 color_matrices[3] = mat3[3](
    mat3(1.0,0.0,1.402, 1.0,-0.344,-0.71414, 1.0,1.772,0.0),
    mat3(1.0,0.0,1.5748, 1.0,-0.1873,-0.4681, 1.0,1.8556,0.0),
    mat3(1.0,0.0,1.4746, 1.0,-0.16455312684366,-0.57135312684366, 1.0,1.8814,0.0)
);
const vec3 luma_coefficients[3] = vec3[3](
    vec3(0.299,0.587,0.114), vec3(0.2126,0.7152,0.0722), vec3(0.2627,0.6780,0.0593)
);
mat3 ubo_mat3(vec4 c0, vec4 c1, vec4 c2) { return mat3(c0.xyz, c1.xyz, c2.xyz); }

// ── CSB / Levels ────────────────────────────────────────────────────────
vec3 ContrastSaturationBrightness(vec4 color, float brt_v, float sat_v, float con_v) {
    vec3 lc = luma_coefficients[color_space_index];
    if (color.a > 0.0) color.rgb /= color.a;
    vec3 brtC = color.rgb * brt_v;
    vec3 intens = vec3(dot(brtC, lc));
    vec3 satC = mix(intens, brtC, sat_v);
    vec3 conC = mix(vec3(0.5), satC, con_v);
    return conC * color.a;
}
#define GammaCorrection(c, g) pow(c, vec3(1.0/g))
#define LevelsControl(c, mi, g, mx, mo, mxo) mix(vec3(mo), vec3(mxo), GammaCorrection(min(max(c-vec3(mi),vec3(0.0))/(vec3(mx)-vec3(mi)),vec3(1.0)), g))

// ── HSL/HSV ─────────────────────────────────────────────────────────────
vec3 RGBToHSL(vec3 c) {
    float mn=min(min(c.r,c.g),c.b), mx=max(max(c.r,c.g),c.b), d=mx-mn;
    vec3 h; h.z=(mx+mn)/2.0;
    if(d==0.0){h.x=0.0;h.y=0.0;}else{
        h.y=(h.z<0.5)?d/(mx+mn):d/(2.0-mx-mn);
        float dR=(((mx-c.r)/6.0)+(d/2.0))/d, dG=(((mx-c.g)/6.0)+(d/2.0))/d, dB=(((mx-c.b)/6.0)+(d/2.0))/d;
        if(c.r==mx)h.x=dB-dG;else if(c.g==mx)h.x=1.0/3.0+dR-dB;else h.x=2.0/3.0+dG-dR;
        if(h.x<0.0)h.x+=1.0;else if(h.x>1.0)h.x-=1.0;
    } return h;
}
float HueToRGB(float f1, float f2, float hue) {
    if(hue<0.0)hue+=1.0;else if(hue>1.0)hue-=1.0;
    if(6.0*hue<1.0)return f1+(f2-f1)*6.0*hue;if(2.0*hue<1.0)return f2;
    if(3.0*hue<2.0)return f1+(f2-f1)*(2.0/3.0-hue)*6.0;return f1;
}
vec3 HSLToRGB(vec3 h) {
    if(h.y==0.0)return vec3(h.z);
    float f2=(h.z<0.5)?h.z*(1.0+h.y):(h.z+h.y)-(h.y*h.z); float f1=2.0*h.z-f2;
    return vec3(HueToRGB(f1,f2,h.x+1.0/3.0), HueToRGB(f1,f2,h.x), HueToRGB(f1,f2,h.x-1.0/3.0));
}
vec3 rgb2hsv(vec3 c) {
    vec4 K=vec4(0.0,-1.0/3.0,2.0/3.0,-1.0);
    vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));
    vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));
    float d=q.x-min(q.w,q.y),e=1e-10;
    return vec3(abs(q.z+(q.w-q.y)/(6.0*d+e)),d/(q.x+e),q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K=vec4(1.0,2.0/3.0,1.0/3.0,3.0);
    vec3 p=abs(fract(c.xxx+K.xyz)*6.0-K.www);
    return c.z*mix(K.xxx,clamp(p-K.xxx,0.0,1.0),c.y);
}

// ── Blend modes ─────────────────────────────────────────────────────────
float BlendAddf(float b,float l){ return min(b+l,1.0); }
float BlendSubf(float b,float l){ return max(b+l-1.0,0.0); }
float BlendScreenf(float b,float l){ return 1.0-((1.0-b)*(1.0-l)); }
float BlendOverlayf(float b,float l){ return b<0.5?(2.0*b*l):(1.0-2.0*(1.0-b)*(1.0-l)); }
float BlendSoftLightf(float b,float l){ return (l<0.5)?(2.0*b*l+b*b*(1.0-2.0*l)):(sqrt(b)*(2.0*l-1.0)+2.0*b*(1.0-l)); }
float BlendColorDodgef(float b,float l){ return (l==1.0)?l:min(b/(1.0-l),1.0); }
float BlendColorBurnf(float b,float l){ return (l==0.0)?l:max(1.0-((1.0-b)/l),0.0); }
float BlendVividLightf(float b,float l){ return (l<0.5)?BlendColorBurnf(b,2.0*l):BlendColorDodgef(b,2.0*(l-0.5)); }
float BlendLinearLightf(float b,float l){ return l<0.5?BlendSubf(b,2.0*l):BlendAddf(b,2.0*(l-0.5)); }
float BlendPinLightf(float b,float l){ return (l<0.5)?min(l*2.0,b):max(2.0*(l-0.5),b); }
float BlendHardMixf(float b,float l){ return (BlendVividLightf(b,l)<0.5)?0.0:1.0; }
float BlendReflectf(float b,float l){ return (l==1.0)?l:min(b*b/(1.0-l),1.0); }

vec3 Blend3Screen(vec3 b,vec3 l){ return vec3(BlendScreenf(b.r,l.r),BlendScreenf(b.g,l.g),BlendScreenf(b.b,l.b)); }
vec3 Blend3Overlay(vec3 b,vec3 l){ return vec3(BlendOverlayf(b.r,l.r),BlendOverlayf(b.g,l.g),BlendOverlayf(b.b,l.b)); }
vec3 Blend3ColorDodge(vec3 b,vec3 l){ return vec3(BlendColorDodgef(b.r,l.r),BlendColorDodgef(b.g,l.g),BlendColorDodgef(b.b,l.b)); }
vec3 Blend3ColorBurn(vec3 b,vec3 l){ return vec3(BlendColorBurnf(b.r,l.r),BlendColorBurnf(b.g,l.g),BlendColorBurnf(b.b,l.b)); }
vec3 Blend3LinearLight(vec3 b,vec3 l){ return vec3(BlendLinearLightf(b.r,l.r),BlendLinearLightf(b.g,l.g),BlendLinearLightf(b.b,l.b)); }
vec3 Blend3VividLight(vec3 b,vec3 l){ return vec3(BlendVividLightf(b.r,l.r),BlendVividLightf(b.g,l.g),BlendVividLightf(b.b,l.b)); }
vec3 Blend3PinLight(vec3 b,vec3 l){ return vec3(BlendPinLightf(b.r,l.r),BlendPinLightf(b.g,l.g),BlendPinLightf(b.b,l.b)); }
vec3 Blend3HardMix(vec3 b,vec3 l){ return vec3(BlendHardMixf(b.r,l.r),BlendHardMixf(b.g,l.g),BlendHardMixf(b.b,l.b)); }
vec3 Blend3Reflect(vec3 b,vec3 l){ return vec3(BlendReflectf(b.r,l.r),BlendReflectf(b.g,l.g),BlendReflectf(b.b,l.b)); }

vec3 BlendHue(vec3 b,vec3 l){vec3 bh=RGBToHSL(b);return HSLToRGB(vec3(RGBToHSL(l).r,bh.g,bh.b));}
vec3 BlendSaturation(vec3 b,vec3 l){vec3 bh=RGBToHSL(b);return HSLToRGB(vec3(bh.r,RGBToHSL(l).g,bh.b));}
vec3 BlendColor(vec3 b,vec3 l){vec3 lh=RGBToHSL(l);return HSLToRGB(vec3(lh.r,lh.g,RGBToHSL(b).b));}
vec3 BlendLuminosity(vec3 b,vec3 l){vec3 bh=RGBToHSL(b);return HSLToRGB(vec3(bh.r,bh.g,RGBToHSL(l).b));}

vec3 get_blend_color(vec3 b, vec3 fg) {
    switch(blend_mode) {
    case 0: return fg;
    case 1: return max(b,fg);
    case 2: return min(b,fg);
    case 3: return b*fg;
    case 4: return (b+fg)/2.0;
    case 5: return min(b+fg,vec3(1.0));
    case 6: return max(b+fg-vec3(1.0),vec3(0.0));
    case 7: return abs(b-fg);
    case 8: return vec3(1.0)-abs(vec3(1.0)-b-fg);
    case 9: return b+fg-2.0*b*fg;
    case 10: return Blend3Screen(b,fg);
    case 11: return Blend3Overlay(b,fg);
    case 13: return Blend3Overlay(fg,b);
    case 14: return Blend3ColorDodge(b,fg);
    case 15: return Blend3ColorBurn(b,fg);
    case 16: return min(b+fg,vec3(1.0));
    case 17: return max(b+fg-vec3(1.0),vec3(0.0));
    case 18: return Blend3LinearLight(b,fg);
    case 19: return Blend3VividLight(b,fg);
    case 20: return Blend3PinLight(b,fg);
    case 21: return Blend3HardMix(b,fg);
    case 22: return Blend3Reflect(b,fg);
    case 23: return Blend3Reflect(fg,b);
    case 24: return min(b,fg)-max(b,fg)+vec3(1.0);
    case 25: return BlendHue(b,fg);
    case 26: return BlendSaturation(b,fg);
    case 27: return BlendColor(b,fg);
    case 28: return BlendLuminosity(b,fg);
    } return fg;
}
vec4 blend_op(vec4 fore) {
    vec4 back = flag2(F2_OUTPUT_BGRA) ? subpassLoad(background).bgra : subpassLoad(background);
    if(blend_mode!=0) fore.rgb = get_blend_color(back.rgb/(back.a+1e-7), fore.rgb/(fore.a+1e-7))*fore.a;
    switch(keyer){ case 1: return fore+back; default: return fore+(1.0-fore.a)*back; }
}

// ── Chroma keying ───────────────────────────────────────────────────────
float AngleDiff(float a,float b){return 0.5-abs(abs(a-b)-0.5);}
float AngleDiffDir(float a,float b){float d=a-b;return d<-0.5?d+1.0:(d>0.5?d-1.0:d);}
float ColorDist(vec3 h){return -(AngleDiff(h.x,chroma_target_hue)*2-chroma_hue_width)*max(min(0.0,chroma_min_brightness-h.z),min(0.0,chroma_min_saturation-h.y));}
float alpha_map(float d){return 1.0-smoothstep(1.0,chroma_softness,d);}
vec3 supress_spill(vec3 c){float d=AngleDiffDir(c.x,chroma_target_hue);float dist=abs(d)/chroma_spill_suppress;if(dist<1){c.x=d<0?chroma_target_hue-chroma_spill_suppress:chroma_target_hue+chroma_spill_suppress;c.y*=min(1.0,dist+chroma_spill_suppress_saturation);}return c;}
vec4 ChromaKey(vec4 c,bool sm){vec3 h=rgb2hsv(c.rgb);float d=ColorDist(h)*-2.0+1.0;vec4 s=vec4(hsv2rgb(supress_spill(h)),1.0)*alpha_map(d);return sm?vec4(s.a,s.a,s.a,1):s;}

// ── YCbCr ───────────────────────────────────────────────────────────────
vec4 ycbcra_to_rgba(float Y,float Cb,float Cr,float A){
    mat3 cm=transpose(color_matrices[color_space_index]);
    vec3 v=vec3(Y,Cb,Cr)*255-vec3(16,128,128); v*=vec3(255.0/219.0,255.0/224.0,255.0/224.0);
    return vec4(cm*v/255,A);
}

// ── Color grading ───────────────────────────────────────────────────────
float eotf_srgb(float x){return x<=0.04045?x/12.92:pow((x+0.055)/1.055,2.4);}
float eotf_rec709(float x){return pow(max(x,0.0),2.4);}  // BT.1886 display EOTF
float eotf_pq(float x){const float m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875;float xp=pow(max(x,0.0),1.0/m2);return pow(max(xp-c1,0.0)/(c2-c3*xp),1.0/m1);}
float eotf_hlg(float x){const float a=0.17883277,b=0.28466892,c=0.55991073;return x<=0.5?(x*x)/3.0:(exp((x-c)/a)+b)/12.0;}
float eotf_logc3(float x){const float a=5.555556,b=0.052272,c=0.247190,d=0.385537,e=5.367655,f=0.092809;return x>e*0.010591+f?(pow(10.0,(x-d)/c)-b)/a:(x-f)/e;}
float eotf_slog3(float x){const float cut=171.2102946929/1023.0;return x>=cut?pow(10.0,(x-0.410557184750733)/0.341132524981570)*0.18+0.01:(x-95.0/1023.0)*0.01/(cut-95.0/1023.0);}
float eotf_gamma24(float x){return pow(max(x,0.0),2.4);}  // Pure gamma 2.4 inverse
float eotf_gamma26(float x){return pow(max(x,0.0),2.6);}  // Pure gamma 2.6 inverse
vec3 apply_eotf(vec3 r,int t){switch(t){case 1:return vec3(eotf_srgb(r.r),eotf_srgb(r.g),eotf_srgb(r.b));case 2:return vec3(eotf_rec709(r.r),eotf_rec709(r.g),eotf_rec709(r.b));case 3:return vec3(eotf_pq(r.r),eotf_pq(r.g),eotf_pq(r.b));case 4:return vec3(eotf_hlg(r.r),eotf_hlg(r.g),eotf_hlg(r.b));case 5:return vec3(eotf_logc3(r.r),eotf_logc3(r.g),eotf_logc3(r.b));case 6:return vec3(eotf_slog3(r.r),eotf_slog3(r.g),eotf_slog3(r.b));case 7:return r;case 8:return vec3(eotf_gamma24(r.r),eotf_gamma24(r.g),eotf_gamma24(r.b));case 9:return vec3(eotf_gamma26(r.r),eotf_gamma26(r.g),eotf_gamma26(r.b));default:return r;}}
float oetf_srgb(float x){return x<=0.0031308?x*12.92:1.055*pow(max(x,0.0),1.0/2.4)-0.055;}
float oetf_rec709(float x){return pow(max(x,0.0),1.0/2.4);}  // BT.1886 inverse
float oetf_pq(float x){const float m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875;float xn=pow(clamp(x,0.0,1.0),m1);return pow((c1+c2*xn)/(1.0+c3*xn),m2);}
float oetf_hlg(float x){const float a=0.17883277,b=0.28466892,c=0.55991073;x=max(x,0.0);return x<=1.0/12.0?sqrt(3.0*x):a*log(12.0*x-b)+c;}
float oetf_gamma24(float x){return pow(max(x,0.0),1.0/2.4);}  // Pure gamma 2.4 (EBU)
float oetf_gamma26(float x){return pow(max(x,0.0),1.0/2.6);}  // Pure gamma 2.6 (DCI)
vec3 apply_oetf(vec3 r,int t){r=max(r,vec3(0.0));switch(t){case 1:return vec3(oetf_srgb(r.r),oetf_srgb(r.g),oetf_srgb(r.b));case 2:return vec3(oetf_rec709(r.r),oetf_rec709(r.g),oetf_rec709(r.b));case 3:return vec3(oetf_pq(r.r),oetf_pq(r.g),oetf_pq(r.b));case 4:return vec3(oetf_hlg(r.r),oetf_hlg(r.g),oetf_hlg(r.b));case 5:return r;case 6:return vec3(oetf_gamma24(r.r),oetf_gamma24(r.g),oetf_gamma24(r.b));case 7:return vec3(oetf_gamma26(r.r),oetf_gamma26(r.g),oetf_gamma26(r.b));default:return r;}}

vec3 tonemap_reinhard(vec3 v){return v/(v+1.0);}
vec3 tonemap_aces_filmic(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0);}
vec3 tonemap_aces_rrt(vec3 v){v*=0.6;vec3 a=v*(v+0.0245786)-0.000090537;vec3 b=v*(0.983729*v+0.432951)+0.238081;return clamp(a/b,0.0,1.0);}
vec3 tonemap_hlg_ootf(vec3 v,float npl){float gamma=1.2*pow(1.111,log2(npl/1000.0));float Ys=dot(v,vec3(0.2627,0.6780,0.0593));vec3 r=v*pow(max(Ys,1e-6),gamma-1.0);if(npl<1000.0){float Yd=dot(r,vec3(0.2627,0.6780,0.0593));float k=0.85;if(Yd>k){float c=k+(1.0-k)*tanh((Yd-k)/(1.0-k));r*=c/max(Yd,1e-6);}}return r;}
float log10_f(float x){return log(x)*0.4342944819032518;}
float spline_c5(float x){
    const float cL[6]=float[6](-4.0,-4.0,-3.1573765773,-0.4852499958,1.8477324706,1.8477324706);
    const float cH[6]=float[6](-0.7185482425,2.0810307172,3.6681241237,4.0,4.0,4.0);
    float xC=clamp(x,0.0001,65504.0),lx=log10_f(xC),ly;
    if(lx<=log10_f(0.18)){float k=(lx-log10_f(0.0001))/(log10_f(0.18)-log10_f(0.0001))*4.0;int j=int(clamp(k,0.0,3.0));float t=clamp(k-float(j),0.0,1.0);ly=mix(mix(mix(cL[j],cL[j+1],0.5),cL[j+1],t),mix(cL[j+1],mix(cL[j+1],cL[min(j+2,5)],0.5),t),t);}
    else{float k=(lx-log10_f(0.18))/(log10_f(65504.0)-log10_f(0.18))*4.0;int j=int(clamp(k,0.0,3.0));float t=clamp(k-float(j),0.0,1.0);ly=mix(mix(mix(cH[j],cH[j+1],0.5),cH[j+1],t),mix(cH[j+1],mix(cH[j+1],cH[min(j+2,5)],0.5),t),t);}
    return pow(10.0,ly);
}
float spline_c9(float x,float mnY,float mdY,float mxY){
    const float cL[10]=float[10](-1.6989700043,-1.6989700043,-1.4779,-1.2291,-0.8648,-0.4480,0.00518,0.4511080334,0.9113744414,0.9113744414);
    const float cH[10]=float[10](0.5154386965,0.8470437783,1.1358,1.3802,1.5197,1.5985,1.6467,1.6746091357,1.6878733390,1.6878733390);
    float mnX=0.18*exp2(-6.5),mdX=0.18,mxX=0.18*exp2(6.5);
    float xC=clamp(x,mnX,mxX),lx=log10_f(xC),ly;
    if(lx<=log10_f(mdX)){float k=(lx-log10_f(mnX))/(log10_f(mdX)-log10_f(mnX))*8.0;int j=int(clamp(k,0.0,7.0));float t=clamp(k-float(j),0.0,1.0);ly=mix(mix(mix(cL[j],cL[j+1],0.5),cL[j+1],t),mix(cL[j+1],mix(cL[j+1],cL[min(j+2,9)],0.5),t),t);}
    else{float k=(lx-log10_f(mdX))/(log10_f(mxX)-log10_f(mdX))*8.0;int j=int(clamp(k,0.0,7.0));float t=clamp(k-float(j),0.0,1.0);ly=mix(mix(mix(cH[j],cH[j+1],0.5),cH[j+1],t),mix(cH[j+1],mix(cH[j+1],cH[min(j+2,9)],0.5),t),t);}
    return pow(10.0,ly);
}
vec3 aces_rrt_s(vec3 v){return vec3(spline_c5(v.r),spline_c5(v.g),spline_c5(v.b));}
vec3 aces_odt_srgb(vec3 v){v=vec3(spline_c9(v.r,.0001,4.8,48.0),spline_c9(v.g,.0001,4.8,48.0),spline_c9(v.b,.0001,4.8,48.0));return clamp(v/48.0,0.0,1.0);}
vec3 apply_tone_mapping(vec3 r,int op){switch(op){case 1:return tonemap_reinhard(r);case 2:return tonemap_aces_filmic(r);case 3:return tonemap_aces_rrt(r);case 4:return aces_odt_srgb(aces_rrt_s(r));case 5:{vec3 v=aces_rrt_s(r);v=vec3(spline_c9(v.r,.0001,4.8,48.0),spline_c9(v.g,.0001,4.8,48.0),spline_c9(v.b,.0001,4.8,48.0));return clamp(v/48.0,0.0,1.0);}case 6:{vec3 v=aces_rrt_s(r);v=vec3(spline_c9(v.r,.005,4.8,800.0),spline_c9(v.g,.005,4.8,800.0),spline_c9(v.b,.005,4.8,800.0));return clamp(v/1000.0,0.0,1.0);}case 7:return tonemap_hlg_ootf(r,display_peak_luminance);default:return r;}}

// ── Effect helpers ──────────────────────────────────────────────────────
// White balance in RGB working space: warm (+t) boosts red, cuts blue; tint (+ti) boosts green.
// (Matches the OGL shader, which expresses the same gains in its BGRA convention.)
vec3 apply_white_balance(vec3 c,float t,float ti){c.r*=1.0+t*0.20;c.g*=1.0+ti*0.10;c.b*=1.0-t*0.20;return c;}
vec3 apply_lmg(vec3 c,vec3 l,vec3 m,vec3 g){return pow(max(c*g+l,vec3(0.0)),max(vec3(0.01),1.0/m));}
vec3 apply_hue_shift(vec3 c,float deg){float pk=max(max(c.r,c.g),max(c.b,0.0001));vec3 h=rgb2hsv(clamp(c/pk,0.0,1.0));h.x=fract(h.x+deg/360.0);return hsv2rgb(h)*pk;}
vec3 apply_tone_balance(vec3 c,float s,float h){float l=dot(c,vec3(0.2126,0.7152,0.0722));c+=vec3(s*0.5*(1.0-smoothstep(0.0,0.6,l)));c+=vec3(h*0.5*smoothstep(0.4,1.0,l));return c;}
vec3 apply_linear_sat(vec3 c,float s){float l=dot(c,vec3(0.2126,0.7152,0.0722));return mix(vec3(l),c,s);}
vec3 apply_cdl(vec3 c,vec3 sl,vec3 of,vec3 pw,float s){c=pow(max(c*sl+of,vec3(0.0)),pw);float l=dot(c,vec3(0.2126,0.7152,0.0722));return mix(vec3(l),c,s);}
vec3 apply_split_tone(vec3 c,vec3 sc,vec3 hc,float b){float l=dot(c,vec3(0.2126,0.7152,0.0722));c+=sc*(1.0-smoothstep(b-0.3,b+0.3,l));c+=hc*smoothstep(b-0.3,b+0.3,l);return c;}
vec3 apply_gamut_compress(vec3 c,vec3 lim){float a=max(max(c.r,c.g),max(c.b,0.0));if(a<=0.0)return c;vec3 d=(a-c)/abs(a);float thr=0.815;for(int i=0;i<3;++i)if(d[i]>thr&&lim[i]>1.0001){float nd=(d[i]-thr)/(lim[i]-thr);d[i]=thr+nd/(1.0+nd)*(lim[i]-thr);}return a-d*abs(a);}
vec3 apply_lut3d(vec3 c,float st){return mix(c,texture(lut3d_tex,clamp(c,0.0,1.0)).rgb,st);}
vec3 apply_hue_curves(vec3 c){vec3 h=rgb2hsv(clamp(c,0.0,1.0));vec4 o=texture(hue_curve_tex,vec2(h.x,0.5));h.x=fract(h.x+o.r);h.y*=o.g;vec4 so=texture(hue_curve_tex,vec2(h.y,0.5));h.y*=so.a;return hsv2rgb(h)+o.b;}
vec3 apply_qualifier(vec3 c,float th,float hw,float ms,float xs,float ml,float xl,float sf,float eo,float so,float ho){
    vec3 h=rgb2hsv(clamp(c,0.0,1.0));float hd=AngleDiff(h.x,th)*2.0;
    float hm=1.0-smoothstep(hw-sf,hw+sf,hd);float sm=smoothstep(ms-sf,ms+sf,h.y)*(1.0-smoothstep(xs-sf,xs+sf,h.y));
    float lm=smoothstep(ml-sf,ml+sf,h.z)*(1.0-smoothstep(xl-sf,xl+sf,h.z));float mk=hm*sm*lm;if(mk<0.001)return c;
    vec3 g=c*(1.0+eo);float gl=dot(g,vec3(0.2126,0.7152,0.0722));g=mix(vec3(gl),g,1.0+so);if(abs(ho)>0.01)g=apply_hue_shift(g,ho);return mix(c,g,mk);}
vec3 apply_rgb_levels(vec3 c){
    vec3 mi=rgb_min_input_pad.xyz,mx=rgb_max_input_pad.xyz,gm=rgb_gamma_pad.xyz,mo=rgb_min_output_pad.xyz,mxo=rgb_max_output_pad.xyz;
    c=clamp((c-mi)/max(mx-mi,vec3(0.0001)),0.0,1.0);c=pow(c,max(1.0/gm,vec3(0.01)));return mix(mo,mxo,c);}
float sample_lut(float v,int ch){float s=clamp(v,0.0,1.0)*255.0;int lo=int(s),hi=min(lo+1,255);float f=fract(s);vec4 l4=texelFetch(curve_lut_tex,ivec2(lo,0),0);vec4 h4=texelFetch(curve_lut_tex,ivec2(hi,0),0);vec4 v4=mix(l4,h4,f);if(ch==0)return v4.r;if(ch==1)return v4.g;if(ch==2)return v4.b;return v4.a;}
vec3 apply_curves(vec3 c){c.r=sample_lut(c.r,0);c.g=sample_lut(c.g,1);c.b=sample_lut(c.b,2);c.r=sample_lut(c.r,3);c.g=sample_lut(c.g,3);c.b=sample_lut(c.b,3);return c;}
float grain_hash(vec2 p,int fs){vec3 p3=fract(vec3(p.xyx)*0.1031);p3+=dot(p3,p3.yzx+float(fs)*0.00137);return fract((p3.x+p3.y)*p3.z);}
vec3 apply_grain(vec3 c,vec2 uv,float in_v,float sz,int fs){vec2 gu=uv*target_size/max(sz,0.5);float n=grain_hash(gu,fs)*2.0-1.0;float l=dot(c,vec3(0.2126,0.7152,0.0722));float r=smoothstep(0.0,0.15,l)*(1.0-smoothstep(0.8,1.0,l));return c+vec3(n*in_v*r);}

// ── 360° / Curved ───────────────────────────────────────────────────────
vec2 apply_curve_warp(vec2 uv){if(abs(screen_arc)<0.0001)return uv;vec2 ndc=uv*2.0-1.0;float ha=abs(screen_arc)*0.5;bool cv=screen_arc>=0.0;
    if(screen_curve_type==1){ndc.x=cv?tan(ndc.x*ha)/tan(ha):atan(ndc.x*tan(ha))/ha;}
    else if(screen_curve_type==2){float vh=ha/aspect_ratio;if(cv){ndc.x=tan(ndc.x*ha)/tan(ha);ndc.y=tan(ndc.y*vh)/tan(vh);}else{ndc.x=atan(ndc.x*tan(ha))/ha;ndc.y=atan(ndc.y*tan(vh))/vh;}}
    else if(screen_curve_type==3){float r=length(ndc);if(r>0.0001){float wr=cv?tan(r*ha)/tan(ha):atan(r*tan(ha))/ha;ndc*=wr/r;}}
    return ndc*0.5+0.5;}
vec2 get_equirect_uv(vec2 suv){
    vec2 ndc=suv*2.0-1.0-vec2(view_offset_x,view_offset_y);
    if(lens_k1!=0.0||lens_k2!=0.0||lens_k3!=0.0){float r2=dot(ndc,ndc);ndc*=1.0+lens_k1*r2+lens_k2*r2*r2+lens_k3*r2*r2*r2;}
    float sc=tan(view_fov*0.5);vec3 dir;
    if(screen_curve_type==1){float ha=screen_arc*0.5;dir=vec3(sin(ndc.x*ha),ndc.y*ha/aspect_ratio,-cos(ndc.x*ha));}
    else if(screen_curve_type==2){float ha=screen_arc*0.5,vh=ha/aspect_ratio;float ah=ndc.x*ha,av=ndc.y*vh;dir=vec3(sin(ah)*cos(av),sin(av),-cos(ah)*cos(av));}
    else if(screen_curve_type==3){float ha=screen_arc*0.5;float r=length(ndc);if(r<0.0001)dir=vec3(0,0,-1);else{float th=r*ha;dir=vec3(ndc.x/r*sin(th),ndc.y/r*sin(th),-cos(th));}}
    else{vec2 sh=ndc+vec2(frustum_h,frustum_v);dir=vec3(sh.x*sc*aspect_ratio,sh.y*sc,-1.0);}
    dir=normalize(dir);
    float cr=cos(view_roll),sr=sin(view_roll);mat3 rz=mat3(cr,-sr,0,sr,cr,0,0,0,1);
    float cp=cos(view_pitch),sp=sin(view_pitch);mat3 rx=mat3(1,0,0,0,cp,-sp,0,sp,cp);
    float cy=cos(view_yaw),sy=sin(view_yaw);mat3 ry=mat3(cy,0,sy,0,1,0,-sy,0,cy);
    dir=ry*rx*rz*dir;
    return vec2(0.5+atan(dir.x,-dir.z)/(2.0*PI), 0.5+asin(dir.y)/PI);}

// ── Pixel format ────────────────────────────────────────────────────────
vec4 get_rgba_color(vec2 uv);
vec4 sample_wrap(vec2 s){if(flag(F_360))s.x=fract(s.x);return get_rgba_color(s);}
vec3 apply_sharpen(vec2 uv,vec3 cc,float am,float rd){vec2 ts=1.0/target_size*rd;return cc+(cc-(sample_wrap(uv+vec2(0,-ts.y)).rgb+sample_wrap(uv+vec2(0,ts.y)).rgb+sample_wrap(uv+vec2(ts.x,0)).rgb+sample_wrap(uv+vec2(-ts.x,0)).rgb)*0.25)*am;}

vec4 get_rgba_color(vec2 uv){
    switch(pixel_format){
    case 0: return vec4(texture(textures[PLANE0],uv).rrr*precision_factor[0],1.0);
    case 1: return texture(textures[PLANE0],uv).bgra*precision_factor[0];
    case 2: return texture(textures[PLANE0],uv).rgba*precision_factor[0];
    case 3: return texture(textures[PLANE0],uv).gbar*precision_factor[0];
    case 4: return texture(textures[PLANE0],uv).abgr*precision_factor[0];
    case 5:{float y=texture(textures[PLANE0],uv).r*precision_factor[0];float cb=texture(textures[PLANE1],uv).r*precision_factor[1];float cr=texture(textures[PLANE2],uv).r*precision_factor[2];return ycbcra_to_rgba(y,cb,cr,1.0);}
    case 6:{float y=texture(textures[PLANE0],uv).r*precision_factor[0];float cb=texture(textures[PLANE1],uv).r*precision_factor[1];float cr=texture(textures[PLANE2],uv).r*precision_factor[2];float a=texture(textures[PLANE3],uv).r*precision_factor[3];return ycbcra_to_rgba(y,cb,cr,a);}
    case 7:{vec3 y3=texture(textures[PLANE0],uv).rrr*precision_factor[0];return vec4((y3-0.0627451)/0.858824,1.0);}
    case 8: return vec4(texture(textures[PLANE0],uv).bgr*precision_factor[0],1.0);
    case 9: return vec4(texture(textures[PLANE0],uv).rgb*precision_factor[0],1.0);
    case 10:{float y=texture(textures[PLANE0],uv).g*precision_factor[0];float cb=texture(textures[PLANE1],uv).r*precision_factor[1];float cr=texture(textures[PLANE1],uv).b*precision_factor[1];return ycbcra_to_rgba(y,cb,cr,1.0);}
    case 11:{float g=texture(textures[PLANE0],uv).r*precision_factor[0];float b=texture(textures[PLANE1],uv).r*precision_factor[1];float r=texture(textures[PLANE2],uv).r*precision_factor[2];return vec4(r,g,b,1.0);}
    case 12:{float g=texture(textures[PLANE0],uv).r*precision_factor[0];float b=texture(textures[PLANE1],uv).r*precision_factor[1];float r=texture(textures[PLANE2],uv).r*precision_factor[2];float a=texture(textures[PLANE3],uv).r*precision_factor[3];return vec4(r,g,b,a);}
    case 13:{vec4 c=texture(textures[PLANE0],uv);float scale=(c.b*(255.0/8.0))+1.0;float Co=(c.r-0.5)/scale;float Cg=(c.g-0.5)/scale;float Y=c.a;return vec4(clamp(Y+Co-Cg,0.0,1.0),clamp(Y+Cg,0.0,1.0),clamp(Y-Co-Cg,0.0,1.0),1.0);}
    case 14:{vec4 c=texture(textures[PLANE0],uv);float scale=(c.b*(255.0/8.0))+1.0;float Co=(c.r-0.5)/scale;float Cg=(c.g-0.5)/scale;float Y=c.a;float a=texture(textures[PLANE1],uv).a;return vec4(clamp(Y+Co-Cg,0.0,1.0),clamp(Y+Cg,0.0,1.0),clamp(Y-Co-Cg,0.0,1.0),a);}
    } return vec4(0.0);
}

// ── Blur ────────────────────────────────────────────────────────────────
vec4 get_blurred_color(vec2 uv){
    if(!flag(F_BLUR)||blur_radius<0.5)return get_rgba_color(uv);
    vec2 ts=1.0/target_size;vec4 tc=vec4(0.0);float tw=0.0;
    if(blur_type==0){int n=int(clamp(blur_radius*3.0,16.0,120.0));float sg=blur_radius/2.0;for(int i=0;i<n;i++){float t=float(i)/max(float(n-1),1.0);float r=sqrt(t)*blur_radius;float th=float(i)*2.39996323;float w=exp(-r*r/(2.0*sg*sg));tc+=sample_wrap(uv+vec2(cos(th),sin(th))*r*ts)*w;tw+=w;}}
    else if(blur_type==1){int s=min(int(blur_radius),6);float ss=blur_radius/max(float(s),1.0);for(int y=-s;y<=s;y++)for(int x=-s;x<=s;x++){tc+=sample_wrap(uv+vec2(float(x),float(y))*ss*ts);tw+=1.0;}}
    else if(blur_type==2){float ar=radians(blur_angle);vec2 d=vec2(cos(ar),sin(ar));int n=int(clamp(blur_radius*2.0,16.0,100.0));for(int i=0;i<n;i++){float t=(float(i)/max(float(n-1),1.0))-0.5;tc+=sample_wrap(uv+d*(t*blur_radius*2.0*ts));tw+=1.0;}}
    else if(blur_type==3){vec2 tp=uv-blur_center;float st=blur_radius*0.01;int n=int(clamp(blur_radius*2.0,16.0,100.0));for(int i=0;i<n;i++){float sc=1.0-st*(float(i)/max(float(n-1),1.0));tc+=sample_wrap(blur_center+tp*sc);tw+=1.0;}}
    else if(blur_type==4){float ar=radians(blur_angle);vec2 nm=vec2(sin(ar),cos(ar));float dst=abs(dot(uv-blur_center,nm)-(blur_tilt.x-0.5));float ba=smoothstep(blur_tilt.y*0.5,blur_tilt.y*0.5+0.2,dst)*blur_radius;if(ba<0.5)return get_rgba_color(uv);int n=int(clamp(ba*2.0,16.0,100.0));for(int i=0;i<n;i++){float t=float(i)/max(float(n-1),1.0);float r=sqrt(t)*ba;float th=float(i)*2.39996323;tc+=sample_wrap(uv+vec2(cos(th),sin(th))*r*ts);tw+=1.0;}}
    else if(blur_type==5){float ns=fract(sin(dot(uv,vec2(12.9898,78.233)))*43758.5453)*6.2831853;int n=int(clamp(blur_radius*8.0,32.0,400.0));for(int i=0;i<n;i++){float t=float(i)/max(float(n-1),1.0);float r=sqrt(t)*blur_radius;float th=ns+float(i)*2.39996323;vec4 cl=sample_wrap(uv+vec2(cos(th),sin(th))*r*ts);float l=dot(cl.rgb,vec3(0.299,0.587,0.114));float w=(1.0+pow(max(l-0.3,0.0),3.0)*15.0)*mix(0.3,1.0,t);tc+=cl*w;tw+=w;}}
    return tw>0.0?tc/tw:get_rgba_color(uv);
}

// ── Shape SDF ───────────────────────────────────────────────────────────
float sdf_box(vec2 p,vec2 b){vec2 d=abs(p)-b;return length(max(d,0.0))+min(max(d.x,d.y),0.0);}
float sdf_rbox(vec2 p,vec2 b,float r){vec2 d=abs(p)-b+r;return length(max(d,0.0))+min(max(d.x,d.y),0.0)-r;}
float sdf_circle(vec2 p,float r){return length(p)-r;}
float sdf_ellipse(vec2 p,vec2 ab){p=abs(p);if(p.x>p.y){p=p.yx;ab=ab.yx;}float l=ab.y*ab.y-ab.x*ab.x,m=ab.x*p.x/l,n=ab.y*p.y/l,m2=m*m,n2=n*n,c=(m2+n2-1.0)/3.0,c3=c*c*c,q=c3+m2*n2*2.0,d=c3+m2*n2,g=m+m*n2;float co;if(d<0.0){float h=acos(q/c3)/3.0,s=cos(h),t=sin(h)*sqrt(3.0);float rx=sqrt(-c*(s+t+2.0)+m2),ry=sqrt(-c*(s-t+2.0)+m2);co=(ry+sign(l)*rx+abs(g)/(rx*ry)-m)/2.0;}else{float h=2.0*m*n*sqrt(d);float s=sign(q+h)*pow(abs(q+h),1.0/3.0),u=sign(q-h)*pow(abs(q-h),1.0/3.0);float rx=-s-u-c*4.0+2.0*m2,ry=(s-u)*sqrt(3.0),rm=sqrt(rx*rx+ry*ry);co=(ry/sqrt(rm-rx)+2.0*g/rm-m)/2.0;}vec2 r=ab*vec2(co,sqrt(1.0-co*co));return length(r-p)*sign(p.y-r.y);}
vec4 shape_fill(vec2 uv){if(shape_fill_type==0)return shape_color1;float t;if(shape_fill_type==1){float r=shape_gradient_angle*PI/180.0;t=dot(uv-0.5,vec2(cos(r),sin(r)))+0.5;}else if(shape_fill_type==2)t=length(uv-shape_gradient_center)/0.5;else{vec2 d=uv-shape_gradient_center;t=atan(d.y,d.x)/(2.0*PI)+0.5;}return mix(shape_color1,shape_color2,clamp(t,0.0,1.0));}

// ── Main ────────────────────────────────────────────────────────────────
void main(){
    vec2 buv=TexCoord.st/TexCoord.q;vec4 col;
    if(flag(F_360)){vec2 uv=get_equirect_uv(buv);if(flag(F_FLIP_H))uv.s=1.0-uv.s;if(flag(F_FLIP_V))uv.t=1.0-uv.t;col=get_blurred_color(uv);}
    else if(flag(F_CURVED)){vec2 uv=apply_curve_warp(buv);if(flag(F_FLIP_H))uv.s=1.0-uv.s;if(flag(F_FLIP_V))uv.t=1.0-uv.t;col=get_blurred_color(uv);}
    else{vec2 uv=buv;if(flag(F_FLIP_H))uv.s=1.0-uv.s;if(flag(F_FLIP_V))uv.t=1.0-uv.t;col=get_blurred_color(uv);}

    if(flag(F_SHARPEN)){vec2 su=buv;if(flag(F_360)){su=get_equirect_uv(buv);if(flag(F_FLIP_H))su.s=1.0-su.s;if(flag(F_FLIP_V))su.t=1.0-su.t;}else if(flag(F_CURVED)){su=apply_curve_warp(buv);if(flag(F_FLIP_H))su.s=1.0-su.s;if(flag(F_FLIP_V))su.t=1.0-su.t;}else{if(flag(F_FLIP_H))su.s=1.0-su.s;if(flag(F_FLIP_V))su.t=1.0-su.t;}col.rgb=apply_sharpen(su,col.rgb,sharpen_amount,sharpen_radius);}
    if(flag(F_STRAIGHT_ALPHA))col.rgb*=col.a;

    if(flag(F_COLOR_GRADING)){col.rgb=apply_eotf(col.rgb,input_transfer);col.rgb*=exposure;col.rgb=ubo_mat3(input_to_working_c0,input_to_working_c1,input_to_working_c2)*col.rgb;if(flag(F_GAMUT_COMPRESS))col.rgb=apply_gamut_compress(col.rgb,gc_limit_pad.xyz);}
    if(flag(F_CDL))col.rgb=apply_cdl(col.rgb,cdl_slope_sat.xyz,cdl_offset_pad.xyz,cdl_power_pad.xyz,cdl_slope_sat.w);
    if(flag(F_LUT3D))col.rgb=apply_lut3d(col.rgb,lut3d_strength);
    if(flag(F_LINEAR_SAT))col.rgb=apply_linear_sat(col.rgb,linear_sat_value);
    if(flag(F_WHITE_BALANCE))col.rgb=apply_white_balance(col.rgb,wb_temperature,wb_tint);
    if(flag(F_LMG))col.rgb=apply_lmg(col.rgb,lmg_lift_pad.xyz,lmg_midtone_pad.xyz,lmg_gain_pad.xyz);
    if(flag(F_SPLIT_TONE))col.rgb=apply_split_tone(col.rgb,split_shadow_balance.xyz,split_highlight_pad.xyz,split_shadow_balance.w);
    if(flag(F_QUALIFIER))col.rgb=apply_qualifier(col.rgb,qual_target_hue,qual_hue_width,qual_min_sat,qual_max_sat,qual_min_lum,qual_max_lum,qual_softness,qual_exposure,qual_sat_offset,qual_hue_offset);
    if(flag(F_HUE_SHIFT))col.rgb=apply_hue_shift(col.rgb,hue_shift_degrees);
    if(flag(F_HUE_CURVE))col.rgb=apply_hue_curves(col.rgb);
    if(flag(F_TONEBALANCE))col.rgb=apply_tone_balance(col.rgb,tb_shadows,tb_highlights);
    if(flag(F_RGB_LEVELS))col.rgb=apply_rgb_levels(col.rgb);
    if(flag(F_CURVES))col.rgb=apply_curves(col.rgb);
    if(flag(F_LEVELS))col.rgb=LevelsControl(col.rgb,min_input,gamma_val,max_input,min_output,max_output);
    if(flag(F_CSB))col.rgb=ContrastSaturationBrightness(col,brt,sat,con);
    if(flag(F_INVERT))col.rgb=1.0-col.rgb;

    if(flag(F_SHAPE)){vec2 uv=TexCoord2.st;vec2 p=uv-shape_center;vec2 hs=shape_size*0.5;float d;
        if(shape_type==0)d=sdf_box(p,hs);else if(shape_type==1)d=sdf_rbox(p,hs,shape_corner_radius);else if(shape_type==2)d=sdf_circle(p,hs.x);else d=sdf_ellipse(p,hs);
        float fa=1.0-smoothstep(-shape_softness,0.0,d);vec4 fl=shape_fill(uv);
        if(flag(F_SHAPE_STROKE)&&shape_stroke_width>0.0){float ra=(1.0-smoothstep(-shape_softness,0.0,abs(d)-shape_stroke_width))*fa;fl=mix(fl,shape_stroke_color,ra*shape_stroke_color.a);}
        float ca=fl.a*fa;col.rgb=col.rgb*(1.0-ca)+fl.rgb*ca;col.a=col.a*(1.0-ca)+ca;}

    col*=opacity;
    if(flag(F_LOCAL_KEY))col.a*=texture(textures[LOCAL_KEY],TexCoord2.st).r;
    if(flag(F_LAYER_KEY))col.a*=texture(textures[LAYER_KEY],TexCoord2.st).r;
    col=blend_op(col);
    if(flag(F_CHROMA))col=ChromaKey(col,flag(F_CHROMA_MASK));

    if(flag(F_COLOR_GRADING)){if(tone_mapping_op>0)col.rgb=apply_tone_mapping(col.rgb,tone_mapping_op);col.rgb=ubo_mat3(working_to_output_c0,working_to_output_c1,working_to_output_c2)*col.rgb;if(tone_mapping_op==0)col.rgb=clamp(col.rgb,0.0,1.0);col.rgb=apply_oetf(col.rgb,output_transfer);}
    if(flag(F_GRAIN))col.rgb=apply_grain(col.rgb,TexCoord.st/TexCoord.q,grain_intensity,grain_size,grain_frame);
    if(flag(F_EDGE_BLEND)){vec2 ub=TexCoord.st/TexCoord.q;float ba=1.0;if(edge_blend_left>0.0)ba*=pow(clamp(ub.x/edge_blend_left,0.0,1.0),edge_blend_gamma);if(edge_blend_right>0.0)ba*=pow(clamp((1.0-ub.x)/edge_blend_right,0.0,1.0),edge_blend_gamma);if(edge_blend_top>0.0)ba*=pow(clamp(ub.y/edge_blend_top,0.0,1.0),edge_blend_gamma);if(edge_blend_bottom>0.0)ba*=pow(clamp((1.0-ub.y)/edge_blend_bottom,0.0,1.0),edge_blend_gamma);col*=ba;}
    fragColor=flag2(F2_OUTPUT_BGRA)?col.bgra:col;
}
