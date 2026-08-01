/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf_shader.h"

#include "isf_gl_context.h"
#include "isf_image_load.h"

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/log.h>
#include <common/utf.h>

#include <GL/glew.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <sstream>

namespace caspar { namespace isf {

namespace {

/// Extract the ISF JSON header (the first {...} block, which lives inside a leading /* */ comment).
/// Brace counting is string-aware so that braces inside JSON string values do not unbalance it.
std::string extract_json(const std::string& source)
{
    const auto open = source.find('{');
    if (open == std::string::npos)
        return {};
    int         depth   = 0;
    bool        in_str  = false;
    bool        escaped = false;
    std::size_t i       = open;
    for (; i < source.size(); ++i) {
        const char c = source[i];
        if (in_str) {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                in_str = false;
            continue;
        }
        if (c == '"')
            in_str = true;
        else if (c == '{')
            ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) {
                ++i;
                break;
            }
        }
    }
    return source.substr(open, i - open);
}

/// GLSL uniform type for an ISF input type.
const char* gl_type_of(const std::string& t)
{
    if (t == "float")
        return "float";
    if (t == "bool" || t == "event")
        return "bool";
    if (t == "long")
        return "int";
    if (t == "color")
        return "vec4";
    if (t == "point2D")
        return "vec2";
    return nullptr; // image / unsupported -> not declared as a scalar uniform
}

GLuint compile(GLenum type, const std::string& src, std::string& log)
{
    GLuint s   = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string l(len > 0 ? len : 1, '\0');
        glGetShaderInfoLog(s, static_cast<GLsizei>(l.size()), nullptr, l.data());
        log = l;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

/// Minimal recursive-descent evaluator for ISF PASSES WIDTH/HEIGHT equations.
/// Supports + - * / %, unary +/-, parentheses, numbers, $VAR identifiers and a few functions
/// (floor/ceil/abs/sqrt/min/max/mod/pow/sin/cos/clamp).
class expr_eval
{
    const std::string&                                s_;
    std::size_t                                       p_ = 0;
    const std::function<double(const std::string&)>&  var_;

    // The grammar recurses once per '(' and once per unary +/-, both of which an
    // ISF file controls without limit, so a long run of either overflows the
    // stack. Bail out at a depth no legitimate size expression reaches.
    static constexpr int MAX_DEPTH = 64;
    int                  depth_    = 0;

    // RAII depth counter: every recursive entry point takes one.
    struct depth_guard
    {
        expr_eval& e;
        bool       ok;
        explicit depth_guard(expr_eval& ev)
            : e(ev)
            , ok(ev.depth_ < MAX_DEPTH)
        {
            ++e.depth_;
        }
        ~depth_guard() { --e.depth_; }
    };

  public:
    expr_eval(const std::string& s, const std::function<double(const std::string&)>& var)
        : s_(s)
        , var_(var)
    {
    }
    double parse() { return expr(); }

  private:
    void skip()
    {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_])))
            ++p_;
    }
    bool match(char c)
    {
        skip();
        if (p_ < s_.size() && s_[p_] == c) {
            ++p_;
            return true;
        }
        return false;
    }
    double expr()
    {
        depth_guard g(*this);
        if (!g.ok)
            return 0.0;
        double v = term();
        for (;;) {
            skip();
            if (match('+'))
                v += term();
            else if (match('-'))
                v -= term();
            else
                break;
        }
        return v;
    }
    double term()
    {
        double v = factor();
        for (;;) {
            skip();
            if (match('*'))
                v *= factor();
            else if (match('/')) {
                double d = factor();
                v        = d != 0.0 ? v / d : 0.0;
            } else if (match('%')) {
                double d = factor();
                v        = d != 0.0 ? std::fmod(v, d) : 0.0;
            } else
                break;
        }
        return v;
    }
    double factor()
    {
        depth_guard g(*this);
        if (!g.ok)
            return 0.0;
        skip();
        if (match('+'))
            return factor();
        if (match('-'))
            return -factor();
        if (match('(')) {
            double v = expr();
            match(')');
            return v;
        }
        if (p_ < s_.size() &&
            (std::isalpha(static_cast<unsigned char>(s_[p_])) || s_[p_] == '$' || s_[p_] == '_'))
            return ident();
        return number();
    }
    double number()
    {
        skip();
        std::size_t start = p_;
        while (p_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[p_])) || s_[p_] == '.'))
            ++p_;
        if (p_ == start)
            return 0.0;
        try {
            return std::stod(s_.substr(start, p_ - start));
        } catch (...) {
            return 0.0;
        }
    }
    double ident()
    {
        skip();
        std::size_t start = p_;
        if (p_ < s_.size() && s_[p_] == '$')
            ++p_;
        while (p_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[p_])) || s_[p_] == '_'))
            ++p_;
        std::string name = s_.substr(start, p_ - start);
        skip();
        if (p_ < s_.size() && s_[p_] == '(') {
            std::vector<double> args;
            match('(');
            if (!match(')')) {
                args.push_back(expr());
                while (match(','))
                    args.push_back(expr());
                match(')');
            }
            return call_func(name, args);
        }
        if (!name.empty() && name[0] == '$')
            name = name.substr(1);
        return var_(name);
    }
    static double call_func(const std::string& f, const std::vector<double>& a)
    {
        auto A = [&](std::size_t i) { return i < a.size() ? a[i] : 0.0; };
        if (f == "floor")
            return std::floor(A(0));
        if (f == "ceil")
            return std::ceil(A(0));
        if (f == "abs")
            return std::fabs(A(0));
        if (f == "sqrt")
            return std::sqrt(A(0));
        if (f == "min")
            return std::min(A(0), A(1));
        if (f == "max")
            return std::max(A(0), A(1));
        if (f == "mod")
            return A(1) != 0.0 ? std::fmod(A(0), A(1)) : 0.0;
        if (f == "pow")
            return std::pow(A(0), A(1));
        if (f == "sin")
            return std::sin(A(0));
        if (f == "cos")
            return std::cos(A(0));
        if (f == "clamp")
            return std::min(std::max(A(0), A(1)), A(2));
        return 0.0;
    }
};

int eval_size_expr(const std::string&                                expr,
                   int                                               fallback,
                   int                                               render_w,
                   int                                               render_h,
                   const std::map<std::string, std::vector<double>>& values)
{
    if (expr.empty())
        return fallback;
    std::function<double(const std::string&)> var = [&](const std::string& n) -> double {
        if (n == "WIDTH")
            return render_w;
        if (n == "HEIGHT")
            return render_h;
        auto it = values.find(n);
        if (it != values.end() && !it->second.empty())
            return it->second[0];
        return 0.0;
    };
    expr_eval e(expr, var);
    const double d = e.parse();

    // The result becomes a render-target dimension, so it has to survive a
    // hostile expression: lround() on a non-finite or out-of-range double is
    // undefined, and an arbitrarily large finite result would be handed
    // straight to a texture allocation.
    if (!std::isfinite(d))
        return fallback;
    constexpr double MAX_DIM = 16384.0; // beyond any GL_MAX_TEXTURE_SIZE in practice
    if (d < 1.0 || d > MAX_DIM)
        return fallback;
    const int v = static_cast<int>(std::lround(d));
    return v > 0 ? v : fallback;
}

} // namespace

namespace {

/// Runtime info for one bound sampler during a render.
struct bound_image
{
    GLuint tex  = 0;
    int    w    = 1;
    int    h    = 1;
    bool   flip = false;
    bool   bgra = false;
};

} // namespace

struct shader::impl
{
    std::string  source_;
    std::wstring base_path_;
    std::string  description_;

    std::vector<input>                         inputs_;
    std::vector<std::string>                   image_names_; ///< declared image input names (order)
    std::map<std::string, std::vector<double>> values_;
    shader_role                                role_ = shader_role::generator;

    std::string vertex_source_; ///< custom .vs body (empty = generated)

    struct pass_desc
    {
        std::string target;
        bool        persistent = false;
        bool        is_float   = false;
        std::string w_expr;
        std::string h_expr;
    };
    std::vector<pass_desc> passes_;

    struct imported_desc
    {
        std::string  name;
        std::wstring path;
    };
    std::vector<imported_desc> imported_;

    std::vector<std::string> sampler_names_; ///< all declared samplers (images + imported + targets)

    // GL state (created lazily on the device thread).
    GLuint program_  = 0;
    GLuint vao_      = 0;
    bool   compiled_ = false;
    bool   failed_   = false;

    // Cached uniform locations (populated once after link; avoids per-frame glGetUniformLocation
    // and string building in the render hot path).
    struct sampler_loc
    {
        GLint tex  = -1;
        GLint size = -1;
        GLint rect = -1;
        GLint flip = -1;
        GLint bgra = -1;
    };
    GLint u_rendersize_ = -1, u_time_ = -1, u_timedelta_ = -1, u_frameindex_ = -1, u_passindex_ = -1,
          u_date_ = -1;
    std::vector<sampler_loc> sampler_locs_; ///< parallel to sampler_names_
    std::vector<GLint>       input_locs_;   ///< parallel to inputs_ (-1 for image inputs)

    std::map<std::string, GLuint> upload_tex_;   ///< per-image-input CPU upload textures
    std::map<std::string, GLuint> imported_tex_; ///< imported name -> GL tex
    std::map<std::string, std::pair<int, int>> imported_size_;
    bool                                       imported_loaded_ = false;

    struct gl_buffer
    {
        GLuint tex[2]    = {0, 0};
        int    front     = 0;
        int    w         = 0;
        int    h         = 0;
        bool   is_float  = false;
        bool   persistent= false;
        bool   written   = false;
    };
    std::map<std::string, gl_buffer> buffers_;
    GLuint                           final_tex_ = 0; ///< scratch output for empty-target passes
    int                              final_w_   = 0;
    int                              final_h_   = 0;

    std::weak_ptr<accelerator::ogl::device> device_;

    explicit impl(const std::string& source, const std::wstring& base_path, const std::string& vertex_source)
        : source_(source)
        , base_path_(base_path)
        , vertex_source_(vertex_source)
    {
        parse();
    }

    // -- parsing ---------------------------------------------------------------------------------

    void parse()
    {
        const auto json = extract_json(source_);
        boost::property_tree::ptree pt;
        if (!json.empty()) {
            try {
                std::istringstream is(json);
                boost::property_tree::read_json(is, pt);
            } catch (...) {
                pt.clear();
            }
        }

        description_ = pt.get<std::string>("DESCRIPTION", "");

        if (auto inputs = pt.get_child_optional("INPUTS")) {
            for (const auto& kv : *inputs) {
                const auto& node = kv.second;
                input       in;
                in.name  = node.get<std::string>("NAME", "");
                in.type  = node.get<std::string>("TYPE", "");
                in.label = node.get<std::string>("LABEL", in.name);
                if (in.name.empty() || in.type.empty())
                    continue;
                in.is_image = (in.type == "image");
                read_num_array(node, "MIN", in.min_value);
                read_num_array(node, "MAX", in.max_value);
                read_num_array(node, "DEFAULT", in.default_value);
                if (in.type == "long") {
                    if (auto vals = node.get_child_optional("VALUES"))
                        for (const auto& v : *vals)
                            in.values.push_back(v.second.get_value<long>(0));
                    if (auto labs = node.get_child_optional("LABELS"))
                        for (const auto& l : *labs)
                            in.labels.push_back(l.second.get_value<std::string>(""));
                }
                if (in.default_value.empty())
                    in.default_value.push_back(0.0);
                if (in.is_image)
                    image_names_.push_back(in.name);
                else
                    values_[in.name] = in.default_value;
                inputs_.push_back(std::move(in));
            }
        }

        // Role from ISF conventions.
        auto has_image = [&](const std::string& n) {
            return std::find(image_names_.begin(), image_names_.end(), n) != image_names_.end();
        };
        if (has_image("startImage") && has_image("endImage"))
            role_ = shader_role::transition;
        else if (has_image("inputImage"))
            role_ = shader_role::filter;
        else
            role_ = shader_role::generator;

        // IMPORTED images.
        if (auto imp = pt.get_child_optional("IMPORTED")) {
            for (const auto& kv : *imp) {
                imported_desc d;
                d.name = kv.first;
                d.path = u16(kv.second.get<std::string>("PATH", ""));
                if (!d.name.empty() && !d.path.empty())
                    imported_.push_back(std::move(d));
            }
        }

        // PASSES.
        if (auto passes = pt.get_child_optional("PASSES")) {
            for (const auto& kv : *passes) {
                const auto& node = kv.second;
                pass_desc   p;
                p.target     = node.get<std::string>("TARGET", "");
                p.persistent = node.get<bool>("PERSISTENT", false) || node.get<int>("PERSISTENT", 0) != 0;
                p.is_float   = node.get<bool>("FLOAT", false) || node.get<int>("FLOAT", 0) != 0;
                p.w_expr     = node.get<std::string>("WIDTH", "");
                p.h_expr     = node.get<std::string>("HEIGHT", "");
                passes_.push_back(std::move(p));
            }
        }
        if (passes_.empty())
            passes_.push_back(pass_desc{}); // implicit single output pass

        // Union of sampler names to declare: image inputs + imported + pass targets (+ inputImage).
        std::set<std::string> seen;
        auto add_sampler = [&](const std::string& n) {
            if (!n.empty() && seen.insert(n).second)
                sampler_names_.push_back(n);
        };
        for (const auto& n : image_names_)
            add_sampler(n);
        for (const auto& d : imported_)
            add_sampler(d.name);
        for (const auto& p : passes_)
            add_sampler(p.target);
        add_sampler("inputImage"); // lenient: also available to non-conformant generators
    }

    static void read_num_array(const boost::property_tree::ptree& node, const char* key, std::vector<double>& out)
    {
        if (auto n = node.get_child_optional(key)) {
            if (n->empty())
                out.push_back(n->get_value<double>(0.0));
            else
                for (const auto& e : *n)
                    out.push_back(e.second.get_value<double>(0.0));
        }
    }

    // -- fragment / program ----------------------------------------------------------------------

    std::string build_fragment() const
    {
        std::ostringstream f;
        f << "#version 330 core\n"
             "uniform vec2 RENDERSIZE;\n"
             "uniform float TIME;\n"
             "uniform float TIMEDELTA;\n"
             "uniform int FRAMEINDEX;\n"
             "uniform int PASSINDEX;\n"
             "uniform vec4 DATE;\n";
        for (const auto& n : sampler_names_) {
            f << "uniform sampler2D " << n << ";\n";
            f << "uniform vec2 _" << n << "_imgSize;\n";
            f << "uniform vec4 _" << n << "_imgRect;\n";
            f << "uniform bool _" << n << "_flip;\n";
            f << "uniform bool _" << n << "_bgra;\n";
        }
        for (const auto& in : inputs_) {
            if (const char* t = gl_type_of(in.type))
                f << "uniform " << t << " " << in.name << ";\n";
        }
        f << "in vec2 isf_FragNormCoord;\n"
             "out vec4 isf_out_color;\n"
             "vec4 _isf_fetch(sampler2D s, vec2 nc, bool flp, bool bgr) {\n"
             "  vec4 c = texture(s, flp ? vec2(nc.x, 1.0 - nc.y) : nc);\n"
             "  return bgr ? c.bgra : c;\n"
             "}\n"
             "#define gl_FragColor isf_out_color\n"
             "#define vv_FragNormCoord isf_FragNormCoord\n"
             "#define isf_FragCoord (isf_FragNormCoord * RENDERSIZE)\n"
             "#define IMG_SIZE(image) (_ ## image ## _imgSize)\n"
             "#define IMG_NORM_PIXEL(image, nc) _isf_fetch(image, vec2(nc), _ ## image ## _flip, _ ## image ## "
             "_bgra)\n"
             "#define IMG_PIXEL(image, pc) IMG_NORM_PIXEL(image, (pc) / IMG_SIZE(image))\n"
             "#define IMG_THIS_NORM_PIXEL(image) IMG_NORM_PIXEL(image, isf_FragNormCoord)\n"
             "#define IMG_THIS_PIXEL(image) IMG_THIS_NORM_PIXEL(image)\n"
             "#line 1\n"
          << source_;
        return f.str();
    }

    std::string build_vertex() const
    {
        std::ostringstream v;
        v << "#version 330 core\n"
             "out vec2 isf_FragNormCoord;\n"
             "void isf_vertShaderInit() {\n"
             "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
             "  isf_FragNormCoord = p;\n"
             "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
             "}\n";
        if (vertex_source_.empty())
            v << "void main() { isf_vertShaderInit(); }\n";
        else
            v << "#line 1\n" << vertex_source_;
        return v.str();
    }

    bool ensure_compiled()
    {
        if (compiled_)
            return program_ != 0;
        compiled_ = true;

        std::string log;
        GLuint      vs = compile(GL_VERTEX_SHADER, build_vertex(), log);
        if (!vs) {
            CASPAR_LOG(warning) << L"[isf] vertex shader compile failed: " << u16(log);
            return false;
        }
        GLuint fs = compile(GL_FRAGMENT_SHADER, build_fragment(), log);
        if (!fs) {
            CASPAR_LOG(warning) << L"[isf] shader compile failed: " << u16(log);
            glDeleteShader(vs);
            return false;
        }
        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok = GL_FALSE;
        glGetProgramiv(program_, GL_LINK_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &len);
            std::string l(len > 0 ? len : 1, '\0');
            glGetProgramInfoLog(program_, static_cast<GLsizei>(l.size()), nullptr, l.data());
            CASPAR_LOG(warning) << L"[isf] shader link failed: " << u16(l);
            glDeleteProgram(program_);
            program_ = 0;
            return false;
        }
        glGenVertexArrays(1, &vao_);
        cache_locations();
        return true;
    }

    void cache_locations()
    {
        u_rendersize_ = glGetUniformLocation(program_, "RENDERSIZE");
        u_time_       = glGetUniformLocation(program_, "TIME");
        u_timedelta_  = glGetUniformLocation(program_, "TIMEDELTA");
        u_frameindex_ = glGetUniformLocation(program_, "FRAMEINDEX");
        u_passindex_  = glGetUniformLocation(program_, "PASSINDEX");
        u_date_       = glGetUniformLocation(program_, "DATE");
        sampler_locs_.resize(sampler_names_.size());
        for (std::size_t i = 0; i < sampler_names_.size(); ++i) {
            const auto& n     = sampler_names_[i];
            sampler_locs_[i].tex  = glGetUniformLocation(program_, n.c_str());
            sampler_locs_[i].size = glGetUniformLocation(program_, ("_" + n + "_imgSize").c_str());
            sampler_locs_[i].rect = glGetUniformLocation(program_, ("_" + n + "_imgRect").c_str());
            sampler_locs_[i].flip = glGetUniformLocation(program_, ("_" + n + "_flip").c_str());
            sampler_locs_[i].bgra = glGetUniformLocation(program_, ("_" + n + "_bgra").c_str());
        }
        input_locs_.resize(inputs_.size());
        for (std::size_t i = 0; i < inputs_.size(); ++i)
            input_locs_[i] = inputs_[i].is_image ? -1 : glGetUniformLocation(program_, inputs_[i].name.c_str());
    }

    // -- texture helpers -------------------------------------------------------------------------

    static void set_tex_params()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    GLuint upload_named(const std::string& name, const unsigned char* rgba, int w, int h)
    {
        GLuint& t = upload_tex_[name];
        if (t == 0) {
            glGenTextures(1, &t);
            glBindTexture(GL_TEXTURE_2D, t);
            set_tex_params();
        } else {
            glBindTexture(GL_TEXTURE_2D, t);
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        glBindTexture(GL_TEXTURE_2D, 0);
        return t;
    }

    static GLuint make_buffer_tex(int w, int h, bool is_float)
    {
        GLuint t = 0;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        set_tex_params();
        if (is_float)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        return t;
    }

    gl_buffer& ensure_buffer(const std::string& name, int w, int h, bool is_float, bool persistent)
    {
        gl_buffer& b = buffers_[name];
        const int  n = persistent ? 2 : 1;
        if (b.w != w || b.h != h || b.is_float != is_float || b.tex[0] == 0) {
            for (int i = 0; i < 2; ++i) {
                if (b.tex[i]) {
                    glDeleteTextures(1, &b.tex[i]);
                    b.tex[i] = 0;
                }
            }
            for (int i = 0; i < n; ++i)
                b.tex[i] = make_buffer_tex(w, h, is_float);
            b.w          = w;
            b.h          = h;
            b.is_float   = is_float;
            b.front      = 0;

            // Persistent buffers are read (as the previous frame) before they are first written,
            // so they must start cleared rather than with undefined texture contents.
            if (persistent) {
                GLint prev_fbo = 0;
                glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
                GLuint cfbo = 0;
                glGenFramebuffers(1, &cfbo);
                glBindFramebuffer(GL_FRAMEBUFFER, cfbo);
                for (int i = 0; i < n; ++i) {
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, b.tex[i], 0);
                    glClearColor(0.f, 0.f, 0.f, 0.f);
                    glClear(GL_COLOR_BUFFER_BIT);
                }
                glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
                glDeleteFramebuffers(1, &cfbo);
            }
        }
        b.persistent = persistent;
        return b;
    }

    GLuint ensure_final(int w, int h)
    {
        if (final_tex_ == 0 || final_w_ != w || final_h_ != h) {
            if (final_tex_)
                glDeleteTextures(1, &final_tex_);
            final_tex_ = make_buffer_tex(w, h, false);
            final_w_   = w;
            final_h_   = h;
        }
        return final_tex_;
    }

    void ensure_imported()
    {
        if (imported_loaded_)
            return;
        imported_loaded_ = true;
        namespace fs     = std::filesystem;
        for (const auto& d : imported_) {
            std::vector<unsigned char> rgba;
            int                        w = 0, h = 0;
            bool                       ok = false;
            if (!base_path_.empty()) {
                const std::wstring full = (fs::path(base_path_) / fs::path(d.path)).wstring();
                ok                      = load_rgba_image(full, rgba, w, h);
            }
            if (!ok)
                ok = load_rgba_image(d.path, rgba, w, h);

            GLuint t = 0;
            glGenTextures(1, &t);
            glBindTexture(GL_TEXTURE_2D, t);
            set_tex_params();
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            if (ok) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
                imported_size_[d.name] = {w, h};
            } else {
                const unsigned char black[4] = {0, 0, 0, 0};
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, black);
                imported_size_[d.name] = {1, 1};
                CASPAR_LOG(warning) << L"[isf] IMPORTED '" << u16(d.name) << L"' could not be loaded from '"
                                    << d.path << L"'.";
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            imported_tex_[d.name] = t;
        }
    }

    // -- uniforms --------------------------------------------------------------------------------

    void set_scalar_uniforms(int width, int height, double time, double time_delta, int frame_index, int pass_index)
    {
        glUniform2f(u_rendersize_, static_cast<float>(width), static_cast<float>(height));
        glUniform1f(u_time_, static_cast<float>(time));
        glUniform1f(u_timedelta_, static_cast<float>(time_delta));
        glUniform1i(u_frameindex_, frame_index);
        glUniform1i(u_passindex_, pass_index);

        std::time_t now = std::time(nullptr);
        std::tm     lt{};
#if defined(_WIN32)
        localtime_s(&lt, &now);
#else
        localtime_r(&now, &lt);
#endif
        glUniform4f(u_date_,
                    static_cast<float>(lt.tm_year + 1900),
                    static_cast<float>(lt.tm_mon + 1),
                    static_cast<float>(lt.tm_mday),
                    static_cast<float>(lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec));

        for (std::size_t k = 0; k < inputs_.size(); ++k) {
            const auto& in  = inputs_[k];
            const GLint loc = input_locs_[k];
            if (in.is_image || loc < 0)
                continue;
            const auto& v = values_.at(in.name);
            auto        g = [&](std::size_t i) { return i < v.size() ? static_cast<float>(v[i]) : 0.0f; };
            if (in.type == "float")
                glUniform1f(loc, g(0));
            else if (in.type == "bool" || in.type == "event" || in.type == "long")
                glUniform1i(loc, static_cast<int>(g(0)));
            else if (in.type == "point2D")
                glUniform2f(loc, g(0), g(1));
            else if (in.type == "color")
                glUniform4f(loc, g(0), g(1), g(2), g(3));
        }
    }

    void bind_samplers(const std::map<std::string, bound_image>& images)
    {
        int unit = 0;
        for (std::size_t i = 0; i < sampler_names_.size(); ++i) {
            const auto& n  = sampler_names_[i];
            const auto& sl = sampler_locs_[i];
            bound_image bi;
            if (auto it = images.find(n); it != images.end())
                bi = it->second;
            else if (auto it2 = imported_tex_.find(n); it2 != imported_tex_.end()) {
                bi.tex  = it2->second;
                bi.w    = imported_size_[n].first;
                bi.h    = imported_size_[n].second;
                bi.flip = true; // imported images are uploaded top-down
            } else if (auto it3 = buffers_.find(n); it3 != buffers_.end()) {
                bi.tex = it3->second.tex[it3->second.front];
                bi.w   = it3->second.w;
                bi.h   = it3->second.h;
            }
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, bi.tex);
            glUniform1i(sl.tex, unit);
            glUniform2f(sl.size, static_cast<float>(bi.w > 0 ? bi.w : 1), static_cast<float>(bi.h > 0 ? bi.h : 1));
            if (sl.rect >= 0)
                glUniform4f(sl.rect, 0.f, 0.f, 1.f, 1.f);
            glUniform1i(sl.flip, bi.flip ? 1 : 0);
            glUniform1i(sl.bgra, bi.bgra ? 1 : 0);
            ++unit;
        }
        glActiveTexture(GL_TEXTURE0);
    }

    // -- render core (context-agnostic; runs on whichever GL context/thread is current) ----------

    /// Run all passes on the current GL context. Returns the final raw GL texture (bottom-up RGBA)
    /// and its size via last_w/last_h, or 0 on failure.
    GLuint render_gl(int                               width,
                     int                               height,
                     double                            time,
                     double                            time_delta,
                     int                               frame_index,
                     const std::vector<image_binding>& images,
                     int&                              last_w,
                     int&                              last_h)
    {
        while (glGetError() != GL_NO_ERROR) {}
        if (!ensure_compiled()) {
            failed_ = true;
            return 0;
        }
        ensure_imported();

        std::map<std::string, bound_image> bound;
        for (const auto& b : images) {
            bound_image bi;
            bi.w    = b.width > 0 ? b.width : width;
            bi.h    = b.height > 0 ? b.height : height;
            bi.flip = b.flip;
            bi.bgra = b.bgra;
            if (b.tex_id != 0)
                bi.tex = b.tex_id;
            else if (b.rgba != nullptr)
                bi.tex = upload_named(b.name, b.rgba, bi.w, bi.h);
            bound[b.name] = bi;
        }

        GLint prev_fbo = 0, prev_vp[4] = {0, 0, 0, 0};
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        glGetIntegerv(GL_VIEWPORT, prev_vp);

        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glDisable(GL_BLEND);
        glUseProgram(program_);
        glBindVertexArray(vao_);

        GLuint last_tex = 0;
        last_w          = width;
        last_h          = height;

        for (std::size_t i = 0; i < passes_.size(); ++i) {
            const auto& ps = passes_[i];
            const int   pw = eval_size_expr(ps.w_expr, width, width, height, values_);
            const int   ph = eval_size_expr(ps.h_expr, height, width, height, values_);

            GLuint write_tex = 0;
            if (ps.target.empty()) {
                write_tex = ensure_final(pw, ph);
            } else {
                auto& b   = ensure_buffer(ps.target, pw, ph, ps.is_float, ps.persistent);
                write_tex = ps.persistent ? b.tex[1 - b.front] : b.tex[0];
                b.written = true;
            }

            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, write_tex, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                CASPAR_LOG(warning) << L"[isf] framebuffer incomplete (unsupported buffer format?); "
                                    << L"disabling shader.";
                failed_ = true;
                break;
            }
            glViewport(0, 0, pw, ph);
            glClearColor(0.f, 0.f, 0.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT);

            set_scalar_uniforms(pw, ph, time, time_delta, frame_index, static_cast<int>(i));
            bind_samplers(bound);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            last_tex = write_tex;
            last_w   = pw;
            last_h   = ph;
        }

        glBindVertexArray(0);
        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
        glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
        glDeleteFramebuffers(1, &fbo);

        for (auto& kv : buffers_) {
            if (kv.second.persistent && kv.second.written)
                kv.second.front = 1 - kv.second.front;
            kv.second.written = false;
        }
        if (failed_)
            return 0;
        return last_tex;
    }

    /// Scratch for readback_bgra, kept across frames.
    ///
    /// This used to be a local, so a 1080p shader allocated and freed 8.3 MB on
    /// every single frame.
    std::vector<unsigned char> readback_tmp_;

    /// Read a raw GL texture (bottom-up RGBA) back into a tightly-packed, top-down BGRA CPU buffer
    /// (the layout a CPU const_frame labelled bgra expects).
    void readback_bgra(GLuint tex, int tw, int th, int out_w, int out_h, unsigned char* dst, int dst_stride)
    {
        const std::size_t need = static_cast<std::size_t>(tw) * th * 4;
        if (readback_tmp_.size() < need)
            readback_tmp_.resize(need);

        glBindTexture(GL_TEXTURE_2D, tex);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        // GL_BGRA, not GL_RGBA: the driver reorders during the transfer, which
        // costs nothing, and it is the order the frame wants. Reading RGBA and
        // swapping afterwards meant a scalar loop over every pixel -- two
        // million iterations a frame at 1080p, four byte moves each -- purely to
        // exchange two channels.
        glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, readback_tmp_.data());
        glBindTexture(GL_TEXTURE_2D, 0);

        const int w = std::min(tw, out_w);
        const int h = std::min(th, out_h);
        // GL hands back bottom-up, the frame wants top-down, so the rows are
        // walked in reverse -- but each row is now a straight copy.
        const std::size_t row = static_cast<std::size_t>(w) * 4;
        for (int y = 0; y < h; ++y) {
            const unsigned char* s = readback_tmp_.data() + static_cast<std::size_t>(th - 1 - y) * tw * 4;
            unsigned char*       d = dst + static_cast<std::size_t>(y) * dst_stride;
            std::memcpy(d, s, row);
        }
    }

    bool render_readback(int                               width,
                         int                               height,
                         double                            time,
                         double                            time_delta,
                         int                               frame_index,
                         const std::vector<image_binding>& images,
                         unsigned char*                    dst,
                         int                               dst_stride)
    {
        int    lw = 0, lh = 0;
        GLuint ft = render_gl(width, height, time, time_delta, frame_index, images, lw, lh);
        if (!ft)
            return false;
        readback_bgra(ft, lw, lh, width, height, dst, dst_stride);
        while (glGetError() != GL_NO_ERROR) {}
        return true;
    }

    void release_gl()
    {
        if (program_)
            glDeleteProgram(program_);
        if (vao_)
            glDeleteVertexArrays(1, &vao_);
        for (auto& kv : upload_tex_)
            if (kv.second)
                glDeleteTextures(1, &kv.second);
        for (auto& kv : imported_tex_)
            if (kv.second)
                glDeleteTextures(1, &kv.second);
        for (auto& kv : buffers_)
            for (int i = 0; i < 2; ++i)
                if (kv.second.tex[i])
                    glDeleteTextures(1, &kv.second.tex[i]);
        if (final_tex_)
            glDeleteTextures(1, &final_tex_);
        program_ = 0;
        vao_     = 0;
        upload_tex_.clear();
        imported_tex_.clear();
        buffers_.clear();
        final_tex_ = 0;
    }
};

shader::shader(const std::string& source, const std::wstring& base_path, const std::string& vertex_source)
    : impl_(std::make_unique<impl>(source, base_path, vertex_source))
{
}
shader::~shader()
{
    if (auto dev = impl_->device_.lock()) {
        auto* p = impl_.get();
        try {
            dev->dispatch_sync([p]() { p->release_gl(); });
        } catch (...) {
        }
    }
}

const std::vector<input>& shader::inputs() const { return impl_->inputs_; }
const std::string&        shader::description() const { return impl_->description_; }
shader_role               shader::role() const { return impl_->role_; }
std::vector<std::string>  shader::image_input_names() const { return impl_->image_names_; }

bool shader::set_value(const std::string& name, const std::vector<double>& values)
{
    auto it = impl_->values_.find(name);
    if (it == impl_->values_.end())
        return false;
    it->second = values;
    return true;
}

void shader::reset_events()
{
    for (const auto& in : impl_->inputs_)
        if (in.type == "event")
            impl_->values_[in.name] = {0.0};
}

std::shared_ptr<core::texture> shader::render(const std::shared_ptr<accelerator::ogl::device>& device,
                                              int                                               width,
                                              int                                               height,
                                              double                                            time,
                                              double                                            time_delta,
                                              int                                               frame_index,
                                              const std::vector<image_binding>&                 images)
{
    if (impl_->failed_ || width <= 0 || height <= 0 || !device)
        return nullptr;

    impl_->device_ = device;

    return device->dispatch_sync([&]() -> std::shared_ptr<core::texture> {
        auto* p = impl_.get();
        int   last_w = width, last_h = height;
        GLuint last_tex = p->render_gl(width, height, time, time_delta, frame_index, images, last_w, last_h);
        if (!last_tex)
            return nullptr;

        // ISF renders bottom-up; the mixer is top-down. Y-flip the final pass into a mixer texture.
        auto   out_tex = device->create_texture(width, height, 4, common::bit_depth::bit8, false);
        GLuint fbos[2] = {0, 0};
        glCreateFramebuffers(2, fbos);
        glNamedFramebufferTexture(fbos[0], GL_COLOR_ATTACHMENT0, last_tex, 0);
        glNamedFramebufferTexture(fbos[1], GL_COLOR_ATTACHMENT0, out_tex->id(), 0);
        glBlitNamedFramebuffer(fbos[0],
                               fbos[1],
                               0,
                               0,
                               last_w,
                               last_h,
                               0,
                               height,
                               width,
                               0,
                               GL_COLOR_BUFFER_BIT,
                               (last_w == width && last_h == height) ? GL_NEAREST : GL_LINEAR);
        glDeleteFramebuffers(2, fbos);

        while (glGetError() != GL_NO_ERROR) {}
        return std::static_pointer_cast<core::texture>(out_tex);
    });
}

bool shader::render_readback(gl_context&                       ctx,
                             int                               width,
                             int                               height,
                             double                            time,
                             double                            time_delta,
                             int                               frame_index,
                             const std::vector<image_binding>& images,
                             unsigned char*                    dst,
                             int                               dst_stride)
{
    if (impl_->failed_ || width <= 0 || height <= 0)
        return false;
    if (!ctx.make_current())
        return false;
    return impl_->render_readback(width, height, time, time_delta, frame_index, images, dst, dst_stride);
}

}} // namespace caspar::isf
