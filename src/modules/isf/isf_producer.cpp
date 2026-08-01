/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf_producer.h"
#include "isf_gl_context.h"
#include "isf_shader.h"

#include <accelerator/ogl/image/image_mixer.h>
#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/array.h>
#include <common/env.h>
#include <common/log.h>
#include <common/utf.h>

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/frame_visitor.h>
#include <core/frame/pixel_format.h>
#include <core/producer/frame_producer_registry.h>

#include <boost/algorithm/string.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace caspar { namespace isf {

namespace {

std::wstring fmt_num(double v)
{
    std::wostringstream os;
    os << v;
    return os.str();
}

/// Extracts the first (top-most) const_frame carrying image data from a draw_frame.
class extract_visitor : public core::frame_visitor
{
  public:
    core::const_frame frame;

    void push(const core::frame_transform&) override {}
    void pop() override {}
    void visit(const core::const_frame& f) override
    {
        if (!frame && f)
            frame = f;
    }
};

/// Convert an 8-bit single-plane BGRA/RGBA top-down source frame into a tightly-packed RGBA,
/// bottom-up buffer (the convention the ISF shader samples inputImage in; matches the generator
/// Y-flip so filter output lands top-down in the mixer).
void to_rgba_bottom_up(const std::uint8_t* src, int src_stride, bool src_is_rgba, std::uint8_t* dst, int w, int h)
{
    const int dst_stride = w * 4;
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* s = src + static_cast<std::size_t>(y) * src_stride;
        std::uint8_t*       d = dst + static_cast<std::size_t>(h - 1 - y) * dst_stride;
        if (src_is_rgba) {
            std::memcpy(d, s, static_cast<std::size_t>(dst_stride));
        } else {
            for (int x = 0; x < w; ++x) {
                d[x * 4 + 0] = s[x * 4 + 2]; // R <- B
                d[x * 4 + 1] = s[x * 4 + 1]; // G
                d[x * 4 + 2] = s[x * 4 + 0]; // B <- R
                d[x * 4 + 3] = s[x * 4 + 3]; // A
            }
        }
    }
}

/// Resolve an ISF shader path: as given, then relative to the media folder, trying common
/// ISF/GLSL extensions. Returns the file contents, or empty on failure. On success resolved_name is
/// the file name and resolved_path is the full path that was opened.
std::string load_shader(const std::wstring& token, std::wstring& resolved_name, std::wstring& resolved_path)
{
    namespace fs = std::filesystem;
    std::vector<std::wstring> candidates;
    candidates.push_back(token);
    candidates.push_back(env::media_folder() + token);
    for (const auto* ext : {L".fs", L".frag", L".glsl", L".isf", L".isf.fs"}) {
        candidates.push_back(token + ext);
        candidates.push_back(env::media_folder() + token + ext);
    }
    for (const auto& c : candidates) {
        std::ifstream f(u8(c), std::ios::binary);
        if (f.good()) {
            std::ostringstream ss;
            ss << f.rdbuf();
            resolved_name = fs::path(u8(c)).filename().wstring();
            resolved_path = fs::path(u8(c)).wstring();
            return ss.str();
        }
    }
    return {};
}

/// Read a sibling ISF vertex shader (.vs) next to the resolved shader path, if present.
std::string load_vertex(const std::wstring& shader_path)
{
    namespace fs = std::filesystem;
    if (shader_path.empty())
        return {};
    fs::path vs = fs::path(shader_path);
    vs.replace_extension(L".vs");
    std::ifstream f(vs, std::ios::binary);
    if (!f.good())
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

class isf_producer : public core::frame_producer
{
    spl::shared_ptr<core::frame_factory>      frame_factory_;
    std::shared_ptr<accelerator::ogl::device> ogl_device_;
    std::unique_ptr<shader>                   shader_;
    std::wstring                              name_;
    int                                       width_;
    int                                       height_;
    double                                    fps_ = 25.0;
    std::uint32_t                             frame_ = 0;
    std::mutex                                mutex_;

    // Filter mode: wraps a source producer whose frame is fed to the shader's inputImage.
    std::shared_ptr<core::frame_producer> source_;
    std::vector<std::uint8_t>             src_rgba_; ///< reusable RGBA (bottom-up) upload buffer

    // Transition mode: blends two sources (startImage / endImage) over transition_frames_.
    std::shared_ptr<core::frame_producer> from_;
    std::shared_ptr<core::frame_producer> to_;
    int                                   transition_frames_ = 25;
    std::vector<std::uint8_t>             from_rgba_;
    std::vector<std::uint8_t>             to_rgba_;

    // Non-OpenGL mixer (e.g. Vulkan): self-contained GL context + CPU readback.
    std::unique_ptr<isf::gl_context> gl_ctx_;
    bool                             gl_ctx_failed_ = false;
    std::vector<unsigned char>       readback_;

  public:
    isf_producer(const core::frame_producer_dependencies& deps,
                 const std::string&                        source,
                 std::wstring                              name,
                 const std::wstring&                       base_path,
                 const std::string&                        vertex_source,
                 std::shared_ptr<core::frame_producer>     src = nullptr)
        : frame_factory_(deps.frame_factory)
        , shader_(std::make_unique<shader>(source, base_path, vertex_source))
        , name_(std::move(name))
        , width_(deps.format_desc.width)
        , height_(deps.format_desc.height)
        , source_(std::move(src))
    {
        fps_ = deps.format_desc.duration != 0
                   ? static_cast<double>(deps.format_desc.time_scale) / static_cast<double>(deps.format_desc.duration)
                   : 25.0;
        if (auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get()))
            ogl_device_ = ogl_mixer->get_ogl_device();
    }

    /// Transition constructor: blends from_ -> to_ over `frames`.
    isf_producer(const core::frame_producer_dependencies& deps,
                 const std::string&                        source,
                 std::wstring                              name,
                 const std::wstring&                       base_path,
                 const std::string&                        vertex_source,
                 std::shared_ptr<core::frame_producer>     from,
                 std::shared_ptr<core::frame_producer>     to,
                 int                                       frames)
        : frame_factory_(deps.frame_factory)
        , shader_(std::make_unique<shader>(source, base_path, vertex_source))
        , name_(std::move(name))
        , width_(deps.format_desc.width)
        , height_(deps.format_desc.height)
        , from_(std::move(from))
        , to_(std::move(to))
        , transition_frames_(frames > 0 ? frames : 1)
    {
        fps_ = deps.format_desc.duration != 0
                   ? static_cast<double>(deps.format_desc.time_scale) / static_cast<double>(deps.format_desc.duration)
                   : 25.0;
        if (auto* ogl_mixer = dynamic_cast<accelerator::ogl::image_mixer*>(frame_factory_.get()))
            ogl_device_ = ogl_mixer->get_ogl_device();
    }

    /// Build an image binding from a source frame: sample a texture-backed frame directly, else
    /// convert+stage a CPU frame into `buf`. Returns false if the frame format is unsupported.
    bool build_binding(const core::const_frame&  cf,
                       const char*               input_name,
                       std::vector<std::uint8_t>& buf,
                       isf::image_binding&       out)
    {
        const int w = static_cast<int>(cf.width());
        const int h = static_cast<int>(cf.height());
        out.name    = input_name;
        out.width   = w;
        out.height  = h;
        if (auto t = cf.texture()) {
            if (auto ogl = std::dynamic_pointer_cast<accelerator::ogl::texture>(t)) {
                out.tex_id = ogl->id();
                out.flip   = true;
                out.bgra   = false;
                return true;
            }
            return false; // texture-backed but not an OGL texture (e.g. Vulkan) - can't sample here
        }
        const auto& pfd = cf.pixel_format_desc();
        const bool  ok  = (pfd.format == core::pixel_format::bgra || pfd.format == core::pixel_format::rgba) &&
                         pfd.planes.size() == 1 && pfd.planes[0].depth == common::bit_depth::bit8;
        if (!ok)
            return false;
        buf.resize(static_cast<std::size_t>(w) * h * 4);
        to_rgba_bottom_up(
            cf.image_data(0).data(), pfd.planes[0].linesize, pfd.format == core::pixel_format::rgba, buf.data(), w, h);
        out.rgba = buf.data();
        return true;
    }

    /// Wrap a rendered texture in a BGRA-labelled, texture-backed frame (zero-copy; the mixer
    /// consumes the GPU texture directly). Optional audio is carried through from the source.
    core::draw_frame wrap_texture(std::shared_ptr<core::texture> tex, int w, int h, const core::const_frame* src)
    {
        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(w, h, 4, common::bit_depth::bit8));

        auto                                   store = std::make_shared<std::vector<std::uint8_t>>(0);
        array<const std::uint8_t>              dummy(store->data(), 0, std::move(store));
        std::vector<array<const std::uint8_t>> img;
        img.push_back(std::move(dummy));

        std::shared_ptr<std::vector<std::int32_t>> astore;
        if (src) {
            const auto& sa = src->audio_data();
            astore         = std::make_shared<std::vector<std::int32_t>>(sa.data(), sa.data() + sa.size());
        } else {
            astore = std::make_shared<std::vector<std::int32_t>>(0);
        }
        const std::size_t         asize = astore->size();
        const std::int32_t* const aptr  = astore->data();
        array<const std::int32_t> audio(aptr, asize, std::move(astore));

        return core::draw_frame(core::const_frame(this, std::move(img), std::move(audio), pfd, std::move(tex)));
    }
    /// Render `images` through the shader and deliver a frame: zero-copy texture on the OpenGL mixer,
    /// or a self-contained GL context + CPU readback on any other mixer. Assumes mutex_ is held.
    core::draw_frame produce(int w, int h, int frame_index, const std::vector<isf::image_binding>& images,
                             const core::const_frame* audio_src)
    {
        const double time_delta = fps_ > 0.0 ? 1.0 / fps_ : 0.0;
        const double time       = static_cast<double>(frame_index) / fps_;

        if (ogl_device_) {
            auto tex = shader_->render(ogl_device_, w, h, time, time_delta, frame_index, images);
            shader_->reset_events();
            if (!tex)
                return core::draw_frame::empty();
            return wrap_texture(std::move(tex), w, h, audio_src);
        }

        // Non-OpenGL mixer (e.g. Vulkan): render on a self-contained GL context and read back to CPU.
        if (!gl_ctx_ && !gl_ctx_failed_) {
            try {
                gl_ctx_ = std::make_unique<isf::gl_context>();
            } catch (...) {
                gl_ctx_failed_ = true;
                CASPAR_LOG(warning) << L"[isf] '" << name_ << L"' could not create a GL context.";
            }
        }
        if (!gl_ctx_)
            return core::draw_frame::empty();
        const bool rendered =
            shader_->render_readback(*gl_ctx_, w, h, time, time_delta, frame_index, images, readback_);
        // Hand the context back before returning, on every path. It is created
        // and used here on the channel thread but destroyed on the producer
        // destroyer pool, and an SFML context left active on one thread cannot
        // be destroyed from another. See gl_context::release.
        gl_ctx_->release();
        if (!rendered)
            return core::draw_frame::empty();
        shader_->reset_events();

        core::pixel_format_desc pfd(core::pixel_format::bgra);
        pfd.planes.push_back(core::pixel_format_desc::plane(w, h, 4, common::bit_depth::bit8));
        auto      frame  = frame_factory_->create_frame(this, pfd);
        const int stride = frame.pixel_format_desc().planes[0].linesize;
        for (int y = 0; y < h; ++y)
            std::memcpy(frame.image_data(0).data() + static_cast<std::size_t>(y) * stride,
                        readback_.data() + static_cast<std::size_t>(y) * w * 4,
                        static_cast<std::size_t>(w) * 4);
        if (audio_src) {
            const auto& a = audio_src->audio_data();
            if (a.size() > 0)
                frame.audio_data() = std::vector<std::int32_t>(a.data(), a.data() + a.size());
        }
        return core::draw_frame(std::move(frame));
    }

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        // Transition mode: blend two sources (startImage / endImage) by progress.
        if (from_ && to_) {
            auto from_frame = from_->receive(field, nb_samples);
            auto to_frame   = to_->receive(field, nb_samples);
            extract_visitor vf, vt;
            if (from_frame)
                from_frame.accept(vf);
            if (to_frame)
                to_frame.accept(vt);
            const auto cf_from = vf.frame;
            const auto cf_to   = vt.frame;
            if (!cf_from || !cf_to)
                return from_frame ? from_frame : to_frame;

            core::draw_frame out;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                isf::image_binding start_bind, end_bind;
                if (!build_binding(cf_from, "startImage", from_rgba_, start_bind) ||
                    !build_binding(cf_to, "endImage", to_rgba_, end_bind))
                    return from_frame;

                double progress = static_cast<double>(frame_) / static_cast<double>(transition_frames_);
                progress        = progress < 0.0 ? 0.0 : (progress > 1.0 ? 1.0 : progress);
                shader_->set_value("progress", {progress});
                out = produce(start_bind.width, start_bind.height, static_cast<int>(frame_),
                              {start_bind, end_bind}, &cf_from);
            }
            if (!out)
                return from_frame;
            ++frame_;
            return out;
        }

        // Filter mode: pull a frame from the source and feed it to the shader's inputImage.
        if (source_) {
            auto source_frame = source_->receive(field, nb_samples);
            if (!source_frame)
                return source_frame;

            extract_visitor v;
            source_frame.accept(v);
            const auto cf = v.frame;
            if (!cf)
                return source_frame;

            core::draw_frame out;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                isf::image_binding in;
                if (!build_binding(cf, "inputImage", src_rgba_, in)) {
                    static bool warned_fmt = false;
                    if (!warned_fmt) {
                        warned_fmt = true;
                        CASPAR_LOG(warning) << L"[isf] '" << name_
                                            << L"' source is not single-plane 8-bit BGRA/RGBA; passing it through.";
                    }
                    return source_frame;
                }
                out = produce(in.width, in.height, static_cast<int>(source_->frame_number()), {in}, &cf);
            }
            if (!out)
                return source_frame; // render failed -> pass the source through
            ++frame_;
            return out;
        }

        // Generator mode: render straight from the shader (no source).
        core::draw_frame out;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            out = produce(width_, height_, static_cast<int>(frame_), {}, nullptr);
        }
        if (!out)
            return core::draw_frame::empty();
        ++frame_;
        return out;
    }

    core::monitor::state state() const override { return {}; }
    std::wstring         print() const override { return L"isf[" + name_ + L"]"; }
    std::wstring         name() const override { return L"isf"; }
    uint32_t             frame_number() const override { return frame_; }
    uint32_t             nb_frames() const override
    {
        if (from_ && to_)
            return static_cast<uint32_t>(transition_frames_);
        return source_ ? source_->nb_frames() : std::numeric_limits<uint32_t>::max();
    }
    bool is_ready() override
    {
        if (from_ && to_)
            return from_->is_ready() && to_->is_ready();
        return source_ ? source_->is_ready() : true;
    }

    std::future<std::wstring> call(const std::vector<std::wstring>& params) override
    {
        std::wstring result;
        if (!params.empty() && boost::iequals(params.at(0), L"ISF")) {
            const std::wstring sub = params.size() > 1 ? params.at(1) : L"";
            if (boost::iequals(sub, L"LIST")) {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const auto& in : shader_->inputs()) {
                    result += u16(in.name) + L" " + u16(in.type) + L" \"" + u16(in.label) + L"\"";
                    if (!in.min_value.empty())
                        result += L" min=" + fmt_num(in.min_value[0]);
                    if (!in.max_value.empty())
                        result += L" max=" + fmt_num(in.max_value[0]);
                    if (!in.default_value.empty()) {
                        result += L" def=";
                        for (std::size_t i = 0; i < in.default_value.size(); ++i)
                            result += (i ? L"," : L"") + fmt_num(in.default_value[i]);
                    }
                    if (!in.values.empty()) {
                        result += L" values=";
                        for (std::size_t i = 0; i < in.values.size(); ++i)
                            result += (i ? L"," : L"") + std::to_wstring(in.values[i]);
                    }
                    result += L"\r\n";
                }
            } else if (boost::iequals(sub, L"SET") && params.size() >= 3) {
                std::vector<double> values;
                for (std::size_t i = 3; i < params.size(); ++i) {
                    try {
                        values.push_back(std::stod(params.at(i)));
                    } catch (...) {
                        values.push_back(0.0);
                    }
                }
                std::lock_guard<std::mutex> lock(mutex_);
                if (!shader_->set_value(u8(params.at(2)), values))
                    result = L"402 CALL ERROR (unknown ISF input)\r\n";
            }
        }
        std::promise<std::wstring> pr;
        pr.set_value(result);
        return pr.get_future();
    }
};

} // namespace

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies,
                                                      const std::vector<std::wstring>&         params)
{
    if (params.empty() || !boost::iequals(params.at(0), L"[ISF]"))
        return core::frame_producer::empty();

    if (params.size() < 2) {
        CASPAR_LOG(warning) << L"[isf] Usage: [ISF] <shader-file> [<source-producer...>]";
        return core::frame_producer::empty();
    }

    std::wstring name;
    std::wstring path;
    const auto   source = load_shader(params.at(1), name, path);
    if (source.empty()) {
        CASPAR_LOG(warning) << L"[isf] Could not load shader '" << params.at(1) << L"'.";
        return core::frame_producer::empty();
    }
    const std::wstring base_path     = std::filesystem::path(path).parent_path().wstring();
    const std::string  vertex_source = load_vertex(path);

    std::vector<std::wstring> source_params(params.begin() + 2, params.end());

    // Transition mode: [ISF] <shader> TRANSITION <from-source> <to-source> [frames]
    if (!source_params.empty() && boost::iequals(source_params.at(0), L"TRANSITION")) {
        if (source_params.size() < 3) {
            CASPAR_LOG(warning) << L"[isf] Usage: [ISF] <shader> TRANSITION <from-source> <to-source> [frames]";
            return core::frame_producer::empty();
        }
        auto from = dependencies.producer_registry->create_producer(dependencies, {source_params.at(1)});
        auto to   = dependencies.producer_registry->create_producer(dependencies, {source_params.at(2)});
        if (from == core::frame_producer::empty() || to == core::frame_producer::empty()) {
            CASPAR_LOG(warning) << L"[isf] Could not create both transition sources for '" << params.at(1) << L"'.";
            return core::frame_producer::empty();
        }
        int frames = 25;
        if (source_params.size() >= 4) {
            try {
                frames = std::stoi(source_params.at(3));
            } catch (...) {
                frames = 25;
            }
        }
        return spl::make_shared<isf_producer>(dependencies, source, name, base_path, vertex_source, from, to, frames);
    }

    // Filter mode: a source producer follows the shader file.
    if (!source_params.empty()) {
        auto src = dependencies.producer_registry->create_producer(dependencies, source_params);
        if (src == core::frame_producer::empty()) {
            CASPAR_LOG(warning) << L"[isf] Could not create source producer for shader '" << params.at(1) << L"'.";
            return core::frame_producer::empty();
        }
        return spl::make_shared<isf_producer>(dependencies, source, name, base_path, vertex_source, src);
    }

    return spl::make_shared<isf_producer>(dependencies, source, name, base_path, vertex_source);
}

}} // namespace caspar::isf
