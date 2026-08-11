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

#include "ocio_config.h"

#include <common/log.h>
#include <common/utf.h>

#ifdef CASPAR_ENABLE_OCIO
#include <OpenColorIO/OpenColorIO.h>
#include <algorithm>
#include <mutex>
namespace OCIO_NS = OCIO_NAMESPACE;
#endif

namespace caspar { namespace accelerator { namespace ocio {

#ifdef CASPAR_ENABLE_OCIO

namespace {

/// The config pinned by default.
///
/// A built-in URI rather than a file on disk: nothing to install, nothing to get out of
/// step between machines, and the version is part of the identifier. Pinned explicitly
/// rather than using ocio://default, which follows "the latest ACES 2 CG config" and would
/// therefore change what a colour space name means when the library is upgraded. A server
/// whose looks have been approved must not have that happen underneath it.
constexpr const char* DEFAULT_CONFIG_URI = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5";

std::mutex                     g_mutex;
OCIO_NS::ConstConfigRcPtr      g_config;
std::string                    g_uri;

/// Load on first use rather than at startup: a server with no OCIO channel should not pay
/// to parse a config, and this is only reached by a MIXER OCIO command or an INFO query.
/// Caller must hold g_mutex.
void ensure_loaded_locked()
{
    if (g_config)
        return;

    try {
        g_config = OCIO_NS::Config::CreateFromBuiltinConfig(DEFAULT_CONFIG_URI);
        g_uri    = DEFAULT_CONFIG_URI;
        CASPAR_LOG(info) << L"[ocio] loaded " << u16(g_uri) << L" (OpenColorIO "
                         << u16(OCIO_NS::GetVersion()) << L")";
    } catch (const OCIO_NS::Exception& e) {
        // Leaves g_config null. Every accessor below then answers empty, and a MIXER OCIO
        // command validating against it fails -- which is the correct outcome: refuse the
        // command rather than render with a colour transform nobody chose.
        CASPAR_LOG(error) << L"[ocio] could not load the built-in config " << u16(DEFAULT_CONFIG_URI) << L": "
                          << u16(e.what());
    }
}

} // namespace

bool available() { return true; }

std::string version() { return OCIO_NS::GetVersion(); }

std::string config_uri()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    return g_uri;
}

bool load_config(const std::string& uri)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        // A built-in URI and a filesystem path are different constructors in OCIO, and
        // guessing wrong gives a confusing error, so dispatch on the scheme.
        auto cfg = uri.rfind("ocio://", 0) == 0 ? OCIO_NS::Config::CreateFromBuiltinConfig(uri.c_str())
                                                : OCIO_NS::Config::CreateFromFile(uri.c_str());
        g_config = cfg;
        g_uri    = uri;
        CASPAR_LOG(info) << L"[ocio] loaded " << u16(uri);
        return true;
    } catch (const OCIO_NS::Exception& e) {
        CASPAR_LOG(error) << L"[ocio] could not load " << u16(uri) << L": " << u16(e.what())
                          << L" -- keeping " << (g_uri.empty() ? std::wstring(L"no config") : u16(g_uri));
        return false;
    }
}

std::vector<std::string> colorspaces()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    std::vector<std::string> out;
    if (!g_config)
        return out;

    const int n = g_config->getNumColorSpaces();
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        out.emplace_back(g_config->getColorSpaceNameByIndex(i));
    return out;
}

std::vector<std::string> displays()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    std::vector<std::string> out;
    if (!g_config)
        return out;

    const int n = g_config->getNumDisplays();
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        out.emplace_back(g_config->getDisplay(i));
    return out;
}

std::vector<std::string> views(const std::string& display)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    std::vector<std::string> out;
    if (!g_config)
        return out;

    const int n = g_config->getNumViews(display.c_str());
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        out.emplace_back(g_config->getView(display.c_str(), i));
    return out;
}

bool has_colorspace(const std::string& name)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config || name.empty())
        return false;
    // getColorSpace resolves roles and aliases too, which is what an operator typing a
    // name expects -- "scene_linear" should work as well as its target space.
    return static_cast<bool>(g_config->getColorSpace(name.c_str()));
}

bool has_display_view(const std::string& display, const std::string& view)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config || display.empty() || view.empty())
        return false;

    const int n = g_config->getNumViews(display.c_str());
    for (int i = 0; i < n; ++i) {
        if (view == g_config->getView(display.c_str(), i))
            return true;
    }
    return false;
}

bool build_input_transform(const std::string& source_space, gpu_shader& out)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config) {
        CASPAR_LOG(error) << L"[ocio] cannot build a transform: no config is loaded";
        return false;
    }

    try {
        // The mixer grades in ACEScg. Named by role where possible so the same code works
        // against a config that spells the space differently; ACES - ACEScg is the studio
        // config's name for it.
        const char* working = g_config->getColorSpace(OCIO_NS::ROLE_SCENE_LINEAR)
                                  ? OCIO_NS::ROLE_SCENE_LINEAR
                                  : "ACES - ACEScg";

        auto processor = g_config->getProcessor(source_space.c_str(), working);
        auto gpu       = processor->getDefaultGPUProcessor();

        auto desc = OCIO_NS::GpuShaderDesc::CreateShaderDesc();
        // GLSL 4.0 matches the OGL mixer's own shader. The Vulkan mixer will ask for
        // GPU_LANGUAGE_GLSL_VK_4_6 instead, which differs in how samplers and uniforms are
        // declared, so the two cannot share a generated program.
        desc->setLanguage(OCIO_NS::GPU_LANGUAGE_GLSL_4_0);
        desc->setFunctionName("caspar_ocio_input");
        // Prefixing every generated identifier keeps a 55-space config from colliding with
        // the ~200 uniforms the mixer's own shader already declares.
        desc->setResourcePrefix("ocio_in_");

        gpu->extractGpuShaderInfo(desc);

        out = gpu_shader{};
        out.cache_id      = desc->getCacheID();
        out.source        = desc->getShaderText();
        out.function_name = desc->getFunctionName();

        const unsigned num_2d = desc->getNumTextures();
        for (unsigned i = 0; i < num_2d; ++i) {
            const char*                        tex_name    = nullptr;
            const char*                        samp_name   = nullptr;
            unsigned                           w           = 0;
            unsigned                           h           = 0;
            OCIO_NS::GpuShaderDesc::TextureType channel     = OCIO_NS::GpuShaderDesc::TEXTURE_RGB_CHANNEL;
            OCIO_NS::GpuShaderCreator::TextureDimensions dims = OCIO_NS::GpuShaderCreator::TEXTURE_2D;
            OCIO_NS::Interpolation             interp      = OCIO_NS::INTERP_LINEAR;
            desc->getTexture(i, tex_name, samp_name, w, h, channel, dims, interp);

            const float* values = nullptr;
            desc->getTextureValues(i, values);

            gpu_texture t;
            t.sampler_name       = samp_name ? samp_name : "";
            t.dimensions         = dims == OCIO_NS::GpuShaderCreator::TEXTURE_1D ? 1 : 2;
            t.width              = static_cast<int>(w);
            t.height             = static_cast<int>(h);
            t.channels           = channel == OCIO_NS::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1 : 3;
            t.interpolate_linear = interp != OCIO_NS::INTERP_NEAREST;
            if (values) {
                const size_t n = static_cast<size_t>(w) * std::max(1u, h) * t.channels;
                t.values.assign(values, values + n);
            }
            out.textures.push_back(std::move(t));
        }

        const unsigned num_3d = desc->getNum3DTextures();
        for (unsigned i = 0; i < num_3d; ++i) {
            const char*            tex_name  = nullptr;
            const char*            samp_name = nullptr;
            unsigned               edge      = 0;
            OCIO_NS::Interpolation interp    = OCIO_NS::INTERP_LINEAR;
            desc->get3DTexture(i, tex_name, samp_name, edge, interp);

            const float* values = nullptr;
            desc->get3DTextureValues(i, values);

            gpu_texture t;
            t.sampler_name       = samp_name ? samp_name : "";
            t.dimensions         = 3;
            t.edge_len           = static_cast<int>(edge);
            t.channels           = 3;
            t.interpolate_linear = interp != OCIO_NS::INTERP_NEAREST;
            if (values) {
                const size_t n = static_cast<size_t>(edge) * edge * edge * 3;
                t.values.assign(values, values + n);
            }
            out.textures.push_back(std::move(t));
        }

        CASPAR_LOG(info) << L"[ocio] built " << u16(source_space) << L" -> " << u16(working) << L": cache_id "
                         << u16(out.cache_id) << L", " << out.source.size() << L" chars of GLSL, " << num_2d
                         << L" 1D/2D + " << num_3d << L" 3D textures";

        // The generated source at debug level, not info: it is thousands of characters and
        // would bury the log on every colour change. But when a transform looks wrong, this
        // is the only place the actual arithmetic is visible -- the C++ that assembled it
        // says nothing about what OCIO decided to emit.
        CASPAR_LOG(debug) << L"[ocio] generated GLSL for " << u16(source_space) << L":\n" << u16(out.source);
        for (const auto& t : out.textures) {
            CASPAR_LOG(debug) << L"[ocio]   texture " << u16(t.sampler_name) << L" " << t.dimensions << L"D "
                              << (t.dimensions == 3 ? t.edge_len : t.width) << L"x"
                              << (t.dimensions == 3 ? t.edge_len : t.height) << L" ch=" << t.channels
                              << L" values=" << t.values.size();
        }

        return true;
    } catch (const OCIO_NS::Exception& e) {
        CASPAR_LOG(error) << L"[ocio] could not build a transform from " << u16(source_space) << L": "
                          << u16(e.what());
        return false;
    }
}

#else // CASPAR_ENABLE_OCIO

// Built without OCIO. Every query answers "nothing", so callers need no #ifdef of their
// own: an AMCP discovery command returns an empty list and a MIXER OCIO command fails
// validation, both of which are honest.

bool                     available() { return false; }
std::string              version() { return {}; }
std::string              config_uri() { return {}; }
bool                     load_config(const std::string&) { return false; }
std::vector<std::string> colorspaces() { return {}; }
std::vector<std::string> displays() { return {}; }
std::vector<std::string> views(const std::string&) { return {}; }
bool                     has_colorspace(const std::string&) { return false; }
bool                     has_display_view(const std::string&, const std::string&) { return false; }
bool                     build_input_transform(const std::string&, gpu_shader&) { return false; }

#endif // CASPAR_ENABLE_OCIO

}}} // namespace caspar::accelerator::ocio
