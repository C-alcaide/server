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
#include <set>
#include <sstream>
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

/// Cache IDs whose build has already been described in the log.
///
/// build_input_transform() is called per layer per frame -- both kernels ask for the
/// transform first and consult their own caches afterwards, because the cache ID is what
/// they key on and only OCIO can produce it. Logging the build unconditionally therefore
/// wrote a line per frame per layer, which at 25 fps buries the log within seconds. The
/// line is worth keeping at info: it is the only place the shape of what OCIO generated is
/// visible. It is worth keeping *once*.
std::set<std::string> g_logged_builds;

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
        // A new config means new transforms worth describing, including a reload of the same
        // URI after the file changed underneath it.
        g_logged_builds.clear();
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

bool load_cdl(const std::string& path,
              const std::string& cccid,
              double             slope[3],
              double             offset[3],
              double             power[3],
              double&            saturation)
{
    // No g_mutex and no ensure_loaded_locked(): a CDL file is self-contained ASC XML and
    // owes nothing to the loaded config. Taking the lock would serialise file reads behind
    // config access for no reason, and loading a config to read a .cdl would be surprising
    // on a server that never uses OCIO otherwise.
    try {
        // AMBIGUITY IS REFUSED, and this is not what OCIO does on its own.
        //
        // `CreateFromFile(path, "")` on a collection holding several corrections silently
        // returns the FIRST one. An operator pointing at a 200-shot `.ccc` without an id
        // would get shot 1's grade, correctly applied, with nothing anywhere to say so --
        // the plausible-but-wrong failure this tree keeps paying for. So count first, and
        // make the operator choose.
        auto group = OCIO_NS::CDLTransform::CreateGroupFromFile(path.c_str());
        if (group && group->getNumTransforms() > 1 && cccid.empty()) {
            std::wstring ids;
            for (int i = 0; i < group->getNumTransforms(); ++i) {
                auto cdl = OCIO_NS::DynamicPtrCast<const OCIO_NS::CDLTransform>(group->getTransform(i));
                const char* id = cdl ? cdl->getID() : "";
                ids += (ids.empty() ? L"" : L", ") + std::wstring(L"'") + u16(id ? id : "") + L"'";
            }
            CASPAR_LOG(error) << L"[ocio] " << u16(path) << L" holds " << group->getNumTransforms()
                              << L" corrections and no id was given, so which one to apply is "
                              << L"undefined. Pass one of: " << ids;
            return false;
        }

        auto t = OCIO_NS::CDLTransform::CreateFromFile(path.c_str(), cccid.c_str());
        if (!t) {
            CASPAR_LOG(error) << L"[ocio] no CDL in " << u16(path)
                              << (cccid.empty() ? std::wstring() : L" with id '" + u16(cccid) + L"'");
            return false;
        }

        double s[3]{}, o[3]{}, p[3]{};
        t->getSlope(s);
        t->getOffset(o);
        t->getPower(p);
        const double sat = t->getSat();

        // Written only once everything has been read, so a throw part-way leaves the
        // caller's grade untouched rather than half-applied.
        for (int i = 0; i < 3; ++i) {
            slope[i]  = s[i];
            offset[i] = o[i];
            power[i]  = p[i];
        }
        saturation = sat;

        CASPAR_LOG(info) << L"[ocio] read CDL " << u16(path)
                         << (cccid.empty() ? std::wstring() : L" id '" + u16(cccid) + L"'")
                         << L": slope " << s[0] << L"," << s[1] << L"," << s[2]
                         << L" offset " << o[0] << L"," << o[1] << L"," << o[2]
                         << L" power " << p[0] << L"," << p[1] << L"," << p[2]
                         << L" sat " << sat;
        return true;
    } catch (const OCIO_NS::Exception& e) {
        // Covers the whole family: unreadable file, malformed XML, an id that is not in the
        // collection, and a collection with several corrections and no id to choose between
        // them -- OCIO refuses that rather than picking one, which is the behaviour we want.
        CASPAR_LOG(error) << L"[ocio] could not read CDL " << u16(path)
                          << (cccid.empty() ? std::wstring() : L" id '" + u16(cccid) + L"'")
                          << L": " << u16(e.what());
        return false;
    }
}

namespace {

/// Split an `amf_transform_ids` attribute, which OCIO stores as whitespace-separated URNs.
bool has_amf_id(const char* attr, const std::string& id)
{
    if (!attr)
        return false;
    std::istringstream in{std::string(attr)};
    std::string        token;
    while (in >> token) {
        if (token == id)
            return true;
    }
    return false;
}

} // namespace

bool resolve_amf_transform_id(const std::string& transform_id, amf_resolution& out)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config || transform_id.empty())
        return false;

    out = amf_resolution{};

    // EVERY kind is searched and none short-circuits. An output transform ID names a colour
    // space AND a view transform, and taking the first hit is how an early version of this
    // reported 2 view transforms where the config has 11 -- it hid the very pairing the
    // display/view mapping depends on.
    try {
        for (int i = 0; i < g_config->getNumColorSpaces(); ++i) {
            const char* name = g_config->getColorSpaceNameByIndex(i);
            auto        cs   = g_config->getColorSpace(name);
            if (cs && has_amf_id(cs->getInterchangeAttribute("amf_transform_ids"), transform_id))
                out.colorspace = name;
        }
        for (int i = 0; i < g_config->getNumLooks(); ++i) {
            const char* name = g_config->getLookNameByIndex(i);
            auto        lk   = g_config->getLook(name);
            if (lk && has_amf_id(lk->getInterchangeAttribute("amf_transform_ids"), transform_id))
                out.look = name;
        }
        for (int i = 0; i < g_config->getNumViewTransforms(); ++i) {
            const char* name = g_config->getViewTransformNameByIndex(i);
            auto        vt   = g_config->getViewTransform(name);
            if (vt && has_amf_id(vt->getInterchangeAttribute("amf_transform_ids"), transform_id))
                out.view_transform = name;
        }
    } catch (const OCIO_NS::Exception& e) {
        CASPAR_LOG(error) << L"[ocio] could not resolve AMF id " << u16(transform_id) << L": "
                          << u16(e.what());
        return false;
    }

    if (out.colorspace.empty() && out.look.empty() && out.view_transform.empty()) {
        CASPAR_LOG(error) << L"[ocio] AMF transform id " << u16(transform_id)
                          << L" is not in " << u16(g_uri)
                          << L". The config carries these ids under `interchange: "
                          << L"amf_transform_ids`, so a file from a different ACES version "
                          << L"will not resolve against it.";
        return false;
    }
    return true;
}

std::vector<std::string> looks()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    std::vector<std::string> out;
    if (!g_config)
        return out;

    const int n = g_config->getNumLooks();
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        out.emplace_back(g_config->getLookNameByIndex(i));
    return out;
}

bool has_look(const std::string& name)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config || name.empty())
        return false;
    // getLook() returns null for an unknown name rather than throwing, which is what makes
    // this usable as an AMCP argument check.
    return static_cast<bool>(g_config->getLook(name.c_str()));
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

namespace {

/// The mixer grades in ACEScg. Named by role where possible so the same code works against a
/// config that spells the space differently; ACES - ACEScg is the studio config's name.
/// Caller must hold g_mutex and have a config loaded.
const char* working_space_locked()
{
    return g_config->getColorSpace(OCIO_NS::ROLE_SCENE_LINEAR) ? OCIO_NS::ROLE_SCENE_LINEAR
                                                               : "ACES - ACEScg";
}

/// Generate source and collect resources from an already-built processor.
///
/// Shared by the input and display transforms because everything after "which processor"
/// is identical -- the language selection, the descriptor set index, the texture walk and
/// the once-per-cache-ID logging. They differed only in the two names, and keeping two
/// copies of the texture walk is how one of them would quietly stop reading the binding
/// index from OCIO.
bool fill_from_processor(const OCIO_NS::ConstProcessorRcPtr& processor,
                         const char*                        function_name,
                         const char*                        resource_prefix,
                         const std::wstring&                what,
                         gpu_target                         target,
                         unsigned                           texture_binding_start,
                         gpu_shader&                        out)
{
    auto gpu = processor->getDefaultGPUProcessor();

    auto desc = OCIO_NS::GpuShaderDesc::CreateShaderDesc();
    // GLSL 4.0 matches the OGL mixer's own shader; GLSL_VK_4_6 matches the Vulkan one.
    // They differ in how samplers and uniforms are declared, so the two cannot share a
    // generated program -- and do not, because the language leads OCIO's cache ID.
    if (target == gpu_target::vulkan) {
        desc->setLanguage(OCIO_NS::GPU_LANGUAGE_GLSL_VK_4_6);
        // Set 1 for the generated transform's own resources. Binding 0 is OCIO's uniform
        // buffer by its own contract, and textures start where the caller says -- because a
        // layer can carry an input transform AND a display transform, both spliced into the
        // same shader, and both would otherwise declare their first sampler at binding 1.
        desc->setDescriptorSetIndex(1, texture_binding_start);
    } else {
        desc->setLanguage(OCIO_NS::GPU_LANGUAGE_GLSL_4_0);
    }
    desc->setFunctionName(function_name);
    // Prefixing every generated identifier keeps a 55-space config from colliding with
    // the ~200 uniforms the mixer's own shader already declares -- and keeps an input
    // and a display transform, which are spliced into the SAME shader, from colliding
    // with each other.
    desc->setResourcePrefix(resource_prefix);

    gpu->extractGpuShaderInfo(desc);

    out = gpu_shader{};
    out.cache_id            = desc->getCacheID();
    out.source              = desc->getShaderText();
    out.function_name       = desc->getFunctionName();
    out.uniform_buffer_size = desc->getUniformBufferSize();

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
        // Asked for rather than computed as textureBindingStart + i: OCIO writes the
        // binding into the source, and the two only coincide while nothing else claims
        // an index.
        t.binding            = static_cast<int>(desc->getTextureShaderBindingIndex(i));
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
        t.binding            = static_cast<int>(desc->get3DTextureShaderBindingIndex(i));
        if (values) {
            const size_t n = static_cast<size_t>(edge) * edge * edge * 3;
            t.values.assign(values, values + n);
        }
        out.textures.push_back(std::move(t));
    }

    // Once per distinct transform, not once per frame. See g_logged_builds.
    if (g_logged_builds.insert(out.cache_id).second) {
        CASPAR_LOG(info) << L"[ocio] built " << what << L" ("
                         << (target == gpu_target::vulkan ? L"vulkan" : L"opengl") << L"): cache_id "
                         << u16(out.cache_id) << L", " << out.source.size() << L" chars of GLSL, " << num_2d
                         << L" 1D/2D + " << num_3d << L" 3D textures, uniform buffer "
                         << out.uniform_buffer_size << L" bytes";

        // The generated source at debug level, not info: it is thousands of characters and
        // would bury the log on every colour change. But when a transform looks wrong, this
        // is the only place the actual arithmetic is visible -- the C++ that assembled it
        // says nothing about what OCIO decided to emit.
        CASPAR_LOG(debug) << L"[ocio] generated GLSL for " << what << L":\n" << u16(out.source);
        for (const auto& t : out.textures) {
            CASPAR_LOG(debug) << L"[ocio]   texture " << u16(t.sampler_name) << L" " << t.dimensions << L"D "
                              << (t.dimensions == 3 ? t.edge_len : t.width) << L"x"
                              << (t.dimensions == 3 ? t.edge_len : t.height) << L" ch=" << t.channels
                              << L" binding=" << t.binding << L" values=" << t.values.size();
        }
    }

    return true;
}

} // namespace

bool build_input_transform(const std::string& source_space, gpu_shader& out, gpu_target target)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config) {
        CASPAR_LOG(error) << L"[ocio] cannot build a transform: no config is loaded";
        return false;
    }

    try {
        const char* working = working_space_locked();
        return fill_from_processor(g_config->getProcessor(source_space.c_str(), working),
                                   "caspar_ocio_input",
                                   "ocio_in_",
                                   u16(source_space) + L" -> " + u16(working),
                                   target,
                                   INPUT_TEXTURE_BINDING_START,
                                   out);
    } catch (const OCIO_NS::Exception& e) {
        CASPAR_LOG(error) << L"[ocio] could not build a transform from " << u16(source_space) << L": "
                          << u16(e.what());
        return false;
    }
}

bool build_display_transform(const std::string& display,
                             const std::string& view,
                             gpu_shader&        out,
                             gpu_target         target,
                             const std::string& looks)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_loaded_locked();
    if (!g_config) {
        CASPAR_LOG(error) << L"[ocio] cannot build a display transform: no config is loaded";
        return false;
    }

    try {
        const char* working = working_space_locked();

        // No look: the 4-argument convenience, exactly as before this parameter existed, so
        // the generated source for every existing configuration is unchanged.
        //
        // FORWARD is working space -> display. The inverse direction is what a *display* used
        // as a source needs, and that is build_input_transform's job via the colour space of
        // the same name -- not this.
        OCIO_NS::ConstProcessorRcPtr proc;
        if (looks.empty()) {
            proc = g_config->getProcessor(working, display.c_str(), view.c_str(),
                                          OCIO_NS::TRANSFORM_DIR_FORWARD);
        } else {
            // LOOK FIRST, THEN THE VIEW. That is the ACES order -- an LMT acts on the
            // scene-referred image and the view renders the result -- and it is not a
            // preference: reversing it would apply a working-space transform to
            // display-encoded pixels.
            //
            // The look's own `process_space` is honoured by OCIO, which converts into it and
            // back around the look. So a look declared in ACES2065-1 works from an ACEScg
            // working space without anything here knowing that.
            auto look = OCIO_NS::LookTransform::Create();
            look->setSrc(working);
            look->setDst(working);
            look->setLooks(looks.c_str());

            auto dv = OCIO_NS::DisplayViewTransform::Create();
            dv->setSrc(working);
            dv->setDisplay(display.c_str());
            dv->setView(view.c_str());
            // The view may name looks of its own. Those are the config author's intent for
            // that view and stay ON; this composes with them rather than replacing them.

            auto group = OCIO_NS::GroupTransform::Create();
            group->appendTransform(look);
            group->appendTransform(dv);
            proc = g_config->getProcessor(group, OCIO_NS::TRANSFORM_DIR_FORWARD);
        }

        return fill_from_processor(proc,
                                   "caspar_ocio_display",
                                   "ocio_out_",
                                   u16(working) + L" -> " +
                                       (looks.empty() ? std::wstring() : L"[" + u16(looks) + L"] -> ") +
                                       u16(display) + L" / " + u16(view),
                                   target,
                                   DISPLAY_TEXTURE_BINDING_START,
                                   out);
    } catch (const OCIO_NS::Exception& e) {
        CASPAR_LOG(error) << L"[ocio] could not build a display transform for " << u16(display) << L" / "
                          << u16(view) << L": " << u16(e.what());
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
std::vector<std::string> looks() { return {}; }
bool                     resolve_amf_transform_id(const std::string&, amf_resolution&)
{
    return false;
}
bool                     load_cdl(const std::string&, const std::string&, double*, double*, double*, double&)
{
    return false;
}
std::vector<std::string> views(const std::string&) { return {}; }
bool                     has_colorspace(const std::string&) { return false; }
bool                     has_display_view(const std::string&, const std::string&) { return false; }
bool                     has_look(const std::string&) { return false; }
bool                     build_input_transform(const std::string&, gpu_shader&, gpu_target) { return false; }
bool build_display_transform(const std::string&, const std::string&, gpu_shader&, gpu_target) { return false; }

#endif // CASPAR_ENABLE_OCIO

}}} // namespace caspar::accelerator::ocio
