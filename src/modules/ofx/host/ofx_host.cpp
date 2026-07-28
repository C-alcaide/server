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

#include "ofx_host.h"

#include "ofx_effect_instance.h"
#include "ofx_includes.h"

#include <accelerator/ogl/util/device.h>
#include <accelerator/ogl/util/texture.h>

#include <common/log.h>
#include <common/utf.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace caspar { namespace ofx {

namespace {

// ---------------------------------------------------------------------------------------
// Per-plug-in health: in-process stability layer (crash-guard + hung-render watchdog +
// blocklist). A plug-in that crashes (SEH caught under /EHa) or repeatedly overruns the
// render budget accrues "strikes"; past a threshold it is blocklisted (no longer instantiated)
// so one bad plug-in cannot keep destabilising the server.
// ---------------------------------------------------------------------------------------
constexpr int    kMaxStrikes       = 3;
constexpr double kRenderBudgetMs   = 2000.0; // a render slower than this counts as a strike

std::mutex&                          health_mutex()
{
    static std::mutex m;
    return m;
}
std::unordered_map<std::string, int>& strike_counts()
{
    static std::unordered_map<std::string, int> m;
    return m;
}
std::unordered_set<std::string>&      blocklist()
{
    static std::unordered_set<std::string> s;
    return s;
}

bool is_blocklisted(const std::string& id)
{
    std::lock_guard<std::mutex> lock(health_mutex());
    return blocklist().count(id) != 0;
}

void record_strike(const std::string& id, const wchar_t* reason)
{
    std::lock_guard<std::mutex> lock(health_mutex());
    if (blocklist().count(id) != 0)
        return;
    const int n = ++strike_counts()[id];
    CASPAR_LOG(warning) << L"[ofx] '" << u16(id) << L"' strike " << n << L"/" << kMaxStrikes << L" (" << reason << L").";
    if (n >= kMaxStrikes) {
        blocklist().insert(id);
        CASPAR_LOG(error) << L"[ofx] '" << u16(id) << L"' blocklisted after " << n
                          << L" strikes; it will no longer be instantiated.";
    }
}

void add_to_blocklist(const std::string& id)
{
    std::lock_guard<std::mutex> lock(health_mutex());
    blocklist().insert(id);
}

/// OpenGL render backend is opt-in via the CASPARCG_OFX_ENABLE_GL environment variable (or the
/// <ofx><enable-opengl>true</enable-opengl></ofx> config maps to it). Off by default so the
/// verified CPU path is unaffected.
bool g_config_gl   = false;
bool g_config_cuda = false;

bool ofx_gl_enabled()
{
    const char* v = std::getenv("CASPARCG_OFX_ENABLE_GL");
    return g_config_gl || (v != nullptr && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y'));
}

/// CUDA render backend is opt-in via CASPARCG_OFX_ENABLE_CUDA (Windows + CUDA build only).
bool ofx_cuda_enabled()
{
    const char* v = std::getenv("CASPARCG_OFX_ENABLE_CUDA");
    return g_config_cuda ||
           (v != nullptr && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y'));
}

/// Passed as the OFX "client data" to ImageEffectPlugin::createInstance so that the host's
/// newInstance() callback can size the created instance to the current frame.
struct creation_params
{
    int    width             = 0;
    int    height            = 0;
    double frame_rate        = 25.0;
    int    bytes_per_channel = 1;
};

// --- Multi-thread suite state (OFX_SUPPORTS_MULTITHREAD) -------------------------------
// Per-thread identity for a running multiThread() call, so the plug-in can query its worker
// index / whether it is a spawned worker.
thread_local unsigned int t_ofx_thread_index = 0;
thread_local bool         t_ofx_is_spawned   = false;

/// Run one OFX worker function with a crash guard (with /EHa a hardware fault is caught here so a
/// misbehaving plug-in cannot take down the server from a spawned worker thread).
inline bool run_ofx_worker(OfxThreadFunctionV1 func, unsigned int index, unsigned int max, void* arg, bool spawned)
{
    t_ofx_thread_index = index;
    t_ofx_is_spawned   = spawned;
    try {
        func(index, max, arg);
        return true;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] multiThread worker " << index << L" crashed; frame may be incomplete.";
        return false;
    }
}

/// CasparCG's concrete OFX image-effect host.
class caspar_ofx_host : public OFX::Host::ImageEffect::Host
{
  public:
    caspar_ofx_host()
    {
        auto& p = getProperties();

        p.setStringProperty(kOfxPropName, "com.casparcg.ofx.host");
        p.setStringProperty(kOfxPropLabel, "CasparCG");

        // We composite RGBA (and single-channel alpha) images.
        p.setStringProperty(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA, 0);
        p.setStringProperty(kOfxImageEffectPropSupportedComponents, kOfxImageComponentAlpha, 1);

        // Supported working pixel depths (CPU render): 8/16-bit int and 32-bit float.
        p.setStringProperty(kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthByte, 0);
        p.setStringProperty(kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthShort, 1);
        p.setStringProperty(kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthFloat, 2);

        // Contexts we intend to support (filter first; generator/general enumerated for discovery).
        p.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextFilter, 0);
        p.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGeneral, 1);
        p.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGenerator, 2);

        p.setIntProperty(kOfxImageEffectPropSupportsMultipleClipDepths, 0);
        p.setIntProperty(kOfxImageEffectHostPropIsBackground, 0);
        p.setIntProperty(kOfxImageEffectPropSupportsOverlays, 0);

#ifdef OFX_SUPPORTS_OPENGLRENDER
        p.setStringProperty(kOfxImageEffectPropOpenGLRenderSupported, ofx_gl_enabled() ? "true" : "false");
#endif
#ifdef CASPAR_OFX_CUDA
        p.setStringProperty(kOfxImageEffectPropCudaRenderSupported, ofx_cuda_enabled() ? "true" : "false");
        // The host owns a CUDA stream and passes it via kOfxImageEffectPropCudaStream at render time.
        // Plug-ins that ignore it and use the legacy default stream still order correctly (the default
        // stream serialises with our blocking stream).
        p.setStringProperty(kOfxImageEffectPropCudaStreamSupported, ofx_cuda_enabled() ? "true" : "false");
#endif
    }

    OFX::Host::ImageEffect::Instance* newInstance(void*                                      clientData,
                                                  OFX::Host::ImageEffect::ImageEffectPlugin* plugin,
                                                  OFX::Host::ImageEffect::Descriptor&        desc,
                                                  const std::string&                         context) override
    {
        const auto* cp = static_cast<const creation_params*>(clientData);

        const int    w     = cp ? cp->width : 1920;
        const int    h     = cp ? cp->height : 1080;
        const double fps   = cp ? cp->frame_rate : 25.0;
        const int    bytes = cp ? cp->bytes_per_channel : 1;

        return new ofx_effect_instance(plugin, desc, context, w, h, fps, bytes);
    }

    OFX::Host::ImageEffect::Descriptor* makeDescriptor(OFX::Host::ImageEffect::ImageEffectPlugin* plugin) override
    {
        return new OFX::Host::ImageEffect::Descriptor(plugin);
    }

    OFX::Host::ImageEffect::Descriptor* makeDescriptor(const OFX::Host::ImageEffect::Descriptor&  rootContext,
                                                       OFX::Host::ImageEffect::ImageEffectPlugin* plug) override
    {
        return new OFX::Host::ImageEffect::Descriptor(rootContext, plug);
    }

    OFX::Host::ImageEffect::Descriptor* makeDescriptor(const std::string&                         bundlePath,
                                                       OFX::Host::ImageEffect::ImageEffectPlugin* plug) override
    {
        return new OFX::Host::ImageEffect::Descriptor(bundlePath, plug);
    }

    // --- Message suite (OFX::Host::Host) ---
    OfxStatus vmessage(const char* type, const char* /*id*/, const char* format, va_list args) override
    {
        char buf[1024];
        if (format)
            std::vsnprintf(buf, sizeof(buf), format, args);
        else
            buf[0] = '\0';
        const bool err = type && (std::strstr(type, "Error") || std::strstr(type, "Fatal"));
        if (err)
            CASPAR_LOG(warning) << L"[ofx] plug-in message: " << u16(buf);
        else
            CASPAR_LOG(info) << L"[ofx] plug-in message: " << u16(buf);
        return kOfxStatOK;
    }

    OfxStatus setPersistentMessage(const char* /*type*/,
                                   const char* /*id*/,
                                   const char* format,
                                   va_list     args) override
    {
        char buf[1024];
        if (format)
            std::vsnprintf(buf, sizeof(buf), format, args);
        else
            buf[0] = '\0';
        CASPAR_LOG(warning) << L"[ofx] plug-in persistent message: " << u16(buf);
        return kOfxStatOK;
    }

    OfxStatus clearPersistentMessage() override { return kOfxStatOK; }

    // --- Multi-thread suite (OFX_SUPPORTS_MULTITHREAD) ---
    // Real parallel implementation so heavy CPU plug-ins can split their render window across
    // worker threads. Each worker runs under a crash guard (see run_ofx_worker).
    OfxStatus multiThread(OfxThreadFunctionV1 func, unsigned int nThreads, void* customArg) override
    {
        if (func == nullptr)
            return kOfxStatFailed;

        unsigned int maxThreads = 1;
        multiThreadNumCPUS(&maxThreads);

        unsigned int n = (nThreads == 0) ? maxThreads : nThreads;
        if (n > maxThreads)
            n = maxThreads;
        if (n < 1)
            n = 1;

        if (n == 1)
            return run_ofx_worker(func, 0, 1, customArg, false) ? kOfxStatOK : kOfxStatFailed;

        std::atomic<bool>        ok{true};
        std::vector<std::thread> workers;
        workers.reserve(n - 1);
        for (unsigned int i = 1; i < n; ++i) {
            workers.emplace_back([func, i, n, customArg, &ok]() {
                if (!run_ofx_worker(func, i, n, customArg, true))
                    ok.store(false);
            });
        }
        // Run index 0 on the calling thread.
        if (!run_ofx_worker(func, 0, n, customArg, false))
            ok.store(false);
        for (auto& w : workers)
            w.join();

        return ok.load() ? kOfxStatOK : kOfxStatFailed;
    }

    OfxStatus multiThreadNumCPUS(unsigned int* nCPUs) const override
    {
        if (nCPUs == nullptr)
            return kOfxStatFailed;
        unsigned int hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 1;
        // Cap so a single plug-in's internal threading does not oversubscribe the machine (the
        // mixer, decoders and other channels are already running their own threads).
        constexpr unsigned int cap = 8;
        *nCPUs = hc > cap ? cap : hc;
        return kOfxStatOK;
    }

    OfxStatus multiThreadIndex(unsigned int* threadIndex) const override
    {
        if (threadIndex == nullptr)
            return kOfxStatFailed;
        *threadIndex = t_ofx_thread_index;
        return kOfxStatOK;
    }

    int multiThreadIsSpawnedThread() const override { return t_ofx_is_spawned ? 1 : 0; }

    OfxStatus mutexCreate(OfxMutexHandle* mutex, int lockCount) override
    {
        if (mutex == nullptr)
            return kOfxStatFailed;
        auto* m = new std::recursive_mutex();
        for (int i = 0; i < lockCount; ++i)
            m->lock();
        *mutex = reinterpret_cast<OfxMutexHandle>(m);
        return kOfxStatOK;
    }

    OfxStatus mutexDestroy(const OfxMutexHandle mutex) override
    {
        if (mutex == nullptr)
            return kOfxStatErrBadHandle;
        delete reinterpret_cast<std::recursive_mutex*>(mutex);
        return kOfxStatOK;
    }

    OfxStatus mutexLock(const OfxMutexHandle mutex) override
    {
        if (mutex == nullptr)
            return kOfxStatErrBadHandle;
        reinterpret_cast<std::recursive_mutex*>(mutex)->lock();
        return kOfxStatOK;
    }

    OfxStatus mutexUnLock(const OfxMutexHandle mutex) override
    {
        if (mutex == nullptr)
            return kOfxStatErrBadHandle;
        reinterpret_cast<std::recursive_mutex*>(mutex)->unlock();
        return kOfxStatOK;
    }

    OfxStatus mutexTryLock(const OfxMutexHandle mutex) override
    {
        if (mutex == nullptr)
            return kOfxStatErrBadHandle;
        return reinterpret_cast<std::recursive_mutex*>(mutex)->try_lock() ? kOfxStatOK : kOfxStatFailed;
    }

#ifdef OFX_SUPPORTS_OPENGLRENDER
    // OpenGL render suite (Phase 6). Resources are managed per-render for now, so there is
    // nothing global to flush.
    OfxStatus flushOpenGLResources() const override { return kOfxStatOK; }
#endif
};

} // namespace

// ---------------------------------------------------------------------------------------
// effect
// ---------------------------------------------------------------------------------------

struct effect::impl
{
    OFX::Host::ImageEffect::Instance* instance    = nullptr; ///< owned
    ofx_effect_instance*              eff         = nullptr; ///< same object, typed
    bool                              render_open = false;
    std::string                       plugin_id;             ///< for health/blocklist accounting
    bool                              gl_zerocopy_unsupported = false; ///< plug-in needs a compatibility GL profile

    // Zero-copy GL source-convert (swizzle/flip/premultiply on the GPU). Created lazily on the
    // mixer's GL thread and owned by that context (freed on context teardown).
    unsigned int zc_convert_prog_ = 0;
    unsigned int zc_convert_vao_  = 0;
    unsigned int zc_upload_tex_   = 0;

    ~impl()
    {
        // A misbehaving plug-in must never let an exception (incl. SEH under /EHa) escape a
        // destructor, which would call std::terminate.
        try {
            if (instance) {
                if (render_open) {
                    OfxPointD rs{1.0, 1.0};
                    instance->endRenderAction(0, 0, 1, false, rs, true, false);
                }
                delete instance;
            }
        } catch (...) {
        }
    }
};

effect::effect()
    : impl_(std::make_unique<impl>())
{
}

effect::~effect() = default;

bool effect::valid() const { return impl_ && impl_->instance != nullptr; }

int effect::working_bytes() const
{
    if (!valid())
        return 1;
    try {
        auto* clip = impl_->instance->getClip("Source");
        if (clip == nullptr)
            clip = impl_->instance->getClip("Output");
        if (clip == nullptr)
            return 1;
        const std::string& d = clip->getPixelDepth();
        if (d == kOfxBitDepthFloat)
            return 4;
        if (d == kOfxBitDepthShort)
            return 2;
        return 1;
    } catch (...) {
        return 1;
    }
}

bool effect::render(const std::uint8_t* src_rgba, std::uint8_t* dst_rgba, int width, int height, double time, field_kind field)
{
    if (!valid() || dst_rgba == nullptr)
        return false;

    const int wb = working_bytes();

    auto& c         = impl_->eff->ctx();
    c.source_rgba   = src_rgba;
    c.output_rgba   = dst_rgba;
    c.width         = width;
    c.height        = height;
    c.working_bytes = wb;
    c.row_bytes     = width * 4 * wb;
    c.time          = time;

    OfxPointD renderScale{1.0, 1.0};
    OfxRectI  window{0, 0, width, height};

    const char* field_str = kOfxImageFieldBoth;
    switch (field) {
        case field_kind::none: field_str = kOfxImageFieldNone; break;
        case field_kind::lower: field_str = kOfxImageFieldLower; break;
        case field_kind::upper: field_str = kOfxImageFieldUpper; break;
        case field_kind::both:
        default: field_str = kOfxImageFieldBoth; break;
    }

    // Crash guard: with /EHa a misbehaving plug-in's hardware exception is caught here so it
    // cannot take down the server; the frame is dropped (source passes through) instead.
    try {
        const auto      t0 = std::chrono::steady_clock::now();
        const OfxStatus st =
            impl_->instance->renderAction(time, field_str, window, renderScale, true, false, false);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > kRenderBudgetMs)
            record_strike(impl_->plugin_id, L"render over time budget");
        return st == kOfxStatOK || st == kOfxStatReplyDefault;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] render action threw; dropping frame.";
        record_strike(impl_->plugin_id, L"render crashed");
        return false;
    }
}

bool effect::opengl_capable() const
{
    return valid() && impl_->eff->opengl_requested();
}

bool effect::render_transition(const std::uint8_t* src_from,
                               const std::uint8_t* src_to,
                               std::uint8_t*       dst_rgba,
                               int                 width,
                               int                 height,
                               double              time,
                               double              transition,
                               field_kind          field)
{
    if (!valid() || dst_rgba == nullptr)
        return false;

    const int wb = working_bytes();

    // The transition progress is delivered through the mandatory "Transition" double parameter.
    set_param("Transition", {transition}, time);

    auto& c          = impl_->eff->ctx();
    c.source_rgba    = src_from;
    c.source_to_rgba = src_to;
    c.output_rgba    = dst_rgba;
    c.width          = width;
    c.height         = height;
    c.working_bytes  = wb;
    c.row_bytes      = width * 4 * wb;
    c.time           = time;

    OfxPointD renderScale{1.0, 1.0};
    OfxRectI  window{0, 0, width, height};

    const char* field_str = kOfxImageFieldBoth;
    switch (field) {
        case field_kind::none: field_str = kOfxImageFieldNone; break;
        case field_kind::lower: field_str = kOfxImageFieldLower; break;
        case field_kind::upper: field_str = kOfxImageFieldUpper; break;
        case field_kind::both:
        default: field_str = kOfxImageFieldBoth; break;
    }

    try {
        const auto      t0 = std::chrono::steady_clock::now();
        const OfxStatus st =
            impl_->instance->renderAction(time, field_str, window, renderScale, true, false, false);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > kRenderBudgetMs)
            record_strike(impl_->plugin_id, L"transition render over time budget");
        return st == kOfxStatOK || st == kOfxStatReplyDefault;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] transition render action threw; dropping frame.";
        record_strike(impl_->plugin_id, L"transition render crashed");
        return false;
    }
}

bool effect::zerocopy_gl_supported() const
{
    return valid() && !impl_->gl_zerocopy_unsupported;
}

bool effect::cuda_capable() const
{
    return valid() && impl_->eff->cuda_requested();
}

void* effect::render_cuda(const std::uint8_t* raw_source,
                          int                 src_stride,
                          bool                is_bgra,
                          bool                straight_alpha,
                          int                 width,
                          int                 height,
                          double              time,
                          field_kind          field)
{
    if (!valid() || raw_source == nullptr)
        return nullptr;

    auto& c             = impl_->eff->ctx();
    // On-device convert path: hand the raw top-down source to the host; it uploads it once and does
    // the swizzle/flip/premultiply on the device (NPP), then mirrors the output back to top-down.
    c.raw_source        = raw_source;
    c.raw_src_stride    = src_stride;
    c.raw_is_bgra       = is_bgra;
    c.raw_straight      = straight_alpha;
    c.source_rgba       = nullptr;
    c.output_rgba       = nullptr; // skip host readback — caller consumes the returned device ptr directly
    c.width             = width;
    c.height            = height;
    c.working_bytes     = 1;
    c.bytes_per_channel = 1;
    c.row_bytes         = width * 4;
    c.time              = time;

    OfxPointD renderScale{1.0, 1.0};
    OfxRectI  window{0, 0, width, height};

    const char* field_str = kOfxImageFieldBoth;
    switch (field) {
        case field_kind::none: field_str = kOfxImageFieldNone; break;
        case field_kind::lower: field_str = kOfxImageFieldLower; break;
        case field_kind::upper: field_str = kOfxImageFieldUpper; break;
        case field_kind::both:
        default: field_str = kOfxImageFieldBoth; break;
    }

    try {
        const auto      t0 = std::chrono::steady_clock::now();
        const OfxStatus st =
            impl_->instance->renderAction(time, field_str, window, renderScale, true, false, false);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > kRenderBudgetMs)
            record_strike(impl_->plugin_id, L"cuda render over time budget");
        c.raw_source = nullptr; // consumed; do not leak into a later render
        if (st != kOfxStatOK && st != kOfxStatReplyDefault)
            return nullptr;
        // Top-down RGBA device buffer (already un-flipped by the host), ready for one contiguous copy.
        return impl_->eff->ctx().output_dev_topdown;
    } catch (...) {
        c.raw_source = nullptr;
        CASPAR_LOG(warning) << L"[ofx] CUDA render action threw; dropping frame.";
        record_strike(impl_->plugin_id, L"cuda render crashed");
        return nullptr;
    }
}

namespace {
GLuint ofx_compile_shader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        glDeleteShader(s);
        return 0;
    }
    return s;
}
} // namespace

std::shared_ptr<core::texture> effect::render_gl_zerocopy(accelerator::ogl::device& device,
                                                          const std::uint8_t*       src,
                                                          int                       src_stride,
                                                          bool                      src_is_bgra,
                                                          bool                      straight_alpha,
                                                          int                       width,
                                                          int                       height,
                                                          double                    time,
                                                          field_kind                field)
{
    if (!valid() || src == nullptr || width <= 0 || height <= 0)
        return nullptr;

    const char* field_str = kOfxImageFieldBoth;
    switch (field) {
        case field_kind::none: field_str = kOfxImageFieldNone; break;
        case field_kind::lower: field_str = kOfxImageFieldLower; break;
        case field_kind::upper: field_str = kOfxImageFieldUpper; break;
        case field_kind::both:
        default: field_str = kOfxImageFieldBoth; break;
    }

    try {
        // Everything from here runs on the OGL device's dedicated GL thread, where the mixer's
        // GL context is current — so the plug-in renders directly into a mixer-owned texture.
        return device.dispatch_sync([&]() -> std::shared_ptr<core::texture> {
            // Drain any stale GL error so the device's GL_CHECK-wrapped allocations start clean.
            while (glGetError() != GL_NO_ERROR) {}

            // Allocate the textures up front, before the plug-in runs — the device wraps these in
            // GL_CHECK, and an OFX plug-in's GL render often leaves a benign GL error behind that
            // would otherwise trip the next GL_CHECK. source: uploaded from CPU; render: the plug-in
            // renders into it and the mixer consumes it directly.
            auto source_tex = device.create_texture(width, height, 4, common::bit_depth::bit8, false);
            auto render_tex = device.create_texture(width, height, 4, common::bit_depth::bit8, true);

            // Compile the source-convert program once (swizzle is free via the GL_BGRA upload; this
            // pass does the vertical flip + premultiply that the CPU path used to do).
            if (impl_->zc_convert_prog_ == 0) {
                static const char* kVs =
                    "#version 330 core\nout vec2 vUV;\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),"
                    "float(gl_VertexID&2)); vUV=p; gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n";
                static const char* kFs =
                    "#version 330 core\nin vec2 vUV;\nout vec4 o;\nuniform sampler2D uSrc;\nuniform bool "
                    "uStraight;\nvoid main(){ vec4 c=texture(uSrc, vec2(vUV.x, 1.0-vUV.y)); if(uStraight) "
                    "c.rgb*=c.a; o=c; }\n";
                GLuint vs = ofx_compile_shader(GL_VERTEX_SHADER, kVs);
                GLuint fs = ofx_compile_shader(GL_FRAGMENT_SHADER, kFs);
                if (vs && fs) {
                    GLuint prog = glCreateProgram();
                    glAttachShader(prog, vs);
                    glAttachShader(prog, fs);
                    glLinkProgram(prog);
                    GLint ok = GL_FALSE;
                    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
                    if (ok) {
                        impl_->zc_convert_prog_ = prog;
                        glGenVertexArrays(1, &impl_->zc_convert_vao_);
                    } else {
                        glDeleteProgram(prog);
                    }
                }
                if (vs)
                    glDeleteShader(vs);
                if (fs)
                    glDeleteShader(fs);
            }
            if (impl_->zc_convert_prog_ == 0)
                return nullptr; // convert program unavailable -> fall back to CPU path

            // Upload the RAW source (top-down); GL_BGRA gives a free channel swap on upload.
            if (impl_->zc_upload_tex_ == 0) {
                glGenTextures(1, &impl_->zc_upload_tex_);
                glBindTexture(GL_TEXTURE_2D, impl_->zc_upload_tex_);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            } else {
                glBindTexture(GL_TEXTURE_2D, impl_->zc_upload_tex_);
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, src_stride / 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                         src_is_bgra ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, src);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glBindTexture(GL_TEXTURE_2D, 0);

            // GPU convert pass: sample the upload texture flipped + premultiplied into source_tex.
            {
                GLint prev_fbo = 0, prev_vp[4] = {0, 0, 0, 0};
                glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
                glGetIntegerv(GL_VIEWPORT, prev_vp);
                GLuint cfbo = 0;
                glGenFramebuffers(1, &cfbo);
                glBindFramebuffer(GL_FRAMEBUFFER, cfbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, source_tex->id(), 0);
                glViewport(0, 0, width, height);
                glDisable(GL_BLEND);
                glUseProgram(impl_->zc_convert_prog_);
                glBindVertexArray(impl_->zc_convert_vao_);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, impl_->zc_upload_tex_);
                glUniform1i(glGetUniformLocation(impl_->zc_convert_prog_, "uSrc"), 0);
                glUniform1i(glGetUniformLocation(impl_->zc_convert_prog_, "uStraight"), straight_alpha ? 1 : 0);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glBindVertexArray(0);
                glUseProgram(0);
                glBindTexture(GL_TEXTURE_2D, 0);
                glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
                glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
                glDeleteFramebuffers(1, &cfbo);
            }

            auto& c         = impl_->eff->ctx();
            c.external_gl   = true;
            c.source_rgba   = nullptr;
            c.output_rgba   = nullptr;
            c.width         = width;
            c.height        = height;
            c.working_bytes = 1;
            c.bytes_per_channel = 1;
            c.row_bytes     = width * 4;
            c.time          = time;
            c.source_tex    = static_cast<unsigned int>(source_tex->id());
            c.output_tex    = static_cast<unsigned int>(render_tex->id());

            OfxPointD renderScale{1.0, 1.0};
            OfxRectI  window{0, 0, width, height};

            OfxStatus st = kOfxStatFailed;
            const auto t0 = std::chrono::steady_clock::now();
            try {
                st = impl_->instance->renderAction(time, field_str, window, renderScale, true, false, false);
            } catch (...) {
                c.external_gl = false;
                while (glGetError() != GL_NO_ERROR) {}
                CASPAR_LOG(warning) << L"[ofx] zero-copy GL render threw; dropping frame.";
                record_strike(impl_->plugin_id, L"gl render crashed");
                return nullptr;
            }
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms > kRenderBudgetMs)
                record_strike(impl_->plugin_id, L"gl render over time budget");

            c.external_gl = false;

            // Detect a legacy fixed-function plug-in: such a plug-in issues GL calls (glBegin,
            // glOrtho, ...) that are INVALID in the mixer's core profile and generate a GL error
            // without drawing anything. If that happened, this plug-in cannot render zero-copy on
            // the core device — flag it so the producer falls back to the self-contained
            // compatibility path (which renders it correctly, at the cost of a readback).
            const bool gl_err = glGetError() != GL_NO_ERROR;
            while (glGetError() != GL_NO_ERROR) {} // fully drain so the mixer stays clean
            if (gl_err) {
                impl_->gl_zerocopy_unsupported = true;
                CASPAR_LOG(info) << L"[ofx] '" << u16(impl_->plugin_id)
                                 << L"' uses a non-core GL profile; using the compatibility "
                                 << L"render path (readback) instead of zero-copy.";
                return nullptr;
            }

            if (st != kOfxStatOK && st != kOfxStatReplyDefault)
                return nullptr;

            // OFX renders bottom-up (origin bottom-left): render_tex row 0 holds the image's bottom
            // row, matching the reference (self-contained) path whose source is uploaded bottom-up.
            // The mixer's texture convention is top-down (row 0 = image top), so flip render_tex into
            // the output texture. Use DSA (glBlitNamedFramebuffer) so we never touch the mixer's
            // bound-framebuffer state — a bind-based blit here corrupted the mixer and crashed it.
            auto out_tex = device.create_texture(width, height, 4, common::bit_depth::bit8, false);
            GLuint fbos[2] = {0, 0};
            glCreateFramebuffers(2, fbos);
            glNamedFramebufferTexture(fbos[0], GL_COLOR_ATTACHMENT0, render_tex->id(), 0);
            glNamedFramebufferTexture(fbos[1], GL_COLOR_ATTACHMENT0, out_tex->id(), 0);
            glBlitNamedFramebuffer(fbos[0], fbos[1],
                                   0, 0, width, height,
                                   0, height, width, 0, // flip Y: src [0,h] -> dst [h,0]
                                   GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glDeleteFramebuffers(2, fbos);
            while (glGetError() != GL_NO_ERROR) {} // keep the mixer's GL_CHECK clean

            static bool logged = false;
            if (!logged) {
                logged = true;
                CASPAR_LOG(info) << L"[ofx] OpenGL zero-copy producer path active (no readback).";
            }

            return std::static_pointer_cast<core::texture>(out_tex);
        });
    } catch (const std::exception& e) {
        CASPAR_LOG(warning) << L"[ofx] zero-copy GL dispatch threw (" << u16(e.what()) << L"); dropping frame.";
        record_strike(impl_->plugin_id, L"gl dispatch crashed");
        return nullptr;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] zero-copy GL dispatch threw (unknown); dropping frame.";
        record_strike(impl_->plugin_id, L"gl dispatch crashed");
        return nullptr;
    }
}

bool effect::output_premultiplied() const
{
    if (!valid())
        return true;
    try {
        return impl_->instance->getOutputPreMultiplication() != kOfxImageUnPreMultiplied;
    } catch (...) {
        return true;
    }
}

bool effect::is_identity(int width, int height, double time)
{
    if (!valid())
        return false;
    try {
        OfxTime     t = time;
        std::string clip;
        OfxPointD   rs{1.0, 1.0};
        OfxRectI    window{0, 0, width, height};
        // isIdentityAction returns kOfxStatOK (with the identity input clip name) when the effect
        // is a no-op for this frame; kOfxStatReplyDefault means "not identity, please render".
        const OfxStatus st = impl_->instance->isIdentityAction(t, kOfxImageFieldBoth, window, rs, clip);
        return st == kOfxStatOK;
    } catch (...) {
        return false;
    }
}

std::vector<effect::param> effect::params() const
{
    std::vector<effect::param> result;
    if (!valid())
        return result;

    for (auto* p : impl_->instance->getParamList()) {
        if (p == nullptr)
            continue;
        const std::string& type = p->getType();
        // Skip structural/non-value params.
        if (type == kOfxParamTypeGroup || type == kOfxParamTypePage)
            continue;

        effect::param pp;
        pp.name  = p->getName();
        pp.type  = type;
        pp.label = p->getLabel();

        // Read metadata (dimension, range, default, choice options) from the param properties.
        try {
            auto& props = p->getProperties();

            const bool is_double = type == kOfxParamTypeDouble || type == kOfxParamTypeDouble2D ||
                                   type == kOfxParamTypeDouble3D;
            const bool is_int = type == kOfxParamTypeInteger || type == kOfxParamTypeInteger2D ||
                                type == kOfxParamTypeInteger3D;

            if (type == kOfxParamTypeDouble2D || type == kOfxParamTypeInteger2D)
                pp.dimension = 2;
            else if (type == kOfxParamTypeDouble3D || type == kOfxParamTypeInteger3D ||
                     type == kOfxParamTypeRGB)
                pp.dimension = 3;
            else if (type == kOfxParamTypeRGBA)
                pp.dimension = 4;

            if (is_double) {
                pp.min       = props.getDoubleProperty(kOfxParamPropDisplayMin, 0);
                pp.max       = props.getDoubleProperty(kOfxParamPropDisplayMax, 0);
                // Fall back to the hard limits when no display range was declared.
                if (!(pp.min > -1e37 && pp.min < 1e37))
                    pp.min = props.getDoubleProperty(kOfxParamPropMin, 0);
                if (!(pp.max > -1e37 && pp.max < 1e37))
                    pp.max = props.getDoubleProperty(kOfxParamPropMax, 0);
                pp.def       = props.getDoubleProperty(kOfxParamPropDefault, 0);
                pp.has_range = true;
            } else if (is_int) {
                pp.min       = static_cast<double>(props.getIntProperty(kOfxParamPropDisplayMin, 0));
                pp.max       = static_cast<double>(props.getIntProperty(kOfxParamPropDisplayMax, 0));
                if (!(pp.min > -1e37 && pp.min < 1e37))
                    pp.min = static_cast<double>(props.getIntProperty(kOfxParamPropMin, 0));
                if (!(pp.max > -1e37 && pp.max < 1e37))
                    pp.max = static_cast<double>(props.getIntProperty(kOfxParamPropMax, 0));
                pp.def       = static_cast<double>(props.getIntProperty(kOfxParamPropDefault, 0));
                pp.has_range = true;
            } else if (type == kOfxParamTypeBoolean) {
                pp.min = 0.0;
                pp.max = 1.0;
                pp.def = static_cast<double>(props.getIntProperty(kOfxParamPropDefault, 0));
                pp.has_range = true;
            } else if (type == kOfxParamTypeChoice) {
                pp.def = static_cast<double>(props.getIntProperty(kOfxParamPropDefault, 0));
                const int n = props.getDimension(kOfxParamPropChoiceOption);
                for (int i = 0; i < n; ++i)
                    pp.choices.push_back(props.getStringProperty(kOfxParamPropChoiceOption, i));
            }
        } catch (...) {
            // Missing/typed-differently properties are non-fatal; report what we have.
        }

        result.push_back(std::move(pp));
    }
    return result;
}

bool effect::set_param(const std::string& name, const std::vector<double>& values, double time)
{
    if (!valid())
        return false;

    auto* p = impl_->instance->getParam(name);
    if (p == nullptr)
        return false;

    const auto v = [&](std::size_t i) { return i < values.size() ? values[i] : 0.0; };

    const std::string& type = p->getType();
    bool               ok   = false;

    try {
        if (type == kOfxParamTypeDouble) {
            if (auto* d = dynamic_cast<OFX::Host::Param::DoubleInstance*>(p))
                ok = d->set(v(0)) == kOfxStatOK;
        } else if (type == kOfxParamTypeInteger) {
            if (auto* d = dynamic_cast<OFX::Host::Param::IntegerInstance*>(p))
                ok = d->set(static_cast<int>(v(0))) == kOfxStatOK;
        } else if (type == kOfxParamTypeBoolean) {
            if (auto* d = dynamic_cast<OFX::Host::Param::BooleanInstance*>(p))
                ok = d->set(v(0) != 0.0) == kOfxStatOK;
        } else if (type == kOfxParamTypeChoice) {
            if (auto* d = dynamic_cast<OFX::Host::Param::ChoiceInstance*>(p))
                ok = d->set(static_cast<int>(v(0))) == kOfxStatOK;
        } else if (type == kOfxParamTypeRGBA) {
            if (auto* d = dynamic_cast<OFX::Host::Param::RGBAInstance*>(p))
                ok = d->set(v(0), v(1), v(2), v(3)) == kOfxStatOK;
        } else if (type == kOfxParamTypeRGB) {
            if (auto* d = dynamic_cast<OFX::Host::Param::RGBInstance*>(p))
                ok = d->set(v(0), v(1), v(2)) == kOfxStatOK;
        } else if (type == kOfxParamTypeDouble2D) {
            if (auto* d = dynamic_cast<OFX::Host::Param::Double2DInstance*>(p))
                ok = d->set(v(0), v(1)) == kOfxStatOK;
        } else if (type == kOfxParamTypeInteger2D) {
            if (auto* d = dynamic_cast<OFX::Host::Param::Integer2DInstance*>(p))
                ok = d->set(static_cast<int>(v(0)), static_cast<int>(v(1))) == kOfxStatOK;
        }

        if (ok) {
            OfxPointD rs{1.0, 1.0};
            impl_->instance->paramInstanceChangedAction(name, kOfxChangeUserEdited, time, rs);
        }
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] Exception setting parameter '" << u16(name) << L"'.";
        return false;
    }

    return ok;
}

bool effect::set_param_string(const std::string& name, const std::string& value, double time)
{
    if (!valid())
        return false;

    auto* p = impl_->instance->getParam(name);
    if (p == nullptr)
        return false;

    try {
        auto* s = dynamic_cast<OFX::Host::Param::StringInstance*>(p);
        if (s == nullptr)
            return false;
        if (s->set(value.c_str()) != kOfxStatOK)
            return false;
        OfxPointD rs{1.0, 1.0};
        impl_->instance->paramInstanceChangedAction(name, kOfxChangeUserEdited, time, rs);
        return true;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] Exception setting string parameter '" << u16(name) << L"'.";
        return false;
    }
}

// ---------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------

struct host::impl
{
    caspar_ofx_host                          host_;
    OFX::Host::ImageEffect::PluginCache       ie_cache_{host_};
    std::vector<plugin_info>                  plugins_;
    bool                                      registered_ = false;

    void ensure_registered()
    {
        if (!registered_) {
            ie_cache_.registerInCache(*OFX::Host::PluginCache::getPluginCache());
            registered_ = true;
        }
    }
};

host::host()
    : impl_(std::make_unique<impl>())
{
    impl_->ensure_registered();
}

host::~host() = default;

void host::add_search_path(const std::string& dir)
{
    OFX::Host::PluginCache::getPluginCache()->prependFileToPath(dir);
}

std::vector<std::string> host::search_paths() const
{
    std::vector<std::string> paths;
    for (const auto& p : OFX::Host::PluginCache::getPluginCache()->getPluginPath())
        paths.push_back(p);
    return paths;
}

void host::scan()
{
    impl_->ensure_registered();

    auto* cache = OFX::Host::PluginCache::getPluginCache();
    cache->setCacheVersion("casparcg.ofx.v1");

    // Persistent plug-in cache: describing every .ofx bundle on each boot is slow when many
    // plug-ins are installed. Load the previous cache (if any) so scanPluginFiles() only
    // re-describes bundles whose binaries changed, then write the refreshed cache back.
    const std::string cache_path = "casparcg_ofx_cache.xml"; // relative to the server working dir
    bool              cache_loaded = false;
    {
        std::ifstream is(cache_path, std::ios::binary);
        if (is.good()) {
            try {
                cache->readCache(is);
                cache_loaded = true;
            } catch (...) {
                // Corrupt/incompatible cache -> ignore and fall back to a full scan.
            }
        }
    }

    cache->scanPluginFiles();

    {
        std::ofstream os(cache_path, std::ios::binary | std::ios::trunc);
        if (os.good()) {
            try {
                cache->writePluginCache(os);
            } catch (...) {
                // Non-fatal: a missing cache just means a full describe next boot.
            }
        }
    }
    CASPAR_LOG(info) << L"[ofx] Plug-in cache " << (cache_loaded ? L"loaded from '" : L"created at '")
                     << u16(cache_path) << L"'.";

    impl_->plugins_.clear();

    for (auto* plugin : impl_->ie_cache_.getPlugins()) {
        if (plugin == nullptr)
            continue;

        plugin_info info;
        info.identifier    = plugin->getIdentifier();
        info.version_major = plugin->getVersionMajor();
        info.version_minor = plugin->getVersionMinor();

        if (const auto* binary = plugin->getBinary())
            info.bundle_path = binary->getBundlePath();

        // getContexts() forces the plug-in to be described so its labels/contexts are populated.
        for (const auto& ctx : plugin->getContexts())
            info.contexts.push_back(ctx);

        auto& desc     = plugin->getDescriptor();
        info.label     = desc.getLabel();
        info.grouping  = desc.getPluginGrouping();

        impl_->plugins_.push_back(std::move(info));
    }
}

const std::vector<plugin_info>& host::plugins() const
{
    return impl_->plugins_;
}

std::unique_ptr<effect>
host::create_effect(const std::string& identifier, effect_context context, int width, int height, double frame_rate, int bytes_per_channel)
{
    impl_->ensure_registered();

    if (is_blocklisted(identifier)) {
        CASPAR_LOG(warning) << L"[ofx] '" << u16(identifier) << L"' is blocklisted; not instantiating.";
        return nullptr;
    }

    auto* plugin = impl_->ie_cache_.getPluginById(identifier);
    if (plugin == nullptr)
        return nullptr;

    creation_params cp;
    cp.width             = width;
    cp.height            = height;
    cp.frame_rate        = frame_rate;
    cp.bytes_per_channel = bytes_per_channel;

    // Choose the OFX context. Honor the requested style (filter with a source clip, or generator),
    // but fall back to the General context, which many commercial plug-ins declare instead of the
    // more specific Filter/Generator contexts. General plug-ins that use the standard Source/Output
    // clips are handled like a filter.
    auto supports = [&](const char* c) {
        for (const auto& ctx : plugin->getContexts())
            if (ctx == c)
                return true;
        return false;
    };

    std::string ofx_context;
    if (context == effect_context::generator) {
        if (supports(kOfxImageEffectContextGenerator))
            ofx_context = kOfxImageEffectContextGenerator;
        else if (supports(kOfxImageEffectContextGeneral))
            ofx_context = kOfxImageEffectContextGeneral;
        else
            ofx_context = kOfxImageEffectContextGenerator;
    } else if (context == effect_context::transition) {
        ofx_context = kOfxImageEffectContextTransition;
    } else {
        if (supports(kOfxImageEffectContextFilter))
            ofx_context = kOfxImageEffectContextFilter;
        else if (supports(kOfxImageEffectContextGeneral))
            ofx_context = kOfxImageEffectContextGeneral;
        else
            ofx_context = kOfxImageEffectContextFilter;
    }

    try {
        OFX::Host::ImageEffect::Instance* instance = plugin->createInstance(ofx_context, &cp);
        if (instance == nullptr)
            return nullptr;

        const OfxStatus create_stat = instance->createInstanceAction();
        if (create_stat != kOfxStatOK && create_stat != kOfxStatReplyDefault) {
            delete instance;
            return nullptr;
        }

        if (!instance->getClipPreferences()) {
            delete instance;
            return nullptr;
        }

        OfxPointD rs{1.0, 1.0};
        instance->beginRenderAction(0, 0, 1, false, rs, true, false);

        std::unique_ptr<effect> e(new effect());
        e->impl_->instance    = instance;
        e->impl_->eff         = static_cast<ofx_effect_instance*>(instance);
        e->impl_->render_open = true;
        e->impl_->plugin_id   = identifier;

        // Only enable a GPU backend if BOTH the host (env flag) and the plug-in advertise it;
        // otherwise a CPU-only plug-in would be handed textures/device-pointers it can't use.
        bool plugin_gl   = false;
        bool plugin_cuda = false;
        try {
            auto& dprops = instance->getDescriptor().getProps();
#ifdef OFX_SUPPORTS_OPENGLRENDER
            plugin_gl = dprops.getStringProperty(kOfxImageEffectPropOpenGLRenderSupported) != "false";
#endif
#ifdef CASPAR_OFX_CUDA
            plugin_cuda = dprops.getStringProperty(kOfxImageEffectPropCudaRenderSupported) != "false";
#endif
        } catch (...) {
        }

        e->impl_->eff->enable_opengl(ofx_gl_enabled() && plugin_gl);
#ifdef CASPAR_OFX_CUDA
        e->impl_->eff->enable_cuda(ofx_cuda_enabled() && plugin_cuda);
#endif
        return e;
    } catch (...) {
        CASPAR_LOG(warning) << L"[ofx] Exception while creating effect '" << u16(identifier) << L"'.";
        record_strike(identifier, L"create crashed");
        return nullptr;
    }
}

host& global_host()
{
    static host instance;
    return instance;
}

void configure_opengl(bool enabled) { g_config_gl = enabled; }
void configure_cuda(bool enabled) { g_config_cuda = enabled; }
void blocklist_plugin(const std::string& identifier) { add_to_blocklist(identifier); }

}} // namespace caspar::ofx
