
#include "context.h"

#include <common/log.h>

#include <SFML/Window/Context.hpp>
#if SFML_VERSION_MAJOR >= 3
#include <SFML/Window/ContextSettings.hpp>
#endif

#include <tuple> // std::ignore

#ifdef _MSC_VER
#include <windows.h>
#include <GL/gl.h>
#endif

#ifndef _MSC_VER
#include <EGL/egl.h>
#include <common/gl/gl_check.h>
#include <stdlib.h>
#endif

namespace caspar::accelerator::ogl {

struct device_context::impl
{
    virtual ~impl() {}
    virtual void bind()   = 0;
    virtual void unbind() = 0;
    virtual void* native_handle() const { return nullptr; }
    virtual void* native_egl_display() const { return nullptr; }
};
struct impl_sfml : public device_context::impl
{
    sf::Context device_;

    impl_sfml()
#if SFML_VERSION_MAJOR >= 3
        : device_(sf::ContextSettings{.depthBits         = 0,
                                      .stencilBits       = 0,
                                      .antiAliasingLevel = 0,
                                      .majorVersion      = 4,
                                      .minorVersion      = 5,
                                      .attributeFlags    = sf::ContextSettings::Attribute::Core},
                  {1, 1})
#else
        : device_(sf::ContextSettings(0, 0, 0, 4, 5, sf::ContextSettings::Attribute::Core), 1, 1)
#endif
    {
        CASPAR_LOG(info) << L"Initializing OpenGL Device (sfml).";
    }

    virtual ~impl_sfml()
    {
#ifdef _MSC_VER
        // Never made current, so nothing can be using it; and it is destroyed after the device
        // thread has unbound, so the share group is going away anyway.
        if (seed_hglrc_)
            wglDeleteContext(reinterpret_cast<HGLRC>(seed_hglrc_));
#endif
    }

#ifdef _MSC_VER
    void* cached_hglrc_ = nullptr;

    /// A context in the mixer's share group that is NEVER MADE CURRENT, handed to consumers
    /// instead of the mixer's own.
    ///
    /// WHY THE MIXER'S OWN CONTEXT CANNOT BE SHARED AGAINST. `device::impl` binds it on the
    /// "OpenGL Device" thread and leaves it current for that thread's whole life
    /// (`context_->bind()` then `io_context_.run()`). A consumer calling
    /// `wglCreateContextAttribsARB(hdc, thatHGLRC, ...)` from its OWN thread is then sharing
    /// against a context current in another thread, which NVIDIA refuses: it returns null and
    /// leaves `GetLastError()` at **0**, so the failure carries no diagnosis at all.
    ///
    /// Measured 2026-08-23 on an RTX A4000, driver 582.53: `Shared GL context creation failed
    /// (error=0), falling back to standalone context`, after which the screen consumer takes its
    /// PBO upload path and the channel keeps reading the composite back to host memory. The
    /// Spout consumer has the identical code and the identical failure, so BOTH GPU-texture
    /// paths on the OpenGL mixer have been unreachable.
    ///
    /// Share groups are transitive, so a context sharing with the mixer's puts a consumer in the
    /// mixer's group just as well -- and because this one is never current anywhere, sharing
    /// against it from any thread is legal. Created here rather than in the consumers because
    /// there is exactly one right moment for it: inside `bind()`, where the mixer's context is
    /// current on THIS thread, which is the one case WGL allows.
    void* seed_hglrc_ = nullptr;
    bool  seed_tried_ = false;
#endif

    virtual void bind() override
    {
        std::ignore = device_.setActive(true);
#ifdef _MSC_VER
        if (!cached_hglrc_)
            cached_hglrc_ = wglGetCurrentContext();

        if (!seed_tried_) {
            seed_tried_ = true; // once, whether or not it works
            auto create = reinterpret_cast<HGLRC(WINAPI*)(HDC, HGLRC, const int*)>(
                wglGetProcAddress("wglCreateContextAttribsARB"));
            auto hdc = wglGetCurrentDC();
            auto cur = wglGetCurrentContext();
            if (create && hdc && cur) {
                // Same version and profile as the mixer's, because a share group spans one
                // GL version family and a mismatch is another silent null return.
                const int attribs[] = {0x2091 /* MAJOR_VERSION_ARB */,
                                       4,
                                       0x2092 /* MINOR_VERSION_ARB */,
                                       5,
                                       0x9126 /* PROFILE_MASK_ARB  */,
                                       0x00000001 /* CORE_PROFILE_BIT */,
                                       0};
                seed_hglrc_ = create(hdc, cur, attribs);
            }
            if (seed_hglrc_) {
                CASPAR_LOG(info) << L"[ogl] share-group seed context created; consumers can take "
                                    L"the zero-copy GPU texture path.";
            } else {
                CASPAR_LOG(warning)
                    << L"[ogl] could not create a share-group seed context (error="
                    << static_cast<unsigned int>(GetLastError())
                    << L"); consumers will share against the mixer's own context, which is "
                       L"current on the device thread and which some drivers refuse.";
            }
        }
#endif
    }
    virtual void unbind() override { std::ignore = device_.setActive(false); }
    virtual void* native_handle() const override
    {
#ifdef _MSC_VER
        // The seed when there is one. Falling back to the mixer's own context keeps the previous
        // behaviour on a driver where the seed could not be made -- that path was already
        // failing, so it cannot be made worse by trying it.
        return seed_hglrc_ ? seed_hglrc_ : cached_hglrc_;
#else
        return nullptr;
#endif
    }
};

#ifndef _MSC_VER
struct impl_egl : public device_context::impl
{
    EGLDisplay eglDisplay_;
    EGLContext eglContext_;

    impl_egl()
        : eglDisplay_(EGL_NO_DISPLAY)
        , eglContext_(EGL_NO_CONTEXT)
    {
        CASPAR_LOG(info) << L"Initializing OpenGL Device (EGL).";

        eglDisplay_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);

        EGLint major, minor;
        eglInitialize(eglDisplay_, &major, &minor);

        const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                        EGL_PBUFFER_BIT,
                                        EGL_BLUE_SIZE,
                                        8,
                                        EGL_GREEN_SIZE,
                                        8,
                                        EGL_RED_SIZE,
                                        8,
                                        EGL_RENDERABLE_TYPE,
                                        EGL_OPENGL_BIT,
                                        EGL_NONE};

        EGLint    numConfigs;
        EGLConfig eglConfig;
        if (!eglChooseConfig(eglDisplay_, configAttribs, &eglConfig, 1, &numConfigs)) {
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize OpenGL: eglChooseConfig"));
        }

        if (!eglBindAPI(EGL_OPENGL_API)) {
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize OpenGL: eglBindAPI"));
        }

        eglContext_ = eglCreateContext(eglDisplay_, eglConfig, EGL_NO_CONTEXT, NULL);
        if (eglContext_ == EGL_NO_CONTEXT) {
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize OpenGL: eglCreateContext"));
        }

        if (!eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, eglContext_)) {
            CASPAR_THROW_EXCEPTION(gl::ogl_exception() << msg_info("Failed to initialize OpenGL: eglMakeCurrent"));
        }
    }

    virtual ~impl_egl()
    {
        eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

        if (eglContext_ != EGL_NO_CONTEXT) {
            eglDestroyContext(eglDisplay_, eglContext_);
        }

        eglTerminate(eglDisplay_);
    }

    virtual void bind() override { eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, eglContext_); }
    virtual void unbind() override { eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); }
    virtual void* native_handle() const override { return eglContext_; }
    virtual void* native_egl_display() const override { return eglDisplay_; }
};
#endif

#ifndef _MSC_VER
device_context::device_context()
    : impl_(std::getenv("DISPLAY") == nullptr ? spl::make_shared<device_context::impl, impl_egl>()
                                              : spl::make_shared<device_context::impl, impl_sfml>())
{
}
#else
device_context::device_context()
    : impl_(new impl_sfml())
{
}
#endif

device_context::~device_context() {}

void device_context::bind() { impl_->bind(); }
void device_context::unbind() { impl_->unbind(); }
void* device_context::native_handle() const { return impl_->native_handle(); }
void* device_context::native_egl_display() const { return impl_->native_egl_display(); }

} // namespace caspar::accelerator::ogl
