#pragma once

#include <memory>

namespace caspar::accelerator::ogl {

class device_context final
{
  public:
    device_context();
    ~device_context();

    device_context(const device_context&) = delete;

    device_context& operator=(const device_context&) = delete;

    void bind();
    void unbind();

    /// Return the platform-native GL context handle (HGLRC on Windows, EGLContext on Linux).
    void* native_handle() const;

    /// Return the EGL display handle (Linux only, nullptr on Windows).
    void* native_egl_display() const;

    struct impl;

  private:
    spl::shared_ptr<impl> impl_;
};

} // namespace caspar::accelerator::ogl
