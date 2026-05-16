#pragma once

#ifdef _WIN32

#include <memory>

struct AVBufferRef;
struct AVFrame;

namespace caspar { namespace ffmpeg {

class d3d11_gl_bridge
{
  public:
    d3d11_gl_bridge();
    ~d3d11_gl_bridge();

    d3d11_gl_bridge(const d3d11_gl_bridge&) = delete;
    d3d11_gl_bridge& operator=(const d3d11_gl_bridge&) = delete;

    bool init(AVBufferRef* hw_device_ctx, void* ogl_device_ptr);

    /// Convert a D3D11VA decoded frame (NV12) to a GPU texture (BGRA) via video processor + WGL interop.
    /// Returns shared_ptr<core::texture> as shared_ptr<void> to avoid namespace issues in MSVC.
    std::shared_ptr<void> convert(AVFrame* d3d11_frame, void* ogl_device_ptr);

    bool is_active() const;
    void cleanup();

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::ffmpeg

#endif // _WIN32
