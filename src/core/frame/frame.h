#pragma once

#include <common/array.h>

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace caspar { namespace core {

class texture
{
  public:
    virtual ~texture() {}
    virtual void bind(int index) = 0;
    virtual void unbind()        = 0;

    /// Opaque identity of the GPU device that owns this texture's memory
    /// (the VkDevice for Vulkan, the ogl::device for OpenGL).
    ///
    /// A mixer may only bind a texture natively when this matches its own
    /// device. Using a VkImage that belongs to a different VkDevice — which
    /// happens as soon as two channels are pinned to different GPUs and one
    /// routes into the other — is undefined behaviour, and the wrapper types
    /// alone cannot distinguish the two cases. Consumers that import via
    /// external memory should use export_native_handle() instead and do not
    /// need to match.
    ///
    /// Returns nullptr when unknown, which callers must treat as "not mine".
    virtual const void*        owner_device() const { return nullptr; }

    /// Export a platform-native handle for the texture's GPU memory.
    /// Windows: Win32 HANDLE, Linux: file descriptor cast to void*.
    /// Returns nullptr if not supported. Caller must NOT close the handle.
    virtual void*              export_native_handle() const { return nullptr; }
    /// Size of the GPU memory allocation backing the texture (bytes).
    virtual unsigned long long export_alloc_size() const { return 0; }
    /// Wait for any pending GPU rendering to complete before reading.
    /// No-op for OGL textures; overridden by VK texture_wrapper.
    virtual void               ensure_render_complete() const {}
    /// Returns a platform-native handle to a VkSemaphore signaled on render completion.
    /// For GPU-side waiting (e.g. CUDA interop) instead of CPU fence wait.
    virtual void*              render_semaphore_handle() const { return nullptr; }
    /// Timeline semaphore value that will be signaled on render completion.
    virtual uint64_t           render_semaphore_value() const { return 0; }
    /// Width/height of the texture.
    virtual int                tex_width() const { return 0; }
    virtual int                tex_height() const { return 0; }
    /// True if the texture uses 16-bit components.
    virtual bool               tex_is_hbd() const { return false; }
    /// On-demand GPU→CPU readback. Returns pixel data or empty if unsupported.
    /// Only called by consumers that explicitly need CPU pixels (e.g. PRINT RAW).
    /// Default: returns empty (no readback capability).
    virtual std::vector<std::uint8_t> read_pixels() const { return {}; }
};

class mutable_frame final
{
    friend class const_frame;

  public:
    using commit_t = std::function<std::any(std::vector<array<const std::uint8_t>>)>;

    explicit mutable_frame(const void*                      tag,
                           std::vector<array<std::uint8_t>> image_data,
                           array<std::int32_t>              audio_data,
                           const struct pixel_format_desc&  desc,
                           commit_t                         commit = nullptr);
    mutable_frame(const mutable_frame&) = delete;
    mutable_frame(mutable_frame&& other) noexcept;

    ~mutable_frame();

    mutable_frame& operator=(const mutable_frame&) = delete;
    mutable_frame& operator=(mutable_frame&& other);

    void swap(mutable_frame& other);

    const struct pixel_format_desc& pixel_format_desc() const;

    array<std::uint8_t>&       image_data(std::size_t index);
    const array<std::uint8_t>& image_data(std::size_t index) const;

    array<std::int32_t>&       audio_data();
    const array<std::int32_t>& audio_data() const;

    std::size_t width() const;

    std::size_t height() const;

    const void* stream_tag() const;

    class frame_geometry&       geometry();
    const class frame_geometry& geometry() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// Whether a frame can give you host (CPU) pixels, and at what cost.
///
/// Before this existed, all three cases below looked identical at the call site:
/// image_data() returned an empty array both when readback had been deliberately
/// skipped (no consumer asked for CPU pixels) and when it simply had not
/// happened yet. Code that treated "empty" as "black" therefore silently
/// produced black frames, and code that treated it as "has pixels" read a null
/// pointer. Ask host_image_state() when you need to know.
enum class host_image_availability
{
    /// No host pixels and none obtainable: a GPU-only frame whose readback was
    /// skipped. image_data() returns an empty array. Use texture() instead, or
    /// declare needs_cpu_frame_data() so the mixer produces pixels for you.
    unavailable,

    /// A readback is in flight. image_data() will block until it lands and then
    /// yield real pixels. Never call this on the channel thread.
    deferred,

    /// Host pixels are present now; image_data() is a cheap accessor.
    available,
};

class const_frame final
{
  public:
    const_frame();
    explicit const_frame(const void*                            tag,
                         std::vector<array<const std::uint8_t>> image_data,
                         array<const std::int32_t>              audio_data,
                         const struct pixel_format_desc&        desc,
                         std::shared_ptr<core::texture>         texture = nullptr);
    /// Lazy-readback constructor: the image_data future is only evaluated when
    /// image_data() is called.  GPU→CPU readback is deferred until a consumer
    /// actually needs CPU pixels.
    explicit const_frame(const void*                                           tag,
                         std::shared_future<array<const std::uint8_t>>         lazy_image,
                         array<const std::int32_t>                             audio_data,
                         const struct pixel_format_desc&                       desc,
                         std::shared_ptr<core::texture>                        texture);
    /// Multi-plane GPU frame: the producer decoded straight into GPU memory and
    /// the planes are still separate (Y + interleaved CbCr for a hardware NV12
    /// decode, say). The mixer binds them as the planes of `desc`, so the shader
    /// does the colour conversion -- which is the whole point, since a
    /// producer-side conversion cannot use the channel's colour management.
    explicit const_frame(const void*                                tag,
                         std::vector<array<const std::uint8_t>>     image_data,
                         array<const std::int32_t>                  audio_data,
                         const struct pixel_format_desc&            desc,
                         std::vector<std::shared_ptr<core::texture>> textures);
    const_frame(const const_frame& other);
    const_frame(mutable_frame&& other);

    ~const_frame();

    const_frame& operator=(const const_frame& other);

    const struct pixel_format_desc& pixel_format_desc() const;

    const array<const std::uint8_t>& image_data(std::size_t index) const;

    /// Can this frame give you host pixels, and at what cost? Resolving a
    /// readback that has already completed is free, so this is cheap to call —
    /// but it does not block on one that is still in flight (that reports
    /// `deferred`).
    host_image_availability host_image_state() const;

    /// Shorthand for host_image_state() != unavailable, i.e. "image_data() will
    /// eventually yield real pixels".
    bool has_host_image() const;

    const array<const std::int32_t>& audio_data() const;

    /// The frame's first GPU plane, or nullptr. Kept for the many callers that
    /// only ever deal with single-plane (already converted) GPU frames.
    std::shared_ptr<core::texture> texture() const;

    /// All GPU planes, in `pixel_format_desc` plane order. Empty when the frame
    /// has no GPU-side representation.
    const std::vector<std::shared_ptr<core::texture>>& textures() const;

    std::size_t width() const;

    std::size_t height() const;

    std::size_t size() const;

    const void* stream_tag() const;
    const_frame with_tag(const void* new_tag) const;

    const std::any& opaque() const;

    const class frame_geometry& geometry() const;

    bool operator==(const const_frame& other) const;
    bool operator!=(const const_frame& other) const;
    bool operator<(const const_frame& other) const;
    bool operator>(const const_frame& other) const;

    explicit operator bool() const;

  private:
    struct impl;
    std::shared_ptr<impl> impl_;
};

}} // namespace caspar::core
