# Vulkan Screen Consumer — Implementation Plan (Draft)

> **Status:** Prototype / Planning  
> **Date:** 2026-06-08  
> **Replaces:** OGL-based `screen_consumer` when VK mixer is active  
> **Prerequisite:** Vulkan mixer operational on target platform

---

## Motivation

The current screen consumer renders via OpenGL, even when the mixer is Vulkan. The VK→GL interop path (`GL_EXT_memory_object`) achieves zero-copy but requires:

- A full GL context + GLEW extension loading
- Platform-specific handle import (`DuplicateHandle` / `dup()`)
- GL_EXT_semaphore for cross-API timeline sync
- Fallback PBO path when extensions are unavailable
- EGL on Linux (complex, driver-dependent)

A native Vulkan screen consumer eliminates all GL dependencies for the preview path, simplifies synchronization (same API), and reuses the existing `vulkan_output` infrastructure.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  VK Mixer                                                     │
│  ┌──────────────────────┐                                     │
│  │ Render attachment     │──── exportable VkDeviceMemory       │
│  │ (RGBA8/16, external)  │──── timeline semaphore (signaled)   │
│  └──────────────────────┘                                     │
└──────────────────────────────────────────────────────────────┘
          │
          │  const_frame.texture() → texture_wrapper
          ▼
┌──────────────────────────────────────────────────────────────┐
│  vk_screen_consumer                                           │
│                                                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │ Import VK   │───▶│ Blit/Render pass │───▶│ Swapchain   │ │
│  │ texture     │    │ (tone-map shader │    │ present     │ │
│  │ (cached)    │    │  + downscale)    │    │ (MAILBOX)   │ │
│  └─────────────┘    └──────────────────┘    └─────────────┘ │
│                                                               │
│  Window: GLFW VkSurfaceKHR (resizable, borderless optional)   │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Window Backend: GLFW

| Option | Pros | Cons |
|--------|------|------|
| GLFW | Cross-platform, VkSurfaceKHR native, resize callbacks, input events, lightweight | New dependency |
| SDL2 | Also cross-platform | Heavier, SDL_Vulkan_* API less idiomatic |
| Win32 raw | No dependency | Windows-only, lots of boilerplate |
| vk_display (current vulkan_output) | No window manager | Not windowed, exclusive display only |

**Decision:** GLFW — minimal, well-tested, gives us `glfwCreateWindowSurface()`. Already Vulkan-native. Single dependency for both Win32 and Linux/Wayland.

### 2. Rendering Strategy: Fragment Shader Pass

Rather than a raw `vkCmdBlitImage` (no color processing), use a simple render pass with a fullscreen triangle and a fragment shader that provides:

- Tone mapping (same operators as current screen.frag)
- EOTF decode + OETF re-encode  
- Key-only mode
- DataVideo color space output (if needed for preview monitors)

This matches current screen consumer feature parity.

### 3. Frame Sync: Fence-Only (No Timeline Semaphore Import)

The `texture_wrapper` already provides `ensure_render_complete()` (CPU fence wait). Since the screen consumer is **non-critical preview** that drops frames freely, we don't need GPU-side semaphore sync. Just:

1. Call `ensure_render_complete()` (returns immediately if mixer already finished)
2. Import VK memory (cached)
3. Blit/render
4. Present (MAILBOX — non-blocking)

This avoids the entire semaphore import/export dance.

### 4. Swapchain Format

| Channel Transfer | Swapchain Format | Notes |
|-----------------|------------------|-------|
| SDR (rec709) | `VK_FORMAT_B8G8R8A8_SRGB` | Most compatible |
| HDR (PQ/HLG) | `VK_FORMAT_A2B10G10R10_UNORM_PACK32` | If surface supports, else tonemap to SDR |
| Linear/Preview | `VK_FORMAT_B8G8R8A8_UNORM` | When shader handles gamma |

Default: `B8G8R8A8_UNORM` with shader-applied sRGB OETF (matches current behavior).

---

## File Structure

```
src/modules/vk_screen/
├── CMakeLists.txt
├── vk_screen.cpp                    # Module init + registration
├── vk_screen.h
├── consumer/
│   ├── vk_screen_consumer.cpp       # Main consumer implementation
│   └── vk_screen_consumer.h         # Factory functions
├── shaders/
│   ├── screen_preview.vert          # Fullscreen triangle vertex shader
│   ├── screen_preview.frag          # Tone-map + colorspace fragment shader
│   ├── screen_preview_vert_spv.h    # Embedded SPIR-V (generated)
│   └── screen_preview_frag_spv.h    # Embedded SPIR-V (generated)
└── util/
    ├── swapchain.h                  # Swapchain lifecycle management
    ├── swapchain.cpp
    ├── vk_texture_cache.h           # LRU cache for imported VkImages
    └── vk_texture_cache.cpp
```

---

## Implementation Phases

### Phase 1: Minimal Windowed Present (No Shader)

**Goal:** VK swapchain in a resizable GLFW window, `vkCmdBlitImage` from imported mixer texture.

**Tasks:**
- [ ] Add GLFW dependency to CMake (FetchContent or find_package)
- [ ] `vk_screen_consumer` class implementing `core::frame_consumer`
- [ ] GLFW window creation + VkSurfaceKHR
- [ ] Swapchain creation (MAILBOX, B8G8R8A8_UNORM, recreate on resize)
- [ ] VK texture import from `texture_wrapper::export_native_handle()` (same pattern as vulkan_output)
- [ ] `vkCmdBlitImage` (source → swapchain image, with scaling)
- [ ] Frame drop policy: capacity-1 `tbb::concurrent_bounded_queue`
- [ ] Present thread (dedicated, frame-rate decoupled)
- [ ] `needs_cpu_frame_data() = false`
- [ ] AMCP: `ADD 1 VK_SCREEN`
- [ ] XML: `<consumers><vk-screen/></consumers>`

**Estimated complexity:** ~600 LOC

### Phase 2: Fragment Shader Pipeline

**Goal:** Replace raw blit with render pass + fragment shader for tone mapping and color space.

**Tasks:**
- [ ] Port `screen.frag` to Vulkan GLSL 450 (push constants instead of uniforms)
- [ ] Compile to SPIR-V via `glslc`, embed via `bin2c`
- [ ] Fullscreen triangle vertex shader (no VBO needed — `gl_VertexIndex` trick)
- [ ] `VkDescriptorSet` with combined image sampler (the imported texture)
- [ ] Push constants: `tone_map_op`, `channel_transfer`, `display_peak_luminance`, `colour_space`, `key_only`
- [ ] `VkRenderPass` targeting swapchain image
- [ ] Pipeline creation (with dynamic viewport/scissor for resize)

**Estimated complexity:** ~400 LOC

### Phase 3: Window Features

**Goal:** Feature parity with current screen consumer UX.

**Tasks:**
- [ ] Borderless mode (GLFW decorated/undecorated)
- [ ] Fullscreen toggle (GLFW monitor mode switch)
- [ ] Always-on-top option
- [ ] Window title: channel name + format + fps
- [ ] Keyboard shortcuts: `F` fullscreen, `Esc` close, `T` toggle always-on-top
- [ ] DPI awareness (GLFW content scale callback → swapchain recreate)
- [ ] Multi-monitor placement (config: `<x>`, `<y>`, `<screen-index>`)
- [ ] Graceful close → removes consumer from channel

**Estimated complexity:** ~300 LOC

### Phase 4: Diagnostics & Polish

**Tasks:**
- [ ] `diagnostics::graph` integration (frame time, present time, drops)
- [ ] TDR recovery: `VK_ERROR_DEVICE_LOST` → recreate swapchain + device
- [ ] OGL mixer fallback: when mixer is OGL, receive CPU frame data + `vkCmdCopyBufferToImage` (staging buffer)
- [ ] `monitor::state` reporting (resolution, present mode, fps, drops)
- [ ] Config options: `<tone-map>`, `<display-peak-nits>`, `<key-only>`

**Estimated complexity:** ~250 LOC

---

## Core Class Design

```cpp
namespace caspar::vk_screen {

struct configuration
{
    int         screen_index  = 0;       // Monitor placement
    int         x             = 0;
    int         y             = 0;
    int         width         = 0;       // 0 = auto (half channel res)
    int         height        = 0;
    bool        borderless    = false;
    bool        always_on_top = false;
    bool        key_only      = false;
    int         tone_map_op   = 0;       // 0=none, 1=reinhard, 2=aces, 7=hlg_ootf
    float       display_peak  = 100.0f;  // SDR default
    std::wstring name         = L"CasparCG Preview";
};

class vk_screen_consumer final : public core::frame_consumer
{
public:
    // --- frame_consumer interface ---
    void initialize(const core::video_format_desc&,
                    const core::channel_info&, int port_index) override;
    std::future<bool> send(core::video_field, core::const_frame) override;
    std::wstring print() const override;
    std::wstring name() const override { return L"vk-screen"; }
    int  index() const override { return 700 + config_.screen_index; }
    bool has_synchronization_clock() const override { return false; }
    bool needs_cpu_frame_data() const override { return !use_vulkan_; }

private:
    configuration config_;
    core::video_format_desc format_desc_;
    bool use_vulkan_ = false;

    // Window
    GLFWwindow* window_ = nullptr;

    // Vulkan objects
    VkInstance       instance_   = VK_NULL_HANDLE;
    VkDevice         device_     = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev_   = VK_NULL_HANDLE;
    VkQueue          queue_      = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_    = VK_NULL_HANDLE;
    VkCommandPool    cmd_pool_   = VK_NULL_HANDLE;

    // Swapchain
    swapchain swapchain_;

    // Texture import cache (LRU, keyed by native handle)
    vk_texture_cache texture_cache_;

    // Render pipeline (Phase 2)
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkRenderPass     render_pass_    = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_      = VK_NULL_HANDLE;
    VkSampler        sampler_        = VK_NULL_HANDLE;

    // Presentation
    static constexpr int kMaxFramesInFlight = 2;
    struct frame_sync {
        VkSemaphore image_available;
        VkSemaphore render_finished;
        VkFence     in_flight;
        VkCommandBuffer cmd;
    };
    std::array<frame_sync, kMaxFramesInFlight> frames_;
    uint32_t current_frame_ = 0;

    // Threading
    tbb::concurrent_bounded_queue<core::const_frame> frame_buffer_;
    std::thread present_thread_;
    std::atomic<bool> running_{false};

    // Diagnostics
    spl::shared_ptr<diagnostics::graph> graph_;

    void present_loop();
    void present_frame(const core::const_frame& frame);
    void recreate_swapchain();
    VkImage import_texture(const std::shared_ptr<core::texture>& tex);
};

} // namespace caspar::vk_screen
```

---

## Swapchain Helper

```cpp
struct swapchain
{
    VkSwapchainKHR   handle = VK_NULL_HANDLE;
    VkFormat         format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D       extent{};
    std::vector<VkImage>     images;
    std::vector<VkImageView> views;
    std::vector<VkFramebuffer> framebuffers;  // Phase 2

    void create(VkDevice dev, VkPhysicalDevice phys, VkSurfaceKHR surface,
                uint32_t width, uint32_t height, VkSwapchainKHR old = VK_NULL_HANDLE);
    void destroy(VkDevice dev);
    bool needs_recreate = false;  // Set by resize callback
};
```

---

## CMakeLists.txt (Draft)

```cmake
# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Shader compilation
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(GLSLC)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/screen_preview_vert.spv"
        COMMAND ${GLSLC} -fshader-stage=vert
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/screen_preview.vert"
                -o "${CMAKE_CURRENT_BINARY_DIR}/screen_preview_vert.spv"
        DEPENDS "shaders/screen_preview.vert")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/screen_preview_frag.spv"
        COMMAND ${GLSLC} -fshader-stage=frag
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/screen_preview.frag"
                -o "${CMAKE_CURRENT_BINARY_DIR}/screen_preview_frag.spv"
        DEPENDS "shaders/screen_preview.frag")
    bin2c("${CMAKE_CURRENT_BINARY_DIR}/screen_preview_vert.spv"
          "screen_preview_vert_spv.h" "caspar::vk_screen" "vert_spv")
    bin2c("${CMAKE_CURRENT_BINARY_DIR}/screen_preview_frag.spv"
          "screen_preview_frag_spv.h" "caspar::vk_screen" "frag_spv")
endif()

set(SOURCES
    vk_screen.cpp
    vk_screen.h
    consumer/vk_screen_consumer.cpp
    consumer/vk_screen_consumer.h
    util/swapchain.cpp
    util/swapchain.h
    util/vk_texture_cache.cpp
    util/vk_texture_cache.h
)

casparcg_add_module_project(vk_screen
    SOURCES ${SOURCES}
    INIT_FUNCTION "vk_screen::init"
)

target_link_libraries(vk_screen PRIVATE
    Vulkan::Vulkan
    glfw
    accelerator
)

# Platform-specific
if(WIN32)
    target_link_libraries(vk_screen PRIVATE dwmapi)
endif()
```

---

## Present Loop Pseudocode

```cpp
void vk_screen_consumer::present_loop()
{
    while (running_) {
        core::const_frame frame;
        if (!frame_buffer_.try_pop(frame)) {
            // No frame available — poll GLFW events and retry
            glfwPollEvents();
            if (glfwWindowShouldClose(window_)) { running_ = false; break; }
            std::this_thread::sleep_for(1ms);
            continue;
        }

        if (swapchain_.needs_recreate) {
            recreate_swapchain();
            swapchain_.needs_recreate = false;
        }

        present_frame(frame);
        glfwPollEvents();

        if (glfwWindowShouldClose(window_))
            running_ = false;
    }
}

void vk_screen_consumer::present_frame(const core::const_frame& frame)
{
    auto& sync = frames_[current_frame_];

    // Wait for this frame slot to be available
    vkWaitForFences(device_, 1, &sync.in_flight, VK_TRUE, UINT64_MAX);

    // Acquire swapchain image
    uint32_t image_index;
    auto result = vkAcquireNextImageKHR(device_, swapchain_.handle,
                                         UINT64_MAX, sync.image_available,
                                         VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        return;
    }

    // Import mixer texture (cached by handle)
    auto tex = frame.texture();
    tex->ensure_render_complete();  // CPU fence — mixer done rendering
    VkImage src_image = import_texture(tex);

    // Record commands
    vkResetFences(device_, 1, &sync.in_flight);
    vkResetCommandBuffer(sync.cmd, 0);
    vkBeginCommandBuffer(sync.cmd, &begin_info);

    // Transition src: GENERAL → TRANSFER_SRC
    // Transition dst: UNDEFINED → TRANSFER_DST
    // vkCmdBlitImage(src → swapchain[image_index])  — Phase 1
    // OR: bind descriptor, vkCmdBeginRenderPass, vkCmdDraw(3) — Phase 2
    // Transition dst: TRANSFER_DST → PRESENT_SRC

    vkEndCommandBuffer(sync.cmd);

    // Submit
    VkSubmitInfo submit{};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &sync.image_available;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &sync.cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &sync.render_finished;
    vkQueueSubmit(queue_, 1, &submit, sync.in_flight);

    // Present
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &sync.render_finished;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_.handle;
    present.pImageIndices = &image_index;
    vkQueuePresentKHR(queue_, &present);

    current_frame_ = (current_frame_ + 1) % kMaxFramesInFlight;
}
```

---

## Texture Import Cache

```cpp
class vk_texture_cache
{
    struct entry {
        platform::native_handle_t handle;   // Duplicated handle (owned)
        VkDeviceMemory memory;
        VkImage        image;
        VkDeviceSize   alloc_size;
        int            width, height;
    };

    static constexpr size_t kMaxEntries = 4;  // Ring buffer size from mixer
    std::deque<entry> entries_;
    VkDevice device_;

public:
    // Returns cached VkImage or imports new one
    VkImage get_or_import(const std::shared_ptr<core::texture>& tex);
    void    clear();
};
```

Cache is keyed by `(native_handle, alloc_size, width, height)`. The mixer uses a fixed ring of 3 attachments, so 4 cache entries guarantees no re-import.

---

## Fragment Shader (Phase 2)

```glsl
#version 450

layout(set = 0, binding = 0) uniform sampler2D src_texture;

layout(push_constant) uniform PushConstants {
    int   tone_map_op;          // 0=none, 1=reinhard, 2=aces, 3=aces_rrt, 7=hlg
    int   channel_transfer;     // 2=SDR, 3=PQ, 4=HLG, 5=linear
    float display_peak_nits;
    int   key_only;
    int   colour_space;         // 0=RGB, 1=DTV full, 2=DTV limited
} pc;

layout(location = 0) out vec4 out_color;

// Fullscreen triangle UVs from gl_VertexIndex
vec2 get_uv() {
    // Vertex 0: (-1,-1), Vertex 1: (3,-1), Vertex 2: (-1,3)
    vec2 uv;
    uv.x = (gl_VertexIndex == 1) ? 2.0 : 0.0;
    uv.y = (gl_VertexIndex == 2) ? 2.0 : 0.0;
    return uv;
}

// ... (same EOTF/tone-map/OETF functions as screen.frag, ported to push constants) ...

void main() {
    vec2 uv = get_uv();
    vec4 color = texture(src_texture, uv);

    if (pc.key_only != 0) {
        out_color = vec4(color.aaa, 1.0);
        return;
    }

    if (pc.tone_map_op > 0)
        color.rgb = apply_display_tone_map(color.rgb);

    out_color = color;
}
```

---

## Configuration (XML)

```xml
<channel>
  <consumers>
    <vk-screen>
      <device>1</device>              <!-- screen index / monitor -->
      <windowed>true</windowed>
      <borderless>false</borderless>
      <always-on-top>false</always-on-top>
      <width>960</width>              <!-- 0 = auto -->
      <height>540</height>
      <x>100</x>
      <y>100</y>
      <key-only>false</key-only>
      <tone-map>none</tone-map>       <!-- none|reinhard|aces|aces-rrt|hlg -->
      <display-peak-nits>100</display-peak-nits>
    </vk-screen>
  </consumers>
</channel>
```

---

## AMCP Commands

```
ADD 1 VK_SCREEN                     # Default preview window
ADD 1 VK_SCREEN 1                   # On monitor index 1
ADD 1 VK_SCREEN 0 BORDERLESS        # Borderless on primary
REMOVE 1 VK_SCREEN
```

---

## Reuse from vulkan_output

| Component | Reuse | Notes |
|-----------|-------|-------|
| `vk_texture_cache` pattern | Copy + simplify | vulkan_output uses LRU-8; screen needs LRU-4 |
| `platform_config.h` | Direct include | Handle types, native_handle_t |
| `texture_wrapper` interface | Direct use | `export_native_handle()`, `ensure_render_complete()` |
| SPIR-V compilation CMake | Adapt | Same `glslc` + `bin2c` pattern |
| TDR recovery pattern | Adapt (Phase 4) | Simpler — just recreate swapchain + reimport |
| VkDevice creation | **Don't reuse** | Use GLFW's required extensions + vk-bootstrap directly |
| Present barrier / Quadro Sync | **Skip** | Not needed for preview |
| CUDA peer transfer | **Skip** | Preview doesn't need cross-GPU |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GLFW + Vulkan loader conflicts | Low | GLFW uses its own loader; we pass our instance |
| Swapchain recreation stutter | Medium | Pre-create double-buffered; resize only on GLFW idle |
| OGL mixer fallback complexity | Medium | Phase 4 — just use CPU staging; preview tolerates 1 copy |
| Driver differences (AMD/Intel) | Medium | Stick to Vulkan 1.1 core features; no exotic extensions |
| GLFW on Wayland (Linux) | Low | GLFW 3.4 has native Wayland; VkSurfaceKHR works |

---

## Performance Expectations

| Metric | Current (GL interop) | VK Screen (Phase 1) | Difference |
|--------|---------------------|---------------------|------------|
| Frame latency | ~0.3ms (blit + present) | ~0.2ms (single API) | -0.1ms |
| CPU overhead | ~50µs (GL dispatch + ext calls) | ~20µs (pre-recorded cmd) | -30µs |
| VRAM | +4MB (GL context) | +0 (shared VkDevice possible) | -4MB |
| Startup | ~80ms (GL init + ext probe) | ~40ms (swapchain create) | -40ms |
| Drop behavior | Identical | Identical | 0 |

**Net:** Marginal latency improvement. Real value is architectural simplification and Linux portability without EGL.

---

## Open Questions

1. **Share VkDevice with mixer or create independent?**  
   Sharing avoids memory import entirely (same device = same VkImage directly readable). But couples consumer lifetime to mixer device. Recommendation: **share** — the mixer device already has the graphics queue; consumer can acquire a second queue or share with mutex.

2. **GLFW vs headless + platform window?**  
   GLFW is simpler but adds a dependency. Alternative: Use the same `vkCreateWin32SurfaceKHR`/`vkCreateXlibSurfaceKHR` direct approach as vulkan_output. Decision deferred to prototyping.

3. **Coexist with OGL screen consumer or replace?**  
   Recommend: **coexist** — register as `vk-screen` alongside existing `screen`. Users choose based on mixer type. Auto-select in future when OGL mixer is deprecated.

4. **Phase 2 shader: port screen.frag or strip down?**  
   The current screen.frag has DataVideo colorspace output (analog monitor feature). For VK screen preview, we could strip this to just tone-map + OETF. Decision: port fully for feature parity, but DataVideo path is low priority.
