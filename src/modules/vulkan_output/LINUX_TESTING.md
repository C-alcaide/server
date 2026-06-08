# Vulkan Output — Linux Testing Checklist

## Prerequisites

- NVIDIA GPU with proprietary driver ≥ 535 (for VK_KHR_display and EGL device)
- Vulkan SDK 1.4+ installed (`vulkaninfo` works)
- Physical display connected directly to GPU (not through USB or DP MST hub)
- CasparCG built with `vulkan_output` module (verify in cmake output)

## 1. Basic Compilation (CI — no GPU needed)

```bash
docker build --target build-casparcg \
  -f tools/linux/Dockerfile.vulkan-output .
```

- [ ] Build completes with no errors
- [ ] `libvulkan_output.a` is produced
- [ ] No undefined symbols at link time

## 2. Module Loading

```bash
./casparcg --log-level debug
```

- [ ] Log shows: `[vulkan_output] Module initialized`
- [ ] No EGL errors on startup
- [ ] `vulkaninfo --summary` shows the expected GPU(s)

## 3. Single-GPU Output (VK_KHR_display)

```
ADD 1 vulkan_output 1
PLAY 1-1 DECKLINK_TEST_PATTERN
```

- [ ] Display surface created via VK_KHR_display (log: `create_display_surface`)
- [ ] Physical output shows test pattern at correct resolution
- [ ] Correct refresh rate (verify with `VK_KHR_display_properties`)
- [ ] No tearing or flickering
- [ ] `REMOVE 1` cleanly releases the display

## 4. OGL → VK Interop (EGL shared context)

```
ADD 1 vulkan_output 1
PLAY 1-1 AMB LOOP
```

- [ ] Log shows: `[interop_context] Created shared EGL context`
- [ ] Log shows: `[shared_texture_pool] GL_EXT_memory_object_fd` extensions loaded
- [ ] No `glImportMemoryFdEXT` errors
- [ ] fd handles properly consumed (no fd leaks — check `/proc/<pid>/fd` count)
- [ ] Smooth playback without GL errors

## 5. Multi-GPU — GPU Affinity (EGL device enumeration)

Requires 2+ NVIDIA GPUs. CasparCG mixer on GPU 0, output on GPU 1.

```
# In casparcg.config, set vulkan output to GPU index 1
ADD 1 vulkan_output 1 GPU 1
PLAY 1-1 AMB LOOP
```

- [ ] Log shows: `[gpu_affinity] GPU 1: /dev/dri/renderD129` (or similar)
- [ ] Separate EGL context created on GPU 1
- [ ] Upload texture + PBOs allocated on affinity context
- [ ] LUID/UUID matches between affinity GL context and VK physical device

## 6. Multi-GPU — CUDA Peer Transfer

Requires 2+ NVIDIA GPUs with peer access (NVLink or same IOMMU group).

- [ ] Log shows: `[cuda_peer_transfer] Initializing peer transfer: device 0 → device 1`
- [ ] `cudaDeviceCanAccessPeer` returns true (or graceful fallback to PBO)
- [ ] Frame data transfers without CPU staging
- [ ] No CUDA errors in log
- [ ] Performance: peer DMA path faster than PBO fallback

## 7. HDR Output

Requires HDR-capable display connected.

```
ADD 1 vulkan_output 1 HDR PQ
PLAY 1-1 HDR_TEST_CLIP
```

- [ ] `VK_EXT_hdr_metadata` set on swapchain
- [ ] Display switches to HDR mode (EOTF = PQ)
- [ ] Correct color volume metadata sent
- [ ] HLG mode also works: `ADD 1 vulkan_output 1 HDR HLG`

## 8. Display Enumeration

```
# AMCP command to list available displays
INFO VULKAN_DISPLAYS
```

- [ ] All connected physical displays enumerated
- [ ] Display names, resolutions, refresh rates reported correctly
- [ ] DRM connector names match (`xrandr` or `modetest` output)

## 9. Stress / Stability

- [ ] Run continuous playback for 1+ hours — no fd leaks, memory growth, or crashes
- [ ] Hot-remove and re-add consumer multiple times without leaks
- [ ] `RESTART` command cleanly cycles the vulkan output
- [ ] Process shutdown (SIGTERM) releases all VK/EGL/DRM resources

## 10. Error Handling

- [ ] Unplugging display mid-playback: logged gracefully, no crash
- [ ] Invalid GPU index: clear error message, fallback or abort
- [ ] VK_ERROR_DEVICE_LOST: TDR watchdog fires, process terminates cleanly
- [ ] Missing Vulkan driver: module reports unavailable at init

## Performance Baseline

On a typical system (RTX 4000 Ada, 1080p60):

| Metric | Target |
|--------|--------|
| OGL→VK interop latency | < 0.5 ms/frame |
| Present-to-scanout | < 1 vsync |
| CPU usage (steady state) | < 5% one core |
| GPU memory overhead | < 100 MB |
| fd count growth over 1h | 0 (no leaks) |
