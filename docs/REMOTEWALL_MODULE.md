# CasparCG `remotewall` Module — User Guide

The `remotewall` module is a native CasparCG **producer** that ingests the cloudXR tile‑wall stream
(HEVC/H.264 tiles over UDP/RTP+FEC with in‑band `SyncMeta`), decodes and composites it on the GPU, and —
on the Vulkan mixer — hands the wall to the compositor **zero‑copy** (no GPU→CPU→GPU round trip). It also
surfaces the stream's per‑frame metadata (colour space, SMPTE timecode, and full source camera) over OSC
and AMCP.

It is the native counterpart to the third‑party `RemoteWallSource.ofx` plug‑in; unlike the OFX route it
keeps the picture on the GPU and can carry the stream's metadata.

## Requirements

- Windows x64 + an NVIDIA GPU with a recent driver (NVDEC via `nvcuvid`).
- Built with `BUILD_CUDA_MODULES=ON`. Decode uses the **NVIDIA Video Codec SDK**, referenced at build
  time via the CMake cache var `REMOTEWALL_NVCODEC_SDK` (see *Building*).
- Zero‑copy uses the CUDA↔Vulkan interop, so the channel must run the **Vulkan mixer**
  (`<accelerator>vulkan</accelerator>`). On the OpenGL mixer it falls back to a CPU‑readback path.

## Playing a wall

```
PLAY 1-10 REMOTEWALL [PORT <n>] [TILES <n>] [CODEC hevc|h264]
                     [DEVICE <cuda-index>] [BINDIP <ip>] [SYNCGROUP <name>]
```

| Option | Default | Meaning |
|---|---|---|
| `PORT` | 9000 (or config) | UDP port the receiver binds |
| `TILES` | 0 (auto) | Expected tile count; 0 = detect from the first `SyncMeta` |
| `CODEC` | hevc | `hevc`, `h264`, or `av1` |
| `DEVICE` | auto | CUDA device index for decode/convert. Default: **auto‑match** the Vulkan mixer's GPU by UUID (multi‑GPU hosts); pass an index to pin it |
| `BINDIP` | all | Local interface to bind (multi‑homed hosts) |
| `SYNCGROUP` | (none) | Cross‑zone frame‑lock group. **Note:** enabling a sync group uses the receiver's host‑side reorder buffer, so playback is CPU‑readback (not zero‑copy). |

Examples:
```
PLAY 1-10 REMOTEWALL PORT 9000
PLAY 1-10 REMOTEWALL PORT 9002 CODEC h264 BINDIP 10.0.0.5
PLAY 1-10 REMOTEWALL PORT 9000 CODEC av1
PLAY 1-10 REMOTEWALL PORT 9000 SYNCGROUP wall
```

The producer outputs the wall at its native resolution; the mixer scales it to the channel. To see a 4K
wall 1:1, set the channel video mode to a matching 2160p mode. Until the stream is receiving, a dark
marker frame is shown.

## Control & metadata (`CALL`)

```
CALL 1-10 REMOTEWALL INFO      # geometry, fps, timecode, colorspace, frame index, FEC/drops
CALL 1-10 REMOTEWALL CAMERA    # camera intrinsics/fov/frustum + the 4x4 projection matrix
CALL 1-10 REMOTEWALL SET port <n> | syncgroup <name> | bindip <ip>   # live reconfigure (rebinds)
```

The same metadata is published continuously as **monitor state** (and therefore over OSC) under the
`remotewall/…` keys of the layer's producer, including:

- `remotewall/wall`, `/grid`, `/fps`, `/colorspace`, `/frame`, `/timecode`, `/fec-recovered`, `/drops`
- `remotewall/bit-depth` (8 or 16), `/device` (CUDA index in use)
- `remotewall/camera/camtoworld`, `/worldtocam`, `/proj` (4×4 matrices), `/intrinsics` (fx,fy,cx,cy),
  `/fov`, `/frustum` (l,r,b,t,near,far)

This lets a client (e.g. the 360‑client) drive camera tracking / previz frustums / projection
calibration directly from the live stream. (A synthetic test pattern carries no timecode/colour/camera
SEI; real Unreal senders populate them.)

## Colour / HDR

The stream's `colorSpace` tag (`BT709`, `BT2020/PQ`, `BT2020/HLG`, …) is mapped to the frame's
colour‑space / transfer so the mixer applies the correct colour handling and tone‑mapping.

**Bit depth is auto‑detected** from the NVDEC surface: 8‑bit 4:2:0 decodes to RGBA8, while a 10/12‑bit
(P016) surface decodes to a **16‑bit RGBA** composite and the producer allocates a `R16G16B16A16`
exportable texture (true HDR, zero‑copy on the Vulkan mixer). For 10‑bit the YUV→RGB matrix follows
the colour tag (BT.2020 for HDR walls). `CALL … REMOTEWALL INFO` and `remotewall/bit-depth` report the
active depth. (10‑bit is validated against an HDR sender; the demo test pattern is 8‑bit.)

## Configuration (`casparcg.config`)

```xml
<configuration>
  <remotewall>
    <listen-port>9000</listen-port>
    <buffer-frames>4</buffer-frames>
    <cuda-device>0</cuda-device>
  </remotewall>
</configuration>
```
These are defaults; per‑`PLAY` options override them.

## Building

The receiver + cloudXR wire/SEI headers are vendored under `src/modules/remotewall/vendored/`
(permissive). The NVIDIA Video Codec SDK is **referenced, not vendored** — point CMake at it:

```
-DREMOTEWALL_NVCODEC_SDK="<path>/Video_Codec_SDK_13.0.37"
```
(the default in `CMakeLists.txt` may already match your checkout). CMake compiles `NvDecoder` +
`ColorSpace.cu` in place and links `nvcuvid`.

## Mixer support & limitations

- **Vulkan mixer:** zero‑copy (CUDA→VK exportable texture, no readback), 8‑bit or 16‑bit. *Primary path.*
- **OpenGL mixer:** CPU‑readback fallback (correct, not zero‑copy). Zero‑copy on GL is deferred: it needs
  a CUDA‑registerable GL texture exposed by `ogl::device` plus an external‑texture input path in the GL
  `image_mixer` and cross‑thread GL/CUDA context sharing with the async writer — the CPU fallback is
  already correct, so this is intentionally out of scope for now.
- **Sync groups:** CPU‑readback (host‑side reorder buffer).
- **Multi‑GPU:** decode auto‑matches the Vulkan mixer's GPU by UUID; override with `DEVICE`.
- **Codecs:** HEVC, H.264, AV1 (AV1 metadata rides an ITU‑T T.35 metadata OBU).
- 8‑bit and 10/12‑bit HDR supported; OpenGL zero‑copy is the remaining follow‑up.
