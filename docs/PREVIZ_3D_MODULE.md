# 3D Previz Module

The previz module adds real-time 3D venue visualisation to CasparCG VP. It renders LED screen surfaces with live channel textures inside a 3D scene, enabling accurate spatial previews of virtual production stages, concert LED walls, and broadcast environments — directly inside CasparCG without external tools.

The system is split between **server-side rendering** (C++/OpenGL) for production output and a **client-side fat application** (Python/PyQt6) for interactive scene authoring and LED wall management.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline](#pipeline)
4. [Server-Side Implementation](#server-side-implementation)
5. [AMCP Command Reference](#amcp-command-reference)
6. [Client-Side Implementation (360 Client)](#client-side-implementation-360-client)
7. [LED Wall Manager](#led-wall-manager)
8. [Scene Objects](#scene-objects)
9. [3D Viewport Controls](#3d-viewport-controls)
10. [Configuration](#configuration)
11. [Auto-Projection](#auto-projection)
12. [Session Management](#session-management)
13. [Operation Guide](#operation-guide)
14. [Best Practices](#best-practices)
15. [Known Limitations](#known-limitations)

---

## Overview

CasparCG VP's previz system provides:

- **3D venue simulation** — place LED screens, reference objects, and venue geometry in a 3D space
- **Live channel texture mapping** — bind CasparCG channel outputs to screen surfaces in real time
- **LED panel database** — built-in specs for ROE, Absen, Unilumin, Samsung, Sony, INFiLED, Desay panels
- **Procedural screen creation** — define screens by tile count, pixel pitch, and physical dimensions
- **Curved screen support** — cylindrical curvature with configurable radius and arc angle
- **Auto-projection** — derive `MIXER PROJECTION` parameters from virtual camera position relative to screens
- **Interactive authoring** — full 3D viewport with orbit/pan/zoom, drag-to-move, and reference objects
- **Import support** — load external .glb and .obj venue models
- **Session persistence** — auto-save/restore of complete scene state

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CasparCG 360 Client                        │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │ Scene Tree   │  │ Properties    │  │   PrevizViewport      │ │
│  │  & Outliner  │  │   Panel       │  │   (QOpenGLWidget)     │ │
│  │             │  │ - Transform   │  │ - OpenGL 3.3 Core     │ │
│  │ 🏠 Venue    │  │ - Screen cfg  │  │ - Orbit/Pan/Zoom      │ │
│  │ 🖥 Screens  │  │ - Channel map │  │ - Node picking        │ │
│  │ 🧍 Refs     │  │ - Mesh info   │  │ - Drag-to-move        │ │
│  │ ⬜ Prims    │  │               │  │ - Screen gizmos       │ │
│  └──────┬──────┘  └───────┬───────┘  │ - Bezel lines         │ │
│         │                 │          │ - Measurements         │ │
│         └────────┬────────┘          └───────────┬───────────┘ │
│                  │ scene_changed                  │             │
│                  ▼                                │             │
│         ┌────────────────┐                        │             │
│         │   StageTab     │◄───────────────────────┘             │
│         │  (Controller)  │  camera_changed / node_moved         │
│         └───────┬────────┘                                      │
│                 │ AMCP (TCP port 5250)                           │
└─────────────────┼───────────────────────────────────────────────┘
                  │
    ┌─────────────▼──────────────────────────────────────────────┐
    │                   CasparCG VP Server                        │
    │                                                             │
    │  ┌──────────────────────────────────────────────────┐       │
    │  │              AMCP Protocol Layer                  │       │
    │  │  PREVIZ SCENE / MAP / CAMERA / SCREEN / ...      │       │
    │  └───────────────────────┬──────────────────────────┘       │
    │                          │                                   │
    │  ┌───────────────────────▼──────────────────────────┐       │
    │  │            previz_renderer (OpenGL 4.5)           │       │
    │  │                                                   │       │
    │  │  ┌─────────────┐  ┌────────────────────────────┐ │       │
    │  │  │ previz_scene │  │  Channel Texture Binding   │ │       │
    │  │  │  - meshes    │  │  CH1 output ──► "MainWall" │ │       │
    │  │  │  - camera    │  │  CH2 output ──► "Floor"    │ │       │
    │  │  │  - screens   │  │  CH3 output ──► "Ceiling"  │ │       │
    │  │  └─────────────┘  └────────────────────────────┘ │       │
    │  │                                                   │       │
    │  │  previz.vert + previz.frag                        │       │
    │  │  (Emissive screens / Diffuse geometry)            │       │
    │  └───────────────────────────────────────────────────┘       │
    │                          │                                   │
    │                          ▼                                   │
    │  ┌──────────────────────────────────────────────────┐       │
    │  │  Output: DeckLink / Vulkan / Spout / NDI         │       │
    │  └──────────────────────────────────────────────────┘       │
    └─────────────────────────────────────────────────────────────┘
```

---

## Pipeline

The rendering pipeline for a previz channel follows this order:

1. **Scene setup** — Meshes loaded from glTF/OBJ or created procedurally via AMCP SCREEN commands
2. **Channel texture binding** — Each channel's `image_mixer::render()` output (already an OpenGL texture) is bound to named screen meshes. This is **zero-copy** — no readback or upload.
3. **Camera update** — Virtual camera positioned via AMCP or client sync
4. **Render pass** — Separate OpenGL pass with perspective projection and depth testing:
   - LED screens: **emissive** — channel texture sampled directly, no lighting applied (screens are light sources)
   - Non-screen geometry: **diffuse** — ambient + N·L directional lighting with glTF base color
5. **Post-processing** — Standard CasparCG output pipeline (consumers, Spout, etc.)

### Data Flow (Client → Server)

```
User edits scene ──► StageTab controller
                          │
                    ┌─────┴──────────────────────────────────┐
                    │ Live Link mode (real-time sync)         │
                    │                                         │
                    │  PREVIZ 4 SCREEN ADD "Main" FLAT 5 3    │
                    │  PREVIZ 4 SCREEN "Main" POSITION 0 2 0  │
                    │  PREVIZ 4 SCREEN "Main" CHANNEL 1       │
                    │  PREVIZ 4 CAMERA 0 1.7 8 180 0 0 60     │
                    └─────────────────────────────────────────┘
                          │ TCP/AMCP
                          ▼
                    CasparCG Server processes commands,
                    generates geometry, binds textures,
                    renders next frame
```

---

## Server-Side Implementation

### Core Data Structures

Located in `src/accelerator/ogl/image/previz_scene.h`:

| Struct | Purpose |
|--------|---------|
| `previz_vertex` | Vertex data: position (xyz), normal (xyz), texcoord (uv) |
| `previz_mesh` | Named mesh with vertex buffer, base colour, screen flag, visibility |
| `previz_camera` | Camera state: position, rotation (yaw/pitch/roll), FOV, clip planes |
| `screen_meta` | Procedural screen metadata: dimensions, curvature, resolution, channel mapping |
| `previz_scene` | Top-level container: meshes, camera, screen registry, display toggles |

### Renderer

Located in `src/accelerator/ogl/image/previz_renderer.cpp`:

- `set_scene()` — Load glTF/OBJ, parse meshes, upload GPU buffers
- `set_camera()` — Update camera; triggers `update_projections()` for auto-projection
- `add_screen()` / `remove_screen()` — Procedural screen management with mesh generation
- `render()` — Full 3D render pass: grid → geometry → screen textures → gizmos
- `compute_frustum()` — Derives MIXER PROJECTION parameters from virtual camera vs screen geometry

### Shaders

**Vertex shader** (`previz.vert`):
```glsl
uniform mat4 u_mvp;       // Model-View-Projection matrix
uniform mat4 u_model;     // Model matrix (for world-space normals)

in vec3 a_Position;
in vec3 a_Normal;
in vec2 a_TexCoord;

out vec3 v_WorldPos;
out vec3 v_Normal;
out vec2 v_TexCoord;
```

**Fragment shader** (`previz.frag`):
```glsl
uniform bool u_is_screen;       // LED screen → emissive mode
uniform bool u_has_texture;     // Channel texture bound?
uniform sampler2D u_texture;    // Channel output texture
uniform vec3 u_base_color;      // Fallback material colour
uniform vec3 u_light_dir;       // Directional light direction
uniform float u_ambient;        // Ambient intensity (0.25 default)
```

LED screens sample the channel texture directly with no lighting applied — they represent physical light sources. All other geometry receives basic diffuse shading (ambient + N·L) using the mesh's base colour.

---

## AMCP Command Reference

All commands are prefixed with `PREVIZ <channel>`.

### Scene Management

| Command | Description |
|---------|-------------|
| `PREVIZ <ch> SCENE <path.glb\|obj>` | Load venue model from file |
| `PREVIZ <ch> SCENE NONE` | Clear loaded scene |
| `PREVIZ <ch> SCENE` | Query current scene path |
| `PREVIZ <ch> INFO` | Full scene status: active, path, mesh list, channel mappings |

### Channel Mapping

| Command | Description |
|---------|-------------|
| `PREVIZ <ch> MAP <mesh_name> <target_channel>` | Bind mesh to channel output texture |
| `PREVIZ <ch> UNMAP <mesh_name>` | Revert mesh to default material |

### Camera Control

| Command | Description |
|---------|-------------|
| `PREVIZ <ch> CAMERA <x> <y> <z> <yaw> <pitch> <roll> <fov>` | Set camera position and orientation |
| `PREVIZ <ch> CAMERA RESET` | Reset to default (0, 1.5, 5, 0, 0, 0, 60) |
| `PREVIZ <ch> CAMERA` | Query current camera state |
| `PREVIZ <ch> PRESET SAVE <name>` | Save camera state as named preset |
| `PREVIZ <ch> PRESET RECALL <name>` | Restore camera from preset |
| `PREVIZ <ch> PRESET LIST` | List all saved presets |

### Procedural Screens

| Command | Description |
|---------|-------------|
| `PREVIZ <ch> SCREEN ADD <name> FLAT <w_m> <h_m>` | Create flat screen |
| `PREVIZ <ch> SCREEN ADD <name> CURVED <w_m> <h_m> <radius_m> <arc_deg>` | Create curved screen |
| `PREVIZ <ch> SCREEN <name> POSITION <x> <y> <z>` | Set screen position (metres) |
| `PREVIZ <ch> SCREEN <name> ROTATION <yaw> <pitch> <roll>` | Set screen rotation (degrees) |
| `PREVIZ <ch> SCREEN <name> RESOLUTION <w_px> <h_px>` | Override output resolution |
| `PREVIZ <ch> SCREEN <name> CHANNEL <ch_num>` | Map screen to channel output |
| `PREVIZ <ch> SCREEN <name> REMOVE` | Delete screen |
| `PREVIZ <ch> SCREEN LIST` | List all procedural screens |

### Display Toggles

| Command | Description |
|---------|-------------|
| `PREVIZ <ch> SHOW <mesh_name> [1\|0]` | Toggle mesh visibility |
| `PREVIZ <ch> GRID [1\|0]` | Show/hide reference grid |
| `PREVIZ <ch> WIREFRAME [1\|0]` | Toggle wireframe overlay |
| `PREVIZ <ch> GIZMO [1\|0]` | Show/hide gizmos and axis indicators |

---

## Client-Side Implementation (360 Client)

The CasparCG 360 Client (`casparcg-360-client`) is a Python/PyQt6 application that provides the interactive 3D authoring interface. The previz system is integrated as a **Stage tab** alongside existing Motion, Color, Shape, and Tracking tabs.

### Key Source Files

| File | Purpose |
|------|---------|
| `previz_viewport.py` | `PrevizViewport(QOpenGLWidget)` — OpenGL 3.3 Core 3D viewport |
| `previz_scene_model.py` | Data model: `PrevizScene`, `SceneNode`, `PrevizCamera`, `ScreenConfig`, `MeshData` |
| `mesh_gen.py` | Procedural mesh generators: screens, primitives, reference models, grid |
| `stage_tab.py` | `StageTab(QWidget)` — UI controller: scene tree, properties panel, toolbar |
| `led_wall_manager.py` | `ScreenBuilderDialog` — LED screen creation wizard |

### Scene Data Model

The scene is represented by `PrevizScene`, a client-side data structure that holds:

- **nodes** — A flat list of `SceneNode` objects (screens, primitives, references, venues)
- **camera** — Interactive viewport camera (orbit/pan/zoom state)
- **virtual_camera** — Separate camera used for auto-projection calculations
- **display toggles** — Grid, wireframe, labels, gizmos, virtual camera visibility
- **camera_presets** — Named viewpoints for quick recall

Each `SceneNode` carries:

| Field | Type | Description |
|-------|------|-------------|
| `uid` | `str` | Unique 8-char hex identifier |
| `name` | `str` | Display name |
| `node_type` | `NodeType` | VENUE, SCREEN, REFERENCE, PRIMITIVE, CAMERA, or GROUP |
| `pos_x/y/z` | `float` | World position in metres |
| `rot_yaw/pitch/roll` | `float` | Rotation in degrees |
| `scale` | `float` | Uniform scale factor |
| `visible` | `bool` | Visibility toggle |
| `locked` | `bool` | Prevents transform edits |
| `mesh` | `MeshData` | Vertex data (positions, normals, UVs) |
| `screen_config` | `ScreenConfig` | LED panel specs (for SCREEN nodes) |
| `source_file` | `str` | Import path (for VENUE nodes) |

### Rendering Pipeline (Client)

The `PrevizViewport` uses OpenGL 3.3 Core Profile with two shader programs:

1. **Main shader** — Per-node rendering with diffuse lighting (ambient + N·L). LED screens rendered as emissive surfaces. Interleaved VBO: position(3) + normal(3) + uv(2) = 32 bytes/vertex.

2. **Grid/gizmo shader** — Position-only vertices (12 bytes/vertex) with uniform colour. Used for ground grid, screen outlines (green), bezel/tile boundary lines (dim orange), virtual camera frustum (yellow body + green up indicator), and measurement lines.

Frame render order:
```
1. Clear (dark grey background)
2. Ground grid
3. Scene nodes (VAO per node, diffuse or emissive)
4. Screen outline gizmos + bezel lines
5. Virtual camera frustum gizmo
6. Measurement overlays
7. 2D text labels (QPainter over GL)
```

---

## LED Wall Manager

### Built-In Panel Database

The client ships with manufacturer presets for common LED panels:

| Manufacturer | Models |
|-------------|--------|
| ROE Visual | Black Pearl BP2, Carbon CB5, Black Marble BM4, Ruby RB1.5 |
| Absen | PL2.5 Pro, A27 |
| Unilumin | UpadIII 2.6, UMini 0.9 |
| Samsung | The Wall IW008J |
| Sony | Crystal LED C-series |
| INFiLED | DB1.9 |
| Desay | HB1.9 |

Each preset stores: pixel pitch (mm), tile dimensions (mm), pixels per tile, weight (kg), peak brightness (nits), refresh rate (Hz).

### Screen Builder

The LED Screen wizard computes everything from three inputs:

1. **Panel type** — Select from database or enter custom pixel pitch + tile dimensions
2. **Tile count** — Columns × Rows
3. **Curvature** — Flat or Cylindrical (radius + arc angle)

Auto-computed fields:
- **Total physical size** = tile_size × tile_count + bezel_gaps
- **Total pixel resolution** = (tile_size_mm / pixel_pitch_mm) × tile_count
- **Mesh geometry** — Flat quad or cylindrical arc with correct UV mapping

### Array Tool

Creates N LED screens along an arc with automatic angular distribution:

- **Panel count** (2–50)
- **Total arc angle** (10°–360°)
- **Radius** (0.5–50 m)
- **Panel dimensions** (width × height)
- **Bezel gap** (0–50 mm)
- **Start channel** — Sequential channel assignment

Panels are evenly spaced along the arc, each rotated to face the centre.

---

## Scene Objects

### Primitives

Generated procedurally from parameters:

| Primitive | Default Size | Use Case |
|-----------|-------------|----------|
| Cube | 1m³ | Set piece stand-in, occlusion reference |
| Sphere | 1m diameter | Lighting reference, scale marker |
| Cylinder | 1m × 1m | Column, truss approximation |
| Plane | 1m² | Floor marker, projection surface |
| Cone | 1m × 1m | Directional indicator |

### Reference Models

Built-in low-poly procedural models at real-world scale:

| Model | Dimensions | Purpose |
|-------|-----------|---------|
| Person (standing) | 1.80m tall | Scale reference, audience sightlines |
| Person (seated) | 1.20m tall | Seated audience layout |
| Car | 4.5m × 1.8m × 1.4m | Automotive event previz |
| Camera + Tripod | 1.6m tall | Camera position planning |
| Truss Section | 0.3m × 0.3m × 1.0m | Rigging layout |
| Chair | 0.45m × 0.45m × 0.85m | Seating arrangement |
| Flight Case | 0.6m × 0.6m × 0.8m | Backstage clearance |
| Lectern | 0.6m × 0.5m × 1.2m | Corporate event staging |

### Import

Supported formats:
- **GLB** (binary glTF 2.0) — recommended, single file, full mesh + material data
- **OBJ** — simple triangle meshes with normals

> **Note**: JSON-based `.gltf` files (with external binary buffers) are not currently supported. Convert to `.glb` first using tools like Blender or `gltf-pipeline`.

---

## 3D Viewport Controls

### Camera

| Input | Action |
|-------|--------|
| LMB drag | Orbit around target point |
| MMB drag | Pan (horizontal + vertical) |
| Scroll wheel | Zoom (dolly in/out) |
| F key (scene tree) | Focus camera on selected node |

### Object Manipulation

| Input | Action |
|-------|--------|
| LMB click on object | Select node (syncs tree + properties) |
| RMB drag on object | Move on XZ ground plane |
| Shift + RMB drag | Move along Y axis (vertical) |

Locked nodes (🔒) cannot be moved by drag or properties panel edits.

### Scene Tree

| Input | Action |
|-------|--------|
| Click | Select (syncs viewport + properties) |
| Ctrl+Click / Shift+Click | Multi-select |
| Delete key | Delete selected nodes |
| Ctrl+D | Duplicate selected nodes |
| F | Focus viewport on selection |
| Right-click | Context menu: Delete, Duplicate, Lock/Unlock, Reset Transform, Group, Focus |

---

## Configuration

### Server Config (casparcg.config)

A previz channel is a standard CasparCG channel. Designate it for previz use by leaving consumers empty or adding a Spout/screen consumer for monitoring:

```xml
<channels>
    <!-- CH1-3: Production outputs -->
    <channel>
        <video-mode>2160p2500</video-mode>
        <consumers>
            <decklink><device>1</device></decklink>
        </consumers>
    </channel>
    <!-- ... CH2, CH3 ... -->

    <!-- CH4: Previz (3D venue simulation) -->
    <channel>
        <video-mode>1080p2500</video-mode>
        <consumers>
            <screen/>  <!-- Local preview window -->
        </consumers>
    </channel>
</channels>
```

The previz state is established at runtime via AMCP commands — there is no static XML for scene configuration. This allows dynamic reconfiguration without server restarts.

---

## Auto-Projection

Auto-projection automatically derives `MIXER PROJECTION` parameters from the virtual camera's position relative to each LED screen. This enables tracked camera workflows where the virtual camera follows a physical camera rig.

### How It Works

1. The **virtual camera** (separate from the viewport camera) represents the physical camera position
2. For each screen with a channel mapping, the server computes the frustum intersection — what portion of the channel content is visible from the virtual camera's perspective
3. These parameters are applied as `MIXER PROJECTION` transforms on the corresponding channel/layer
4. When the virtual camera moves, projections update automatically

### Workflow

1. Set up LED screens with channel mappings
2. Position the virtual camera at the physical camera location
3. Enable auto-projection (🎯 button in the Stage tab toolbar)
4. Move the virtual camera — projection parameters update in real time
5. Connect camera tracking to drive the virtual camera position automatically

The virtual camera has independent X/Y/Z/Yaw/Pitch/FOV controls in the Stage tab toolbar, separate from the viewport's orbit camera.

---

## Session Management

### Auto-Save

The client debounces scene changes (5-second timeout) and writes `session_autosave.json` automatically. On startup, the last auto-saved session is restored.

### Session File Format

```json
{
    "channel_tabs": [
        {
            "channel": 4,
            "layer": 10,
            "previz": {
                "nodes": [
                    {
                        "uid": "a1b2c3d4",
                        "name": "Main Wall",
                        "node_type": "SCREEN",
                        "pos_x": 0.0, "pos_y": 2.0, "pos_z": -5.0,
                        "rot_yaw": 0.0, "rot_pitch": 0.0, "rot_roll": 0.0,
                        "scale": 1.0,
                        "visible": true, "locked": false,
                        "screen_config": {
                            "preset": "ROE Black Pearl BP2",
                            "tiles_h": 8, "tiles_v": 4,
                            "bezel_mm": 0.0,
                            "channel": 1
                        }
                    }
                ],
                "camera": { "x": 0, "y": 1.7, "z": 8, "yaw": 180, "pitch": 0, "fov": 60 },
                "virtual_camera": { "x": 0, "y": 1.7, "z": 0, "yaw": 0, "pitch": 0, "fov": 60 },
                "show_grid": true,
                "show_wireframe": false
            }
        }
    ]
}
```

### Manual Save/Load

**File → Save Session…** saves a named `.json` file that can be shared between machines or version-controlled.

---

## Operation Guide

### Quick Start: Single LED Wall

1. **Start the server** with at least 2 channels (1 production + 1 previz)
2. **Open the 360 Client** and connect to the server (AMCP port 5250)
3. **Switch to the Stage tab** on the previz channel (e.g. CH4)
4. **Create a screen**: Click `Create ▾ → LED Screen…`
   - Select panel type (e.g. ROE Black Pearl BP2)
   - Set tile count (e.g. 8×4)
   - Set channel mapping to CH1
   - Click Create
5. **Enable Live Link** (🔗 toggle) — screen appears on server
6. **Play content** on CH1 — it appears on the 3D screen surface
7. **Orbit the viewport** to inspect from different angles

### Quick Start: Curved LED Stage

1. **Create an array**: Click `Create ▾ → Array Tool…`
   - Count: 6, Arc: 180°, Radius: 8m
   - Panel size: 1m × 2m
   - Start channel: 1
2. Six screens appear arranged in a semicircle
3. Adjust individual screen positions by RMB-dragging in the viewport
4. Add reference objects (Person, Camera) for scale context

### Multi-Channel Setup

```
CH1 → Main LED wall (front)     ←  PREVIZ 4 SCREEN "Front" CHANNEL 1
CH2 → Left wing                 ←  PREVIZ 4 SCREEN "Left" CHANNEL 2
CH3 → Right wing                ←  PREVIZ 4 SCREEN "Right" CHANNEL 3
CH4 → Previz output             ←  The previz renderer itself
```

### Adding Camera Tracking

1. Set up previz screens as above
2. Configure camera tracking on the desired channel (see CAMERA_TRACKING.md)
3. Position the virtual camera at the tracked camera's physical location
4. Enable auto-projection — tracked camera movements now drive content mapping

> **Tracking lens & timing realism in previz.** When a binding runs in `MODE PREVIZ`, the virtual camera is driven by the same processed pose as a live layer, so the tracking realism features apply here too:
> - `TRACKING DELAY` time-aligns the virtual camera with a delayed video feed (the interpolated pose drives the previz camera).
> - `TRACKING NODAL` shifts the virtual camera's position onto the lens entrance pupil for correct parallax.
> - `TRACKING LENS` feeds the profile's field of view (and nodal offset) into the previz camera FOV as the lens zooms.
> - `OPENTRACKIO` can drive the previz camera just like FreeD — `TRACKING 1-4 BIND OPENTRACKIO PORT 55555 HOST 239.135.1.100 MODE PREVIZ`.
>
> Distortion (`k1`–`k3`, `p1`/`p2`) and focus-driven depth of field (`TRACKING DOF`) are **not** applied to the previz camera — they affect the 2D/360 layer output only.

---

## Best Practices

### Scene Setup

- **Use real-world units** — All dimensions are in metres, rotations in degrees. A 5m × 3m LED wall is `FLAT 5.0 3.0`.
- **Name screens clearly** — Use descriptive names like "Front_Wall", "Floor_Screen". Names are used in AMCP commands and session files.
- **Lock reference objects** — After positioning reference models (people, cameras), lock them (🔒) to prevent accidental movement.
- **Use the panel database** — Built-in presets ensure accurate pixel pitch and resolution calculations. Custom specs should only be used for panels not in the database.

### Performance

- **Keep previz channel at 1080p** — The previz output doesn't need to match production resolution. 1080p25/50 is sufficient for preview purposes.
- **Limit scene complexity** — The previz renderer is not a game engine. Keep imported models under 50K triangles total.
- **Prefer GLB over OBJ** — GLB files are binary, faster to parse, and include material data.
- **Avoid excessive reference models** — Each node adds a draw call. 20-30 objects is a reasonable ceiling.

### Live Production

- **Test auto-projection before going live** — Verify that channel textures appear correctly on all mapped screens from the expected camera position.
- **Save camera presets** — Store key viewpoints (FOH, backstage, camera positions) for quick switching during rehearsals.
- **Use Live Link selectively** — Live Link sends AMCP commands on every change. Disable it during bulk edits, then re-enable and push once.
- **Save sessions before major changes** — Manual save provides a rollback point.

### Coordinate System

```
        Y (up)
        │
        │    Z (forward, into screen)
        │   /
        │  /
        │ /
        └──────── X (right)

  Origin (0,0,0) = centre of stage floor
  Y=0 = floor level
  Positive Z = upstage (away from audience)
  Positive X = stage right
  Yaw 0° = facing +Z, Yaw 90° = facing +X
```

---

## Known Limitations

### Server

- **No global illumination** — Screens illuminate themselves (emissive) but do not cast light onto surrounding geometry. For light-bounce simulation, use Unreal Engine via Spout.
- **No shadow mapping** — Geometry does not cast or receive shadows.
- **Single-mesh glTF** — The GLB loader extracts only the first mesh from multi-mesh files. Complex venue models should be combined into a single mesh before export.
- **No animation** — Mesh animation (skeletal or morph targets) is not supported. All geometry is static.
- **Channel texture latency** — Zero-copy binding is same-GPU only. Cross-GPU setups may require Spout with a one-frame latency.

### Client

- **JSON-based .gltf not supported** — Only binary `.glb` files are supported for import. Convert `.gltf` files to `.glb` using Blender (`File → Export → glTF Binary`) or the `gltf-pipeline` CLI tool.
- **No hierarchical re-parenting** — The Group node creates a logical container but drag-to-reparent in the scene tree is not yet implemented. Nodes remain flat.
- **No undo/redo** — Scene edits are immediate and cannot be undone. Use session save as a checkpoint.
- **No transform gizmo overlay** — Object movement is via RMB drag (XZ plane) or Shift+RMB (Y axis). There is no visual translate/rotate/scale gizmo widget.
- **Viewport picking uses AABB** — Node selection uses axis-aligned bounding boxes, not per-triangle ray casting. Overlapping nodes may require tree selection instead of viewport clicking.
- **No multi-mesh import** — Imported OBJ/GLB files are loaded as a single mesh. Scene hierarchy from the source file is not preserved.
- **Spout texture reception not yet integrated** — The client viewport currently renders with base colours only. Live channel texture display on screen surfaces requires Spout integration (planned).
- **Reference models are box approximations** — Built-in reference objects use simple box geometry for fast rendering, not detailed meshes.
