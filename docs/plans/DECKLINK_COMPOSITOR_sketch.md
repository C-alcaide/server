# Plan: DeckLink Compositor — Multiple Channels to One Output

## Current State (verified)
- NOT supported. Each `decklink_consumer` calls `EnableVideoOutput` + `StartScheduledPlayback` on its device — these are exclusive operations on a hardware port. A second consumer opening the same device would fail at the DeckLink SDK level.
- `index()` = `300 + device_index`, but this is per-channel output map — not the real blocker.
- Existing `<ports>` / secondaries = 1 channel → N *different* devices (sync group). Inverse of what's needed.
- Existing subregion geometry (`src_x`, `dest_x`, etc.) handles N-of-same-channel to region, not cross-channel composition.

## Architecture Decision
New **DeckLink Compositor** system: one entity owns the hardware port exclusively; multiple channels contribute frames to their assigned sub-regions of a composited output frame.

Three components:
1. `decklink_compositor` — owns device, runs DeckLink scheduled playback, composites slot frames each tick
2. `decklink_compositor_slot_consumer` — implements `frame_consumer`, stores latest frame per channel, feeds compositor
3. Module-level static registry — maps device_index → compositor, created at init time from config

## Config Schema

```xml
<!-- NEW top-level section in <configuration> -->
<decklink-compositors>
  <compositor>
    <device>1</device>
    <video-mode>1080p2500</video-mode>
    <embedded-audio>false</embedded-audio>
    <!-- hdr, latency, pixel-format, color-space supported same as <decklink> -->
  </compositor>
</decklink-compositors>

<!-- Per-channel consumer (instead of <decklink>) -->
<channel>
  <video-mode>300x200p2500</video-mode>   <!-- must match compositor framerate -->
  <consumers>
    <decklink-composite>
      <device>1</device>    <!-- must match a declared compositor above -->
      <dest-x>0</dest-x>
      <dest-y>0</dest-y>
      <!-- region-w / region-h optional: inferred from channel frame size -->
    </decklink-composite>
  </consumers>
</channel>

<channel>
  <video-mode>300x200p2500</video-mode>
  <consumers>
    <decklink-composite>
      <device>1</device>
      <dest-x>320</dest-x>
      <dest-y>0</dest-y>
    </decklink-composite>
  </consumers>
</channel>
```

## Implementation Steps

### Phase 1: Config structs & parsing
- Add `compositor_global_config` struct in `config.h` (device_index, format name, audio, latency, pixel_format, hdr, color_space)
- Add `compositor_slot_config` struct in `config.h` (device_index, dest_x, dest_y, region_w, region_h)
- In `config.cpp`: add `parse_compositor_global_config()` and `parse_compositor_slot_config()` functions

### Phase 2: `decklink_compositor` class (new file)
New files: `src/modules/decklink/consumer/decklink_compositor.h` + `.cpp`

**`decklink_compositor`:**
- Constructor: takes `compositor_global_config`; opens DeckLink device (`get_device()`), calls `EnableVideoOutput`, `SetCallback`, `StartScheduledPlayback`; allocates zeroed scratch frame buffer (full output res, BGRA)
- `struct slot { int dest_x, dest_y, region_w, region_h; std::mutex mtx; core::const_frame latest_frame; }`
- `int create_slot(dest_x, dest_y, region_w, region_h)` — appends slot, returns slot_id
- `void release_slot(int slot_id)` — marks slot inactive (sets valid=false)
- `void update_slot(int slot_id, core::const_frame frame)` — mutex-locked store
- `ScheduledFrameCompleted` callback: for each active slot, if latest_frame valid: blit into scratch buffer; schedule scratch buffer; zero scratch for next tick
- Blit: progressive BGRA memcpy row-by-row at (dest_x, dest_y). Simple loop, no format_strategy needed.
- Enforce framerate match at `create_slot()` if output is running

**`decklink_compositor_slot_consumer`** (also in this file):
- Holds `shared_ptr<decklink_compositor>` + `slot_id_`
- `send(field, frame)`: on field1/progressive, calls `compositor_->update_slot(slot_id_, frame)`; returns `make_ready_future(true)`
- `initialize(format_desc, ...)`: validates framerate match; registers slot via `compositor_->create_slot()`; stores slot_id_
- `index()`: `500 + device_index * 100 + slot_id_` — unique, no collision with regular DL (300+)
- Destructor: calls `compositor_->release_slot(slot_id_)`

**Module-level static registry** (in `decklink_compositor.cpp`):
```cpp
static std::mutex g_registry_mutex;
static std::map<int64_t, std::shared_ptr<decklink_compositor>> g_compositor_registry;

void setup_compositors(ptree_optional node);
std::shared_ptr<decklink_compositor> find_compositor(int64_t device_index);
```

### Phase 3: Factory function
In `decklink_consumer.cpp` or new file:
- `create_preconfigured_compositor_consumer(ptree, format_repo, channel_info)`:
  - Calls `parse_compositor_slot_config(ptree)`
  - Calls `find_compositor(slot_config.device_index)` — throws if not found
  - Returns `spl::make_shared<decklink_compositor_slot_consumer>(compositor, slot_config)`

### Phase 4: Registration in `decklink.cpp`
In `decklink::init()`:
- Call `setup_compositors(env::properties().get_child_optional(L"configuration.decklink-compositors"))`
- Register: `dependencies.consumer_registry->register_preconfigured_consumer_factory(L"decklink-composite", create_preconfigured_compositor_consumer)`

### Phase 5: `CMakeLists.txt`
Add to `src/modules/decklink/CMakeLists.txt`:
```
consumer/decklink_compositor.cpp
consumer/decklink_compositor.h
```

### Phase 6: Config example update
Update `src/shell/casparcg.config` with a commented example of the new section.

## Key Design Decisions

| Decision | Choice |
|---|---|
| Framerate match | **Required** — enforced in `create_slot()`, throws if channel fps ≠ compositor fps |
| Scaling | **None** — user must size channels to fit their dest region |
| Field / interlaced | **Progressive-only** for initial impl; interlaced deferred |
| Pixel format | **BGRA** — simplest for CPU compositing; no YUV conversion at composite stage |
| Consumer index | `500 + device_index*100 + slot_id` — no collision with regular DL (300+) |
| Server changes | **None** — `init()` reads `env::properties()` directly, everything stays in the decklink module |
| Compositor lifetime | Static registry holds it; slot consumers hold `shared_ptr`, compositor stays alive while any channel references it |
| Black fill | Scratch buffer zeroed once per frame; each active slot blits over its region |

## Relevant Files

| File | Change |
|---|---|
| `src/modules/decklink/consumer/config.h` | Add `compositor_global_config` and `compositor_slot_config` structs |
| `src/modules/decklink/consumer/config.cpp` | Add `parse_compositor_global_config()` and `parse_compositor_slot_config()` |
| `src/modules/decklink/consumer/decklink_compositor.h` | **NEW** |
| `src/modules/decklink/consumer/decklink_compositor.cpp` | **NEW** |
| `src/modules/decklink/consumer/decklink_consumer.cpp` | Add factory `create_preconfigured_compositor_consumer()` |
| `src/modules/decklink/decklink.cpp` | Call `setup_compositors()`, register `decklink-composite` factory |
| `src/modules/decklink/CMakeLists.txt` | Add new `.cpp`/`.h` files |
| `src/shell/casparcg.config` | Add config example |

## Reference Patterns in Existing Code

- `decklink_consumer_proxy` (decklink_consumer.cpp:1189) — proxy/executor pattern to follow
- `decklink_secondary_port` (decklink_consumer.cpp:474) — secondary device ownership pattern
- `sdr_bgra_strategy::convert_frame` (sdr_bgra_strategy.cpp:100+) — exact blit logic to adapt
- `parse_xml_config()` (config.cpp:105) — XML parsing pattern to follow
- `setup_channels()` (server.cpp:257) — init ordering reference

## Verification Steps

1. `python run_build.py casparcg` — clean build with new files
2. Config: 2 channels (300×200p25) → `<decklink-composite>` on device 1, at (0,0) and (320,0)
3. Play distinct clips on each channel → verify both visible in their regions on 1080p25 output
4. MIXER CLEAR on one channel → other channel continues outputting cleanly
5. Remove consumer from one channel → other channel still outputs (slot release works)
6. Check diagnostics graph / logging for compositor scheduling
