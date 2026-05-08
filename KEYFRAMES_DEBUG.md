# Keyframe Module — Debugging Notes

This document records logic bugs found and fixed in the keyframe animation feature
(`src/modules/keyframes/`).  It is separate from `BUILDING_WORKFLOW.md`, which
covers build/compile errors only.

---

## Bug #1 — KF2 has no visual effect; all ticks show `frame=0 time=0`

**Date fixed:** 2026-03-19

### Symptom

After the sequence **ARM → Set KF1 → Seek → Set KF2**, moving the KF2 slider
causes no change on the output signal.  Server log (after adding `[kf-tick]`
diagnostics to `auto_tick_all`) shows:

```
[kf-tick] ch=0 lay=10 frame=0 time=0 kf_count=2 dur=0 proj_yaw=<KF1 yaw> proj_enable=1
```

… on every render frame, regardless of the clip's actual position.

### Root cause

The layer is in the **paused** state (via AMCP `PAUSE`).  In
`src/core/producer/layer.cpp`:

```cpp
auto frame = paused_ ? core::draw_frame{} : foreground_->receive(field, nb_samples);
```

When paused, `receive_impl()` → `av_producer::next_frame()` is never called.
`next_frame()` is the **only** path that updates `frame_time_`.  Because
`frame_time_` stays at `AV_NOPTS_VALUE` / 0:

- `av_producer::time()` returns `0`
- `ffmpeg_producer::frame_number()` = `time() - start()` = `0`
- `auto_tick_all`: `time_secs = 0 / fps = 0.0`
- `interpolate(0.0)` ≤ KF1.time_secs → always returns **KF1 state**

KF1 appeared to work coincidentally: `interpolate(0)` returns KF1's state, and
editing KF1 changed that state immediately.  KF2, at a later `time_secs`, was
never reached.

### Fix — two parts

**1. Server: `kf_binding` + `auto_tick_all` + `tick()`**
(`src/modules/keyframes/keyframe_registry.h` / `.cpp`)

- Added `double client_time_secs = -1.0` to `kf_binding`.
- `auto_tick_all` now falls back to `client_time_secs` when `frame_number == 0`:

```cpp
double time_secs;
if (frame_number == 0 && b.client_time_secs >= 0.0)
    time_secs = b.client_time_secs;
else
    time_secs = static_cast<double>(frame_number) / fps;
```

- `tick()` now only stores `client_time_secs`; the old `apply_transform()` call
  inside `tick()` was removed because it fought `auto_tick_all` every frame.

**2. Server: `av_producer::Impl::seek()` pre-announces `frame_time_`**
(`src/modules/ffmpeg/producer/av_producer.cpp`)

The primary fix. Inside the `buffer_mutex_` lock in `seek()`, immediately set
`frame_time_ = target_pts` so that `time()` / `frame_number()` / OSC state all
reflect the new seek position before `next_frame()` is ever called:

```cpp
// Under buffer_mutex_ lock, after computing target_pts:
frame_time_ = target_pts;  // pre-announce seek target
```

`target_pts` (TIME_BASE_Q, µs relative to container start) is in the same
coordinate space that `next_frame()` uses when it eventually updates
`frame_time_`, so this is exact. On a paused layer `next_frame()` is never
called, so without this `frame_time_` stayed at the pre-seek position —
causing `frame_number()`, OSC, and `auto_tick_all` to all see the wrong time.

**3. Client: send TICK on every OSC update**
(`casparcg-360-client/keyframes.py`, `KeyframeEngine::_on_frame`)

```python
if self._server_armed:
    self._send(f"KEYFRAMES {self._cl()} TICK {frame / fps:.4f}")
```

Secondary safety net for `frame_number==0` edge cases.

### Diagnostic approach used

Added `[kf-tick]` logging lines to `auto_tick_all` that fire on every kf_count ≥ 2
tick, printing `frame`, `time_secs`, `kf_count`, `dur`, `proj_yaw`, and
`proj_enable`. The first session showed `frame=0 time=0` — pointing to the
paused-layer issue. After adding the `frame_number==0` fallback, the second session
showed `frame=74 time=2.96` stuck — revealing that a seek on a paused layer leaves
`frame_time_` stale at the pre-seek value, not just at 0. The root fix is
pre-announcing `frame_time_` in `seek()`.
