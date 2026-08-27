# HTML / CEF — GPU-direct page compositing

> **State:** shipped
> **Module:** `src/modules/html` — **282 lines** different from upstream across 7 files, plus two
> fork-only ones: `producer/html_gpu_bridge.{h,cpp}`
> **Commands:** upstream's `PLAY [HTML]` and the CG interface; the fork adds configuration only
> **Architecture:** none, deliberately — the D3D11 shared-texture import is the same bridge GStreamer uses, documented in GPU_INTEROP_ARCHITECTURE.md
> **Guide:** none, deliberately — Upstream owns the HTML producer and the CG interface; this fork adds GPU-direct configuration only, documented in §2 here. No separate operator guide.
> **Coverage:** none dedicated — see §4

Takes CEF's composited page as a **D3D11 shared texture** instead of a host-memory bitmap, so a
browser layer reaches the mixer without a CPU copy per frame.

---

## 1. What is implemented today

Upstream CEF renders off-screen and hands back a CPU buffer via `OnPaint`. This fork also binds
`OnAcceleratedPaint` (`html_producer.cpp:399-406`), where the composited page arrives as a D3D11
shared texture, and `html_gpu_bridge.cpp` imports it through
`accelerator/vulkan/util/d3d11_import_bridge.h` — the same bridge the GStreamer GPU route reuses.

**CEF binds `OnPaint` or `OnAcceleratedPaint` once and will not change afterwards**
(`html_producer.cpp:703`), so the choice is made at browser creation and is not a per-frame
fallback. That is why the switch below is a configuration element and not a runtime command.

`html_gpu_bridge.h` carries its own byte-order enum that "mirrors `cef_color_type_t` without
dragging a CEF header into the accelerator layer" — the compositor declares the byte order of its
shared surface, and this fork does not assume it.

---

## 2. Configuration

```
<html>
    enable-gpu                 CEF's own GPU compositing
    gpu-direct                 take the shared texture instead of a host bitmap
    gpu-direct-adapter-luid    which adapter the shared handle belongs to
    angle-backend              CEF's ANGLE backend selection
    cache-path
    remote-debugging-port
</html>
```

**`gpu-direct-adapter-luid` is load-bearing on a two-GPU machine.** A D3D11 shared handle is
adapter-bound; importing one produced on the other adapter fails rather than degrades.

---

## 3. Why this matters beyond speed

HTML is how this fork renders **fill + key lower thirds**, so the key channel is an output of the
page rather than a separate asset. A byte-order or premultiplication error therefore shows up in the
key, not only in the fill — and comparing the two honestly means capturing both.

---

## 4. Known gaps

1. **No battery.** Nothing measures the HTML producer on either route, so the GPU-direct path is not
   compared against `OnPaint` for picture equality — the check that would catch a byte-order or
   alpha-domain difference between them.
2. **The reconciliation in `../audits/CEF_GPU_DIRECT_RECONCILIATION.md` is a code reading**, not a
   measurement.
3. **No fixture.** There is no committed page whose expected output is known, which is what a
   battery would need first.
