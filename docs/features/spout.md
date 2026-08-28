# Spout — GPU texture sharing with other Windows applications

> **State:** shipped, unmeasured
> **Modules:** `src/modules/spout` (producer and consumer)
> **Commands:** none of its own — reached by producer syntax on `PLAY` and by consumer name on `ADD`, with `MAX_WIDTH` / `MAX_HEIGHT` / `EVERY_NTH` / `FPS` as consumer arguments
> **Architecture:** none, deliberately — a thin wrapper over the Spout SDK; the interesting constraint (adapter-bound shared handles) is in the guide
> **Guide:** [`../guides/SPOUT.md`](../guides/SPOUT.md)
> **Coverage:** **none** — but `state()` now reports which path ran, which is what a battery would need

Shares frames with other Spout-aware Windows applications over a shared DirectX texture, with no
host copy. The consumer publishes a channel as a Spout sender; the producer receives another
application's sender as a layer. Windows only.

---

## 1. What is implemented today

| piece | evidence |
| :--- | :--- |
| Consumer — publishes the channel as a named Spout sender | `spout_consumer_impl::name()` returns `SPOUT` |
| Producer — receives a named sender as a layer | `create_spout_producer` |
| Exportable Vulkan texture path (producer) | `create_vk_shared_slots` uses `create_exportable_texture` |
| Default sender name when none is given | `"CasparCG Spout"`, in `spout_consumer_impl`'s constructor |
| **GPU downscale on the OGL mixer** | `spout_consumer_impl::downscale_on_gpu` — `glBlitFramebufferEXT` into a cached small texture |
| **Zero-copy on the Vulkan mixer** | `spout_consumer_impl::try_send_vulkan` + `vulkan::device::blit_to_shared` |
| **Frame-rate divisor** | `every_nth_` / `target_fps_`, gated in `send()` on the calling thread |
| **Which path ran, as monitored state** | `spout_consumer_impl::state()` — `spout/gpu-path`, `spout/gpu-downscale` |

The line numbers that used to be in this table were correct when written and pointed at unrelated
code after the next edit to the file — including the edit that added the rows above. Symbols
survive that; see `CLAUDE.md` on citing a symbol rather than a line.

**The producer accepts three different syntaxes for the same thing**, which is the detail most
likely to waste someone's afternoon (`spout_producer.cpp:573-581`):

```
PLAY 1-1 "[SPOUT] SenderName"
PLAY 1-1 "spout://SenderName"
PLAY 1-1 SPOUT SenderName
```

All three resolve to the same receiver. The `[SPOUT]` and bare `SPOUT` forms take the name as a
*second* parameter; the `spout://` form takes it inline via `substr(8)`.

---

## 2. How to drive it

Publish channel 1 under a name another application can find:

```
ADD 1 SPOUT NAME "Programme Out"
```

Receive another application's sender as a layer:

```
PLAY 1-1 "spout://Resolume - Composition"
```

Omit the name on the consumer and it publishes as `CasparCG Spout`.

### Preview-sized senders

A preview does not need the channel's resolution or its frame rate, and both cost real time:

```
ADD 1 SPOUT preview MAX_WIDTH 256 EVERY_NTH 2
ADD 1 SPOUT preview MAX_WIDTH 256 FPS 25
```

| argument | default | effect |
| :--- | :--- | :--- |
| `MAX_WIDTH` / `MAX_HEIGHT` | `0` (native) | cap the sent size, preserving aspect ratio. The reduction runs on the GPU on both mixers |
| `EVERY_NTH` | `1` | send only every Nth frame |
| `FPS` | `0` (off) | the same, as a rate. Rounds **down** to a whole divisor — `FPS 30` on a 50p channel sends at 25, because only whole frames can be skipped. `EVERY_NTH` wins if both are given |

### Telling the fast path from the slow one

A CPU fallback produces the same picture at the same size, so the only way to know which ran is to
ask:

```
INFO 1
```

| field | meaning |
| :--- | :--- |
| `spout/gpu-path` | the frame never reached host memory |
| `spout/gpu-downscale` | the GPU path also did the resize (rather than sending at native size) |
| `spout/out-width` / `spout/out-height` | what is actually being published |
| `spout/every-nth` | the divisor in force, after `FPS` was resolved against the channel rate |

---

## 3. Design decisions, and what they cost

**Shares a texture, not pixels.** The point of Spout is that no frame crosses host memory, so the
consumer hands over a GPU handle. That is also why this is Windows-only and why it goes through
`create_exportable_texture` — the same allocation path the CUDA producers use.

**It therefore inherited a fix it was never tested against.** That function was returning images
in `eUndefined` to every consumer, including this one, until `0f1c5fb38` transitioned them once at
creation. Spout has no battery, so the fix was verified on the ProRes path and *assumed* here.
Stated plainly because it is exactly the kind of shared-path assumption that this folder exists to
make visible.

**A downscale used to disable the thing that made it affordable.** The GPU path was guarded by
`is_shared() && src_w == out_w_ && src_h == out_h_`, so asking for `MAX_WIDTH` fell through to
swscale and `SendImage`. That left two usable combinations — small-and-CPU, or full-res-and-GPU —
and neither scales to several previews: a 1920×1080 host readback costs **3.12 ms**, 19% of a
16.7 ms frame, and eight of them 25 ms. At 256×144 the same readback is **0.088 ms**. So the only
combination worth having was the one the guard excluded, and the resize itself is a GPU blit that
costs a fraction of the copy it removes. The dimension test now selects *whether to blit*, not
whether to use the GPU at all.

**The two mixers need entirely different mechanisms, and the Vulkan one is not the obvious one.**
On the OGL mixer the frame's texture belongs to the mixer's context, so the consumer's shared
context can name it and blit from it directly. On the Vulkan mixer neither half of that applies:

* `vulkan::image_mixer` does not override `native_gl_context()`, so `channel_info::gl_share_context`
  is **null** on that backend and `is_shared()` can never be true — the cast to
  `ogl::texture` that looks like the culprit is never even reached;
* the composite is allocated through `create_attachment()`, which has no exportable memory, so
  `gl_shared_texture` would reject it outright (`gl_export_bridge.cpp` says as much in its error
  text: *"created with create_texture rather than create_exportable_texture?"*).

So the Vulkan path blits the mixer's attachment into an exportable image **this consumer owns**, and
imports that into its own unshared context — `GL_EXT_memory_object` needs exportable memory, not a
shared context. `device::blit_to_shared` does the blit, and does the downscale in the same pass
because `vkCmdBlitImage` can scale where `vkCmdCopyImage` cannot. The destination is left in
`eGeneral`, which is what a GL importer needs; `previz_texture_bridge` makes the same choice for the
same reason.

**GLEW could not be used for the OGL blit.** GLEW macro-renames the framebuffer entry points and
declares them `dllimport` while `SpoutGLextensions.h` declares the same names `extern`, so the two
cannot share a translation unit — the conflict that forced the *producer's* GL half into
`spout_gl_bridge.cpp`. The Spout SDK's own loader already provides `glBlitFramebufferEXT` and the
FBO calls, so the consumer uses those and stays in one file.

**Three accepted syntaxes** rather than one. Convenient for hand-typed commands and for
compatibility with other CasparCG forks' conventions; the cost is three parse branches and no
single documented form, which is why §1 lists all three rather than picking a favourite.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery starts a Spout sender or receiver.

Coverage here needs a second process, which is why it does not exist — but it is not impossible:
the consumer and producer in one server can be pointed at each other, publishing from channel 1
and receiving on channel 2, and the received frame compared against the source. That is a real
round-trip check and needs no external application.

Until that exists, treat Spout as working because it is in use, not because it is measured — and
note that the `eUndefined` fix above was never confirmed on this path.

**What changed in favour of a battery**: `state()` returned `{}` until now, so a check could not
have distinguished zero-copy from a correct CPU fallback even if it had compared pixels — both give
the right picture, and a check that cannot see the difference cannot fail for a change that breaks
the fast path. `spout/gpu-path` is the missing half. The check worth writing is the one in the
handoff plan: a flat asymmetric colour on a channel with no producer, `MAX_WIDTH 256`, assert the
colour **and** `gpu-path`, on both mixers.

---

## 5. Known gaps

1. **No coverage.** §4 describes a self-round-trip that would need no second application.
2. **The `eUndefined` layout fix is unverified here**, only on the ProRes path.
3. **Windows-only**, with no message explaining that on other platforms.
4. **No documented canonical syntax** — three forms work and none is marked preferred.
5. **The GPU downscale and the Vulkan path are unmeasured.** Both compile and both are argued from
   the source above, which is not the same as a picture having been compared. Two specific things
   are unverified and would show up as exactly the kind of defect this tree keeps producing:
   * **Channel order on the Vulkan path.** `device.h` records that a blit "carries the source's
     channel order into 8 bits unchanged — BGRA from an 8-bit attachment, RGBA from a 16-bit one,
     since only the 8-bit shader path swizzles." So the published texture's red/blue order may
     differ between an 8-bit and a 16-bit channel, and between the two mixers. A grey ramp is
     invariant under that exchange and would pass; only **asymmetric** per-channel values can see
     it.
   * **Vertical orientation on the Vulkan path.** `bInvert=false` is correct for the OGL mixer,
     whose output is already top-down, and was carried over unchanged. Whether the Vulkan
     attachment agrees has not been checked against a picture.
6. **`FPS` rounds down silently** past the log line at `initialize()`. An operator who asks for 30
   on a 50p channel gets 25 and only `spout/every-nth` says so.

---

## 6. Related commits

| commit | why it matters |
| :--- | :--- |
| `0f1c5fb38` | transitioned every exportable texture at creation, including the one this module gets. Verified on ProRes, assumed here |

---

## 7. Diagrams

Not warranted on its own. If a diagram is ever drawn for GPU interop generally, Spout is one arrow
on it — see `../architecture/GPU_INTEROP_ARCHITECTURE.md`.
