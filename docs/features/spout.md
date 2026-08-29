# Spout — GPU texture sharing with other Windows applications

> **State:** shipped and measured — path by battery, pixels by a real receiver
> **Modules:** `src/modules/spout` (producer and consumer)
> **Commands:** none of its own — reached by producer syntax on `PLAY` and by consumer name on `ADD`, with `MAX_WIDTH` / `MAX_HEIGHT` / `EVERY_NTH` / `FPS` as consumer arguments
> **Architecture:** none, deliberately — a thin wrapper over the Spout SDK; the interesting constraint (adapter-bound shared handles) is in the guide
> **Guide:** [`../guides/SPOUT.md`](../guides/SPOUT.md)
> **Coverage:** `cli.py spout-signalling` — **8/8, both mixers**, gating which path was taken; `cli.py preview-cost` — the publishing cost against a screen consumer; **pixels verified by hand** through a real receiver, both mixers byte-exact

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

**The path, not the picture.** `cli.py spout-signalling` starts one server per case, adds a Spout
consumer over AMCP and reads what the consumer reports about itself. **8/8, parity OK**, on both
mixers: `gpu-path` true in every case, `gpu-downscale` true exactly when a `MAX_WIDTH` was given,
the published raster 256×144 where 256 was asked for, and `every-nth` 1/2/5 including `FPS 5`
resolving against a 25 Hz channel.

**No battery receives a Spout sender**, so nothing has compared a pixel. That is not laziness: a
CPU fallback publishes the same frame at the same size as the zero-copy path — they differ by
3.12 ms, not by appearance — so a pixel comparison is structurally incapable of failing for the
change that battery exists to protect. Receiving also needs `SpoutGL`, which ships wheels for
cp38–cp313 while the 360 client's environment is Python 3.14.

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

## 4b. What a preview actually costs, both ends

`cli.py preview-cost` — five channels at 1080p5000 (a 20 ms tick), **full rate, no
`EVERY_NTH`**, one server per arm. `consume` is how long the channel waited for its
consumers to take the frame; `penalty` is that above the same mixer's no-preview floor.

**Compare the MEAN, not the max.** `consume_max` is a tail statistic over the window's
reports and it moves between runs: OGL Spout gave 0.14 then 0.20 ms while the screen
consumer gave 0.23 then 0.14 — **the two swapped order**, so either conclusion could have
been quoted. The means over three runs each are non-overlapping and stable to the third
decimal:

| mixer | arm | penalty (mean of 3 runs) | % of tick | late frames |
| :--- | :--- | ---: | ---: | ---: |
| ogl | `MAX_WIDTH 256` | **+0.030 ms** | 0.15% | 0 / 6265 |
| ogl | screen consumer, same raster | **+0.095 ms** | 0.48% | 0 / 6266 |
| vulkan | `MAX_WIDTH 256` | ~+0.03 ms | ~0.15% | **48–59 / ~6265** |
| vulkan | screen consumer, same raster | ~+0.10 ms | ~0.5% | 3 / 6264 |

So publishing over Spout costs about **a third** of what a screen consumer costs, and both
are a fraction of a percent of the tick.

And the receive end, `casparcg-360-client/tools/preview_bench.py`, five previews at
384×216 in one GL context:

| | Spout `receiveTexture` | `PrintWindow` + upload |
| :--- | ---: | ---: |
| acquire per frame, mean | **1.96 ms** | **57.20 ms** |
| p95 | 2.19 ms | 82.65 ms |
| loop rate achieved | 58.0 fps | 15.7 fps |
| grabs that returned data | 95% | 70% |

**So: cheaper, not free.** Publishing costs the channel about **0.15% of a tick** on
OpenGL, roughly a third of a screen consumer at the same raster. The receive end is where
the difference is decisive: **29× per frame**, and the grab path cannot sustain the
channel's rate at all.

**The preview itself runs at full rate.** The consumer drops a frame when the previous one
is still in flight, so "publishes at 50 Hz" needed checking rather than assuming:
`spout/sent-frames` and `spout/dropped-frames` report **5000 sent, 0 dropped over 20 s
across five channels on both mixers** — exactly 50 fps per channel. `spout/fps` cannot
answer this, because it counts frames *offered* and is incremented before the drop check.

Two honest limits. The box was at **98% CPU on the baseline arm alone** — five 1080p50
channels saturate it — so these are costs under load, which flatters neither arm. And
`acquire` on the receive side is CPU wall time with no GPU fence, an omission that
favours Spout.

---

## 5. Known gaps

1. **No coverage.** §4 describes a self-round-trip that would need no second application.
2. **The `eUndefined` layout fix is unverified here**, only on the ProRes path.
3. **Windows-only**, with no message explaining that on other platforms.
4. **No documented canonical syntax** — three forms work and none is marked preferred.
5. **Measured end to end, and it found a real defect.** `cli.py spout-signalling` gates the
   reported path (8/8, both mixers). Beyond that, a real Spout receiver was built for this
   machine's Python (`casparcg-360-client/tools/build_spoutgl.md`) and used to read the
   published pixels back:

   | | OGL | Vulkan (before) | Vulkan (after) |
   | :--- | :--- | :--- | :--- |
   | `#20A0C0` received as RGBA | (32, 160, 192) | **(192, 160, 32)** | (32, 160, 192) |
   | red band on top quarter | top rows | top rows | top rows |

   **The Vulkan path exchanged red and blue.** Cause, exactly as `device.h` had recorded
   about blits: the mixer's 8-bit attachment is declared `eR8G8B8A8Unorm` but *holds BGRA
   bytes* — only the 8-bit shader path swizzles — and a component-wise blit preserves the
   byte order, so an OpenGL importer reading `GL_RGBA8` gets blue where red should be.
   Fixed by giving the exportable destination the opposite component order
   (`create_exportable_texture(..., swap_rb=true)`), which makes the blit reorder the
   bytes into true RGBA. Both mixers are now byte-identical.

   **A grey test pattern is invariant under that exchange and passed everything.** The
   probe colour has three distinct channels for precisely this reason.

   `bInvert=false` on the Vulkan path, carried over from the OGL mixer unverified, turns
   out to be correct — confirmed with a spatially asymmetric source rather than a flat one,
   since a flat colour cannot show a flip.

6. **The Vulkan path induces late frames past two channels — diagnosed, NOT fixed.**
   `preview-cost --mixer vulkan --channels N`, Spout against a no-preview baseline that ran
   0 late at every N:

   | channels | 1 | 2 | 3 | 5 |
   | :--- | ---: | ---: | ---: | ---: |
   | late frames | 0 / 1003 | 0 / 2008 | **21 / 3008** | **76 / 5009** |
   | rate | 0% | 0% | 0.70% | 1.52% |

   Nothing at one or two, onset at three, doubling by five. **That is contention on
   something shared, not a per-consumer cost** — a per-consumer cost is visible at N=1 and
   scales linearly. `consume_max` stays at 0.03–0.15 ms throughout, the lowest of any arm,
   so the channel is *not* waiting for the consumer to accept the frame.

   **The mechanism.** `device::blit_to_shared` wraps its work in `dispatch_sync`, so every
   consumer makes a **blocking round trip to the Vulkan device's single dispatch thread**
   once per tick — and that is the thread the mixer composites on. Five channels at 50 Hz
   is 250 such round trips a second competing with the compositing they are copying from.
   The OpenGL path does its blit on the consumer's own GL context and runs **0 late at the
   same load**, which is the control that makes this reading rather than a guess.

   **Four fixes tried. All rejected by measurement.** The noise floor was established
   first — five identical runs gave 41, 47, 48, 53, 54 late, so ±7 on a mean of ~48, and
   anything inside that is not a difference.

   | attempt | late / ~5010 | verdict |
   | :--- | ---: | :--- |
   | *(shipping)* blit on the device dispatch thread | **41–54** | the baseline |
   | record + submit on the **caller's thread**, private command pool | 211 | **4× worse** |
   | …the same, without the completion fence | 242 | worse still |
   | **narrow the source barriers** to `eColorAttachmentOutput` from `eAllCommands` | 43 | no change |
   | *(diagnostic)* blit but **never** `SendTexture` | 101–120 | **worse** |
   | *(diagnostic)* `SendTexture` but **never** blit | 59–70 | **worse** |

   The last two are the interesting ones: **removing either half of the work makes it
   worse, reproducibly.** So the cost is not "the blit" and not "the send" — it is the
   phasing of the consumer's GPU work against the mixer's on a shared queue, and doing
   less of it simply moves the collision. That also kills the dispatch-thread hypothesis
   outright: moving recording off that thread put five consumer threads into direct
   contention for `device::submit`'s queue lock, where the dispatch thread had at least
   been serialising them in one place.

   **How bad is it, actually?** Less than "late frames" suggests. The raw TIMING lines show
   the average period stays exactly nominal — `avg=20.00ms (nominal 20.00)` — in every arm.
   What changes is **jitter**: 2.4–6.1 ms with no preview against 4.3–8.7 ms with five
   Spout consumers, which tips ~1% of ticks past 20 ms. The channel does not fall behind
   and no preview frame is dropped. It is added jitter, not lost output.

   **What is left to try**, and why it was not attempted here: this GPU has a separate
   queue family — the server logs `reserved compute queue family 2 for external decoders
   (graphics is 0)` — and moving the blit there would take it off the mixer's queue
   entirely, which is the only lever the measurements have not ruled out. Two obstacles,
   both real: that family is *already reserved* for FFmpeg decoders, so previews would
   reintroduce the contention the reservation exists to prevent; and reading the mixer's
   attachment from another family needs **queue-family ownership transfers**, which means
   the mixer must submit a release barrier every frame — or its attachments must become
   `VK_SHARING_MODE_CONCURRENT`, which slows compositing for every channel whether or not
   a preview is attached. Either is a mixer-wide change of the class this tree records
   producing `VK_ERROR_DEVICE_LOST` at four concurrent producers
   (`av_vulkan_import.cpp`), and it wants its own session with `coexistence` before and
   after.

7. **`FPS` rounds down silently**7. **`FPS` rounds down silently** past the log line at `initialize()`. An operator who asks for 30
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
