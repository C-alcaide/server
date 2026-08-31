# Spout — GPU texture sharing with other Windows applications

> **State:** shipped and measured — path by battery, pixels by a real receiver
> **Modules:** `src/modules/spout` (producer and consumer)
> **Commands:** none of its own — reached by producer syntax on `PLAY` and by consumer name on `ADD`, with `MAX_WIDTH` / `MAX_HEIGHT` / `EVERY_NTH` / `FPS` as consumer arguments
> **Architecture:** none, deliberately — a thin wrapper over the Spout SDK; the interesting constraint (adapter-bound shared handles) is in the guide
> **Guide:** [`../guides/SPOUT.md`](../guides/SPOUT.md)
> **Coverage:** `cli.py spout-pixels` — the RECEIVED picture, through a real Spout receiver: **6/6 at 1080p2500 and 6/6 at 5000x3000p50, both mixers, every cell byte-exact**, gating geometry, channel order, size and aspect; `cli.py spout-signalling` — **8/8, both mixers**, gating which path was taken, now at any raster via `--video-mode`; `cli.py preview-cost` — the publishing cost against a screen consumer

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

**And the picture, since 2026-08-31.** `cli.py spout-pixels` plays a fixture whose geometry is
known — a 3x3 grid of nine colours — and receives it through a real Spout receiver in the harness's
own process, using `SpoutGL.createOpenGL()` rather than a toolkit. It then searches the eight
dihedral transforms crossed with {identity, R↔B exchange} for the one mapping the fixture onto what
arrived, so a wrong picture is **named** (`flip_v`, `rot90_cw`, `identity + exchanged`) rather than
merely failed.

**6/6 at 1080p2500 and 6/6 at 5000x3000p50, both mixers** — native, `MAX_WIDTH 256` and
`MAX_WIDTH 384`: `identity`, no exchange, and all nine cells matching to **0 codes** — which also rules out a gamma or transfer shift on the share path, since
the geometry gate alone would not have seen one. That settles the two items this section used to
carry as unmeasured: channel order and vertical orientation on the Vulkan path.

This section previously said *"no battery receives a Spout sender"* and gave two reasons. **The
first was right and is not an argument against a pixel check**: a CPU fallback publishes the same
frame at the same size as the zero-copy path, so a pixel comparison cannot fail for the change
`spout-signalling` protects — which is why `spout-pixels` is a separate battery gating a separate
question rather than an extension of it. **The second was simply wrong**: it said `SpoutGL` ships
wheels for cp38–cp313 against a Python 3.14 environment. It imports and runs on 3.14.4 on this
machine, and had done for as long as the claim stood.

Still not covered on this path: the `eUndefined` layout fix, alpha, and anything about a second
simultaneous receiver.

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

### The receive end — and a comparison that was wrong

| | Spout `receiveTexture` | `PrintWindow` + upload | **HWND embed** |
| :--- | ---: | ---: | ---: |
| acquire per frame | 1.40–1.96 ms | 57.20 ms | **none** |
| client CPU (5 previews, ~50 Hz) | **17.3%** of a core | — | **0.0%** of a core |

**The 29× figure compared the wrong things and is withdrawn.** `PrintWindow` is what the
client uses for its **scopes**, at 10 Hz — it is not how the embedded preview is
displayed. The preview is displayed by *not* displaying it: the window belongs to the
server, DWM composites it into the client's widget tree, and the client does **no
per-frame work at all**. Measured: an `embed` arm with five reparented windows and an
empty frame loop costs **0.0% of one core**, against **17.3%** for five Spout receivers
at the same cadence.

**So Spout is not a client-side performance win. It is a client-side performance
regression**, bought for architectural reasons: no cross-process `SetWindowLong` /
`QWindow.fromWinId` (whose deadlock risk `window_embedder` documents and guards against
with `IsHungAppWindow`), previews composable in one GL surface at any size rather than N
desktop windows with a 320×180 floor, and no requirement that the windows exist on the
desktop at all.

### The non-Spout mitigations, both measured and both null

If the screen consumer is kept, can it be made cheaper? Two levers, neither of which
moves it:

| lever | consume_mean |
| :--- | ---: |
| window 256 wide | 0.0965 ms |
| window 1920 wide | 0.0950 ms |
| `<gpu-texture>true` | 0.1055 ms |
| `<gpu-texture>false` | 0.1040 ms |

**The cost does not scale with the window**, so "embed a smaller preview" is not a
mitigation, and the texture path makes no difference either. The ~0.10 ms is a fixed
per-consumer frame handoff — the channel waiting for the consumer to accept the frame —
not the cost of drawing pixels, which happens on the consumer's own thread. `<vsync>`
already defaults to false, so there is no win there either.

**There is therefore no configuration-level optimisation left for the screen consumer.**
Every knob it exposes that could plausibly matter has been measured and none moves the
number. It is also the *smooth* option: 0 late frames on either mixer at five consumers.

That is what leaves Spout's 0.034 ms as the only lever that moves the publish cost at
all: it is a cheaper handoff, not a smaller picture.

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

1. ~~**No coverage.**~~ Closed 2026-08-31 by `cli.py spout-pixels` (§4). The self-round-trip §4
   used to propose — consumer on channel 1, producer on channel 2 — was never needed: the receiver
   lives in the harness process. What remains uncovered is **alpha** and **two receivers at once**.
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

6. **The Vulkan path's jitter — diagnosed, then FIXED by deferring the submission 1 ms.**

   | | before | after | screen consumer |
   | :--- | ---: | ---: | ---: |
   | `spout_256`, 5 channels | 59 / 6265 | **14 / 6265** | 3 / 6266 |
   | `spout_native`, 5 channels | 110 / 6267 | **10 / 6266** | 0 / 6262 |

   **What it was.** Each consumer's blit is one extra `vkQueueSubmit` into the queue the
   mixer composites on, issued the instant the channel hands over the frame — while the
   mixer is still working on that same tick. Two sweeps pinned it: holding channels at
   five and varying consumers gave 0/0/8/70 late for 1/2/3/5 (and 8 again at *three*
   channels with three consumers, so it tracks consumers, not channels), while a raster
   sweep from 64 px to 960 px was **flat**. Doubling the submissions per consumer roughly
   doubled the lateness. The OpenGL control settles the mechanism: it costs the channel
   *more* in handoff (0.042–0.045 against Vulkan's 0.023–0.034) and is perfectly smooth,
   so what produces the lateness is not the channel waiting for the consumer.

   **The fix is one `sleep_for` and the duration barely matters**, which is the tell:

   | defer | 0 µs | 250 | 500 | 1000 | 1500 | 2000 |
   | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
   | late / ~5010 | 44–78 | 9 | 23 | 25/17/11 | 17 | 10 |

   It is not about giving the GPU time — it is about **decoupling the submission from the
   moment the frame arrives**, so the mixer's own submission goes in first. The sleep is on
   the *consumer's* executor thread, which is already asynchronous and drops a frame rather
   than stalling the channel, so it costs the preview 1 ms of latency and the channel
   nothing: **5000 sent, 0 dropped over 20 s across five channels, still exactly 50 fps.**
   `CASPARVP_VK_BLIT_DEFER_US` overrides the 1000 µs default.

   **Four richer fixes were built and measured worse**, which is why the shipping answer is
   a sleep:

   | attempt | late / ~5010 |
   | :--- | ---: |
   | *(shipping)* defer 1 ms | **10–25** |
   | do nothing | 44–78 |
   | record + submit on the caller's thread, private command pool | 211 |
   | …the same, without the completion fence | 242 |
   | coalesce blits into one submission (mean batch 1.5, unaided) | 56–78 |
   | …with a 1.5 ms collection window forcing batches of 2.9 | 25 |
   | narrow the source barriers from `eAllCommands` | 43 |

   Batching *does* work — 25 against 71 — but plain deferral beats it at a fraction of the
   complexity, so the batching machinery was removed rather than kept. Noise floor is ±7 on
   a mean of 48 (five identical runs: 41/47/48/53/54).

   **Not zero, and worth saying.** 10–14 late frames in 6265 is ~0.2%, against a baseline
   of 0 and a screen consumer's 0–3. The remaining difference would need the blit off the
   mixer's queue entirely — folding it into the mixer's own submission (the
   `previz_texture_bridge` pattern) or a separate queue family with the ownership transfers
   that implies. Neither is worth a mixer-wide change for 0.2%.

   **`previz_texture_bridge` still has the unfixed version of this**, and worse: it submits
   once per posted channel *and* creates a fence, waits, destroys it and frees the command
   buffer every frame per channel. Unmeasured — nothing drives previz — so it is a
   prediction from the source, and the same one-line deferral would likely apply.

7. **`FPS` rounds down silently**7. **`FPS` rounds down silently**7. **`FPS` rounds down silently**7. **`FPS` rounds down silently** past the log line at `initialize()`. An operator who asks for 30
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
