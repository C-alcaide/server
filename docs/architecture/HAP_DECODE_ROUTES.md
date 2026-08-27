# HAP — three decode routes, and the seek machinery under them

Why this module is shaped the way it is. Operator documentation is in
[`../guides/PIPELINE_EFFICIENCY_GUIDE.md`](../guides/PIPELINE_EFFICIENCY_GUIDE.md) §HAP; state and
measurements are in [`../features/hap.md`](../features/hap.md). This file is the part neither has:
**why there are three routes, what each hands the mixer, and how the seek invalidation works** —
because the last of those cost a full session and two failed fixes.

---

## 1. Three routes, chosen once at construction

`hap_producer_impl`'s constructor `dynamic_cast`s the frame factory and picks a route it never
changes (`hap_producer.cpp:330-350`):

| mixer detected | route | flags | what the mixer receives |
| :--- | :--- | :--- | :--- |
| OpenGL | **GL FBO resolve** | both false | an ordinary RGBA texture, YCoCg already resolved |
| Vulkan | **BC upload** | `use_vk_upload_` | the **compressed BC texture**, resolved in the mixer shader |
| neither | **CPU decode** | `use_cpu_decode_` | BGRA in host memory |

The choice is structural, not a fallback chain: the route is decided before the first frame and
cannot change, because the GL context and the texture pool are built for it.

### Why OpenGL resolves in the module and Vulkan does not

This is the asymmetry that surprises everyone, and it is deliberate.

The **OpenGL** route runs its own fullscreen pass through an FBO (`gl/hap_gl_decode.cpp`) with its
own shaders — `FS_PASSTHROUGH` for DXT1/DXT5 and `FS_YCOCG` for HAP Q — and publishes the resolved
result as `pixel_format::rgba`. Those shaders swizzle to **BGR** because the mixer carries the pixel
that way, and all three say so in a comment.

The **Vulkan** route publishes `pixel_format::ycocg_dxt5` and lets the mixer's own shader (case 13)
resolve the YCoCg while sampling the compressed texture. Nothing is decoded in the module.

The consequence is a real difference in output, not just in code path: because the mixer samples the
**compressed** texture, filtering happens *before* the YCoCg resolve, and `scale` is a per-texel
divisor. Interpolate-then-divide is not divide-then-interpolate. So the two routes agree exactly at
1:1 and diverge where the raster forces resampling — measured at 5 LSB (PAL) to 16 LSB (2160p), and
recorded in `../features/hap.md` §4 as an accepted divergence rather than a defect. The OpenGL route
is the more correct of the two; the Vulkan one is what every HAP implementation does.

### HAP Q Alpha takes none of them

`hap_producer.cpp:977` sends `HapVariant::HapQAlpha` to the CPU decoder even on Vulkan, because the
zero-copy path cannot carry two textures. So `pixel_format::ycocg_dxt5a` — shader case 14 on both
mixers — is **published by nothing**. It is dead code, correct as written, and unreachable.

---

## 2. Snappy on worker threads, not LZ4 on the GPU

The sibling GPU codec module, `cuda_notchlc`, decompresses with **nvcomp LZ4 on the device**. HAP
uses **Snappy on CPU worker threads** with a bounded queue (`hap_producer.cpp:92`). Both choices are
deliberate and were reached independently; `hap_producer.cpp:23` records the contrast.

The bound is the load-bearing part: workers drain the raw queue as fast as they can, and the bound
is what stops a fast disk from growing an unbounded backlog of decompressed frames.

---

## 3. The seek epoch — how invalidation actually works

The part worth reading before touching this module.

A seek arrives on the **decode side** while the consumer is somewhere else entirely, so
invalidation cannot be a simple flush. The mechanism is an **epoch counter**, `seek_epoch_`:

```
io_loop            bump seek_epoch_        <- FIRST
                   clear raw_queue_
                   clear ready_queue_
                   demuxer_->seek_to_frame(target)
                   seek_done_ = true

workers            stamp each packet with the epoch it was read under
                   drop items whose epoch != current, twice:
                     before decompression   (rp.epoch, :750)
                     after  decompression   (:779)

gl_loop            waits on done_cv_, drains stale-epoch items from done_pq_,
                   tags each decoded frame with the epoch it was decoded under

consumer           discards stale-epoch frames AT THE POP
                   (receive_impl before its empty test; last_frame before it
                    consumes seek_done_)
```

**Three properties of this design are load-bearing, and each was learned by getting it wrong:**

**The epoch is bumped BEFORE the queues are cleared.** It used to be bumped after. That left a
window in which `gl_loop` could finish decoding a pre-seek frame, still read the old epoch, and push
it into the queue the seek had just emptied. Measured: `CALL SEEK 7` after `LOAD` left the channel on
frame 0 — exactly the frame `LOAD` leaves — in 5 of 24 Vulkan captures.

**The stale frame is discarded by the CONSUMER, never by the decode thread.** Dropping it in
`gl_loop` is the obvious implementation and it produced **1450 `vk::DeviceLostError` in eight
minutes**. Releasing a `draw_frame` destroys the Vulkan texture it owns, and doing that off the mixer
thread frees a `VkImage` the GPU may still be writing into. Rewriting it to release from the mixer
thread still produced 1749 — the thread was never the issue; the underlying defect was that
`copy_compressed_async` returned a texture whose upload had not retired. That is fixed in
`accelerator/vulkan/util/device.cpp`, and the discard now belongs to the consumer regardless.

**`last_frame()` must discard before it consumes `seek_done_`.** The flag is one-shot: the first
successful pop clears it. So a single stale frame taken there was cached and the flag cleared, and
nothing ever revisited it — a transient race made permanent, presenting as a paused channel holding
the wrong frame for the rest of the run.

Dropped frames are counted in `hap/drop-stale`; non-zero means the invalidation is working.

---

## 4. HAP is the reference case for loop-wrap behaviour

Worth knowing because it is cited from elsewhere. On a four-frame loop, HAP shows all four frames
while the CUDA ProRes producer showed only `{40, 41}` — two lost per iteration. The comparison is
recorded in `av_producer`'s loop-wrap reasoning, and **HAP is the one that was right**. It performs
no flush at the wrap, which is the behaviour the other producer was corrected toward.

---

## 5. What this module has never been measured on

* **No reference-decoder comparison, on any variant.** `mixer-parity --codec hap_q --decoder
  hap_native` compares the two backends against each other; nothing compares either against
  FFmpeg's own HAP decoder, though the ground-truth references now exist.
* **HAP Q Alpha has no fixture and no reachable code path** — §1.
* **The Snappy worker count against channel count has never been profiled**, so there is no
  channel-count ceiling for HAP as there is for ProRes.
* **A `CALL SEEK` race remains possible in principle** on the routes other than Vulkan: the epoch
  machinery is shared, but only the Vulkan path has been driven hard enough to expose it.
