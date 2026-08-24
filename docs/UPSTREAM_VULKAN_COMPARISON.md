# This fork's Vulkan/FFmpeg work against Niklas Andersson's upstream series

Read 2026-08-24 against `niklaspandersson/casparcg-server`, branches `vulkan-refactor-1`
(= upstream PR **#1782**, *"Vulkan step 2: internal refactoring"*) and `vulkan-accel-rework-v2`
(the phase-5 work the PR body describes as still to come). Both were fetched and read; **neither
was built or run here**, so every claim below is either a reading of their source, an arithmetic
consequence of it, or a measurement already recorded in this tree against the *identical
construct*. Where that distinction matters it is stated inline.

PR #1782 is deliberately narrow — *"no code changes outside of the vulkan namespace"* — and the
FFmpeg hardware decoding is phase 5 of a five-phase plan. So the two things worth comparing sit
on different branches, and they have opposite conclusions: the refactor is a **problem we have to
absorb**, and the decode path is **mostly a validation of ours, with one defect going the other
way**.

---

## 1. The one defect: their p010 path lands neutral chroma 0.5 code low

`feat(core): nv12 and p010 pixel formats` (`e48b0f9c3`) adds a semi-planar case to both mixers'
fragment shaders and routes it through the existing `ycbcra_to_rgba`. That function, on their
branch and on upstream `master`, is:

```glsl
vec3 YCbCr = vec3(Y, Cb, Cr) * 255;
YCbCr -= vec3(16.0, 128.0, 128.0);
```

**`* 255` is exact only when the sample IS `code/255`, i.e. an 8-bit texture.** p010 puts its ten
bits in the HIGH bits of a 16-bit sample, and their commit correctly gives it
`precision_factor = 1` for that reason — so neutral chroma, code 512, arrives as
`(512 << 6) / 65535 = 0.5000076`, and the shader computes

```
0.5000076 * 255 - 128 = -0.498
```

Neutral chroma can never be zero. It is a constant offset on every 10-bit YCbCr frame — a slight
cast on greys, not a scaling error, so it survives every ratio-based check.

**This fork hit exactly this and fixed it.** The scale now comes from the texture depth —
`65535/256` for a 16-bit texture, `255.0` for 8-bit — which lands legal black on 16, neutral
chroma on 128 and legal white on 235. For p010 it is *exact*: `32768/65535 × 65535/256 = 128`.
Measured here on both mixers: channel spread on a neutral source **1.48–2.95 LSB before, 0.00
after**; against a closed-form model over 45 unclipped samples, mean 1.58 / max 5.02 LSB before,
**mean 0.009 / max 0.06** after. `CLAUDE.md` carries the account.

### Why their verification could not see it

Their commit records a careful check: *"H.264 and H.265 8-bit output is bit-identical frame for
frame; 10-bit differs by at most 1/255 on ~9.5% of pixels with zero mean"*. That is a **hardware
path against the software path**, and both go through the same `ycbcra_to_rgba`:

| path | texture | sample at neutral chroma | after `* 255 - 128` |
| :--- | :--- | :--- | :--- |
| software `yuv420p10` | 16-bit, LSB-aligned, `precision_factor` 64 | `512/65535 × 64` | −0.498 |
| hardware `p010` | 16-bit, MSB-aligned, `precision_factor` 1 | `(512<<6)/65535` | −0.498 |

**The same offset on both sides, so the comparison cancels it** — and "zero mean" is the
signature of that cancellation rather than of correctness. This is the trap `CLAUDE.md` states as
*an internal check cannot validate against a standard*; the same two arms round-tripped to each
other while both were wrong. Finding it here needed a **flat decoded** fixture — neutral by
construction, `u = v = 512`, so "neutral in, neutral out" needs no model at all — and the harness
had that fixture recorded as impossible until it was built.

Their 8-bit `nv12` case is genuinely exact (`code/255 × 255 − 128`), so the 8-bit half of their
verification is sound. Only the 10-bit half is measuring a shared error.

**Action: offer the shader fix upstream.** It is not specific to their series — upstream `master`
has carried it for every existing 10-bit YCbCr source — but their p010 path is a new instance of
it, and the fix is small and derivable rather than a matter of taste.

---

## 2. Where their implementation is better than ours: the wait is on the GPU

Both trees do the same job — turn an `AV_PIX_FMT_VULKAN` frame into mixer textures — and the
mechanics agree closely, which is itself reassuring: two independent implementations of the
`AVVkFrame` contract that reached the same shape.

| | this fork (`av_vulkan_import.cpp`) | theirs (`vulkan_frame_import.cpp`) |
| :--- | :--- | :--- |
| wait for decode | **host**, `vkWaitSemaphores` with a 1 s timeout, declines the frame on timeout | **GPU**, the copy submits with a timeline wait at `sem_value[i]` |
| signal back | `sem_value + 1` per plane, written back into the `AVVkFrame` by the caller | same, written back in the importer |
| source layout | transitions in, then **restores the incoming layout** | transitions in, then **records the new layout** into `AVVkFrame::layout[]` |
| failure policy | decline the frame, fall back to software | fall back to CPU decoding for the producer's life |

**The host wait is the real difference, and ours is a deliberate trade rather than an oversight —
but the trade was forced by an architecture their refactor removes.** The comment in
`av_vulkan_import.cpp` says it: a wait inside the submission *"would sit on the mixer's single
shared graphics queue, so an unsatisfiable value would block every other channel behind it"*. It
is a correct reading of our device: one queue, one dispatch thread, every channel behind it. They
can afford the GPU wait because PR #1782 gives each `image_kernel` its own `command_context` over
an owning `vulkan_queue`, and the copy runs on a **transfer** queue that is not the mixer's.

So the ordering is: **#1782 is the prerequisite for removing our per-frame host stall**, not a
competing design. Worth stating plainly because it inverts how the refactor reads from here —
it is not overhead to absorb, it buys something we currently pay for every frame.

**What that stall costs here is not measured.** It is one host block per decoded frame per
producer, on the producer's own thread, for the residual of decode work already submitted. Our
`decode-cost` figures (1.16 cores against software's 1.90 on ProRes 422 HQ) were taken *with* the
stall, so the number to beat exists; the improvement does not.

One point where ours is arguably the better design and worth offering back: **restoring the
incoming layout instead of recording the new one.** It costs one extra barrier per plane per
frame and it cannot be forgotten — a missed writeback leaves FFmpeg's `layout[]` lying about a
surface it will reuse, and that is the same class of defect `encode-parity --validate` caught in
our *exporter*, which never wrote `sem_value` back and re-signalled a consumed timeline value on
every frame while producing a correct picture.

---

## 3. Where they are ahead: hardware video decode, which we deferred on purpose

`aef653fc4` decodes H.264/H.265 through **Vulkan Video on the decode queue**, `8faf28dfd` adds
NVDEC handing its surfaces to Vulkan, and both reach the mixer with no host round trip. Reported:
*"Roughly 10x less CPU: 9.9s against 102.3s over 20s with four 1080p50 H.265 producers."*

Our `<vulkan-decode>` covers the **compute** decoders only — ProRes, ProRes RAW, FFV1, DPX — and
excludes h264/hevc/av1/vp9. The reason is recorded in our own plan and it is not a hardware limit:
*"today's decode allowlist excludes h264/hevc/av1/vp9 because we never declared a decode queue,
not because the GPU lacks one. Widening it is a follow-on this plan does not take."* This box has
a `QUEUE_VIDEO_DECODE_BIT_KHR` family.

So this is them doing the follow-on we scoped out, and there is nothing to reconcile: if their
phase 5 lands, we take it rather than build our own.

Two of their choices are worth adopting whether or not it lands:

* **The video path as a strategy.** `cpu_video_strategy` / `vulkan_video_strategy` make virtual
  only the three places CPU and hardware decode genuinely differ — how the decoder is opened,
  what the filter graph may contain, and how a decoded frame reaches the mixer — leaving
  scheduling, timing, seeking and buffering written once. Our `av_producer.cpp` instead threads
  `gpu_direct_video_`, `current_seek_target_` and their friends through the common code, and
  `CLAUDE.md` already records three separate bugs from getting `seek_internal`'s shared-versus-
  per-caller distinction wrong. A strategy boundary is where those distinctions would have been
  forced into the open.
* **`gpu_frame_factory`** (`c68a12060`): a producer-side counterpart to `const_frame::texture()`,
  so a GPU-aware producer downcasts its `core::frame_factory` and builds a frame whose planes
  live on the GPU, carried in `opaque()`. Our CUDA producers reach the same end by other means;
  one named interface is better than two conventions.

---

## 4. Where we are ahead, and it is the whole encode direction

Their five phases are decode, consumers, macOS and plumbing. **None of them encodes.** This fork
has, and none of it has an upstream counterpart:

* `av_vulkan_export.{h,cpp}` — the mirror of the importer: copy the composite into an
  FFmpeg-allocated `AVVkFrame` (never wrap the mixer's, whose pooled handles are recycled while
  an encoder still holds the frame), signal the timeline, restore layouts.
* The encode-queue declaration — `qf[2]` with `video_caps`, which is what lets `h264_vulkan` and
  `hevc_vulkan` reach the NVENC block through Vulkan. Measured here at 15–39% block utilisation
  where the compute encoders read 0%.
* `libplacebo` rather than `scale_vulkan` as the converter, which is not a preference: measured,
  `scale_vulkan` converts RGB to only NV12/YUV420P/YUV444P — all 8-bit — and ProRes encoding
  accepts only `YUV422P10 / YUV444P10 / YUVA444P10`. That finding inverted our own plan's premise.
* Field pairing in the FFmpeg consumer, on the host and on the GPU path.
* And the numbers: 12 concurrent 1080p25 ProRes 422 HQ recordings, 14 H.264 — **corroborated
  against standalone FFmpeg on the same GPU with the server not running**, two of the three arms
  landing on the server's figure exactly (`encode-standalone`, 2026-08-24).

If the intent is to make Vulkan *the preferred accelerator*, recording is half of what an
accelerator is for, and this is a ready set of PRs behind the shader fix in §1.

---

## 5. The cost to us: #1782 removes every hook our GPU-interop code hangs on

This is the actionable item, and it is larger than the PR's 1096/631 line count suggests. `device`
loses, in `vulkan-refactor-1`:

| removed from `device` | who in this tree uses it |
| :--- | :--- |
| `allocateCommandBuffers`, `submit` | `av_vulkan_import`, `av_vulkan_export`, `d3d11_import_bridge`, `previz_texture_bridge` |
| `dispatch_async` / `dispatch_sync` | the same four, plus the CUDA producers and `prores_consumer` |
| `create_attachment` | moved to `image_kernel` behind `renderpass` |
| `get_pipeline`, `upload_vertex_buffer`, `copy_async` | `image_kernel`, `image_mixer` |

Twelve files here call at least one of them. And our `src/accelerator/vulkan/` is **+10456/−865
against upstream master across 38 files**, with `device.cpp`, `pipeline.cpp`, `renderpass.cpp`,
`image_kernel.cpp` and `image_mixer.cpp` heavily modified on both sides — those are five of the
nineteen files #1782 touches.

**One capability the refactor drops the hook for entirely: exportable attachments.** Our
`device::create_attachment` allocates with `vk::ExportMemoryAllocateInfo`, and that is what makes
CUDA interop, the GL export bridge and NVENC GPU-direct recording possible from the Vulkan mixer.
`git grep` finds **no** `ExportMemoryAllocateInfo` anywhere in `src/accelerator/vulkan` on either
of their branches — their external-memory work is on the *import* side, for handing NVDEC
surfaces in. So after #1782 the export allocation has to be re-established at the new home
(`renderpass::create_attachment` / `image_kernel`), and it is worth telling them the hook exists
and what needs it: their own phases 2 and 3 — *"allowing image mixers to completely skip the
gpu-to-host copy if no consumers ever read the frame on host"* and a GPU-frame screen consumer —
are the same direction, and a consumer that wants the composite in another API needs exactly this.

**Recommendation: rebase onto #1782 early rather than after it merges.** The mechanical part is
moving four bridge classes onto `command_context`, which is a better home for their
"one command buffer and one fence for the life of the object" than reaching into `device` was —
and per §2 it is also what unlocks the GPU-side wait.

---

## 6. Two fixes we already had, found independently on both sides

Worth recording because they are evidence the two implementations are being read carefully rather
than a duplication to resolve:

* **`AVClass::item_name` may be null.** Their `0be4b97f2` and our `ffmpeg.cpp:88` are the same
  crash — FFmpeg's own Vulkan context leaves it null, and the first log line from any FFmpeg
  Vulkan code jumps through it. One difference worth passing on: they fall back to
  `avc->class_name`, we call `av_default_item_name(obj)`, which is what FFmpeg's own
  `AV_CLASS_*` machinery does and stays right if the default ever changes.
* **Sync objects must not be destroyed under work in flight.** Their `48ae56f82` defers a
  producer's `command_context` destruction until the GPU drains; our importer and exporter both
  wait on their fence in the destructor for the same reason. Different architectures, same
  hazard, both covered.

One place their queue fix does *not* apply to us but points at a real gap: `d11380d1f`
(*"one queue family, one device queue-create entry"*) is about `vkCreateDevice` rejecting a
duplicated family, which we cannot hit because we never create the device. But the same
uniqueness requirement applies to `pQueueFamilyIndices` on a CONCURRENT image, and that list
comes from our `hwctx->qf[]`. We guard `decode_qf == graphics_qf` by declining outright — there is
**no equivalent guard on `qf[2]`**, the encode family, so a GPU whose video-encode family is also
its compute or graphics family would emit a duplicate index. Not reachable on this box, where the
encode family is separate; worth a guard rather than a comment.

---

## What to do, in order

1. **Offer the `ycbcra_to_rgba` depth-derived scale upstream**, with the derivation and the
   before/after numbers. It affects their p010 path and every existing 10-bit source. Do not
   frame it as a defect in their commit — the shader predates it.
2. **Guard `qf[2]` against a duplicate family index** here, per §6.
3. **Rebase this fork's `src/accelerator/vulkan/` onto #1782** while it is still a PR, and move
   the four bridge classes onto `command_context`.
4. **Re-establish exportable attachments** at the refactor's new home, and tell them what needs
   the hook — their phases 2 and 3 want it too.
5. **Then** try the GPU-side timeline wait in the importer, and measure it against the
   1.16-core `decode-cost` figure the host stall produced.
6. **Offer the encode direction** — exporter, encode queue, `libplacebo`, field pairing — as its
   own series, with the standalone-corroborated ceilings.
