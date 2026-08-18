# Plan: land the BGRA64 capture fix properly

**Status: plan only. 2026-08-18.** The cause is proven and written up in
[`UPSTREAM_SYNC_2026-08-18.md`](UPSTREAM_SYNC_2026-08-18.md) §3.1.1; two attempts at a fix
were reverted in `c8caa2226`. This is what a third attempt should do differently, and what it
must prove before it lands.

**No urgency.** The defect is pre-existing (identical on FFmpeg 7.0.2 and 8.1.2), invisible at
8 bits (0.035 LSB8), and affects only 16-bit captures — which in practice means the harness's
`--bit-depth 16` arms rather than anything on air. Reverting is always available and costs
nothing.

---

## 1. What is being fixed

`image_consumer` hands `convert_image_frame` an `AV_PIX_FMT_BGRA64LE` frame and asks for
`AV_PIX_FMT_RGBA64BE`. That is a channel permutation plus a byte swap, and **swscale gets the
permutation wrong** — measured on one pixel through `ffmpeg` alone:

```
bgra64le -> rgba64be   52428,39321,26214  ->  52420,39327,26205   LOSSY (-8,+6,-9)
bgra64le -> rgba64le   52428,39321,26214  ->  52420,39327,26205   LOSSY (identical)
bgra64le -> bgra64le   exact (no-op)
rgba64le -> rgba64be   exact (byte swap only)
```

Only the OGL mixer is affected, because OGL reports `pixel_format::bgra` and Vulkan reports
`rgba`, so Vulkan's capture is a byte swap and comes out exact.

---

## 2. Step 0 — the cheap fix, already ruled out

**Do not start by trying swscale flags.** Nine flagsets were tested and every one produces the
identical `-8, +6, -9`:

`(default)`, `accurate_rnd`, `accurate_rnd+full_chroma_int`,
`accurate_rnd+full_chroma_int+full_chroma_inp`, `bitexact`, `accurate_rnd+bitexact`,
`neighbor`, `fast_bilinear`.

`bitexact` failing to make a **permutation** exact is the notable one, and it is also the
strongest argument for §6.

---

## 3. Options, and why one is chosen

| # | Option | Verdict |
| :-- | :--- | :--- |
| 1 | Make the OGL mixer report `rgba` so the byte-swap-only path is taken | **No.** The BGR working order runs through the whole OGL shader and every per-channel uniform; `CLAUDE.md` has a section on the traps it already causes. Changing it to fix a capture-path defect is disproportionate and would need the full grading battery re-measured. |
| 2 | Hand-convert in place, into the existing un-premultiply scratch buffer | **Tried, reverted.** See §4. |
| 3 | Hand-convert into a **properly allocated destination frame**, bypassing swscale | **Chosen.** §5. |
| 4 | Report it to FFmpeg | **Yes, in parallel** — §6. Independent of whatever we ship. |

---

## 4. Why attempt 2 failed, in enough detail not to repeat it

Attempt 2 wrote the permuted, byte-swapped pixels back into the `std::vector<uint8_t> buf`
that the un-premultiply loop already allocates, set `av_frame->format = AV_PIX_FMT_RGBA64BE`,
and passed that frame straight to `avcodec_send_frame`.

**It was exactly right on pixels** — `image-convert` 4/4 on ogl16, vk16 and ogl8;
`conformance --mixer ogl --bit-depth 16 --quick` 36/36 with worst 0.96 LSB16 and zero
timeouts.

**And it was unshippable.** On the full 100-conversion matrix it produced **18 capture
timeouts out of 2300** and `85/89` conversions, worst 49780 LSB16. Re-running the reverted tree
confirmed the attribution: **0 timeouts, 0/100, worst 33.00** — the pre-fix baseline exactly.

Two candidate mechanisms, and the plan should assume both:

* **Allocation discipline.** `convert_image_frame` allocates its destination with
  `av_frame_get_buffer(dest, 64)` — aligned and refcounted. A `std::vector`'s data is neither.
  `avcodec_send_frame`'s contract expects a refcounted frame for encoders that retain
  references; handing it a function-local buffer is outside that contract, and "it worked in
  the small run" is exactly how that presents.
* **Cost per capture.** Even discounting the above, the loop now does three byte swaps and a
  conditional exchange per pixel over 2 M pixels, on the consumer's thread.

**A capture that does not complete reports as a wild value, not as a missing one.** That is why
the failure looked like a correctness regression (49780 LSB16) rather than a timing one, and it
is the reason the timing gate in §7 is not optional.

---

## 5. The design to implement

### 5.1 A converter that does not call swscale

Add to [`src/modules/image/util/image_converter.h`](../src/modules/image/util/image_converter.h)
alongside its sibling:

```cpp
/// Packed 16-bit BGRA/RGBA -> RGBA64BE, without swscale.
///
/// swscale's BGRA64 -> RGBA64 permutation is lossy (see docs/PLAN_BGRA64_CAPTURE_FIX.md),
/// and a component permutation plus a byte swap needs no scaler. `swap_br` is true when the
/// source is BGRA-ordered.
std::shared_ptr<AVFrame> pack16_to_rgba64be(const std::shared_ptr<AVFrame>& src, bool swap_br);
```

Implementation requirements, each of which is a thing attempt 2 got wrong or got right by luck:

1. **Allocate the destination the same way `convert_image_frame` does** —
   `ffmpeg::alloc_frame()`, set `width`/`height`/`format`, then `av_frame_get_buffer(dest, 64)`
   and check the return. Aligned and refcounted, so the encoder's contract is satisfied.
2. **Respect both linesizes.** Walk rows using `src->linesize[0]` and `dest->linesize[0]`
   independently; never assume either is `width * 8` or that the buffer is contiguous.
3. **Carry the metadata across**, as `convert_image_frame` does: `sample_aspect_ratio`,
   `colorspace`, `color_trc`, `color_primaries`.
4. **Do not modify the source.** Attempt 2 mutated the scratch buffer in place, which coupled
   this conversion to the un-premultiply loop's internals.
5. **Leave the 8-bit path completely alone.** `BGRA -> RGBA` at 8 bits measures exact
   (`image-convert --bit-depth 8` is 4/4 with the control at 0.00), so it is not in scope and
   touching it would widen the blast radius for nothing.

### 5.2 Call site

In `image_consumer`'s `is_hi_dep && (bgra || rgba)` branch, after the existing un-premultiply,
replace `convert_image_frame(av_frame, target_fmt)` with
`pack16_to_rgba64be(av_frame, pix_desc.format == core::pixel_format::bgra)`.

Everything else in that branch stays, including the un-premultiply itself — that is a separate
concern and is correct.

### 5.3 Keep the diagnostic

The `MIXER_OUT` trace already in `image_consumer` stays. It is what split "the mixer produced
this" from "the consumer mangled it", no battery can distinguish those, and without it the
defect reads as a 16-bit precision fault in the OGL backend.

---

## 6. Report it upstream, independently

This is an FFmpeg defect, not a CasparCG one, and the repro needs none of our code:

```
printf ... > one_pixel_bgra64le.raw
ffmpeg -f rawvideo -pix_fmt bgra64le -s 2x2 -i one_pixel_bgra64le.raw \
       -f rawvideo -pix_fmt rgba64be -
```

Report that an RGB→RGB **component permutation** loses precision, that it is unaffected by
`bitexact`, and that the same conversion is exact when no permutation is involved
(`rgba64le -> rgba64be`). Present on 7.0.2 and 8.1.2 alike. If upstream fixes it, options 2
and 3 both become unnecessary — which is worth knowing before investing in either.

---

## 7. Verification gates

Attempt 2 passed everything it was asked and was still wrong, because it was asked the wrong
things. These are the gates.

### 7.1 Exactness

| battery | requirement |
| :--- | :--- |
| `image-convert --mixer ogl --bit-depth 16` | 4/4, control 0.00, excess +0.00 |
| `image-convert --mixer vulkan --bit-depth 16` | 4/4 (no regression) |
| `image-convert --mixer {ogl,vulkan} --bit-depth 8` | 4/4 (no regression) |
| `conformance --mixer ogl --bit-depth 16`, **FULL matrix** | **100/100 within 1.0 LSB** |
| `conformance --mixer vulkan --bit-depth 16`, full | no regression against its baseline |
| `conformance`/`grading`, both mixers, 8-bit | unchanged from `589b30d63`'s figures |
| `grading --bit-depth 16`, both mixers | improved or unchanged |

### 7.2 Timing — the gate attempt 2 would have failed

* **Zero capture timeouts** on the full `conformance --bit-depth 16` matrix, both mixers. Not
  "few". The reverted tree achieves zero, so zero is the standard.
* **Wall-clock per capture**, measured before and after on a quiet box, must not materially
  increase. Attempt 2 was never timed, which is why a 10 s budget being breached came as a
  surprise.

### 7.3 Hygiene, because two measurements in this investigation were wrong for these reasons

* **Check the box is quiet first** — `cli.py runs`, and look for stray `casparcg.exe`. One
  earlier run was contaminated by another session's battery and read as a clean result.
* **Never conclude from `--quick`.** 36 conversions showed 0 timeouts where the full 100
  showed 18. The defect only appears at the full matrix's parallelism.
* **Capture the whole summary**, not a `grep` of a few lines. One retraction in §3.1.1 exists
  because three grepped lines of a truncated run were read as a verdict.
* **Kill only your own processes.** A blanket `taskkill` in this investigation killed another
  session's GStreamer run.

---

## 8. Harness change owed alongside it

`conformance` currently reports 18 timeouts as `85/89 conversions within 1.0 LSB` plus "not
measured" notes — which reads like a 96 % pass. A timeout is not a soft outcome: it means the
capture never happened, and the surviving numbers are computed from whatever did.

Make capture timeouts a **first-class gating signal**: counted, printed in the summary line,
and non-zero fails the battery. That is what would have caught attempt 2 without anyone
noticing the discrepancy by hand, and it belongs in the harness whether or not this fix lands.

---

## 9. Rollback

The fix is one new function plus a one-line call-site change, so reverting is a single commit.
Given the defect is pre-existing and invisible at 8 bits, **if the §7.2 timing gate fails,
revert and stop** — do not tune the harness's timeout upward to accommodate it. That would
trade a measurement everyone relies on for an error nobody can see.
