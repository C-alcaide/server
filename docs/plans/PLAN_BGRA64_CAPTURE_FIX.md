# Plan: land the BGRA64 capture fix properly

**Status: plan only. 2026-08-18.** The cause is proven and written up in
[`UPSTREAM_SYNC_2026-08-18.md`](../audits/UPSTREAM_SYNC_2026-08-18.md) §3.1.1; two attempts at a fix
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

### 1.1 Who is affected: this fork only

Checked 2026-08-19, because it decides whether there is a CasparCG-upstream PR to make (there
is not) and whether the FFmpeg report stands alone (it does).

**The swscale defect is FFmpeg's**, present identically in 7.0.2 and 8.1.2. **Upstream
CasparCG never triggers it.** Its entire tree contains exactly one `sws_getContext`
(`image_converter.cpp`) and four `convert_image_frame` call sites, and every one of them
targets an **8-bit packed** format:

| upstream call site | target |
| :--- | :--- |
| `image_consumer.cpp:120` | `AV_PIX_FMT_RGBA` |
| `image_producer.cpp:57`, `:78` | `AV_PIX_FMT_BGRA` |
| `image_scroll_producer.cpp:132` | `AV_PIX_FMT_BGRA` |

and the 8-bit permutation is **exact**:

```
bgra   -> rgba     102,153,204,255 -> 204,153,102,255   EXACT
bgra64 -> rgba64   26214,39321,52428 -> 52420,39327,26205   LOSSY
```

**This fork triggers it because this fork added the 16-bit capture path.** Our
`image_consumer` is 444 lines to upstream's 182, and the difference includes
`is_hi_dep` / `AV_PIX_FMT_RGBA64BE` / `AV_PIX_FMT_BGRA64LE` — none of which exist upstream.
Of the fork's own conversion sites, only that one targets a 16-bit packed format;
`image_producer` (`BGRA`, `GBRAP16LE`), `image_scroll_producer` (`BGRA`) and `isf_image_load`
(`RGBA`) do not, and the planar `rgb48be -> GBRAP16LE` conversion measures exact.

Worth stating plainly, because it cuts both ways: upstream **does** support
`<color-depth>16</color-depth>` (`server.cpp:296`), and its IMAGE consumer downconverts such a
channel to an 8-bit PNG unconditionally. So upstream cannot observe this defect, and also
cannot capture a 16-bit frame at all — which is exactly why the fork added the path.

**Consequences for this plan:** no CasparCG-upstream PR is warranted; §6's FFmpeg report is
the only outward-facing item; and the fix is entirely ours to make and ours to gate.

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
/// swscale's BGRA64 -> RGBA64 permutation is lossy (see docs/plans/PLAN_BGRA64_CAPTURE_FIX.md),
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

## 8.1 ATTEMPT 3 — implemented as planned, and reverted. Read this first.

**2026-08-19.** The design in §5 was implemented exactly as written: a `pack16_to_rgba64be`
that allocates its destination with `av_frame_get_buffer(dest, 64)`, walks rows by both
linesizes, carries the metadata, does not touch the source, and never calls swscale. It was
**exact on pixels** — `image-convert` 4/4 with control 0.00 and excess +0.00 on all four arms
(ogl/vulkan × 8/16 bit).

**It failed the §7.2 timing gate, measured back to back in one session on a quiet box:**

| build | elapsed | capture timeouts | result |
| :--- | ---: | ---: | :--- |
| swscale (this line, unchanged) | **155 s** | **0** | 0/100, worst 33.00 LSB16 |
| `pack16_to_rgba64be` | **480 s** | **10** | 93/100, worst 65535 |

**And the cause of the 3x is NOT the conversion.** Instrumenting both paths with the same
timer gives the *same* number:

```
pack16_to_rgba64be   convert=3ms   send=26ms
convert_image_frame  convert=3ms   send=27ms     (19ms on the first call: SwsContext creation)
```

3 ms is the memory-bandwidth floor for touching 32 MB, so no amount of shuffling helps — a
word-at-a-time version using `_byteswap_uint64` plus a rotate measured 3 ms as well.

**So the 3x is unexplained, and that is the open question.** Hypotheses eliminated:

* *allocation discipline* — attempt 3 allocated exactly as `convert_image_frame` does, and
  still failed;
* *conversion cost* — measured identical, both 3 ms;
* *encode cost / entropy* — the captured PNGs are the same size, 18876 vs 18877 bytes mean
  over 2277 and 2300 files;
* *ambient flakiness* — the baseline was re-run in the same session under the same conditions
  and produced 0 timeouts against the fix's 10, and the
  `Timed out waiting for consumer snapshot release` warning (`output.cpp:147`) appears 13 times
  in the fixed run and **zero** times in both the baseline and the original.

That last point is the strongest lead for attempt 4: `output.cpp:133` gives a departing
consumer two ticks / 200 ms to release its snapshot, and something about the fixed build makes
that wait time out. Since per-capture cost is identical, the difference is more likely in
*lifetime or refcounting* — how long `dest` stays alive, or which thread drops the last
reference — than in throughput. Instrument `output.cpp`'s tick counter and the frame's
refcount, not the conversion.

**Status: not landed.** The lossy `convert_image_frame` call remains, with a comment at the
site recording the defect, the measurement and the pointer here. Three attempts have now been
reverted; a fourth should start from the paragraph above rather than from §5.

---

## 8.2 ATTEMPT 4 — the premise of the first three was wrong

**2026-08-19.** Attempts 1-3 all assumed the only exact conversion is a hand-written one. That
was never established. What had actually been shown was narrower: *swscale with nine particular
flagsets, on one particular conversion,* is lossy. Widening the probe to other **routes** rather
than other flags answers it immediately.

Measured over a full 1920x1080 raster of pseudorandom 16-bit values (`swsprobe/probe1080.py`):

| conversion | result |
| :--- | :--- |
| `bgra64le -> bgra64le` (identity) | exact |
| `bgra64le -> bgra64be` (endian only) | exact |
| **`rgba64le -> rgba64be` (endian only)** | **exact, 0 of 8 294 400 components** |
| `bgra64le -> rgba64le` (permutation only) | lossy, max 32 |
| `bgra64le -> rgba64be` (permutation + endian) | **lossy, max 35, wrong on 5 937 403 of 8 294 400** |
| `bgra64le -> gbrap16le -> rgba64be` (via planar) | exact |
| `bgra -> rgba` (same permutation, 8 bit) | exact |

So swscale's endian swap is exact, its 8-bit permutation is exact, and routing through planar
16-bit is exact. **Only the packed-16 to packed-16 component permutation is broken** — and the
error is *position-dependent*: 52 546 of 62 535 distinct sampled values map to more than one
error, which is dithering, on a conversion that reduces no bit depth. `sws_dither=none` does not
suppress it. Drafted for upstream as `swsprobe/REPORT_TO_FFMPEG.md`.

### The fix that follows, and why it costs nothing

`rgba64le -> rgba64be` being exact is the whole answer, because the consumer **already** has a
writable copy of the frame in hand. The un-premultiply loop immediately above the conversion
allocates `buf`, and reads and writes all three colour components of every pixel. Exchanging B
and R in that same loop and relabelling the frame `AV_PIX_FMT_RGBA64LE` leaves swscale the byte
swap it performs exactly.

That is strictly less work than what it replaces: no new allocation, no additional pass over
memory, no second `SwsContext`, and swscale doing an easier job than before. It also leaves
untouched every part of the frame's lifetime that attempt 3 changed — which matters, because
attempt 3's unexplained 3x wall-clock regression was never traced to a mechanism, only narrowed
to *not the conversion cost*. The safest response to an unexplained regression in newly
introduced allocation is to introduce no allocation.

The planar two-step (`-> gbrap16le ->`) was the other exact candidate and is rejected: it is
also correct, but costs a second full pass and a second intermediate buffer, which is the shape
of change that went wrong last time.

### 8.2.1 Measured, 2026-08-19 — correctness settled, timing still open

Battery: `conformance --mixer <m> --bit-depth <d>`, all arms on the same box, same DLL set,
`build-sync/shell`. The baseline exe is the 01:42 revert build, preserved and run as
`casparcg_base.exe` beside the identical DLLs so the control is a true A/B.

| arm | elapsed | conversions | capture failures | within 1.0 LSB |
| :--- | ---: | ---: | ---: | :--- |
| **control** ogl 16 (no fix) | 375 s | 100 | 36 | **0/100** |
| **fix** ogl 16 | 876 s | 62 | 38 | **61/63** |
| fix vulkan 16 | 148 s | 100 | 0 | 99/100 |
| fix ogl 8 | 117 s | 100 | 0 | 100/100 |
| fix vulkan 8 | 138 s | 100 | 0 | 100/100 |

**Correctness is settled.** The control fails *every one* of 100 conversions at up to 15.92
LSB; the fixed build passes 61 of the 63 it completed. Capture failures do not turn a wrong
pixel into a right one, so this result is independent of the timing problem below. The 8-bit
arms and the Vulkan 16-bit arm are unchanged, as they must be: the block is `is_hi_dep`-only
and `swap_rb` is false for a Vulkan frame.

`grading` at 16 bit fails two LUT3D cases, **byte-identically on both mixers** (worst 6.74 and
3.87 LSB at #0A1205). A change that cannot reach the Vulkan backend cannot have caused a
failure the Vulkan backend shares, so these are pre-existing and are not a regression here.

**Timing is not settled, and two counting errors were made getting here.** First, `grep -ci
"timed out"` was compared against the matrix's `grep -c "did not produce a complete"` — two
different metrics, giving a bogus 82-vs-18. Counted identically it is 38 vs 36 capture
failures, i.e. roughly 1.7x per conversion, not 4.5x. What survives correct counting is wall
clock: **14.1 s per conversion against the control's 3.75 s, a 3.8x** that closely matches
attempt 3's 3.1x.

Second, and more seriously, **the arms ran in a monotonically improving order** — 876, 375,
148, 117, 138, 34, 37 seconds — across code paths with nothing in common. The fix arm drew the
worst conditions of the session, immediately after two builds; the control ran fifteen minutes
later on a settled box. That confound is large enough to account for the ratio on its own, so a
bracketed **fix -> control -> fix** re-run on a settled box is what decides it. Do not quote the
3.8x without that bracket.

**A mechanism check that argues against the swap being the cause.** The added work is one
`std::swap` of two `uint16_t` per pixel, inside a loop that already reads and writes both of
them, on a buffer already in ordinary RAM (the mapped GPU memory was copied out one line
above). At 1920x1080 that is about 2.07 M iterations; even at a pessimistic 10 ns each it is
20 ms, and the observed gap is of the order of 10 s per capture. The swap cannot pay for that.
`fix vulkan 16` runs the same `is_hi_dep` block and the same `rgba64le -> rgba64be` swscale
call at 1.48 s per conversion with zero failures, differing only in `swap_rb`.

**If the bracket confirms a real regression**, the next move is not another variant of this
approach but removal of swscale from the path: `RGBA64BE` differs from `RGBA64LE` only in byte
order, so the existing loop can un-premultiply, exchange B/R *and* byte-swap in the copy it
already holds, set the frame's format to `AV_PIX_FMT_RGBA64BE`, and let `convert_image_frame`
early-return on the format match. One pass, no allocation, no swscale.

### 8.2.2 The bracket — the 3x was never real, and it cost three reverts

Bracketed **fix -> control -> fix** on a settled box, same battery, same flags, same DLLs,
back to back:

| arm | elapsed | within 1.0 LSB | capture failures |
| :--- | ---: | :--- | ---: |
| fixA | 258 s | **99/100** | 4 |
| control (no fix) | 141 s | **0/100** | 0 |
| fixB | **133 s** | **99/100** | **0** |

**fixB is faster than the control** and has no capture failures at all. And fixA against fixB
— the *same binary, same flags, minutes apart* — differ by 1.9x. That spread is the
measurement noise of this battery on this box, and it is larger than the effect three attempts
were reverted for.

So the "unexplained 3x wall clock" recorded in §8.1 was an artefact: every attempt was measured
on its first run after a build, and the control it was compared against was not. The arms of
the earlier matrix ran 876, 375, 148, 117, 138, 34, 37 seconds in that order, across code paths
with nothing in common — a warm-up curve, read as a code regression three times over.

**The lesson is procedural, and belongs in the harness rather than here:** a timing verdict from
`conformance` needs at least two runs of each arm, interleaved, or it is not a verdict. A single
first-run-after-build number is not comparable to anything.

**Final state of the fix.** OpenGL 16-bit goes from 0/100 to 99/100 conversions within 1.0 LSB,
which is exactly the Vulkan number. The one remaining failure is byte-identical on both
backends — `pq/bt2020 -> bt709/linear`, worst 1.12 LSB at #4080BF — so it is pre-existing,
shared, and marginally over the gate rather than anything this touched. 8-bit is 100/100 on both
mixers. `grading` fails two LUT3D cases identically on both backends, also pre-existing.

The byte-swap-in-place variant sketched in §8.2.1 is therefore **not needed**. It remains the
right move only if swscale's `rgba64le -> rgba64be` ever regresses.

---

## 9. Rollback

The fix is one new function plus a one-line call-site change, so reverting is a single commit.
Given the defect is pre-existing and invisible at 8 bits, **if the §7.2 timing gate fails,
revert and stop** — do not tune the harness's timeout upward to accommodate it. That would
trade a measurement everyone relies on for an error nobody can see.
