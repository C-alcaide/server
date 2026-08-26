# Vulkan mixer

> **State:** shipped
> **Modules:** `src/accelerator/vulkan`
> **Commands:** none of its own — selected by `<accelerator>vulkan</accelerator>`
> **Coverage:** `conformance`, `grading`, `mixer-parity`, and every battery that takes `--mixer vulkan`

A second mixer backend beside the OpenGL one, doing the same compositing and the same grading chain
through a Vulkan pipeline. It is the backend the fork's GPU-direct work targets: FFmpeg's Vulkan
decoders, the CUDA producers' zero-copy path and the Vulkan encode consumers all require it.

Implementation detail is in
[`../architecture/VULKAN_MIXER_IMPLEMENTATION.md`](../architecture/VULKAN_MIXER_IMPLEMENTATION.md).
This document is the state, the parity position, and the traps that are specific to having two
backends.

---

## 1. What is implemented today

Feature parity with the OpenGL mixer across the compositing and grading surface: all 71
`image_transform` fields are composed by both backends, all pixel formats 0–15 are handled by both
shaders, and both implement the same seven tone-mapping operators.

Verified mechanically on 2026-08-26 rather than asserted: every non-geometry `image_transform`
field appears in **both** `apply_transform_colour_values` implementations, and the shader case
lists match.

**Selected in the config, not at runtime:**

```xml
<accelerator>vulkan [auto|opengl|vulkan]</accelerator>
```

`auto` resolves to **opengl**, so Vulkan is opt-in.

---

## 2. The two-backend traps

These are the reason this document exists rather than only the implementation note. Each has cost
real time in this tree.

**An uploaded texture must not be released before its copy retires, and until 2026-08-27 nothing
enforced that.** `submitSingleTimeCommands` submits and returns a timeline value — it never waits —
and `texture::impl::~impl()` frees at once: `destroyImageView`, `freeMemory`, `destroyImage`, with
no fence, no timeline check and no deferred-destruction sweep. So releasing a texture while its
upload was still in flight freed the `VkImage` mid-write, and the GPU took the device down:
`vk::Queue::submit: ErrorDeviceLost`.

**It survived because nothing ever released one early.** A published frame crosses the mixer, gets
drawn and is released long after its copy has retired, so the margin hid the defect completely. The
first code to release one early — the HAP producer discarding pre-seek frames after a `SEEK` — hit
it immediately: **1450 device-lost errors in eight minutes**, then 1749 on a rewrite that released
from the other thread, against **zero** on binaries without the discard.

`copy_compressed_async` now waits on its timeline value in a deferred async on the **caller's**
thread, which is the idiom `copy_async` in the same file already used. **The general fix is still
owed:** deferring destruction until the timeline passes the value that last touched the image,
which would make every texture safe rather than this one call site. Every other
`submitSingleTimeCommands` caller that hands a texture onward carries the same latent risk.

**A defect that only a new caller can reach is not a dormant defect, it is an undiscovered one** —
and the lesson generalises past Vulkan: when a change makes a previously-unreachable path
reachable, the failure it produces may belong to the path, not the change. Two attempts here were
blamed on the discard before the lifetime was suspected.

**Channel order differs between the backends.** The OpenGL mixer carries the pixel in **BGR**
through the whole grading chain — `col.r` holds blue — while Vulkan grades in **RGB**. So every
per-channel uniform needs `.bgr` at its call site in `shader.frag`, or reversal on upload, and
Vulkan needs neither. Consequences:

- **Greys are invariant under a red/blue exchange**, so a grey ramp or a neutral white balance
  passes every defect of this class. Colour tests must use asymmetric per-channel values.
- **A function's channel order is a property of its call site**, not of the function.
  `ChromaOnCustomColor` is called as `ChromaOnCustomColor(c.bgra).bgra`, so it receives RGB and must
  *not* swizzle — adding one there mirrored the hue wheel.
- This is not theoretical: the ICVFX gain was uploaded RGB and applied to a BGR pixel until
  2026-08-26. See [`projection-and-icvfx.md`](projection-and-icvfx.md).

**Both backends keep their own transform allowlist.** A new `image_transform` field must be named
in `apply_transform_colour_values` in *both* `src/accelerator/ogl/util/transforms.cpp` and
`src/accelerator/vulkan/util/transforms.cpp`, or it is silently dropped — the AMCP command returns
`202`, the query reads the value back correctly, and the kernel sees the default. Add it to one and
not the other and the backends diverge in a way no single-backend test can see. **`per_channel_levels`
was composed by different rules in the two backends until 2026-08-26**, and single-layer tests could
not see it because intersecting against defaults returns the other side's values.

**The Vulkan UBO must match the shader's block exactly.** `uniform_block.h` is std140 and its
offsets were maintained by hand-written comments until `static_assert`s were added. The failure mode
is total and silent: at size 884 instead of a multiple of 16 the mixer logged no error, decoded
normally, and produced **no readback at all** — conformance 0/4, flat-decoded 0/29.

---

## 3. Verification

| what | battery | latest numbers | date |
| :--- | :--- | :--- | :--- |
| Colour-space conversions, 1 LSB gate | `conformance --mixer vulkan` | 23/23 patches per conversion, worst **0.52 LSB** | 2026-08-26 |
| Grading operators, 1 LSB gate | `grading --mixer vulkan` | 8/8 patches, worst **0.55**, neutrals **0.00** | 2026-08-26 |
| Backend agreement | `mixer-parity` | — | — |
| Vulkan API usage | `vk-validation` | **cannot currently fail** — see below | 2026-08-26 |

**`vk-validation` is broken and a clean run from it means nothing.** A deliberately planted
`mipLevels = 0` — an unambiguous stateless VUID, verified compiled in and verified reached —
produced "0 VUID findings". Two causes are fixed (deprecated layer-setting names the layer ignores
silently; a finding regex blind to `kVUID_Core_*` ids) and a third is open: core validation emits
nothing for calls after device creation on this rig, though the layer is demonstrably in the chain.
**So no Vulkan API-usage claim in this fork currently rests on evidence.**

**What the 1 LSB batteries do not cover:** they drive a colour producer, which is BGRA — so they
never call the YCbCr decode at all and cannot see a decode defect. That is how an 8-bit-code scale
error survived in `ycbcra_to_rgba` on both backends: identical on both, so every parity check
passed, and invisible to the colour batteries by construction.

---

## 4. Known gaps

1. **`vk-validation` cannot fail.** Root cause unidentified; until it is, API-usage correctness is
   argued from reading code.
2. **Parity is checked on flat colour, not on decoded video.** A defect shared by both backends
   passes `mixer-parity` by definition.
3. **Tone-mapping operators 4–6 are implemented differently in the two shaders** (`_709`/`_p3`/
   `_2020_pq` named functions versus inlined `aces_odt_srgb(aces_rrt_s(…))`). They were checked
   equivalent for op 4 on 2026-08-26; ops 5 and 6 were not.

---

## 5. Related commits

| commit | why it matters |
| :--- | :--- |
| `b1da34a34` | the `per_channel_levels` composition divergence, the ICVFX channel exchange, and the UBO `static_assert`s |
| `0f1c5fb38` | exportable textures reached the mixer in `eUndefined`; six modules feed it through that path |

---

## 6. Diagrams

Not owed here. The mixer's own stage order belongs to
[`../guides/COLOR_GRADING.md`](../guides/COLOR_GRADING.md), which already carries it, and
duplicating it would create a second thing to keep true.
