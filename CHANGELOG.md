CasparVP — Unreleased
==========================================

### Fixed: three grading defects found reviewing the same code upstream

Ported back from the review of `CasparCG/server#1765`, which is where these six tools
originated. All three are on **both** mixers.

**`MIXER MIDTONE 0` produced `Inf` on HDR samples rather than a clamped curve.**
`apply_lmg` guarded the exponent instead of the value: `max(vec3(0.01), 1.0 / midtone)`. At
`midtone == 0` the reciprocal is `+Inf` and `max(0.01, Inf)` is `Inf`, so the guard does
nothing for the case it looks like it guards — `pow()` returns 0 below white and `+Inf` for
any sample above 1.0, and the `Inf` then reaches `color *= opacity` and the blend stage as
`NaN`. A negative `midtone` was silently turned into exponent `0.01` rather than rejected.
Now `pow(c, 1.0 / clamp(midtone, 0.01, 100.0))`.

**No in-range grade changes.** `[0.01, 100]` was chosen to reproduce the old exponent bounds
exactly: swept `midtone` over `[0.01, 100]` against `c` over `(0, 4]`, old and new agree to
**0.0** — over that range they are the same expression. Only previously broken input moves:
`midtone` of 0 or below now behaves as `0.01` instead of collapsing the channel.

**Accumulated hue shift was unbounded.** Combining transforms added `hue_shift` without
bound, so layer + channel + tween could walk it arbitrarily far from zero. Now wrapped with
`std::remainder(…, 360.0)` — wrapped, not clamped, because 200° of rotation is −160°, not
180°. This changes no rendered pixel: the shader rotates with `fract()`, so 540° already
rendered identically to 180°. What it fixes is precision loss in `fract()` at large
magnitudes, and the `abs(hue_shift) > epsilon` test that reads an accumulated 360 as active
when it is exactly identity. Verified: `0→0, 15→15, 180→180, -180→-180, 200→-160, 360→0,
540→-180, 1080.5→0.5`.

**Split-tone balance was selected by a float equality test.** `other.split_balance != 0.5`
treated a tweened `0.4999999` as an explicit setting. It now keys off whether the incoming
transform actually has split toning active, which is the test the mixer already uses to
decide whether to run the effect. The override itself stays and the comment now says why:
`split_balance` is a crossover *position* in luma, not a strength, so no arithmetic
composition of two of them means anything (0.5 + 0.5 puts everything in shadow; 0.5 × 0.5 =
0.25), and the shader has one crossover for one colour pair — two stacked split tones cannot
both be represented however they are combined.

**Not ported: the `glUniform3f` fix**, which this fork already carries. Upstream's
`shader::set(name, double, double, double)` passes `value1` twice, so every `vec3` uniform
arrives as `(x, y, y)`; that is what made `MIXER CDL`, `MIXER LIFT/MIDTONE/GAIN` and
`MIXER SPLITTONE` miss the model by 31.83, 15.68 and 15.00 LSB on an upstream build, and
0.51, 0.55 and 0.50 LSB after. Recorded here because it is the reason this fork's grading
battery passes and an upstream build's does not.

**Measured on both mixers**, `cli.py grading`: **48/48 inside gate on each**, and the three
affected ops return the same number on both backends —

| op | OpenGL | Vulkan |
| :--- | ---: | ---: |
| `Lmg` — `MIXER LIFT`/`MIDTONE`/`GAIN` | 0.55 LSB | 0.55 LSB |
| `SplitTone` | 0.50 LSB | 0.50 LSB |
| `HueShift` | 0.42 LSB | 0.42 LSB |

each op's neutral row exact at **0.00** on both, and all eight interaction stacks giving the
same pixel when their commands are sent in reverse order. The OpenGL figures also match an
upstream build carrying the same three changes to the digit, which is the cross-check that
the two codebases have not drifted on these ops.

`cli.py conformance` is **100/100 within 1 LSB on both mixers** as well. Both `.frag` files
were edited, so the whole shader binary is rebuilt on both backends and the colour-conversion
path is worth re-gating even though nothing in it was touched.

**What that proves, and what it does not.** It is a regression gate, not a demonstration of
the fixes. Every parameter in the battery sits inside the range where old and new agree by
construction — midtone `1.10/0.95/1.20`, a single-layer hue shift, split balance `0.45` with
non-zero colours — so 48/48 says nothing broke, and the numeric verification quoted above
is what says each fix is right.

**The fixes are not reachable by this battery at all**, which is worth recording rather than
hiding. `MIXER MIDTONE 0` diverges only for a sample above 1.0, and the battery's flat
patches are 8-bit `PLAY #RRGGBB` colours that cannot exceed it — an 8-bit case would pass
identically against the broken build, which is exactly the "proves nothing happened" failure
the battery guards against elsewhere. The hue accumulation needs two stacked transforms
(layer plus channel) and every case here is single-layer. Both are real coverage gaps, not
oversights in this change.

### Fixed: `MIXER PROJECTION_BLEND_MASK` never reached the shader, and had the wrong channels underneath

Two defects, one on top of the other, in a feature that has never worked.

**The mask was dropped during transform composition.** `apply_transform_colour_values` in
`src/accelerator/{ogl,vulkan}/util/transforms.cpp` composes a layer's `image_transform` field
by hand-written field, and `blend_mask` was not named in either copy. So the command returned
`202`, `MIXER PROJECTION_BLEND_MASK` reported the mask back at its correct dimensions, and
the shader saw nothing. Measured before the fix: all four patches **byte-identical to no mask
at all**, on both mixers. It now composes innermost-wins, like the 3D LUT — two masks cannot
be combined without resampling one onto the other's raster, and picking a resampling rule
silently is worse than picking the layer's own.

**And once the mask arrived, OpenGL multiplied the wrong channels.** `col.rgb *=
texture(blend_mask_tex, uv).rgb` on a shader that carries the pixel in **BGR** — the next
line is `fragColor = col.bgra`. Asked for a mask of `(0.8, 0.6, 0.4)`, the picture came back
multiplied by `(0.4, 0.6, 0.8)`. Now `.bgr`. The Vulkan kernel grades in RGB and correctly
does not swizzle; it passed unchanged, which is what says the two agree for a reason.

That second defect had never been reachable, and it is the reason a neutral mask is not an
acceptable test: grey is invariant under a red/blue exchange, so an equal-channel mask passes
whether or not the swizzle is right.

**Measured on both mixers, byte-identical**, `cli.py blend-mask` — reference is the *measured*
unmasked frame times the mask, so no colour model is involved:

| patch | delta | separation | `NONE` restores |
| :--- | ---: | ---: | ---: |
| (0.80, 0.60, 0.40) | 0.20 LSB | 61 LSB | 0.00 LSB |
| (0.70, 0.50, 0.30) | 0.40 LSB | 51 LSB | 0.00 LSB |
| (0.60, 0.45, 0.75) | 0.40 LSB | 115 LSB | 0.00 LSB |
| (0.50, 0.35, 0.65) | 0.40 LSB | 100 LSB | 0.00 LSB |

Behaviour change for anyone who had a blend mask configured: it did nothing before and does
something now. That is the fix, but it will look like a regression to a configuration that
was silently ignoring the mask.

### Added: `MIXER EXPOSURE`, and exposure now reaches an OCIO layer

```
MIXER 1-10 EXPOSURE 0.6 [duration] [tween]
MIXER 1-10 EXPOSURE            # query
```

A linear gain in the working space. Until now exposure existed only as `MIXER COLORSPACE`'s
sixth argument, which lives in the colour-grade state and is mutually exclusive with
`MIXER OCIO` — so on an OCIO layer exposure was **unreachable**. The multiply also sat
inside the input-conversion block the OCIO splice replaces, the same place gamut compression
was, so moving it out was the other half of that fix.

Like gamut compression, it is gated on the pixel having reached the working space by any
route; on a layer with no conversion the command sets its state and the shader does nothing,
because a "linear" gain on a display-encoded pixel is not a gain on light. Where both
`MIXER EXPOSURE` and `MIXER COLORSPACE`'s sixth argument are set they **multiply** — both
are scalars, so composition is the only answer that is not arbitrary, and existing
`MIXER COLORSPACE` behaviour is unchanged. Negative and non-finite gains are refused.

**No rendered-output change for any existing configuration.** Exposure moved past the gamut
matrix on Vulkan and stayed where it was on OpenGL; a scalar commutes with a linear matrix,
so both produce what they produced before. The OpenGL kernel additionally now uploads the
uniform on *every* path — it previously left it unset on the no-conversion branch, which was
harmless only because the multiply was inside a block that branch does not enter.

**A field is not plumbed until it is in the allowlist.** The first working build of this
command was accepted, stored, queryable and completely inert:
`apply_transform_colour_values` in `src/accelerator/{ogl,vulkan}/util/transforms.cpp`
composes a layer's `image_transform` field by hand-written field, and a field not named
there is dropped, so the kernel read the default `1.0` every frame. Both mixers keep their
own copy of that list. `exposure` now composes **multiplicatively**, like `opacity`
alongside it — nested transforms each contribute a gain and the composition of two gains is
their product.

Measured on both mixers, `cli.py ocio-exposure`: **0.51 LSB** against the model on both
source spaces, with the (1.0, 0.6) pair separating by **26.0** and **24.0 LSB**, and the
no-working-space control byte-identical. Plus `cli.py conformance --exposure` at 0.5, 1.6
and 2.5: 100/100 within 1 LSB, both backends.

### Fixed: `MIXER GAMUTCOMPRESS` did nothing on a layer using `MIXER OCIO`

The command was accepted, returned `202`, and the kernel set `gamut_compress_enable` true —
after the OCIO branch had cleared it, so the uniform was correct and the shader still never
executed the compressor. `apply_gamut_compress` sat *inside* the input-conversion block, and
the OCIO splice **replaces** that block rather than following it. A command that reported
success and changed nothing.

The call now lives outside that block on both backends, immediately after the splice, which
is where it belonged: it operates on ACEScg after the matrix, so it is a working-space
operation and not part of the conversion. It is gated on the pixel having reached the
working space by any route — `MIXER OCIO`, `MIXER COLORSPACE`, `<auto-color-convert>` or a
working-space composite. A layer with no conversion at all is still display-encoded, and
compressing there would mean nothing, so the command remains inert there as before.

**Behaviour change beyond the fix.** On OpenGL the compressor ran *before* the exposure
multiply and on Vulkan *after*; it now runs after on both. This changes OpenGL output only
for a layer that sets a non-default exposure — reachable solely through `MIXER COLORSPACE`'s
6th argument — and it moves OpenGL onto Vulkan's answer rather than away from it.

**Measured on both mixers, byte-identical**, `cli.py ocio-gamut-compress`:

| source space | vs. model | (off, on) separation |
| :--- | ---: | ---: |
| ARRI LogC3 (EI800) | 0.53 LSB | 28.0 LSB |
| S-Log3 S-Gamut3.Cine | 0.51 LSB | 25.0 LSB |
| control — layer in no working space | — | **0.00 LSB** |

The separation is the load-bearing number: a shader that ignores the flag matches the "off"
model perfectly, so only an (off, on) pair that moves says the flag was read. The battery
asks for limits `1.20 / 1.35 / 1.50` rather than the ACES defaults, so a build that ignored
the arguments and applied its own constants cannot pass either. The control is what says the
working-space gate holds.

`exposure` remains unavailable on the OCIO path — it has no standalone command, and its only
setter (`MIXER COLORSPACE`'s 6th argument) is mutually exclusive with `MIXER OCIO`. But the
reason it looked *blocked* was wrong and is worth recording: the two mixers apply exposure at
different points (OpenGL after the gamut matrix, Vulkan before), and that was read as a
parity hazard serious enough to pin the argument at 1.0 in the conformance battery. It is not
one. A scalar commutes with the matrix, which is linear, and with `apply_gamut_compress`,
which is homogeneous of degree one — `compress(s·c) == s·compress(c)`, max deviation 1.1e-15
over 200k random colours. Measured at exposure **0.5, 1.6 and 2.5**: 100/100 and 36/36 within
1 LSB on **both** mixers. `cli.py conformance --exposure` now carries it.

### Added: `<ocio-display>` / `<ocio-view>` on the DeckLink and screen consumers

The per-consumer view override existed but only the IMAGE consumer declared one, so the
case it was built for — a channel feeding an LED processor *and* an SDI monitor two views of
one composite — could not actually be configured. It can now:

```xml
<decklink>
    <device>1</device>
    <ocio-display>Gamma 2.2 Rec.709 - Display</ocio-display>
    <ocio-view>ACES 2.0 - SDR 100 nits (Rec.709)</ocio-view>
</decklink>
```

Both elements or neither; the server refuses one alone rather than rendering the channel's
view while looking configured.

**Measured on both mixers**, `cli.py consumer-view --consumer {image,screen,decklink}`:
4/4 patches routed in every case, the two views 28–50 LSB apart.

| consumer | readback | deviation from its own view |
| :--- | :--- | ---: |
| image | the PNG the mixer produced | 0.2–0.4 LSB |
| screen | `PrintWindow(PW_CLIENTONLY)` | 0.2–0.4 LSB |
| decklink | over the SDI wire, second card looped back | 3.2–5.2 LSB |

The DeckLink figure is the wire — RGB → 4:2:2 → back, limited range — not a wrong view, and
it is held to its own gate for that reason while the exact paths stay at 1 LSB.

Nothing else changed: the mixer fan-out and `output` routing are the same code the IMAGE
consumer already used. Each new consumer costs an `ocio_view()` override and two lines of
config parsing.

### Fixed: OCIO transforms no longer compile on the frame path

Selecting an OCIO transform cost a dropped frame. The GPU program — OCIO generation, LUT
upload, GLSL compile, driver pipeline build — was built on the **first draw that needed it**,
~1.2 s for an input transform and worse for a display transform, whose source is ten times
larger. Measured before the fix: a capture **1.6 s after `OCIO_DISPLAY` returned no frame at
all**, because an ACES 2.0 display transform is ~15 KB of GLSL.

`image_mixer::prewarm_ocio()` now builds it on the device thread when the command is
accepted, and returns immediately — the compile still costs what it costs, but it no longer
costs a frame. `MIXER OCIO` and `OCIO_DISPLAY` call it, and **the mixer pre-warms consumer
views itself when the view set changes**, because a consumer has no command to hang it on.

It goes through the same `select_ocio_variant` a draw uses, so the cache key is by
construction the one the draw computes.

**Measured: 0 compiles on the frame path on either mixer**, against one per view before.
`OCIO_DISPLAY` returns in 34 ms.

Getting that number required fixing the measurement first. The warning lives *inside* the
variant builder, so it fired for a pre-warm too and still said "on the frame path" — it
could not distinguish the two states it was being used to compare. It now reports which, so
"compiling ON THE FRAME PATH" means what it says and pre-warms are logged separately.

Unregressed: `ocio` 18/18, `ocio-display` 4/4 and `consumer-view` 4/4 on both mixers,
`vk-validation --render-format fp16` 0 VUIDs.

### Added: per-consumer OCIO views — one composite, several looks

A channel feeding an LED processor and an SDI monitor can now give them different views of
the same composite. The mixer runs one post-composite pass per **distinct** view over the
same working-space composite and hands each consumer the frame it asked for; consumers do no
GPU work of their own.

```
OCIO_DISPLAY 1 "<display>" "<A>"        the channel's own view
ADD 1 IMAGE <name> "<display>" "<B>"    this consumer's view
```

A consumer declares its view by implementing `frame_consumer::ocio_view()`. The IMAGE
consumer is the first to do so — because it is the one a measurement can read back. Any
other consumer gains one by implementing that and parsing it in its factory; nothing else
needs to change.

Requires `<working-space-composite>`, for the same reason the channel display transform
does: a display transform is not invertible, so a second view cannot be derived from the
first once one has been applied. The fan-out has to happen while the composite still exists.

**Measured** with `CasparCG-TestRunner/cli.py consumer-view`, OGL and Vulkan
**byte-identical**: 4/4 patches routed, each frame within **0.5 LSB** of OCIO's CPU model
for its own view, the two views 28–50 LSB apart.

**Implementation.** `image_mixer::render()` now returns `core::render_output` — a primary
plus one result per view. On Vulkan it is one renderpass with N attachments and one fence,
and the float resolve became an explicit draw because `set_resolve_target` is per pass and
singular. That moved the primary's fp16 resolve off the path the validation layers had
already cleared, so it was re-run: **0 VUIDs**, and `conformance --render-format fp16`
**100/100**.

The still-frame cache caches the whole tick, views included, and compares the view **count**
before trusting itself — a consumer attaching or detaching changes the view set without
changing a single layer.

Unregressed: conformance 100/100, grading 48/48, `ocio` 18/18, `ocio-display` 4/4,
`blend-domain` correct in both domains, `mixer-parity` 6/6 rasters identical.

### Added: `<working-space-composite>` — blending in scene-linear light

Both mixers convert each layer to the channel's display encoding **before** the blend, on
purpose, so that blend modes operate on 0–1 display values. This option turns that around:
every layer converts **into** scene-linear ACEScg, none of them out of it, and the channel
applies the display encoding **once** to the composite, immediately ahead of the LED
calibration LUT.

```xml
<channel>
    <render-format>fp16</render-format>
    <auto-color-convert>true</auto-color-convert>
    <working-space-composite>true</working-space-composite>
</channel>
```

**Both preconditions are refused, not warned about.** fp16 because ACEScg carries values
above 1.0 and below 0 that a unorm target clamps away; `auto-color-convert` because every
layer needs a defined route into the working space, and without one a layer reaches an
ACEScg composite still display-encoded with nothing downstream able to tell.

**Default off, because it changes every composite of two or more layers.** Measured with
`CasparCG-TestRunner/cli.py blend-domain`, both mixers byte-identical:

| | verdict | worst Δ from the selected model |
| :--- | :--- | ---: |
| default | display domain — **unchanged** | 0.50 LSB |
| `<working-space-composite>` | **working domain** | 0.60 LSB |

The textbook figure: a 50% mix of black and white reads **128** blending display values and
**191** blending light.

**No shader change was needed.** The input and output halves have been independent uniforms
since A4e split them, so the layer draws simply suppress the output half, and the
post-composite pass reuses the branch the OCIO input transform already uses — input half
off, output half on, `luminance_scale` 1.0. The pass itself follows `apply_calibration_lut`
rather than inventing a shape.

Three consequences, all documented in `COLOR_GRADING.md`:

* **`MIXER COLORSPACE`'s output half is overridden** — the channel owns the output encoding
  now. Its input half still applies.
* **The `k_direct` / `k_direct_cg` shortcuts do not run**; they leave the pixel in the
  output gamut and the composite has to be in AP1.
* **A single layer is unaffected**, which is why `conformance` and `grading` cannot see this
  option at all and `blend-domain` had to exist.

This is the prerequisite for a channel-level OCIO display transform and for per-consumer
views: a display transform is not invertible, so a composite that is already display-encoded
cannot be re-encoded for a second view.

Default path unregressed: `conformance` 100/100 and `ocio` 18/18 on both mixers, `grading`
48/48, and `blend-domain` without the flag still reports the display domain.

### Fixed: four of seven ACEScg gamut matrices, and six auto-convert entries

**`k_to_working` and `k_to_output` were not the matrices they claimed to be for `bt2020`,
`p3_d65`, `arri_wg3` and `sgamut3cine`** — worst deviation 0.41 per element, identical on
both mixers. `k_direct` (auto-color-convert) had **6 of 16 checkable entries wrong**. Found
while designing the working-space composite, by checking the tables it would sit on.

Full account, evidence and the corrected rows: `docs/GAMUT_MATRIX_DEFECT_2026-08-12.md`.

**This changes rendered output, and neutrals are exactly where it does not.** Measured, in
8-bit LSB after the output OETF:

| patch | bt2020 channel | p3-d65 channel | ARRI WG3 source | S-Gamut3.Cine source |
| :--- | ---: | ---: | ---: | ---: |
| mid grey | **0.0** | **0.0** | **0.0** | **0.0** |
| warm skin | 6.1 | 4.5 | 13.2 | 11.5 |
| saturated red | 18.2 | 15.9 | 30.9 | 52.6 |
| saturated green | 66.6 | 9.1 | 58.6 | 60.3 |

Grey invariance is why it survived — the same blind spot `CLAUDE.md` documents for channel
swaps. Anyone with an approved look on a wide-gamut channel or a camera-log source will see
a change; it is a correction, not a regression.

**Affected paths**, all of them ones this fork exists for: `MIXER OCIO` on a BT.2020 or P3
channel, `MIXER COLORSPACE LOGC3 ARRI_WG3 …` (the first usage example in
`COLOR_GRADING.md`), `MIXER COLORSPACE SLOG3 SGAMUT3_CINE …`, and any tone-mapped
conversion. **Not** affected: the default BT.709↔BT.2020 manual path, which uses
`k_direct_cg` — audited and correct in all four entries, which is why `conformance` reported
100/100 throughout.

**Also fixed, a separate defect:** `k_to_output` was addressed with the wrong index space.
`gamut_index()` returns the `k_direct` index (p3_dci=3, adobe_rgb=4) and was used to index
`k_to_output`, whose order is the MIXER COLORSPACE enum (3=AP0, 4=AP1) — so `MIXER OCIO` on
a p3-dci channel applied ACEScg→AP0 and on an adobe-rgb channel applied the identity. Both
tables now carry p3-dci and adobe-rgb rows of their own, addressed by a separately named
`working_gamut_index()`.

**Where the numbers come from:** OCIO 2.5.2 through the pinned studio config, which is what
the server links. The config has no linear P3-DCI space, so those rows come from a
colorimetric derivation validated against OCIO to **1.1e-7** across the gamuts whose
primaries are exact by standard. The derivation is deliberately not used for the camera
gamuts, where it disagrees with OCIO by up to 0.0056 — published primaries vary in the last
digits and the config is authoritative.

**No battery could have caught this**, which is the second finding. The harness transcribes
these same tables, so a wrong matrix was compared against a copy of itself and passed.
`CasparCG-TestRunner/core/gamut_reference.py` now derives them from OCIO instead, and
`cli.py ocio` sweeps the channel gamut — it previously ran on BT.709 only, one of the three
rows that happened to be correct.

Verified: `ocio` **18/18 within 1.0 LSB** (3 channel gamuts × 6 spaces, worst 0.55) on both
mixers, `conformance` 100/100 both, `grading` 48/48 both, 1247 harness tests.

### Verified: `<render-format>fp16</render-format>` renders correctly on both mixers

The element has been parsed since the float-composite work landed and **no configuration
had ever selected it**, on either mixer, so neither the fp16 render target nor the Vulkan
fp16 resolve blit in `renderpass::commit()` had executed once.

Both now have:

| | |
| :--- | :--- |
| fp16 render target, OpenGL and Vulkan | `conformance --render-format fp16` — **100/100 conversions within 1.0 LSB** |
| the Vulkan fp16 resolve blit under the Khronos validation layers | **0 VUIDs**, 12/12 commands accepted, server survived, positive control reporting |

Confirmed against the server's own `[server] Channel 1 render-format fp16 (float working
space)` log line rather than against the battery passing, because a silently dropped element
would have passed too.

**No server code changed.** The barriers and the resolve path were correct as written; what
was missing was any way to aim a measurement at them.

**And the precision cost is now measured rather than calculated.** `cli.py banding` reports
the render target's actual quantisation step per signal level:

| level | unorm | fp16 | predicted `2^(e-10)` |
| ---: | ---: | ---: | ---: |
| 0.9 | 1.0 | **32.0** | 32.0 |
| 0.6 | 1.0 | **32.0** | 32.0 |
| 0.3 | 1.0 | **16.0** | 16.0 |
| 0.1 | 1.0 | **4.0** | 4.0 |
| 0.02 | 1.0 | **1.0** | 1.0 |

LSB16 = 1/65535; a 12-bit output step is 16.0 and a 10-bit one 64.1. So **fp16 near white is
exactly 2× coarser than a 12-bit output and half a 10-bit step** — a real highlight
regression for a 12-bit or better consumer, and comfortably fine below that. Both mixers
byte-identical.

Two corrections to what `OCIO_INTEGRATION_STUDY.md` §4.3.4 assumed. There is **no `bit12`
channel**: `<color-depth>` accepts only 8 or 16 and `bit12` is only ever a source pixel
format, so 12-bit is a consumer property rather than a channel one. And fp16's *shadow*
advantage — the half of §4.3.4 that argues for it — sits below a 16-bit capture's floor and
is neither confirmed nor refuted.

### Added: `<straight-alpha-grading>` — the colour chain on unpremultiplied RGB

Both mixers apply the whole colour chain — the EOTF, `MIXER COLORSPACE`, `MIXER OCIO`, every
grading tool, the OETF — to **premultiplied** RGB. OCIO documents the opposite, and OIIO
exposes `unpremult` on `colorconvert` for exactly this reason: a colour transform describes
the surface colour, and `C(a·c) ≠ a·C(c)` for any non-linear `C`.

**Measured before the change, on both mixers** (`CasparCG-TestRunner/cli.py alpha-domain`;
full account in that repo's `docs/alpha_domain_2026-08-12.md`). Four transforms, five
partial-alpha patches each, compared against two closed-form models:

| transform | worst Δ from the premultiplied model | Δ from the straight model |
| :--- | ---: | ---: |
| `srgb/bt709 → bt709/linear` | 0.46 LSB | 17.3 – 56.4 |
| `sdr/bt709 → bt709/pq` | 0.51 LSB | 13.5 – 23.0 |
| `pq/bt2020 → bt709/sdr` | 0.22 LSB | 20.0 – 116.0 |
| `OCIO ACEScct → bt709/rec709` | 0.49 LSB | 23.0 – 63.0 |

The OpenGL and Vulkan reports are byte-identical apart from the mixer's name. The sharpest
single number: `MIXER COLORSPACE PQ BT2020 NONE BT709 REC709` on `#8073401A` (alpha 0x80)
renders **243.8** where the straight domain gives **128.0**.

**The new channel element turns it around**, unpremultiplying above the chain and
re-premultiplying once below it:

```xml
<channel>
    <straight-alpha-grading>true</straight-alpha-grading>
</channel>
```

Measured after, both mixers, byte-identical to each other again: every discriminating patch
selects the straight model at **≤0.46 LSB**.

**Default off, because it changes rendered output** wherever content has soft edges and a
non-linear transform is configured — most lower thirds. Opaque content is bit-identical
either way, which is why the three flat-patch batteries could never see this: they all drive
alpha 0xFF, and at alpha 1.0 the two domains are algebraically identical.

The re-premultiply lands after `MIXER OPACITY` and both key multiplies rather than straight
after the output conversion, which fixes a second defect in passing: those three scale alpha
alone while the default path's RGB still carries the old alpha, so the two reach the blend
disagreeing.

Default path unregressed, re-measured after the change: conformance **100/100**, grading
**48/48**, `ocio` **6/6** (worst 0.55) on **both** mixers, `mixer-parity` 6/6 rasters
identical, `vk-validation` 0 VUIDs with the control reporting, and `alpha-domain` without
the flag still selects the premultiplied domain.

### Added: `MIXER OCIO` works on the Vulkan mixer (A4f)

The generated transform is spliced into the mixer's fragment shader, compiled to SPIR-V at
runtime, and selected per draw. OCIO input transforms now work on both mixers.

**Measured, 6 colour spaces × 23 patches against OCIO's own CPU processor:**

| | worst delta |
| :--- | ---: |
| ACEScct | 0.51 LSB |
| ACES2065-1 | 0.55 LSB |
| ARRI LogC3 (EI800) | 0.53 LSB |
| S-Log3 S-Gamut3.Cine | 0.39 LSB |
| ADX10 | 0.53 LSB |
| Rec.2100-PQ - Display | 0.54 LSB |

**6/6 within the 1.0 LSB gate on Vulkan, and every figure is identical to OpenGL's** — not
merely the same to two decimals, but the same worst-case patch for each of the six spaces.
Two mixers with different channel orders, different shading languages and different resource
binding models agreeing that precisely means the residual is OCIO's GPU-vs-CPU arithmetic,
not either integration.

Vulkan API usage checked separately under the Khronos validation layers: 12/12 commands
accepted, **0 VUIDs**, with the best-practices control proving the layers were reporting.

That identity also closes the channel-order trap empirically. The patches are asymmetric
(`#3080A0`, `#BF8040`, `#4080BF`), and a red/blue exchange in the Vulkan splice would show
as a large delta on every one of them while leaving greys correct. The splice deliberately
does **not** swizzle: this shader carries true RGB where the OpenGL one carries BGR, so
copying the OGL call site's `.bgr` would have mirrored the hue wheel.

**A pipeline is now chosen per layer rather than per pass.** `renderpass` took one pipeline
at construction and used it for every layer, so a pass compositing two layers with different
colour spaces would have applied one layer's transform to both. `layer_info` carries its own,
null meaning the base pipeline.

**Cost, and it is not small.** A colour space new to the process pays, on the frame path:
OCIO generation, a LUT image creation with a device `waitIdle`, a shaderc compile and a
driver pipeline build — about 1.2 s and one dropped frame, logged as a warning. Every later
frame is a map lookup. Pre-warming at `MIXER OCIO` command time is the fix and is not done
yet; see `docs/OCIO_HANDOFF_2026-08-11.md`.

Unchanged on this path: user `exposure` and gamut compression, which live in the
`color_grade` struct inside the input block OCIO replaces. Neither belongs to an input
transform's job, but the gap is real.

### Fixed: `MIXER OCIO` on a Vulkan channel was accepted and then discarded

`MIXER 1-1 OCIO "ACEScct"` returned `202 OK` on a Vulkan channel, logged its processor
build, and changed nothing — because the value never reached the mixer.

`accelerator/vulkan/util/transforms.cpp`'s `apply_transform_colour_values` merges the
image transform **field by field**, and `ocio` was not one of the fields it copied. The
OpenGL copy of that function has carried the member since the OCIO work began, together
with a comment saying precisely what happens when one is missed:

> Easy to miss, and it fails silently: this merge is explicit field by field, so a new
> `image_transform` member that is not listed here simply never reaches the kernel. The
> symptom is a command that reports 202 and changes nothing.

That is the whole defect. It was found while wiring the Vulkan side of the OCIO input
transform: the kernel's new code did not run, the server log carried the AMCP command's
`(opengl)` validation build and no `(vulkan)` build at all, and every downstream symptom
was of a value that was never delivered rather than of anything the mixer computed.

No rendered output changes yet — the Vulkan shader splice (A4f) is still to come, so an
OCIO transform on that mixer remains a no-op end to end. What changes is that the
transform now reaches the kernel, which is the prerequisite for it doing anything at all.
Verified by the OCIO battery re-run being bit-identical before and after
(0/6 on Vulkan, worst 143.39 LSB, unchanged), with the log showing the Vulkan build and
LUT upload now happening for exactly the two colour spaces that emit a texture.

### Fixed: OCIO logged a full processor description every frame

`build_input_transform` is called per layer per frame — both kernels ask OCIO for the
transform first and consult their own caches afterwards, because OCIO's cache ID is what
they key on and only OCIO can produce it. Its `info`-level "built X -> Y" line therefore
went to the log 25 times a second per layer with an OCIO transform set. It is now emitted
once per distinct cache ID, and reset when a config is loaded.

### Added: Vulkan mixer generates and uploads OCIO's resources (A4e)

`build_input_transform` takes a `gpu_target`, and the Vulkan mixer asks for
`GPU_LANGUAGE_GLSL_VK_4_6` with `setDescriptorSetIndex(1, 1)`. The generated LUTs are
uploaded as Vulkan images and written into descriptor set 1 at the bindings OCIO declared
in the source, read back from `getTextureShaderBindingIndex()` rather than assumed.

Two things were measured across all 55 colour spaces in the pinned studio config before
any of it was wired:

* **Nothing emits a 3D texture or declares `sampler3D`.** Every LUT-bearing space produces
  one 2D single-channel image; the most any space needs is one. The 3D path is implemented
  anyway, on the same two helpers `MIXER LUT3D` already exercises.
* **Nothing emits a uniform buffer** — `getUniformBufferSize()` is 0 everywhere, because an
  input transform has no dynamic property. The reserved binding 0 stays declared and
  written as a zero-filled placeholder, which is legal and is what a display transform with
  a dynamic exposure would use.

Verified under the Khronos validation layers, forced on via the loader because the server
requests them only under `_DEBUG`: **0 VUIDs** across four colour spaces, with a
best-practices positive control proving the layers were reporting rather than merely
loaded (`CasparCG-TestRunner`, `cli.py vk-validation`).

### Fixed: `PRINT RAW` wrote nothing on the GPU-direct path, and said `202 OK` anyway

Two independent bugs, and the second is why the first went unnoticed for so long.

**`write_frame_png` read only the first plane.** A GPU-direct frame is P010 — two plane
textures, Y and UV — but the readback called `frame.texture()`, which returns just the
first, and then measured it against `width x height x 4 components`:

```
write_frame_png: GPU readback returned 4147200 bytes, need 16588800 for 1920x1080
```

4147200 is exactly the luma plane (1920x1080, one component, 16-bit), so the readback had
been working the whole time and only the expectation was wrong. It now reads every
texture in `frame.textures()` and converts them through the semi-planar `nv12` branch the
software decode path already used — the frame carries its own plane geometry, chroma
siting and colour space, so no new colour code was needed.

**And `PRINT RAW` could not report the failure.** The write was dispatched to a detached
thread, so the command returned `202 PRINT RAW OK` before the write was attempted and no
error inside `write_frame_png` could ever reach the caller. It is synchronous now and
returns `404 PRINT RAW FAILED` when the write fails. A 1080p PNG encode costs tens of
milliseconds on an explicitly invoked debug command — a better trade than a status code
that cannot be wrong.

**The Vulkan half needed one more thing.** `d3d11_import_bridge::copy_planes` built its
two plane wrappers without a device pointer, so `texture_wrapper::read_pixels` returned
`{}` on its first line and Vulkan GPU-direct had no `PRINT RAW` at all while OpenGL
worked. That was deliberate — the comment read *"a host copy of one plane is not a
picture anyone can use"* — and it was right until the readback learned to read both
planes. They are `VkReadableTextureWrapper` now.

Measured, `h265_hdr10` / `h265_10bit` / `av1` on both mixers, 16 cases:

| | before | after |
| :--- | ---: | ---: |
| cases with a `PRINT RAW` frame | 8 of 16 | **16 of 16** |
| independent decoder check | half the cases | **every case**, 48.0 dB |
| colour gate on GPU-direct cases | 25 dB vs an FFmpeg decode | **45 dB vs the mixer's own decode**, 53.0 dB |

This affects diagnostics only — `write_frame_png` has exactly one caller, `PRINT RAW`.
Rendering is unchanged: `conformance` 100/100 within 1.0 LSB and `grading` 48/48 on both
mixers, and `mixer-parity` 6/6 rasters byte-identical between the backends.

### ⚠ Fixed: the Vulkan mixer produced garbage on any non-square-pixel video mode

**A PAL channel on the Vulkan mixer rendered unrecognisable striping — not a colour
shift, no picture at all.** OpenGL was correct throughout. If you run SD on Vulkan,
this is the fix.

The Vulkan mixer created its render target at `square_width x square_height` where the
OpenGL mixer uses `width x height`. `square_*` is the size a non-square-pixel raster
would occupy on a square-pixel display — PAL is **720x576 stored, 1024x576 displayed**
— and it belongs in the aspect maths, not in the size of the surface that is rendered
into and read back. The channel's frame is 720x576, so handing the consumer a 1024x576
readback sheared every row by 304 pixels.

Measured, `png_8` through a PAL channel, Vulkan against OpenGL:

| | before | after |
| :--- | ---: | ---: |
| PAL (720x576) | unrecognisable | **byte-identical to OpenGL** |
| PAL SDI capture (`sdi-input`) | 7.96 dB, SSIM 0.0057 | **39.67 dB, SSIM 0.9885** |

NTSC was wrong too and looked plausible: its `square_height` is 540 against a real 486,
so the row pitch matched and only the bottom of the frame was lost — a crop rather than
a shear. It is now byte-identical to OpenGL as well.

**Every other mode is unaffected**, because `square_width == width` for all of them —
1080i5000, 1080p2500 and the custom 2600x1500p25 are byte-identical to OpenGL before
and after. `cli.py conformance` still passes **100/100 conversions within 1.0 LSB** on
Vulkan and `cli.py grading` **48/48**, so the colour chain is untouched.

**Why no battery caught it.** `conformance` reports 36/36 within 1.0 LSB at PAL on the
broken build. Its patches are flat, and a flat patch is invariant under a sampling
displacement — every texel it could wrongly sample holds the same value. A flat-patch
battery certifies a colour chain, not a mixer; catching this needs one case per mode
carrying spatial detail.

### ⚠ Behaviour change: colour grading output

**Six defects in the grading chain are fixed below. Any look built by eye against the
old behaviour — a tone curve, a 3D LUT, a qualifier key, film grain — will render
differently. Re-check saved grades.**

Found by a new mathematical conformance battery in CasparCG-TestRunner
(`cli.py grading`): flat colour patches through a channel with no colour conversion
active, compared against a closed-form model of the shader at a 1 LSB gate, plus
multi-op stacks that check the pipeline order and the independence of the enable
flags. After these fixes it passes **48/48 on both the OpenGL and Vulkan mixers**,
every row at or below 0.56 LSB — the 8-bit quantisation half-step.

##### Grading fixes
* **Tone curves lost their last segment.** `build_curve_lut`'s segment search ran to
  `n - 2`, so the final interval between the last two control points was never
  matched and `seg` kept its initial `0` — the end of every curve with three or more
  points was evaluated with the *first* segment's control points and tangents.
  Measured on `((0,0),(0.3,0.05),(0.35,0.95),(1,1))`: the output fell from 0.9255 at
  x=0.3451 to 0.0824 at x=0.3529, a **215 LSB cliff**, then climbed again. Two-point
  curves were unaffected. Both backends carried the duplicated builder; both fixed.
* **The 3D LUT was sampled half a texel off.** Both shaders used the colour value
  directly as the texture coordinate, putting value `v` at texel position `v*N - 0.5`
  when entry `k` describes input `k/(N-1)` and belongs at `v*(N-1)`. Measured with an
  identity cube, where the correct output is the input: **up to 7 LSB on a 17-cube**
  and 4 on a 33-cube, antisymmetric about v=0.5 and exact only there. Now
  `(v*(N-1) + 0.5)/N`, with N from `textureSize()`. An identity cube is now exact at
  every sampled value on both backends.
* **The secondary qualifier ignored hue entirely.** `MIXER QUALIFIER` takes degrees;
  both kernels uploaded the centre and width raw into uniforms the shader compares
  against `rgb2hsv`'s 0..1 hue, so `hue_mask` evaluated to 1 for every pixel and the
  key selected on saturation and luminance alone. Measured: a 64 LSB shift on a green
  patch from a key targeting 210°. Now `/360` for the centre and `/180` for the width
  — the width is compared against `AngleDiff(...)*2`, which saturates at half a turn.
  The chroma keyer has always divided its target hue by 360; this matches it.
* **Film grain was a flat darkening, not noise (OpenGL).** `target_size` was only
  uploaded inside the blur branch, so with blur off it kept the GLSL default
  `vec2(0,0)`. `grain_hash(0,0,seed)` is exactly 0 for any seed, so `noise` was −1 for
  every pixel: `MIXER GRAIN 0.1` on `#808080` produced a uniform 102/255 frame with
  zero variance. The same uniform made `apply_sharpen` compute `1.0/target_size` =
  infinity, so all four taps clamped to the edge texel. Now uploaded
  unconditionally, as the Vulkan kernel always did. Measured after: grain sigma
  0.0593 against a predicted 0.0577, and the two backends agree to four decimals.

##### Parity fixes (OpenGL / Vulkan)
* **Red and blue tone curves were exchanged on Vulkan.** The OpenGL kernel packs the
  four curve LUTs as `(B, G, R, master)` deliberately, because it carries the pixel
  through grading in BGRA. The Vulkan kernel copied that packing but grades in RGB,
  so the user's blue curve was applied to red and vice versa — 20.8 LSB, OpenGL
  clean. Green and master are their own inverse under the exchange, which is why a
  grey ramp would never have caught it. Vulkan now packs `(R, G, B, master)`.
* **Automatic gamut compression swapped the cyan and yellow limits (OpenGL).** The
  auto path passed the ACES 1.3 limits in RGB order into a uniform this shader
  consumes in BGRA, so the yellow axis was compressed with the cyan limit and vice
  versa. Its own manual path and the Vulkan kernel were both correct. **Now measured**
  (see the behaviour-change entry below): a BT.2020 PQ source on a BT.709 SDR
  auto-converting channel matches the model on both mixers, with the flag off and on.

### ⚠ Behaviour change: still images on the Vulkan mixer render at all

**Any opaque PNG, JPEG or TGA played on a channel using the Vulkan mixer produced a black
frame and took the channel with it. It now renders, and measures identically to the OpenGL
mixer. Nothing that worked before changes; a whole path that did not, now does.**

An image without alpha decodes to `rgb24`. `image_producer` asked
`is_frame_compatible_with_mixer()` whether to convert it, and that function answered by
asking whether **core** can describe the layout — which it can, as `pixel_format::rgb`, one
plane of stride 3. The question that mattered is whether the *mixer* can sample it, and the
Vulkan one cannot: Vulkan does not oblige an implementation to support a 3-component format
as a sampled image, and this GPU (Quadro P4000 / RTX A4000, driver 582.53) does not. So
`device::create_texture` threw — from inside a `copy_async` dispatch whose future the mixer
only awaits during draw, i.e. **on the channel thread, once per frame, for as long as the
layer was on air.**

`av_producer` already excludes `rgb24`/`bgr24` from filter-graph negotiation for exactly
this reason, and the throw in `device.cpp` carried a comment saying producers no longer
offer packed 3-byte RGB so it should be unreachable. The image module was the producer that
still did.

Measured, DeckLink SDI out, FFmpeg reading back on an independent card input, 1080p25,
against the same reference and rig that measured the OpenGL mixer:

| case | OpenGL | Vulkan before | Vulkan after |
| :--- | ---: | ---: | ---: |
| `pixel-format=rgba`, 8-bit channel | 37.46 dB | **blank** | **37.46 dB** |
| `pixel-format=yuv`, 8-bit channel | 48.28 dB | **blank** | **48.28 dB** |
| `pixel-format=yuv`, 16-bit channel | 52.53 dB | **blank** | **52.53 dB** |

Identical to the OpenGL figures in all three, including the +4.26 dB depth gain — the
patches are flat, so a correct implementation quantises identically, and equality is the
parity evidence rather than a coincidence.

Two guards were added so this cannot be silent again:

* The Vulkan `image_mixer` asks `device::can_sample_packed()` **before** uploading a plane,
  and drops the item with one warning naming the format if the answer is no. Losing a layer
  is worse than OpenGL, which samples stride-3 fine — but it is the parity floor the
  codebase already chose, and it no longer costs the channel.
* The device logs at startup which packed component counts it cannot sample, so the
  constraint is visible before something trips over it.

### ⚠ Behaviour change: a channel whose tick throws now keeps its frame rate

**Previously a channel that failed in produce or mix ran as fast as the CPU allowed. The
symptoms were a pegged core, thousands of log lines a second, and diagnostics counters that
looked like unrelated performance faults. It is now bounded to one occurrence per frame.**

Nothing paces the tick loop except its consumers: `output_()` is what blocks on the DeckLink
clock (or a file consumer's queue), and it is the last phase. An exception before it returned
straight to the top of the `while`.

Measured on the Vulkan/`rgb24` failure above, 1080p25, one still-image layer:

| | before | after |
| :--- | ---: | ---: |
| exceptions in a 6.0 s window | 28,997 (~4,833/s) | **148 (24.9/s)** |
| server log lines, same window | 116,258 | **834** |

24.9/s against a nominal 25 is one occurrence per frame. This is also why the underlying
defect read as four: the same failure produced a blank output, an exception storm, a pegged
core, and a mixer render-tick count of 28,000 draws against ~100 delivered frames — which
had been recorded as a separate "90× more compositing than the channel consumes"
performance defect. There was one bug.

The catch sleeps out only the *remainder* of the frame period, so a tick that already
overran is not delayed further and a transient exception costs nothing.

### ⚠ Behaviour change: automatic gamut compression now actually happens (OpenGL)

**A channel with `<auto-gamut-compress>true</auto-gamut-compress>` on the OpenGL mixer was
not compressing anything. It now does, so wide-gamut sources converted to a narrower
channel gamut will render differently — highly saturated colours that used to hard-clip are
now rolled off.**

`image_kernel::draw` writes the gamut-compress uniform in two places. The automatic path
sets it inside the colour-conversion branch when `auto_gamut_compress && ig != og`. The
manual `MIXER GAMUT-COMPRESS` block runs **later and unconditionally**, and its `else` set
`gamut_compress_enable` to `false` — so it overwrote the automatic decision on every draw:

| config | what happened |
| :--- | :--- |
| `auto-gamut-compress` on, no `MIXER GAMUT-COMPRESS` | flag cleared — no compression at all |
| both on | manual limits win, auto limits never used |

The automatic path's limits were therefore unreachable in either case, and
`image_transform.gamut_compress` defaults to `false`, which is the first row. The Vulkan
kernel ORs a flag bit and never clears it, so its automatic path worked — this was an
OpenGL-only divergence, and it is why the cyan/yellow limit-order fix in `1288dc032` was
correct but inert.

An explicit `MIXER GAMUT-COMPRESS` still wins, because it names its own limits. The `else`
now only disables what the automatic path did not already enable.

**Measured 2026-08-04**, and this is the first time anything reached the automatic path.
`CasparCG-TestRunner`'s new `cli.py gamut-compress` plays a BT.2020 PQ source on a BT.709
SDR auto-converting channel and runs each source as an **(off, on) pair**, because a row
compared only against its own model cannot tell a flag that was read from one stuck
permanently either way:

| | OpenGL | Vulkan |
| :--- | :--- | :--- |
| flag off, vs the model | **pass** | **pass** |
| flag on, vs the model | **pass** | **pass** |
| off vs on, same source | **255 LSB** | **255 LSB** |

So automatic gamut compression now demonstrably reaches the shader, and its limits — the
cyan/yellow order fixed in `1288dc032` — are correct on both backends. Both of those fixes
had been reasoned from the source and shipped unverified; they are now measured.

Getting there needed one more fix in the harness, not the server: `TestCase.config_key`
omitted `auto_gamut_compress`, so the two halves of each pair shared a server and the second
ran against the first's config. Both rows captured byte-identical frames and the pair check
reported that the flag changed nothing — which read exactly like these two fixes being
inert. The server's own trace (`gamut_compress=true`, `ig=1 og=0`, no `NO_CONVERT`) is what
separated the two readings.

Still open from the same run, and unrelated to gamut compression: the **HLG** source
(`prores_422_hlg`) misses the 45 dB colour gate at 37.8-40.4 dB on both mixers, with SSIM
0.99999 and with both rows moving when the flag changes. That is a systematic level offset
on the HLG→SDR path, not a compression fault, and it is not yet attributed between the
server and the model.

### ⚠ Behaviour change: DeckLink v210 output on the CPU pack path

**Any DeckLink consumer with `<pixel-format>yuv</pixel-format>` that packed v210 on the
CPU was putting red and blue approximately exchanged on the SDI wire. If a downstream
device, LUT or camera-match was trimmed against that output, re-check it.**

`rgb_to_yuv_avx2` emitted each chroma pair as **(Cr, Cb)** — `_mm256_hadd_epi32` puts its
first argument's sums in the low elements, and the arguments were `(cbcr4[1], cbcr4[0])`.
`pack_v210_avx2`'s multiplier table `[1, 16, 4, 1, 16, 4]` is the correct v210 placement
for **(Cb, Cr)**: Cb at dword 0 bits 0-9, Cr at bits 20-29. So every 48-pixel AVX2 batch
wrote `Cr | Y<<10 | Cb<<20` where the scalar `pack_v210` in the same file — and
`cpu_ref_v210_row` in `ogl_gl_strategy.cpp`, and the v210 specification — write
`Cb | Y<<10 | Cr<<20`.

Exchanging Cb and Cr is **not** an exact red/blue swap: R reconstructs from 1.5748·Cr and
B from 1.8556·Cb, and green mixes both. That is why it could not be undone downstream, and
why it read as two faults rather than one.

Measured on SDI, OpenGL mixer, `pixel-format=yuv`, `gpu-pack=cpu`, 1080p25, against a
reference read back by FFmpeg on an independent card input:

| | before | after |
| :--- | ---: | ---: |
| 8-bit channel | 6.67 dB | **48.28 dB** |
| 16-bit channel | not measured | **52.53 dB** |
| as-is with red/blue exchanged in software | 23.34 dB | — |

The +4.26 dB the 16-bit channel gains over the 8-bit one reproduces the +4.23 dB measured
independently on the GPU packer, which is the cross-check that the two implementations now
agree.

Scope: every v210 path that packs on the CPU — SDR and HDR, 8-bit and 16-bit, both
mixers — since `rgb_to_yuv_avx2` is shared. The GL compute packer (`<gpu-pack>gpu</>`) has
its own implementation and was never affected, which is why `gpu-pack=gpu` was a valid
workaround. The scalar `pack_v210` only ever handles the sub-48-pixel row remainder and
the black fill, and black has Cb == Cr == 512, so nothing in the fill could reveal the
disagreement.

A one-shot self-audit now runs at strategy construction and packs a synthetic
asymmetric-per-channel pattern both ways, logging
`[v210_strategy] AVX2/scalar pack parity PASS` or a mismatch with the first differing
word. Two independent implementations existed all along; nothing compared them.

### ⚠ Behaviour change: ArtNet / sACN fixture colours

**Every existing ArtNet and sACN installation will send different DMX values after
this build. Re-check your fixture positions.**

The fixture rasteriser had a degenerate case on the last scanline of an unrotated
fixture rectangle — which is the common case, since `rotation` defaults to 0. The
active-edge pair selected there shared an endpoint, the slope guard skipped the
interpolation, and the horizontal span silently kept its initial values of `0` and
`width - 1`. Each fixture's bottom row was therefore sampled across the **full width
of the frame** rather than across the fixture.

Measured on the shipped example geometry (ten 50×100 fixtures in a 500×100 box):
7020 pixels were accumulated where 5151 are actually covered, and a fixture over a
region of pure red averaged to 181 instead of 247 — **27% of every fixture's value
came from outside the fixture**. A rotated fixture was unaffected (it has no
horizontal edge, so the table never degenerated): the same test gives 254 before
and after.

Fixture colours are now sampled only from the pixels the fixture actually covers.
Anyone who nudged fixture boxes to compensate for the old behaviour will want to
re-trim them.

Note on the related divide-by-zero: sACN guarded a zero pixel count and ArtNet did
not, but neither could reach it — the old rasteriser always produced at least one
(full-width) scanline. With correct spans a fixture positioned entirely off-frame
genuinely yields no samples, so the guard is now load-bearing and lives in the
shared implementation. Such a fixture emits black instead of an average of a row it
does not touch.

### Consumers
##### Fixes
* OAL (`system-audio`), PortAudio and the ProRes bypass consumer no longer force a
  full-frame GPU→CPU readback on their channel. None of them reads a pixel, but
  `needs_cpu_frame_data()` defaults to true and `any_consumer_needs_cpu_data()`
  short-circuits on the first consumer that says yes — so `<system-audio />` on
  channel 1, which the shipped config includes, was enough to defeat every
  GPU-native consumer sharing that channel.
* ArtNet / sACN: fixture geometry and rasterisation deduplicated into
  `modules/dmx_common`; the two copies had already drifted apart by one bug fix.

### Producers
##### Fixes
* OFX: passing a GPU-only source (HAP, CUDA ProRes, NotchLC, ISF) to a plug-in
  dereferenced a null host pointer. Such frames now pass through unprocessed with a
  warning, except on the OpenGL zero-copy path which handles them natively.
* OFX: a plug-in advertising both the CUDA and OpenGL render extensions rendered a
  black frame on the OpenGL mixer — the host took the CUDA branch, which wrote
  nothing the caller then blitted.
##### Improvements
* DeckLink input: the capture buffer is now reference-counted into the filter
  graph instead of being copied into it, removing a full-frame copy per frame per
  input from the driver's callback thread.
* UYVY sources (DeckLink input, NDI input, ffmpeg software decode) share one
  staging buffer between their luma and chroma plane views instead of having the
  identical bytes copied into two, halving the host cost of every UYVY frame.

CasparCG 2.5.0 Stable
==========================================

### Important

We recommend running CasparCG 2.5 on CPUs which support AVX2. Officially Chrome claims to require AVX2, and it is required for some of our in-progress HDR support.
Intel CPUs based on Haswell or later support this, which were first released to consumers in 2013, or 2014 for servers.

Starting with CasparCG 2.6, this will become a requirement

### Core
##### Improvements
* Initial support for HDR. This is limited to a subset of producers and consumers at this stage.
* Build for Windows with VS2022
* Rework linux builds to produce ubuntu deb files
* Update ffmpeg to 7.0
* Reimplement mixer transforms, to handle routes correctly
* Support more pixel formats from ffmpeg, to preserve colour accuracy better
* Support running on headless linux
* Transitions: Additional behaviours
##### Fixes
* Build with boost 1.85/1.86/1.87/1.88
* Build with ffmpeg 7.1
* Only produce mixed frames on channels which have consumers
* Routed channels not compositing correctly when channel used a MIXER KEY
* Handle audio for fractional framerates properly
* Gracefully exit on SIGINT and SIGTERM

### Producers
##### Improvements
* FFmpeg: Support loading with a scaling-mode, to configure how clips get fit into the channel
* FFmpeg: Support more pixel formats without cpu conversion
* FFmpeg: Enable alpha for webm videos
* Image: Support loading with a scaling-mode, to configure how images get fit into the channel
* Image: Replace freeimage with ffmpeg
* HTML: Update CEF to 142
* HTML: Support audio
##### Fixes
* Route: Use full field rate when performing i->p channel route
* HTML: Gracefully handle page load errors
* HTML: Always set cache path

### Consumers
##### Improvements
* Screen: Set size and position from AMCP
* Screen: Improve performance
* Image: Propagate AMCP parameters from PRINT command
* FFmpeg: Remove unnecessary forced conversion to YUVA422
* Decklink: Support explicit yuv output (requires AVX2)
* Decklink: Allow selecting device by hardware persistent id

##### Fixes
* FFmpeg: Correctly handle PTS on frame drop


CasparCG 2.4.3 Stable
==========================================

### Core
##### Fixes
* Improve error handling for invalid config files #1571
* Flush logs before exit #1571
* Check audio cadence values look sane before accepting format #1588
* Cross-channel routes from progressive to interlaced showing lots of black #1576
* Transition: ignoring some transforms of input frames #1602

### Producers
##### Fixes
* FFmpeg: fix crash on invalid frame header
* Decklink: Crash with ffmpeg 7 #1582
* HTML: Fix crash during uninit on exit
* Image: update state during init #1601

### Consumers
##### Fixes
* FFmpeg: set frame_rate for rtmp streams #1462


CasparCG 2.4.2 Stable
==========================================

### Consumers
##### Fixes
* Decklink: fix support for driver 14.3 and later


CasparCG 2.4.1 Stable
==========================================

### Core
##### Fixes
* Fix bad config file examples
* Fix `casparcg_auto_restart.bat` not starting scanner
* Revert removal of tbbmalloc, due to notable performance loss on windows
* Suppress some cmake build warnings
* Build failure when doxygen installed on system
* Build failures with ffmpeg 7.0
* Revert RPATH linking changes

### Producers
##### Fixes
* FFmpeg: Ignore ndi:// urls
* FFmpeg: Using both in and seek could result in incorrect duration
* Route: Race condition during destruction
* Image: Update freeimage on windows with some CVE fixes and failures with certain pngs
* Image: Respect EXIF rotate flag
* NDI: list local sources

### Consumers
##### Fixes
* Decklink: subregion copy not respecting frame height
* Decklink: subregion vertical offset
* Decklink: subregion height limited with some formats


CasparCG 2.4.0 Stable
==========================================

### Core
##### Improvements
* Custom resolutions can be specified in casparcg.config
* Interlaced mixer pipeline to ensure field accuracy
* Preserve unicode characters in console input/output
* Producers to be run at startup can be defined in casparcg.config
* Support 8K frames
* Support 4K DCI frames
* Remove undocumented CII and CLK protocol implementations
* Config parameter can be an absolute system path, not just relative to the working directory
* AMCP: Add CLEAR ALL command
* AMCP: Command batching syntax
* AMCP: LOAD/LOADBG/PLAY commands accept a CLEAR_ON_404 parameter, to instruct the layer to be cleared when the requested file was not found
* AMCP: Add commands to subscribe and unsubscribe to OSC on any port number
* AMCP: Add CALLBG command to perform CALL on background producer
* Build: Require C++17 for building
* Build: Support newer versions of Boost
* Build: Support newer versions of TBB
* Build: Disable precompiled headers for linux
* Build: Support VS2022
* Build: Replace nuget and locally committed dependencies with direct http downloads
* Build: Allow configuring diag font path at build time
* Linux: Support setting thread priorities
* Linux: Initial ARM64 compatibility
* Linux: Rework build to always use system boost
* Linux: Rework build process to better support being build as a system package
* Logging: add config option to disable logging to file and to disable column alignment
* Transitions: Support additional audio fade properties for STING transition
##### Fixes
* Crash upon exiting if HTML producer was running
* AMCP: Ensure all consumers and producers are reported in `INFO` commands
* AMCP: Deferred mixer operations were not being cleared after being applied
* AMCP: `LOAD` command would show a frame or two of black while new producer was loading
* OpenGL: Fix support for recent Linux drivers
* Linux: Fix endless looping on stdin
* Route: Fix error when clearing layer
* Transitions: Fix wipe duration

### Producers
##### Improvements
* Decklink: Require driver 11.0 or later
* Decklink: Scale received frames on GPU
* FFmpeg: Update to v5.1
* FFmpeg: Improve performance
* FFmpeg: Allow specifying both SEEK and IN for PLAY commands
* HTML: Update to CEF 117
* HTML: `CALL 1-10 RELOAD` to reload a renderer
* HTML: Expose `cache-path` setting
* NDI: Upgrade to NDI5
* System Audio: Allow specifying output device to use
##### Fixes
* Decklink: Log spamming when using some input formats
* FFmpeg: Prevent loading unreadable files
* FFmpeg: Unable to play files with unicode filenames
* FFmpeg: Don't lowercase filter parameters
* FFmpeg: Support parameters with name containing a dash
* HTML: media-stream permission denied
* HTML: Expose angle backend config field, the best backend varies depending on the templates and machine
* HTML: Crash when multiple iframes were loaded within a renderer
* Image: Improve file loading algorithm to match the case insensitive and absolute path support already used by ffmpeg

### Consumers
##### Improvements
* Artnet: New artnet consumer
* Decklink: Configure device duplex modes in casparcg.config
* Decklink: Output a subregion of the channel
* Decklink: Add secondary outputs in a consumer, to ensure sync when used within a single card
* iVGA: Remove consumer
* NDI: Upgrade to NDI5
##### Fixes
* Decklink: Fix stutter when loading clips
* FFmpeg: Fix RTMP streaming missing headers
* NDI: dejitter


CasparCG 2.3.3 LTS Stable
==========================================

### Producers
##### Improvements
* Image Scroll Producer: Ported from 2.1


CasparCG 2.3.2 LTS Stable
==========================================

### Producers
##### Fixes
* Packages: Update TBB library to v2021.1.1 - fixes CPU and memory growth when deleting threads
* FFmpeg: Fix possible deadlock leading to producer not being cleaned up correctly


CasparCG 2.3.2 Beta
==========================================

### Producers
##### Fixes
* Packages: Update TBB library to v2021.1.1 - fixes CPU and memory growth when deleting threads
* FFmpeg: Fix possible deadlock leading to producer not being cleaned up correctly


CasparCG 2.3.1 Stable
==========================================

### Producers
##### Fixes
* Flash: Use proper file urls when loading templates, to allow it to work after Flash Player EOL
* FFmpeg: Various HTTP playback improvements


CasparCG 2.3.0 Stable
==========================================

### Producers
##### Features
* FFmpeg: Add more common file extensions to the supported list
* NDI: Require minimum of NDI v4.0
##### Fixes
* HTML: Minimise performance impact on other producers


CasparCG 2.3.0 RC
==========================================

### Producers
##### Features
* Flash: Disable by default, requires enabling in the config file
* FFmpeg: Remove fixed thread limit to better auto select a number
##### Fixes
* Decklink: Downgrade severity of video-format not supported
* FFmpeg: Correctly handle error codes. Ignore exit errors during initialisation
* Route: Detect circular routes and break the loop

### Consumers
##### Features
* Bluefish: Various improvmements including support for Kronos K8

### General
##### Fixes
* Diag not reflecting channel videoformat changes


CasparCG 2.3.0 Beta 1
==========================================

### Producers
##### Features
* Decklink: Detect and update input format when no format is specified in AMCP
* Decklink: Improve performance (gpu colour conversion & less heavy deinterlacing when possible)
* Decklink: `LOAD DECKLINK` will display live frames instead of black
* FFmpeg: Update to 4.2.2
* HTML: Better performance for gpu-enabled mode
* HTML: `window.remove()` has been partially reimplemented
* NDI: Native NDI producer
* Route: Allow routing first frame of background producer
* Route: zero delay routes when within a channel, with 1 frame when cross-channel
* Transition: Add sting transitions
* Add frames_left field to osc/info for progress towards autonext
##### Fixes
* Colour: parsing too much of amcp string as list of colours
* FFmpeg: Always resample clips to 48khz
* FFmpeg: Ensure frame time reaches the end of the clip
* FFmpeg: RTMP stream playback
* FFmpeg: SEEK and LENGTH parameters causing issues with AUTONEXT
* FFmpeg: Ensure packets/frames after the decided end of the clip are not displayed
* FFmpeg: Incorrect seek for audio when not 48khz
* FFmpeg: Some cases where it would not be destroyed if playing a bad stream
* HTML: unlikely but possible exception when handling frames
* HTML: set autoplay-policy
* HTML: animations being ticked too much
* Route: Sending empty frame into a route would cause the destination to reuse the last frame

### Consumers
##### Features
* Audio: Fix audio crackling
* Audio: Fix memory leak
* Bluefish: Various improvmements including supporting more channels and UHD.
* NDI: Native NDI consumer
* Screen: Add side by side key output
* Screen: Add support for Datavideo TC-100/TC-200
##### Fixes
* Decklink: Tick channel at roughly consistent rate when running interlaced output
* Possible crash when adding/removing consumers

### General
##### Features
* Add mixer colour invert property
* Restore `INFO CONFIG` and `INFO PATHS` commands
* Linux: Update docker images to support running in docker (not recommended for production use)
##### Fixes
* NTSC audio cadence
* Ignore empty lines in console input
* Fix building with clang on linux
* Fix building with vs2019
* Better error when startup fails due to AMCP port being in use
* Backslash is a valid trailing slash for windows

CasparCG 2.2.0
==========================================

General
-------

 * C++14
 * Major refactoring, cleanup, optimization
   and stability improvements.
 * Removed unmaintained documentation API.
 * Removed unmaintained program options API.
 * Removed unused frame age API.
 * Removed misc unused and/or unmaintained APIs.
 * Removed TCP logger.
 * Fixed memory leak in transition producer.
 * Removed PSD Producer (moved to 3.0.0).
 * Removed Text Producer (moved to 3.0.0).
 * Removed SyncTo consumer.
 * Removed channel layout in favor of 8 channel passthrough
    and FFMPEG audio filters.
 * Major stability and performance improvements of GPU code.
 * Requires OpenGL 4.5.
 * Repo cleanup (>2GB => <100MB when cloning).
 * Misc cleanup and fixes.

Build
-----
 * Linux build re done with Docker.
 * Windows build re done with Nuget.

HTML
----
 * Updated to Chromium 63 (Julusian).
 * Allow running templates from arbitrary urls (Julusian).

DECKLINK
--------
 * Fixed broken Linux.
 * Misc cleanup and fixes.
 * Complex FFMPEG filters (VF, AF).

MIXER
-----
 * Performance improvements.
 * Removed straight output (moved to 3.0.0).
 * Proper OpenGL pipelining.
 * Blend modes are always enabled.
 * Misc cleanup and fixes.
 * Removed CPU mixer.
 * Mixer always runs in progressive mode. Consumers are expected to convert to interlaced if required.

IMAGE
-----
 * Correctly apply alpha to base64 encoded pngs from AMCP (Julusian).
 * Unmultiply frame before writing to png (Julusian).
 * Removed scroll producer (moved to 3.0.0)

 ROUTE
 -----

 * Reimplemented, simplified.
 * Cross channel routing will render full stage instead of simply copying channel output.
 * Reduced overhead and latency.

FFMPEG
------
 * Rewritten from scratch for better accuracy, stability and
    performance.
 * Update freezed frame during seeking.
 * FFMPEG 3.4.1.
 * Reduce blocking during initialization.
 * Fixed timestamp handling.
 * Fixed V/A sync.
 * Fixed interlacing.
 * Fixed framerate handling.
 * Fixed looping.
 * Fixed seeking.
 * Fixed duration.
 * Audio resampling to match timestamps.
 * Fixed invalid interlaced YUV (411, 420) handling.
 * Added YUV(A)444.
 * Added IO timeout.
 * Added HTTP reconnect.
 * FFMPEG video filter support.
 * FFMPEG audio filter support.
 * Complex FFMPEG filters (VF, AF).
 * CALL SEEK return actually sought value.
 * All AMCP options are based on channel format.
 * Misc improvements, cleanup and fixes.

Bluefish
--------
 * Misc cleanup and fixes.

OAL
------------
 * Added audio sample compensation to avoid audio distortions
    during time drift.
 * Misc cleanup and fixes.

Screen
---------------
 * Proper OpenGL pipelining.
 * Misc cleanup and fixes.

AMCP
----
 * Added PING command (Julusian).
 * Removed INFO commands in favor of OSC.
 * Moved CLS, CINF, TLS, FLS, TLS, THUMBNAIL implementations into
    a separate NodeJS service which is proxied through
    an HTTP API.
 * Misc cleanup and fixes.

CasparCG 2.1.0 Next (w.r.t 2.1.0 Beta 2)
==========================================

General
-------

 * Removed asmlib dependency in favor of using standard library std::memcpy and
    std::memset, because of better performance.

CasparCG 2.1.0 Beta 2 (w.r.t 2.1.0 Beta 1)
==========================================

General
-------

 * Fail early with clear error message if configured paths are not
    creatable/writable.
 * Added backwards compatibility (with deprecation warning) for using
    thumbnails-path instead of thumbnail-path in casparcg.config.
 * Suppress the logging of full path names in stack traces so that only the
    relative path within the source tree is visible.
 * General stability improvements.
 * Native thread id is now logged in Linux as well. Finally they are mappable
    against INFO THREADS, ps and top.
 * Created automatically generated build number, so that it is easier to see
    whether a build is newer or older than an other.
 * Changed configuration element mipmapping_default_on to mipmapping-default-on
    for consistency with the rest of the configuration (Jesper Stærkær).
 * Handle stdin EOF as EXIT.
 * Added support for RESTART in Linux startup script run.sh.
 * Copy casparcg_auto_restart.bat into Windows releases.
 * Fixed bug with thumbnail generation when there are .-files in the media
    folder.
 * Removed CMake platform specification in Linux build script
    (Krzysztof Pyrkosz).
 * Build script for building FFmpeg for Linux now part of the repository.
    Contributions during development (not w.r.t 2.1.0 Beta 1):
   * Fix ffmpeg build dependencies on clean Ubuntu desktop amd64 14.04.3 or
      higher (Walter Sonius).
 * Added support for video modes 2160p5000, 2160p5994 and 2160p6000
    (Antonio Ruano Cuesta).
 * Fixed serious buffer overrun in FFmpeg logging code.

Consumers
---------

 * FFmpeg consumer:
   * Fixed long overdue bug where HD material was always recorded using the
      BT.601 color matrix instead of the BT.709 color matrix. RGB codecs like
      qtrle was never affected but all the YCbCr based codecs were.
   * Fixed bug in parsing of paths containing -.
   * Fixed bugs where previously effective arguments like -pix_fmt were
      ignored.
   * Fixed bug where interlaced channels where not recorded correctly for
      some codecs.
 * DeckLink consumer:
   * Rewrote the frame hand-off between send() and ScheduledFrameCompleted() in
      a way that hopefully resolves all dead-lock scenarios previously possible.
 * Bluefish consumer:
   * Largely rewritten against newest SDK Driver 5.11.0.47 (Satchit Nambiar and
      James Wise sponsored by Bluefish444):
     * Added support for Epoch Neutron and Supernova CG. All current Epoch
        cards are now supported.
     * Added support for for multiple SDI channels per card. 1 to 4 channels
        per Bluefish444 card depending on model and firmware.
     * Added support for single SDI output, complementing existing external key
        output support.
     * Added support for internal key using the Bluefish444 hardware keyer.
 * Screen consumer:
   * Fixed full screen mode.

Producers
---------

 * FFmpeg producer:
   * Increased the max number of frames that audio/video can be badly
      interleaved with (Dimitry Ishenko).
   * Fixed bug where decoders sometimes requires more than one video packet to
      decode the first frame.
   * Added support for IN and OUT parameters (Dimitry Ishenko).
   * Added DV/HDV video device support under Linux (Walter Sonius).
   * Remove unused flags variable in queued_seek (Dimitry Ishenko).
   * Now recognizes .ts files without probing contents (Ovidijus Striaukas).
   * Fixed uninitialized value causing initial log printout to usually say that
      clips are interlaced when they are not.
 * Destroy producer proxy:
   * Created workaround for bug in FFmpeg where every new thread used to
      cleanup caused handles to leak (not sure why). Reduced the effect by using
      only one thread for all producer destructions.
 * Framerate producer:
   * Fixed bug when INFO was used on a not yet playing framerate producer.
 * HTML producer:
   * Fixed bug where only URL:s with . in them where recognized.
 * Image producer:
   * Added LENGTH parameter to allow for queueing with LOADBG AUTO.
   * Fixed inconsistency in what file extensions are supported vs listed in
      CLS/CINF.
 * Layer producer:
   * Fixed serious bug where a circular reference of layer producers caused a
      stack overflow and server crash.
   * Can now route from layer on a channel with an incompatible framerate.
 * Channel producer:
   * Can now route from channel with an incompatible framerate.
   * Deinterlaces interlaced content from source channel.
   * Added optional NO_AUTO_DEINTERLACE parameter to opt out of the mentioned
      deinterlacing.
 * Scene producer:
   * Added abs(), floor(), to_lower(), to_upper() and length() functions to the
      expression language.
   * Created XML Schema for the *.scene XML format. Allows for IDE-like auto-
      completion, API documentation and validation.
   * Added possibility to specify the width and height of a layer instead of
      letting the producer on the layer decide.
   * Added global variables scene_width, scene_height and fps.
   * Made it possible to use expressions in keyframe values.
   * Fixed serious bug where uninitialized values were used.
   * Created more example scenes.
   * Can now forward CALL, CG PLAY, CG STOP, CG NEXT and CG INVOKE to the
      producer on a layer.
 * CG proxy wrapper producer:
   * New in 2.1.0.
   * Allows all CG producers to be used as an ordinary producer inside a layer
      in a scene.
   * Allows the Scene producer to know what variables are available in a
      template.
 * Color producer:
   * Now has support for gradients.
 * PSD producer:
   * Added support for centered and right justified text.
 * Text producer:
   * Fixed bug where tracking contributed to the overall text width on the
      last character.

Mixer
-----

 * Fixed bug in the contrast/saturation/brightness code where the wrong luma
    coefficients was used.
 * Rewrote the chroma key code to support variable hue, instead of fixed green
    or blue. Threshold setting was removed in favour of separate hue width,
    minimum saturation and minimum brightness constraints. Also a much more
    effective spill suppression method was implemented.
 * Fixed bug where glReadPixels() was done from the last drawn to texture
    instead of always from the target texture. This means that for example a
    MIXER KEYER layer without a layer above to key, as well as a separate alpha
    file with MIXER OPACITY 0 now works as expected.
 * Fixed bug where already drawn GL_QUADS were not composited against, causing
    for example italic texts to be rendered incorrectly in the text_producer.

AMCP
----

 * INFO PATHS now adds all the path elements even if they are using the default
    values.
 * MIXER CHROMA syntax deprecated (still supported) in favour of the more
    advanced syntax required by the rewritten chroma key code.
 * Added special command REQ that can be prepended before any command to
    identify the response with a client specified request id, allowing a client
    to know exactly what asynchronous response matched a specific request.
 * Added support for listing contents of a specific directory for CLS, TLS,
    DATA LIST and THUMBNAIL LIST.
 * Fixed bug where CINF only returned the first match.
 * Fixed bug where a client closing the connection after BYE instead of
    letting the server close the connection caused an exception to be logged.



CasparCG 2.1.0 Beta 1 (w.r.t 2.0.7 Stable)
==========================================

General
-------

 * 64 bit!
 * Linux support!
   * Moved to CMake build system for better platform independence.
     * Contributions before build system switch (not w.r.t 2.0.7 Stable):
       * gitrev.bat adaptions for 2.1 (Thomas Kaltz III).
   * Thanks to our already heavy use of the pimpl idiom, abstracting platform
      specifics was easily done by having different versions of the .cpp files
      included in the build depending on target platform. No #ifdef necessary,
      except for in header only platform specific code.
   * Flash, Bluefish and NewTek modules are not ported to the Linux build.
   * Contributions during development (not w.r.t 2.0.7 Stable):
     * Fixed compilation problems in Linux build (Dimitry Ishenko).
     * Fixed compilation problem in GCC 5 (Krzysztof Pyrkosz).
     * Fixed thumbnail image saving on Linux (Krzysztof Pyrkosz).
     * Fixed compilation problem in PSD module (Krzysztof Pyrkosz).
 * Major code refactoring:
   * Mixer abstraction so different implementations can be created. Currently
      CPU mixer and GPU mixer (previously the usage of the GPU was mandatory)
      exists.
   * Flattened folder structure for easier inclusion of header files.
   * Many classes renamed to better describe the abstractions they provide.
   * Sink parameters usually taken by value and moved into place instead of
      taken by const reference as previously done.
   * Old Windows specific AsyncEventServer class has been replaced by platform
      independent implementation based on Boost.Asio.
   * Pimpl classes are now stack allocated with internal shared_ptr to
      implementation, instead of both handle and body being dynamically
      allocated. This means that objects are now often passed by value instead
      of via safe_ptr/shared_ptr, because they are internally reference counted.
   * Protocol strategies are now easier to implement correctly, because of
      separation of state between different client connections.
   * Complete AMCP command refactoring.
   * On-line help system that forces the developer to document AMCP commands,
      producer syntaxes and consumer syntaxes making the documentation coupled
      to the code, which is great.
     * Added missing help for VERSION command (Jesper Stærkær).
   * Upgraded Windows build to target Visual Studio 2015 making it possible to
      use the C++11 features also supported by GCC 4.8 which is targeted on
      Linux.
     * Fixed compilation problems in Visual Studio 2015 Update 1
        (Roman Tarasov)
   * Created abstraction of the different forms of templates (flash, html, psd
      and scene). Each module registers itself as a CG producer provides. All CG
      commands transparently works with all of them.
   * Audio mixer now uses double samples instead of float samples to fully
      accommodate all int32 samples.
   * Reduced coupling between core and modules (and modules and modules):
     * Modules can register system info providers to contribute to INFO SYSTEM.
     * XML configuration factories for adding support for new consumer elements
        in casparcg.config.
     * Server startup hooks can be registered (used by HTML producer to fork
        its sub process).
     * Version providers can contribute content to the VERSION command.
 * Refactored multichannel audio support to use FFmpeg's PAN filter and
    simplified the configuration a lot.
 * Upgraded most third party libraries we depend on.
 * Some unit tests have been created.
 * Renamed README.txt to README, CHANGES.txt to CHANGELOG and LICENSE.txt to
    LICENSE
 * Created README.md for github front page in addition to README which is
    distributed with builds.
 * README file updates (Jonas Hummelstrand).
 * Created BUILDING file describing how to build the server on Windows and
    Linux.
 * Diagnostics:
   * Now also sent over OSC.
   * Diag window is now scrollable and without squeezing of graphs.
   * Contextual information such as video channel and video layer now included
      in graphs.
 * Logging:
   * Implemented a TCP server, simply sending every log line to each connected
      client. Default port is 3250.
   * Changed default log level to info and moved debug statements that are
      interesting in a production system to info.
   * Try to not log full stack traces when user error is the cause. Stacktraces
      should ideally only be logged when a system error or a programming error
      has occurred.
   * More contextual information about an error added to exceptions. An example
      of this is that XML configuration errors now cause the XPath of the error
      is logged.
   * Improved the readability of the log format.
   * Added optional calltrace.log for logging method calls. Allows for trace
      logging to be enabled while calltracing is disabled etc.

OSC
---

 * Improved message formatting performance.
 * Added possibility to disable sending OSC to connected AMCP clients.
 * Fixed inconsistent element name predefined_client to predefined-client in
    casparcg.config (Krzysztof Pyrkosz).

Consumers
---------

 * System audio consumer:
   * Pushes data to openal instead of being callbacked by SFML when data is
      needed.
   * Added possibility to specify the expected delay in the sound card. Might
      help get better consumer synchronization.
 * Screen consumer:
   * Added mouse interaction support, usable by the producers running on the
      video channel.
 * FFmpeg consumer:
   * Replaced by Streaming Consumer after it was adapted to support everything
      that FFmpeg Consumer did.
   * Added support for recording all audio channels into separate mono audio
      streams.
   * Now sends recording progress via OSC.
 * SyncTo consumer:
   * New in 2.1.0.
   * Allows the pace of a channel to follow another channel. This is useful for
      virtual "precomp" channels without a DeckLink consumer to pace it.
 * DeckLink consumer:
   * Added workaround for timescale bug found in Decklink SDK 10.7.
   * Now ScheduledFrameCompleted is no longer only used for video scheduling
      but for audio as well, simplifying the code a lot.
 * iVGA consumer:
   * No longer provides sync to the video channel.
   * Supports NewTek NDI out of the box just by upgrading the
      Processing.AirSend library.

Producers
---------

 * Scene producer:
   * New in 2.1.0.
   * Utilizes CasparCG concepts such as producers, mixer transforms and uses
      them in a nested way to form infinite number of sub layers. Think movie
      clip in Flash.
   * A scene consists of variables, layers, timelines and marks (intro and
      outro for example).
   * Mostly for use by other producers but comes with a XML based producer that
      is a registered CG producer and shows up in TLS.
   * Enables frame accurate compositions and animations.
   * Has a powerful variable binding system (think expressions in After Effects
      or JavaFX Bindings).
 * PSD producer:
   * New in 2.1.0.
   * Parses PSD files and sets up a scene for the Scene producer to display.
   * Text layers based on CG parameters.
   * Supports Photoshop timeline.
   * Uses Photoshop comment key-frames to describe where intro and outro (CG
      PLAY and CG STOP) should be in the timeline.
   * Shows up as regular templates in TLS.
 * Text producer:
   * New in 2.1.0.
   * Renders text using FreeType library.
   * Is used by the PSD producer for dynamic text layers.
 * Image scroll producer:
   * Speed can be changed while running using a CALL. The speed change can be
      tweened.
   * Added support for an absolute end time so that the duration is calculated
      based on when PLAY is called for shows when an exact end time is
      important.
 * Image producer:
   * Fixed bug where too large (OpenGL limit) images were accepted, causing
      problems during thumbnail generation.
 * Framerate producer:
   * New in 2.1.0.
   * Wraps a producer with one framerate and converts it to another. It is not
      usable on its own but is utilized in the FFmpeg producer and the DeckLink
      consumer.
   * Supports different interpolation algorithms. Currently a no-op
      drop-and-repeat mode and a two different frame blending modes.
   * It also supports changing the speed on demand with tweening support.
 * FFmpeg producer:
   * Supports decoding all audio streams from a clip. Useful with .mxf files
      which usually have separate mono streams for every audio channel.
   * No longer do framerate conversion (half or double), but delegates that
      task to the Framerate producer.
   * Added support for v4l2 devices.
   * Added relative and "from end" seeking (Dimitry Ishenko).
   * Contributions during development (not w.r.t 2.0.7 Stable):
     * Fixed 100% CPU problem on clip EOF (Peter Keuter, Robert Nagy).
     * Constrained SEEK within the length of a clip (Dimitry Ishenko).
     * Fixed a regular expression (Dimitry Ishenko).
 * DeckLink producer:
   * No longer do framerate conversion (half or double), but delegates that
      task to the Framerate producer.
 * Route producer:
   * Added possibility to delay frames routed from a layer or a channel.
 * HTML Producer:
   * Disabled web security in HTML Producer (Robert Nagy).
   * Reimplemented requestAnimationFrame handling in Javascript instead of C++.
   * Implemented cancelAnimationFrame.
   * Increased animation smoothness in HTML Producer with interlaced video
      modes.
   * Added remote debugging support.
   * Added mouse interaction support by utilizing the Screen consumer's new
      interaction support.
 * Flash Producer:
   * Contributions during development (not w.r.t 2.0.7 Stable):
     * Workaround for flickering with high CPU usage and CPU accelerator
        (Robert Nagy)

AMCP
----

 * TLS has a new column for "template type" for clients that want to
    differentiate between html and flash for example.
 * SET CHANNEL_LAYOUT added to be able to change the audio channel layout of a
    video channel at runtime.
 * HELP command added for accessing the new on-line help system.
 * FLS added to list the fonts usable by the Text producer.
 * LOCK command added for controlling/gaining exclusive access to a video
    channel.
 * LOG CATEGORY command added to enable/disable the new log categories.
 * SWAP command now optionally supports swapping the transforms as well as the
    layers.
 * VERSION command can now provide CEF version.



CasparCG Server 2.0.7 Stable (as compared to CasparCG Server 2.0.7 Beta 2)
==========================================================================

General
-------

 * Added support for using a different configuration file at startup than the
    default casparcg.config by simply adding the name of the file to use as the
    first command line argument to casparcg.exe.
 * Upgraded FFmpeg to latest stable.
 * Created build script.
 * Fixed bug where both layer_producer and channel_producer display:s and
    empty/late first frame when the producer is called before the consumer in
    the other end has received the first frame.
 * Added rudimentary support for audio for layer_producer and channel_producer.
 * Upgraded DeckLink SDK to 10.1.4, bringing new 2K and 4K DCI video modes. New
    template hosts also available for those modes.
 * General bug fixes (mostly memory and resource leaks, some serious).
 * Updated Boost to version 1.57
 * Frontend no longer maintained and therefore not included in the release.

Mixer
-----

 * Added support for rotation.
 * Added support for changing the anchor point around which fill_translation,
    fill_scale and rotation will be done from.
 * Added support for perspective correct corner pinning.
 * Added support for mipmapped textures with anisotropic filtering for
    increased downscaling quality. Whether to enable by default can be
    configured in casparcg.config.
 * Added support for cropping a layer. Not the same as clipping.

AMCP
----

 * Added RESUME command to complement PAUSE. (Peter Keuter)
 * To support the new mixer features the following commands has been added:

   * MIXER ANCHOR -- will return or modify the anchor point for a layer
      (default is 0 0 for backwards compatibility). Example:
      MIXER 1-10 ANCHOR 0.5 0.5
      ...for changing the anchor to the middle of the layer
      (a MIXER 1-10 FILL 0.5 0.5 1 1 will be necessary to place the layer at the
      same place on screen as it was before).

   * MIXER ROTATION -- will return or modify the angle of which a layer is
      rotated by (clockwise degrees) around the point specified by ANCHOR.

   * MIXER PERSPECTIVE -- will return or modify the corners of the perspective
      transformation of a layer. One X Y pair for each corner (order upper left,
      upper right, lower right and lower left). Example:
      MIXER 1-10 PERSPECTIVE 0.4 0.4 0.6 0.4 1 1 0 1

   * MIXER MIPMAP -- will return or modify whether to enable mipmapping of
      textures produced on a layer. Only frames produced after a change will be
      affected. So for example image_producer will not be affected while the
      image is displayed.

   * MIXER CROP -- will return or modify how textures on a layer will be
      cropped. One X Y pair each for the upper left corner and for the lower
      right corner.

 * Added INFO QUEUES command for debugging AMCP command queues. Useful for
    debugging command queue overflows, where a command is deadlocked. Hopefully
    always accessible via console, even though the TCP command queue may be
    full.
 * Added GL command:
    - GL INFO prints information about device buffers and host buffers.
    - GL GC garbage collects pooled but unused GL resources.
 * Added INFO THREADS command listing the known threads and their descriptive
    names. Can be matched against the thread id column of log entries.

Consumers
---------

 * Removed blocking_decklink_consumer. It was more like an experiment at best
    and its usefulness was questionable.
 * Added a 10 second time-out for consumer sends, to detect/recover from
    blocked consumers.
 * Some consumers which are usually added and removed during playout (for
    example ffmpeg_consumer, streaming_consumer and channel_consumer) no longer
    affect the presentation time on other consumers. Previously a lag on the SDI
    output could be seen when adding such consumers.

HTML producer
-------------

 * No longer tries to play all files with a . in their name.
    (Georgi Chorbadzhiyski)
 * Reimplemented using CEF3 instead of Berkelium, which enables use of WebGL
    and more. CEF3 is actively maintained, which Berkelium is not. (Robert Nagy)
 * Implements a custom version of window.requestAnimationFrame which will
    follow the pace of the channel, for perfectly smooth animations.
 * No longer manually interlaces frames, to allow for mixer fill transforms
    without artifacts.
 * Now uses CEF3 event loop to avoid 100% CPU core usage.



CasparCG Server 2.0.7 Beta 2 (as compared to CasparCG Server 2.0.7 Beta 1)
==========================================================================

General
-------

 * Added sending of OSC messages for channel_grid channel in addition to
    regular channels.

Producers
---------

 * FFmpeg: Reports correct nb_frames() when using SEEK (Thomas Kaltz III)
 * Flash: Fixed bug where CG PLAY, CG INVOKE did not work.

Consumers
---------

 * channel_consumer: Added support for more than one channel_consumer per
    channel.
 * decklink_consumer: Added support for a single instance of the consumer to
    manage a separate key output for use with DeckLink Duo/Quad cards:

    <decklink>
      <device>1</device>
      <key-device>2</key-device>
      <keyer>external_separate_device</keyer>
    </decklink>

    ...in the configuration will enable the feature. The value of <key-device />
    defaults to the value of <device /> + 1.
 * synchronizing_consumer: Removed in favour of a single decklink_consumer
    managing both fill and key device.
 * streaming_consumer: A new implementation of ffmpeg_consumer with added
    support for streaming and other PTS dependent protocols. Examples:

    <stream>
      <path>udp://localhost:5004</path>
      <args>-vcodec libx264 -tune zerolatency -preset ultrafast -crf 25 -format mpegts -vf scale=240:180</args>
    </stream>

    ...in configuration or:

    ADD 1 STREAM udp://localhost:5004 -vcodec libx264 -tune zerolatency -preset ultrafast -crf 25 -format mpegts -vf scale=240:180

    ...via AMCP. (Robert Nagy sponsored by Ericsson Broadcasting Services)
 * newtek_ivga_consumer: Added support for iVGA consumer to not provide channel
    sync even though connected. Useful for iVGA clients that downloads as fast
    as possible instead of in frame-rate pace, like Wirecast. To enable:

    <newtek-ivga>
      <provide-sync>false</provide-sync>
    </newtek-ivga>

    ...in config to not provide channel sync when connected. The default is
    true.

AMCP
----

 * Added support in ADD and REMOVE for a placeholder <CLIENT_IP_ADDRESS> which
    will resolve to the connected AMCP client's IPV4 address.
 * Fixed bug where AMCP commands split into multiple TCP packets where not
    correctly parsed (http://casparcg.com/forum/viewtopic.php?f=3&t=2480)



CasparCG Server 2.0.7 Beta 1 (as compared to 2.0.6 Stable)
==========================================================

General
-------
 * FFmpeg: Upgraded to master and adapted CasparCG to FFmpeg API changes
    (Robert Nagy sponsored by SVT)
 * FFmpeg: Fixed problem with frame count calculation (Thomas Kaltz III)
 * Fixed broken CG UPDATE.

Producers
---------

 * New HTML producer has been created (Robert Nagy sponsored by Flemish Radio
    and Television Broadcasting Organization, VRT)



CasparCG Server 2.0.6 Stable (as compared to 2.0.4 Stable)
==========================================================

General
-------
 * iVGA: Allow for the server to work without Processing.AirSend.x86.dll to
    prevent a possible GPL violation. It is available as a separate optional
    download.
 * iVGA: Only provide sync to channel while connected, to prevent channel
    ticking too fast.
 * FFmpeg: Fixed bug during deinterlace-bob-reinterlace where output fields
    were offset by one field in relation to input fields.
 * FFmpeg: Fixed bug in ffmpeg_consumer where an access violation occurred
    during destruction.
 * FFmpeg: Improved seeking. (Robert Nagy and Thomas Kaltz III)
 * Frontend: Only writes elements to casparcg.config which overrides a default
    value to keep the file as compact as possible.
 * System audio: Patched sfml-audio to work better with oal-consumer and
    therefore removed PortAudio as the system audio implementation and went back
    to oal.
 * Flash: Changed so that the initial buffer fill of frames is rendered at a
    frame-duration pace instead of as fast as possible. Otherwise time based
    animations render incorrectly. During buffer recovery, a higher paced
    rendering takes place, but still not as fast as possible, which can cause
    animations to be somewhat incorrectly rendered. This is the only way though
    if we want the buffer to be able to recover after depletion.
 * Fixed race condition during server shutdown.
 * OSC: outgoing audio levels from the audio mixer for each audio channel is
    now transmitted (pFS and dBFS). (Thomas Kaltz III)
 * Stage: Fixed bug where tweened transforms were only ticked when a
    corresponding layer existed.
 * Screen consumer: Added borderless option and correct handling of name
    option. (Thomas Kaltz III)
 * AMCP: CLS now reports duration and framerate for MOVIE files were
    information is possible to extract. (Robert Nagy)
 * Version bump to keep up with CasparCG Client version.



CasparCG Server 2.0.4 Stable (as compared to 2.0.4 Beta 1)
==========================================================

General
-------
 * Can now open media with file names that only consist of digits.
    (Cambell Prince)
 * Miscellaneous stability and performance improvements.

Video mixer
-----------
 * Conditional compilation of chroma key support and straight alpha output
    support in shader (just like with blend-modes) because of performance impact
    even when not in use on a layer or on a channel. New <mixer /> element added
    to configuration for turning on mixer features that not everybody would want
    to pay for (performance-wise.) blend-modes also moved into this element.
 * Fixed bug where MIXER LEVELS interpreted arguments in the wrong order, so
    that gamma was interpreted as max_input and vice versa.

Consumers
---------
 * Added support for NewTek iVGA, which enables the use of CasparCG Server
    fill+key output(s) as input source(s) to a NewTek TriCaster without
    requiring video card(s) in the CasparCG Server machine, or taking up inputs
    in the TriCaster. <newtek-ivga /> element in config enables iVGA on a
    channel. (Robert Nagy sponsored by NewTek)
 * DeckLink: Created custom decklink allocator to reduce the memory footprint.
 * Replaced usage of SFML for <system-audio /> with PortAudio, because of
    problems with SFML since change to static linkage. Also PortAudio seems to
    give lower latency.

Producers
---------
 * FFmpeg: Added support for arbitrary FFmpeg options/parameters
    in ffmpeg_producer. (Cambell Prince)
 * Flash: Flash Player 11.8 now tested and fully supported.
 * Flash: No longer starts a Flash Player to service CG commands that mean
    nothing without an already running Flash Player.
 * Flash: globally serialize initialization and destruction of Flash Players,
    to avoid race conditions in Flash.
 * Flash: changed so that the Flash buffer is filled with Flash Player
    generated content at initialization instead of empty frames.

OSC
---
 * Performance improvements. (Robert Nagy sponsored by Boffins Technologies)
 * Never sends old values to OSC receivers. Collects the latest value of each
    path logged since last UDP send, and sends the new UDP packet (to each
    subscribing OSC receiver) with the values collected. (Robert Nagy sponsored
    by Boffins Technologies)
 * Batches as many OSC messages as possible in an OSC bundle to reduce the
    number of UDP packets sent. Breakup into separate packages if necessary to
    avoid fragmentation. (Robert Nagy sponsored by Boffins Technologies)
 * Removed usage of Microsoft Agents library (Server ran out of memory after a
    while) in favour of direct synchronous invocations.



CasparCG Server 2.0.4 Beta 1 (as compared to 2.0.3 Stable)
==========================================================

General
-------
 * Front-end GUI for simplified configuration and easy access to common tasks.
    (Thomas Kaltz III and Jeff Lafforgue)
 * Added support for video and images file thumbnail generation. By default the
    media directory is scanned every 5 seconds for new/modified/removed files
    and thumbnails are generated/regenerated/removed accordingly.
 * Support for new video modes: 1556p2398, 1556p2400, 1556p2500, 2160p2398,
    2160p2400, 2160p2500, 2160p2997 and 2160p3000.
 * Experimental ATI graphics card support by using static linking against SFML
    instead of dynamic. Should improve ATI GPU support, but needs testing.
 * Added support for playback and pass-through of up to 16 audio channels. See
    http://casparcg.com/forum/viewtopic.php?f=3&t=1453 for more information.
 * Optimizations in AMCP protocol implementations for large incoming messages,
    for example base64 encoded PNG images.
 * Logging output now includes milliseconds and has modified format:
    YYYY-MM-DD hh:mm:ss.zzz
 * Improved audio playback with 720p5994 and 720p6000 channels.
 * An attempt to improve output synchronization of consumers has been made. Use
    for example:

    <consumers>
      <synchronizing>
        <decklink>
          <device>1</device>
          <embedded-audio>true</embedded-audio>
        </decklink>
        <decklink>
          <device>2</device>
          <key-only>true</key-only>
        </decklink>
      </synchronizing>
    </consumers>

    ...to instruct the server to keep both DeckLink consumers in sync with each
    other. Consider this experimental, so don't wrap everything in
    <synchronizing /> unless synchronization of consumer outputs is needed. For
    synchronization to be effective all synchronized cards must have genlock
    reference signal connected.
 * Transfer of source code and issue tracker to github. (Thomas Kaltz III)

Layer
-----
 * Fixed a problem where the first frame was not always shown on LOAD.
    (Robert Nagy)

Stage
-----

 * Support for layer consumers for listening to frames coming out of producers.
    (Cambell Prince)

Audio mixer
-----------
 * Added support for a master volume mixer setting for each channel.

Video mixer
-----------
 * Added support for chroma keying. (Cambell Prince)
 * Fixed bug where MIXER CONTRAST set to < 1 can cause transparency issues.
 * Experimental support for straight alpha output.

Consumers
---------
 * Avoid that the FFmpeg consumer blocks the channel output when it can't keep
    up with the frame rate (drops frames instead).
 * Added support for to create a separate key and fill file when recording with
    the FFmpeg consumer. Add the SEPARATE_KEY parameter to the FFmpeg consumer
    parameter list. The key file will get the _A file name suffix to be picked
    up by the separated_producer when doing playback.
 * The Image consumer now writes to the media folder instead of the data
    folder.
 * Fixed bug in DeckLink consumer where we submit too few audio samples to the
    driver when the video format has a frame rate > 50.
 * Added another experimental DeckLink consumer implementation where scheduled
    playback is not used, but a similar approach as in the bluefish consumer
    where we wait for a frame to be displayed and then display the next frame.
    It is configured via a <blocking-decklink> consumer element. The benefits of
    this consumer is lower latency and more deterministic synchronization
    between multiple instances (should not need to be wrapped in a
    <synchronizing> element when separated key/fill is used).

Producers
---------
 * Added support for playing .swf files using the Flash producer. (Robert Nagy)
 * Image producer premultiplies PNG images with their alpha.
 * Image producer can load a PNG image encoded as base64 via:
    PLAY 1-0 [PNG_BASE64] <base64 string>
 * FFmpeg producer can now use a directshow input filters:
    PLAY 1-10 "dshow://video=Some Camera"
    (Cambell Prince, Julian Waller and Robert Nagy)
 * New layer producer which directs the output of a layer to another layer via
    a layer consumer. (Cambell Prince)

AMCP
----
 * The master volume feature is controlled via the MASTERVOLUME MIXER
    parameter. Example: MIXER 1 MASTERVOLUME 0.5
 * THUMBNAIL LIST/RETRIEVE/GENERATE/GENERATE_ALL command was added to support
    the thumbnail feature.
 * ADD 1 FILE output.mov SEPARATE_KEY activates the separate key feature of the
    FFmpeg consumer creating an additional output_a.mov containing only the key.
 * Added KILL command for shutting down the server without console access.
 * Added RESTART command for shutting down the server in the same way as KILL
    except that the return code from CasparCG Server is 5 instead of 0, which
    can be used by parent process to take other actions. The
    'casparcg_auto_restart.bat' script restarts the server if the return code is
    5.
 * DATA RETRIEVE now returns linefeeds encoded as an actual linefeed (the
    single character 0x0a) instead of the previous two characters:
    \ followed by n.
 * MIXER CHROMA command added to control the chroma keying. Example:
    MIXER 1-1 CHROMA GREEN|BLUE 0.10 0.04
    (Cambell Prince)
 * Fixed bug where MIXER FILL overrides any previous MIXER CLIP on the same
    layer. The bug-fix also has the side effect of supporting negative scale on
    MIXER FILL, causing the image to be flipped.
 * MIXER <ch> STRAIGHT_ALPHA_OUTPUT added to control whether to output straight
    alpha or not.
 * Added INFO <ch> DELAY and INFO <ch>-<layer> DELAY commands for showing some
    delay measurements.
 * PLAY 1-1 2-10 creates a layer producer on 1-1 redirecting the output of
    2-10. (Cambell Prince)

OSC
---
 * Support for sending OSC messages over UDP to either a predefined set of
    clients (servers in the OSC sense) or dynamically to the ip addresses of the
    currently connected AMCP clients.
    (Robert Nagy sponsored by Boffins Technologies)
 * /channel/[1-9]/stage/layer/[0-9]
   * always             /paused           [paused or not]
   * color producer     /color            [color string]
   * ffmpeg producer    /profiler/time    [render time]     [frame duration]
   * ffmpeg producer    /file/time        [elapsed seconds] [total seconds]
   * ffmpeg producer    /file/frame       [frame]           [total frames]
   * ffmpeg producer    /file/fps         [fps]
   * ffmpeg producer    /file/path        [file path]
   * ffmpeg producer    /loop             [looping or not]
   * during transitions /transition/frame [current frame]   [total frames]
   * during transitions /transition/type  [transition type]
   * flash producer     /host/path        [filename]
   * flash producer     /host/width       [width]
   * flash producer     /host/height      [height]
   * flash producer     /host/fps         [fps]
   * flash producer     /buffer           [buffered]        [buffer size]
   * image producer     /file/path        [file path]



CasparCG Server 2.0.3 Stable (as compared to 2.0.3 Alpha)
=========================================================

Stage
-----

 * Fixed dead-lock that can occur with multiple mixer tweens. (Robert Nagy)

AMCP
----

 * DATA STORE now supports creating folders of path specified if they does not
    exist. (Jeff Lafforgue)
 * DATA REMOVE command was added. (Jeff Lafforgue)



CasparCG Server 2.0.3 Alpha (as compared to 2.0 Stable)
=======================================================

General
-------

 * Data files are now stored in UTF-8 with BOM. Latin1 files are still
    supported for backwards compatibility.
 * Commands written in UTF-8 to log file but only ASCII characters to console.
 * Added supported video formats:
   * 720p2398 (not supported by DeckLink)
   * 720p2400 (not supported by DeckLink)
   * 1080p5994
   * 1080p6000
   * 720p30 (not supported by DeckLink)
   * 720p29.976 (not supported by DeckLink)

CLK
---

 * CLK protocol implementation can now serve more than one connection at a time
    safely.
 * Added timeline support to the CLK protocol.
 * Refactored parts of the CLK parser implementation.

Consumers
---------

 * Consumers on same channel now invoked asynchronously to allow for proper
    sync of multiple consumers.
 * System audio consumer:
   * no longer provides sync to the video channel.
 * Screen consumer:
   * Support for multiple screen consumers on the same channel
   * No longer spin-waits for vsync.
   * Now deinterlaces to two separate frames so for example 50i will no longer
      be converted to 25p but instead to 50p for smooth playback of interlaced
      content.
 * DeckLink consumer now logs whether a reference signal is detected or not.

Producers
---------

 * Image scroll producer:
   * Field-rate motion instead of frame-rate motion with interlaced video
      formats. This can be overridden by giving the PROGRESSIVE parameter.
   * SPEED parameter now defines pixels per frame/field instead of half pixels
      per frame. The scrolling direction is also reversed so SPEED 0.5 is the
      previous equivalent of SPEED -1. Movements are done with sub-pixel
      accuracy.
   * Fixed incorrect starting position of image.
   * Rounding error fixes to allow for more exact scrolling.
   * Added support for motion blur via a new BLUR parameter
   * Added PREMULTIPLY parameter to support images stored with straight alpha.



CasparCG Server 2.0 Stable (as compared to Beta 3)
==================================================

General
-------

 * Misc stability and performance fixes.

Consumers
---------

 * File Consumer
   * Changed semantics to more closely follow FFmpeg (see forums).
   * Added options, -r, -acodec, -s, -pix_fmt, -f and more.
 * Screen Consumer
   * Added vsync support.



CasparCG Server 2.0 Beta 3 (as compared to Beta 1)
==================================================

Formats
-------

 * ProRes Support
   * Both encoding and decoding.
 * NTSC Support
   * Updated audio-pipeline for native NTSC support. Previous implementation
      did not fully support NTSC audio and could cause incorrect behaviour or
      even crashes.

Consumers
---------

 * File Consumer added
   * See updated wiki or ask in forum for more information.
   * Should support anything FFmpeg supports. However, we will work mainly with
      DNxHD, PRORES and H264.
    - Key-only is not supported.
 * Bluefish Consumer
   * 24 bit audio support.
    - Embedded-audio does not work with Epoch cards.
 * DeckLink Consumer
   * Low latency enabled by default.
   * Added graphs for driver buffers.
 * Screen Consumer
   * Changed screen consumer square PAL to the more common wide-square PAL.
   * Can now be closed.
   * Fixed interpolation artifacts when running non-square video-modes.
   * Automatically deinterlace interlaced input.

Producers
---------

 * DeckLink Producer
   * Improved color quality be avoiding unnecessary conversion to BGRA.
 * FFMPEG Producer
   * Fixed missing alpha for (RGB)A formats when deinterlacing.
   * Updated buffering to work better with files with long audio/video
      interleaving.
   * Seekable while running and after reaching EOF. CALL 1-1 SEEK 200.
   * Enable/disable/query looping while running. CALL 1-1 LOOP 1.
   * Fixed bug with duration calculation.
   * Fixed bug with fps calculation.
   * Improved auto-transcode accuracy.
   * Improved seeking accuracy.
   * Fixed bug with looping and LENGTH.
   * Updated to newer FFmpeg version.
   * Fixed incorrect scaling of NTSC DV files.
   * Optimized color conversion when using YADIF filters.
 * Flash Producer
   * Release Flash Player when empty.
   * Use native resolution TemplateHost.
   * TemplateHosts are now chosen automatically if not configured. The
      TemplateHost with the corresponding video-mode name is now chosen.
   * Use square pixel dimensions.

AMCP
----

 * When possible, commands will no longer wait for rendering pipeline. This
    reduces command execution latencies, especially when sending a lot of
    commands in a short timespan.
 * Fixed CINF command.
 * ADD/REMOVE no longer require subindex,
    e.g. "ADD 1 SCREEN" / "REMOVE 1 SCREEN" instead of "ADD 1-1 SCREEN" / ...
 * PARAM is renamed to CALL.
 * STATUS command is replaced by INFO.
 * INFO command has been extended:
   * INFO (lists channels).
   * INFO 1 (channel info).
   * INFO 1-1 (layer info).
   * INFO 1-1 F (foreground producer info).
   * INFO 1-1 B (background producer info).
   * INFO TEMPLATE mytemplate (template meta-data info, e.g. field names).
 * CG INFO command has been extended.
   * CG INFO 1 (template-host information, e.g. what layers are occupied).

Mixer
-----

 * Fixed alpha with blend modes.
 * Automatically deinterlace for MIXER FILL commands.

Channel
-------

 * SET MODE now reverts back to old video-mode on failure.

Diagnostics
-----------

 * Improved graphs and added more status information.
 * Print configuration into log at startup.
 * Use the same log file for the entire day, instead of one per startup as
    previously.
 * Diagnostics window is now closable.



CasparCG Server 2.0 Beta 1 (as compared to Alpha)
=================================================

 * Blending Modes (needs to be explicitly enabled)
   * overlay
   * screen
   * multiply
   * and many more.
 * Added additive keyer in addition to linear keyer.
 * Image adjustments
   * saturation
   * brightness
   * contrast
   * min input-level
   * max input-level
   * min output-level
   * max output-level
   * gamma
 * Support for FFmpeg-filters such as (ee http://ffmpeg.org/libavfilter.html)
   * yadif deinterlacer (optimized in CasparCG for full multi-core support)
   * de-noising
   * dithering
   * box blur
   * and many more
 * 32-bit SSE optimized audio pipeline.
 * DeckLink-Consumer uses external-key by default.
 * DeckLink-Consumer has 24 bit embedded-audio support.
 * DeckLink-Producer has 24 bit embedded-audio support.
 * LOADBG with AUTO feature which automatically plays queued clip when
    foreground clip has ended.
 * STATUS command for layers.
 * LOG LEVEL command for log filtering.
 * MIX transition works with transparent clips.
 * Freeze on last frame.
 * Producer buffering is now configurable.
 * Consumer buffering is now configurable.
 * Now possible to configure template-hosts for different video-modes.
 * Added auto transcoder for FFmpeg producer which automatically transcodes
    input video into compatible video format for the channel.
   * interlacing (50p -> 50i)
   * deinterlacing (50i -> 25p)
   * bob-deinterlacing (50i -> 50p)
   * bob-deinterlacing and reinterlacing (w1xh150i -> w2xh250i)
   * doubling (25p -> 50p)
   * halfing (50p -> 25p)
   * field-order swap (upper <-> lower)
 * Screen consumer now automatically deinterlaces when receiving interlaced
    content.
 * Optimized renderer.
 * Renderer can now be run asynchronously with producer by using a
    producer-buffer size greater than 0.
 * Improved error and crash recovery.
 * Improved logging.
 * Added Image-Scroll-Producer.
 * Key-only has now near zero performance overhead.
 * Reduced memory requirements.
 * Removed "warm up lag" which occurred when playing the first media clip after
    the server has started.
 * Added read-back fence for OpenGL device for improved multi-channel
    performance.
 * Memory support increased from standard 2 GB to 4 GB on 64 bit Win 7 OS.
 * Added support for 2* DeckLink cards in Full HD.
 * Misc bugs fixes and performance improvements.
 * Color producer now support some color codes in addition to color codes, e.g.
    EMPTY, BLACK, RED etc...
 * Alpha value in color codes is now optional.
 * More than 2 DeckLink cards might be possible but have not yet been tested.



CasparCG Server 2.0 Alpha (as compared to 1.8)
==============================================

General
-------

 * Mayor refactoring for improved readability and maintainability.
 * Some work towards platform-independence. Currently the greatest challenge
    for full platform-independence is flash-producer.
 * Misc improved scalability.
 * XML-configuration.
 * DeckLink
   * Support for multiple DeckLink cards.

Core
----

 * Multiple producers per video_channel.
 * Multiple consumers per video_channel.
 * Swap producers between layers and channels during run-time.
 * Support for upper-field and lower-field interlacing.
 * Add and remove consumers during run-time.
 * Preliminary support for NTSC.

AMCP
----

 * Query flash and template-host version.
 * Recursive media-folder listing.
 * Misc changes.

Mixer
-----

 * Animated tween transforms.
 * Image-Mixer
   * Fully GPU accelerated (all features listed below are done on the GPU),
   * Layer composition.
   * Color spaces (rgba, bgra, argb, yuv, yuva, yuv-hd, yuva-hd).
   * Interlacing.
   * Per-layer image transforms:
     * Opacity
     * Gain
     * Scaling
     * Clipping
     * Translation
 * Audio Mixer
   * Per-layer and per-sample audio transforms:
       * Gain
   * Fully internal audio mixing. Single output video_channel.

Consumers
---------

 * DeckLink Consumer
   * Embedded audio.
   * HD support.
   * Hardware clock.
 * Bluefish Consumer
   * Drivers are loaded on-demand (server now runs on computers without
      installed Bluefish drivers).
   * Embedded audio.
   * Allocated frames are no longer leaked.

Producers
---------

 * Decklink Producer
   * Embedded audio.
   * HD support.
 * Color Producer
   * GPU accelerated.
 * FFMPEG Producer
   * Asynchronous file IO.
   * Parallel decoding of audio and video.
   * Color space transform are moved to GPU.
 * Transition Producer
   * Fully interlaced transition (previously only progressive, even when
      running in interlaced mode).
   * Per-sample mixing between source and destination clips.
   * Tween transitions.
 * Flash Producer
   * DirectDraw access (slightly improved performance).
   * Improved time-sync. Smoother animations and proper interlacing.
 * Image Producer
   * Support for various image formats through FreeImage library.

Diagnostics
-----------

 * Graphs for monitoring performance and events.
 * Misc logging improvements.
 * Separate log file for every run of the server.
 * Error logging provides full exception details, instead of only printing that
    an error has occurred.
 * Console with real-time logging output.
 * Console with AMCP input.

Removed
-------

 * Registry configuration (replaced by XML Configuration).
 * TGA Producer (replaced by Image Producer).
 * TGA Scroll Producer
