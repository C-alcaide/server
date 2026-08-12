# OCIO integration — handoff, 2026-08-11

Resume point for the OCIO work on branch **`feature/ocio-mixer`**, off `CasparVPV`. Plan of
record: [`OCIO_INTEGRATION_STUDY.md`](OCIO_INTEGRATION_STUDY.md).

**A4e and A4f are done: the OCIO input transform now works on both mixers, at parity.**
Updated in place rather than as a second file — four same-dated handoffs once existed in the
harness repo and which was current could only be recovered from git timestamps.

*Covers work through 2026-08-12. Kept as one file rather than split at midnight: the work is
continuous, and four same-dated handoffs in the harness repo once made "which one is current"
answerable only from git timestamps.*

## Resume here: an A5 design note, and one measurement before it

**Do not write more A5 render code yet.** Two decisions taken on 2026-08-12 change the shape
of the work, and one open question is wider than A5.

**Status 2026-08-12.** §1 is closed: the alpha defect was real and the fix has landed behind
`<straight-alpha-grading>` (default off). **§2 step 1 is closed too** — fp16 is measured, and
so are the two paths that had never executed. The next step is §2 step 2, a post-composite
colour stage. Everything from §3 on is unchanged.

### 1. ~~Measure the alpha domain first~~ — MEASURED 2026-08-12, and it is real

OCIO's documented practice is to **unpremultiply, transform, then re-premultiply** — OIIO
exposes `unpremult` on `colorconvert` for exactly this. Both mixers do the opposite:
`if(flag(F_STRAIGHT_ALPHA)) col.rgb *= col.a;` runs **before** the input-convert block and
before the OCIO splice, so every non-linear transform — OCIO's *and* the built-in EOTF — is
applied to premultiplied RGB.

* **Pre-existing.** The built-in chain has always done this; the OCIO work inherited it.
* **Only wrong where `0 < alpha < 1`.** Opaque content is unaffected.
* **No battery could see it.** `conformance`, `grading` and `ocio` are all flat *opaque*
  patches. Structurally the same blind spot as the grey-ramp/channel-swap one, and it lands
  on exactly the content this fork cares about — HTML graphics are fill+key with soft edges.

**No longer a claim from reading.** `CasparCG-TestRunner/cli.py alpha-domain` now measures it,
and both mixers transform **premultiplied RGB** — 20 discriminating patches each, every one
selecting that model at **≤0.51 LSB** while the straight-domain model sits **13.5 to 116 LSB**
away. The OGL and Vulkan reports are byte-identical apart from the mixer's name. Full account:
`CasparCG-TestRunner/docs/alpha_domain_2026-08-12.md`.

The sharpest single number: `MIXER COLORSPACE PQ BT2020 NONE BT709 REC709` on `#8073401A`
(alpha 0x80) renders **243.8** where the straight domain gives **128.0**. Not a tolerance
question.

Two controls, both able to fail, are what make that readable rather than plausible. A pure
gamut matrix separates the two models by exactly **0.00 LSB** — a matrix commutes with a
scalar — so the rig cannot be manufacturing differences. And the ten patches where the two
hypotheses coincide (alpha 0xFF, plus the whole matrix case) hold at **1 LSB, worst 0.49**, so
the rig reduces to `conformance` when alpha is taken out of the picture.

**Confirmed wider than OCIO, as predicted.** The cleanest row has no OCIO, no matrix and no
output encoding in it — `srgb → linear`, a per-channel power law — and it separates by up to
56 LSB. So the built-in EOTF/OETF chain and every non-linear grading tool downstream inherit
this, and it is a correctness issue to fix before building A5 on top.

**What the measurement does not cover:** a source that declares *straight* alpha. The colour
producer declares premultiplied (`color_producer.cpp:46`; `is_straight_alpha` defaults false
at `pixel_format.h:138`), so `col.rgb *= col.a` never ran for these patches — the pixel was in
the premultiplied domain by declaration. `image_producer.cpp:64` passes `true`, so every PNG
and TGA still reaches the same domain via that multiply, one unbranched line above the same
transform. Everything below that point is measured; the line itself is still only read.

**Also flagged, not measured:** `shader.frag:1861`, the shape/fill composite, is commented
*"Porter-Duff 'over' composite (straight alpha)"* while the pixel reaching it is
premultiplied. A fix to the alpha domain has to decide about that block too.

#### The fix — DONE, behind `<straight-alpha-grading>` (default off)

Unpremultiply above the transform chain and premultiply once below it. The
`is_straight_alpha` flag now selects *which* end needs the operation rather than gating one:

```
if (!is_straight_alpha && col.a > 0.0) col.rgb /= col.a;   // premultiplied source -> straight
<input convert / OCIO splice, grading, output convert / OCIO display>
col.a *= opacity; col.a *= local_key; col.a *= layer_key;  // alpha alone, in this domain
col.rgb *= col.a;                                          // premultiply once, before blend
```

Both questions the fix had to settle were answered before it was written:

* **The re-premultiply goes after opacity and both key multiplies**, immediately before
  `blend()` — not straight after the output conversion. It is provably identical to the old
  behaviour at alpha 1.0, and it fixes a second defect in passing: `opacity`, `local_key`
  and `layer_key` each scale col.a while the legacy path's col.rgb still carries the OLD
  alpha, so the two reach the blend disagreeing.
* **Default off**, because it changes rendered output wherever content has soft edges and a
  non-linear transform is configured. `CHANGELOG.md` carries the entry with the measurement.

Measured after, both mixers, reports byte-identical to each other: every discriminating
patch selects the straight model at **≤0.46 LSB**. Default path unregressed — conformance
100/100, grading 48/48, `ocio` 6/6 (worst 0.55) on both mixers, `mixer-parity` 6/6 rasters
identical, `vk-validation` 0 VUIDs.

`cli.py alpha-domain` now asserts the verdict against the channel's setting rather than
reporting a one-way finding: exit **0** when the domain matches `<straight-alpha-grading>`,
**1** when it does not, **2** for inconclusive. Both settings exit 0 on both mixers.

**Still not covered:** a straight-alpha source, and `shader.frag`'s shape/fill composite (see
above). Both unchanged by this work.

### 1b. Four ACEScg gamut matrices were wrong — found AND FIXED 2026-08-12

Found while designing step 2 below, by checking the tables the design was going to build on.
**`k_to_working` and `k_to_output` are not the matrices they claim to be for `bt2020`,
`p3_d65`, `arri_wg3` and `sgamut3cine`** — identical numbers on both mixers. Full account,
evidence and the correct rows: [`GAMUT_MATRIX_DEFECT_2026-08-12.md`](GAMUT_MATRIX_DEFECT_2026-08-12.md).

Three independent lines agree: a colorimetric derivation that reproduces the correct `bt709`
row to 1.5e-5, a round trip needing no external reference, and OCIO's own matrices. Worst
deviation 0.41 per element.

**This blocked step 2** — a working-space composite is defined by the conversion into and
out of ACEScg, and four of the seven routes were wrong. **Now fixed**, so step 2 is
unblocked. Auditing the direct tables as well found `k_direct` had 6 of 16 checkable
entries wrong too, while `k_direct_cg` was clean. Verified after: `ocio` 18/18 across three
channel gamuts, `conformance` 100/100, `grading` 48/48, both mixers.

It also lands on what this fork is for: `MIXER COLORSPACE LOGC3 ARRI_WG3 …` is the first
usage example in `COLOR_GRADING.md`, and `MIXER OCIO` on a BT.2020 channel takes
`k_to_output[1]`.

**No battery could have caught it**, and that is the second finding: the harness's
`color_math` and `ocio_reference` transcribe these same tables, so a wrong matrix is
compared against a copy of itself. `CasparCG-TestRunner/core/gamut_reference.py` now derives
them from OCIO instead, and `tests/test_gamut_matrices.py` holds the four as
`xfail(strict=True)` so they fail loudly when the server is fixed.

The fix changes rendered output for every existing configuration using those paths —
neutrals not at all, saturated colour by up to 67 LSB. `CHANGELOG.md` carries the table.

### 2. The plan is reordered — go straight to the working-space composite

Channel-level as *"applied where the built-in output conversion runs, per layer"* — which is
what the inert code below implements — is a **stepping stone that gets replaced**, not
extended, if consumer-level is the destination. It is, and effort is not the constraint.

New order:

1. ~~**Float composite.**~~ **DONE 2026-08-12, with one correction to the plan.**

   **There is no `bit12` channel to measure on.** `<color-depth>` accepts only 8 or 16
   (`server.cpp:302-304`); `bit12` is only ever a *source* pixel format. So the measurement
   was reframed to the render target's actual quantum per signal level, which is answerable
   and stronger — a 16-bit capture resolves 1/32 of fp16's ulp near white, better than a
   12-bit path could have.

   **fp16's quantum measures 32.0 LSB16 near white** — 2.0x a 12-bit output step, 0.5x a
   10-bit one — halving per octave exactly as `2^(e-10)` predicts (32, 32, 16, 4, 1). unorm
   reads 1.0 at every level; both mixers byte-identical. `cli.py banding`, and
   `CasparCG-TestRunner/docs/render_format_quantum_2026-08-12.md`.

   So **fp16 is a real highlight regression only against a 12-bit or better consumer**, and
   12-bit is a consumer property rather than a channel one. Undecided, and this instrument
   cannot decide it: whether fp16's *shadow* precision is worth having — the half of §4.3.4
   that argues FOR fp16 sits below a 16-bit unorm capture's floor.

   **Both fp16 paths listed under "What is NOT verified" are now executed:**

   | | |
   | :--- | :--- |
   | the fp16 render target, both mixers | `conformance --render-format fp16` **100/100 within 1.0 LSB** |
   | the Vulkan fp16 resolve blit under the layers | `vk-validation --render-format fp16` **0 VUIDs**, 12/12 commands, server survived |

   The first was confirmed against the server's own log line rather than against the battery
   passing, because a silently dropped element would have passed too. `config_generator` can
   now emit `<render-format>`, which is what the previous handoff said to do first.
2. ~~**A post-composite colour stage.**~~ **DONE 2026-08-12** — `<working-space-composite>`,
   default off, both mixers. Every layer converts into ACEScg, none out; the channel applies
   the display encoding once, ahead of the calibration LUT. Preconditions fp16 and
   auto-color-convert, both refused rather than warned about.

   Measured with the new `cli.py blend-domain`, both mixers byte-identical: default reports
   the display domain at worst 0.50 LSB, the option reports the working domain at worst
   0.60. A 50% mix of black and white reads 128 blending display values and 191 blending
   light.

   **`conformance` and `grading` are structurally blind to this** — one layer over black is
   the same pixel either way — which is why `blend-domain` had to exist. Everything below
   was the design, and it held:

   * **No shader change is needed.** The input and output halves are independent uniforms
     already (`do_input_convert` / `do_output_convert`, `F2_INPUT_CONVERT` /
     `F2_OUTPUT_CONVERT`) — that is what A4e's split bought.
   * **The post-composite pass reuses the OCIO input-transform branch verbatim**
     (`image_kernel.cpp:719`): input half off, output half on, `luminance_scale` 1.0,
     `k_to_output[target]`. It is exactly what that branch already sets up, so the change is
     `if (ocio_in || params.output_convert_only)` rather than a fourth branch.
   * **Layer draws need `working_space_composite`** to do two things: suppress the output
     half (an override outside the branch chain, where `ocio_out` already sits), and force
     the input half through the ACEScg route. `k_direct_cg`, `k_direct` and the
     "source already matches target" skip each leave the pixel somewhere other than AP1,
     and a composite of layers in different spaces is not in any space.
   * **It requires fp16**, which §1 verified, and it is opt-in because it changes blending.
3. **Channel-level display transform** in that stage, as the default view.
4. **Consumer override** — a fan-out in the mixer, one pass per distinct view.

Surviving the reorder: `build_display_transform`, the binding-range split, the 1D/NEAREST
fixes, and the per-layer variant machinery (which input transforms still need). Rewritten:
the display splice location and the `draw_params.ocio_display` plumbing.

### 3. Answered: layer-level is wrong for a display transform

An **input** transform describes where the pixels came from — per layer, correctly. A
**display** transform describes what screen they are going to, and every layer in a channel
goes to the same screen. Two layers with different display transforms would blend a
PQ-encoded layer with a Rec.709-encoded one, and that composite is not in any space. The
per-layer "look" people usually mean is the creative grade, which already exists per layer
and operates in ACEScg.

### 4. Consumer-level: two real costs, neither fatal

* **It changes blending permanently for any channel using it.** One composite serving two
  views must stay in working space, so layers blend in scene-linear ACEScg rather than in
  display-encoded values — and the shader comment is right that blend modes were designed
  for 0–1 display values. Unavoidable rather than an implementation choice: display
  transforms are not invertible, so you cannot composite in one display space and convert to
  another. Arguably more correct — it is what film compositors do — but existing looks shift.
* **fp16/fp32 stops being optional.** ACEScg needs range above 1.0 and below 0; unorm clamps
  both.

**Not a problem, despite sounding like one:** consumers need no GPU passes of their own. The
transform runs in the mixer; only the *configuration* lives with the consumer. The mixer does
N passes over one working-space composite and hands each consumer a finished texture. No
cross-device work, no consumer touching the mixer's device.

### 5. What the design note must settle

Beyond the three-level model (input = layer, look = layer, display = channel default +
consumer override) and fp16 vs fp32:

* **Alpha domain** — measured, see §1. The transform runs on premultiplied RGB on both
  mixers; the design note has to state which domain A5's display transform consumes, and the
  fix in §1 should land before it rather than after.
* **`isData` policy.** OCIO bypasses data colour spaces by default and v2 lets the
  application opt in. We never check `isData`. For a server whose key channel is an output,
  that deserves a decision rather than an accident.
* **Config source.** The docs point at `GetCurrentConfig()`, which auto-initialises from
  `$OCIO`. We pin a built-in URI instead. The pin is right — approved looks must not change
  because an environment variable did — but it should be a *stated deviation*, and
  `MIXER OCIO CONFIG` should probably accept `$OCIO` explicitly.
* **Pre-warming**, reframed: OCIO's own guidance is that building a processor is expensive
  and thread-blocking and should happen "as infrequently as is sensible". Following the
  standard, not a performance nicety.
* **Dynamic properties** are the documented mechanism for exposure/gamma without rebuilding
  the processor — and are what the reserved, currently unused UBO at set 1 binding 0 exists
  for. That is where the missing `exposure` on the OCIO path belongs.

The study (`OCIO_INTEGRATION_STUDY.md`) predates all of this and several of its assumptions
have since been measured wrong — no post-composite stage (there is one), the UBO, `sampler3D`.
It needs a successor, not an edit.

## Sources for the practice claims above

* [Developing with OpenColorIO](https://opencolorio.readthedocs.io/en/latest/guides/developing/developing.html)
  — roles, display/view, processor cost
* [Colorspaces — `isData`](https://opencolorio.readthedocs.io/en/latest/guides/authoring/colorspaces.html)
* [OCIO 2.0 release notes — data space handling](https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/main/docs/releases/ocio_2_0.rst)
* [OpenImageIO `colorconvert` / `unpremult`](https://github.com/OpenImageIO/oiio/blob/master/src/include/OpenImageIO/imagebufalgo.h)

### What is done

* **`build_display_transform(display, view, out, target)`** generates the working-space →
  display/view program for either backend, sharing `fill_from_processor` with the input
  transform so the texture walk and the binding-index read exist once.
* **1D LUT images, on both backends.** `GL_TEXTURE_1D` on OpenGL; a real `e1D` image and
  view on Vulkan rather than an Nx1 2D one. See the measurement below — this was a genuine
  gap in what A4e shipped.
* **Per-texture filtering.** OCIO says NEAREST or LINEAR per table and the Vulkan side was
  binding its linear sampler to everything. It now selects `keySampler_` (clamp-to-edge
  nearest) for the tables that ask for it. OpenGL already did this correctly.
* **The second splice site**, `//__CASPAR_OCIO_DISPLAY__`, in both shaders, at the output
  block it replaces — before the blend, where the block it replaces sits. ⚠ The OGL marker
  swizzles, the Vulkan one must not.
* **`draw_params.ocio_display` / `.ocio_view`** on both backends, and **both kernels** build,
  cache, splice and upload both halves. The variant cache key is the **pair** of cache IDs,
  because two source spaces through one display are two programs.
* **The output-half override sits outside the branch chain, on both backends.** An earlier
  attempt anchored it between two `else if` arms, which compiled cleanly and silently gated
  auto colour conversion on there being no display transform. Anchor uniqueness is not
  anchor correctness in a 2000-line function; read the patched region.

### Status of the code above

**Nothing populates `draw_params.ocio_display`, so all of the display work is inert.** It
compiles, it is regression-clean, and it has never executed. Both kernels were brought to
readiness together deliberately, so the operator surface would light OpenGL and Vulkan up at
once rather than shipping a parity gap.

Per the reorder in "Resume here", the splice location and the `draw_params` plumbing are
expected to be **rewritten** for a post-composite stage. Leave them — they cost nothing and
they document the shape. `build_display_transform`, the binding ranges and the 1D/NEAREST
fixes survive unchanged.

Still true whenever the operator surface is built: the LED calibration LUT carries a
**render-fingerprint field**, and a display transform needs one too or the still-frame cache
serves a stale look. `MIXER <ch> OCIO_DISPLAY "<display>" "<view>"` should validate against
`has_display_view()`, which already exists — and quote both arguments, because every display
and view name in this config contains spaces.

A harness battery is still owed: `cli.py ocio` compares an input transform against OCIO's CPU
processor, and the display half needs the same treatment. The oracle already has the pieces.

### Three measurements that shape it

**Display transforms emit `sampler1D`, and input transforms never did.** Two of the three
textures in an ACES 2.0 HDR view are 1D — the reach and gamut-cusp tables, 363×1 and
**`INTERP_NEAREST`**, one of them 3-channel. A4e's Vulkan uploader treated every non-3D LUT
as 2D and bound its linear sampler to all of them, both of which are wrong for those tables:
an `e2D` view does not match a `sampler1D` declaration, and interpolating between entries
that were never meant to be interpolated is wrong rather than soft. **Both fixed**, on both
backends. Unreached until now only because every input-transform LUT is 2D and linear —
which is exactly how "not 3D means 2D" and "one sampler for all of them" survived.

**Input and display transforms collide at binding 1 unless told otherwise.** Both declare
their first sampler there. They now take disjoint ranges — input 1..4, display 5..8, inside
the 8 that descriptor set 1 reserves — via `INPUT_TEXTURE_BINDING_START` /
`DISPLAY_TEXTURE_BINDING_START`. Verified: with start 5, OCIO emits bindings 5, 6, 7 and the
sampler names do not collide either (`ocio_in_` vs `ocio_out_`).

**Everything else A4e built covers display transforms unchanged.** Across all 41
display/view combinations: **at most 3 textures**, **zero 3D LUTs**, **zero dynamic
uniforms**. So no `e3D` view is needed, the 8 reserved bindings are enough, and the
placeholder uniform buffer stays unused. The generated source is ~16 KB against ~1.5 KB for
an input transform, which lands entirely on compile time.

### The architectural point

A display transform must consume **working-space** pixels. The mixer converts to display
space **per layer, before blending** — deliberately, so blend modes operate on display
values — so by the time a composite exists it is already display-encoded, and a
post-composite pass is too late *as the mixer stands today*.

That is what the inert code above works around, and it is why the plan was reordered: the fix
is to stop converting per layer, not to work around the fact that we do. See "Resume here".

### Also still open

**Pre-warming, which A4f made twice as expensive and A5 will make worse again.** A new
colour space costs OCIO generation, a LUT image creation with a `waitIdle`, a shaderc
compile and a driver pipeline build on the frame path — ~1.2 s and one dropped frame, and
the OCIO battery has to sleep 1.5 s on the first patch of each case. A display transform's
source is ten times larger, so its compile will be longer. The fix is to build the variant
when the command is accepted, where the processor is already built for validation.

## What A4e established, and how

The two open questions the previous handoff named are answered, by generating the source and
reading it rather than by reasoning about the API:

| question | answer |
| :--- | :--- |
| do the declarations land at the reserved bindings? | **yes.** `setDescriptorSetIndex(1, 1)` emits `layout(set=1, binding = 1) uniform sampler2D` — exactly the set layout the pipeline declares |
| does OCIO emit `sampler3D` for anything in this config? | **no.** All 55 colour spaces checked: zero 3D textures, zero `sampler3D`, max one texture per space, always 2D single-channel |
| *(new)* does OCIO emit a uniform buffer? | **no.** `getUniformBufferSize()` is 0 for all 55 — an input transform has no dynamic property, so no uniform block is declared at all |

That third one corrects this document's previous assumption that binding 0 held OCIO's UBO
and that A4e would size it from the real value. It stays declared and written as a
zero-filled 256-byte placeholder: a descriptor set may legally carry a binding the shader
never reads, and a display transform with a dynamic exposure (A5) is what would fill it.

The bindings are taken from `getTextureShaderBindingIndex()`, not computed as
`textureBindingStart + i`. They agree today; only one of them is what OCIO actually wrote
into the source.

## The defect A4e found

**`MIXER OCIO` on a Vulkan channel was accepted, logged, and discarded.**
`vulkan/util/transforms.cpp`'s `apply_transform_colour_values` merges the image transform
field by field and did not copy `ocio`. The OGL copy of that function has carried the member
all along, next to a comment predicting this exact failure: *"a new `image_transform` member
that is not listed here simply never reaches the kernel. The symptom is a command that
reports 202 and changes nothing."*

Worth internalising: **every Vulkan-side OCIO symptom was downstream of a value that was
never delivered.** The first battery run after wiring A4e reported 0/6 with plausible
per-space deltas, which is exactly what a missing splice looks like. What distinguished them
was the log — the AMCP command's `(opengl)` validation build appeared and no `(vulkan)` build
did, so the kernel's new code had never run at all.

Also fixed: `build_input_transform` logged a full processor description at `info` **every
frame** — it is called per layer per frame by both kernels, because OCIO's cache ID is what
they key on and only OCIO can produce it. Now once per distinct cache ID.

## Verification

| | evidence |
| :--- | :--- |
| OGL OCIO input transform | **6/6 colour spaces within 1.0 LSB**, worst 0.55 (`cli.py ocio --mixer ogl`) |
| the numeric claim behind it | `#3080A0` → `MIXER OCIO ACEScct` rendered `(0,202,255)`; the model gives `(0,202,255)`, delta `[0 0 0]` |
| the 1 LSB gate itself | **measured**, not assumed: 0.39–0.55 clustered. `CasparCG-TestRunner/docs/ocio_tolerance_2026-08-11.md` |
| LUT-texture upload (OGL) | proven via `ADX10` and `Rec.2100-PQ - Display`, the only spaces that emit a texture |
| **Vulkan OCIO resource path (A4e)** | the log shows the `(vulkan)` `glsl_vk_4.6` build and a LUT upload for **exactly** `ADX10` and `Rec.2100-PQ - Display`, and for no other space — which is the prediction the 55-space survey made |
| **Vulkan API usage** | **0 VUIDs** under the Khronos validation layers across four colour spaces, both LUT-bearing and not (`cli.py vk-validation`) |
| A4e changes no rendered output | the OCIO battery on Vulkan is **bit-identical before and after** — 0/6, worst 143.39 LSB both times. That is the correct result: without the splice the transform is still a no-op |
| base paths unregressed | conformance **100/100** and grading **48/48** on **both** mixers, re-run after this session's changes |
| harness | **1184 tests pass**, including the OCIO oracle and the new `vk-validation` guards |
| runtime SPIR-V (Vulkan) | shaderc rebuilds the real 40 KB shader → 100/100 identical |

**On validation-layer silence.** The server requests validation layers only under
`#ifdef _DEBUG` (`vulkan/util/device.cpp:394`), and every build measured here is Release —
so the debug messenger it installs does not exist and its silence proves nothing. The
previous handoff's "validation layers loaded and silent" claim should be read with that in
mind. `cli.py vk-validation` forces the layer through the loader instead, and runs a
best-practices **positive control** so that "no errors" is distinguishable from "no
reporting"; it exits **2 for inconclusive** rather than folding that into a pass.

## What is NOT verified — compiled but never executed

1. ~~**The fp16 render target on either mixer.**~~ **VERIFIED 2026-08-12.**
   `config_generator` emits `<render-format>` now, and `conformance --render-format fp16`
   is **100/100 within 1.0 LSB on both mixers** — confirmed against the server's own
   `[server] Channel 1 render-format fp16 (float working space)` log line, because a
   silently dropped element would have passed the battery too.
2. ~~**The Vulkan fp16 resolve blit** (`renderpass::commit()`).~~ **VERIFIED 2026-08-12.**
   `vk-validation --render-format fp16`: **0 VUIDs**, 12/12 commands accepted, server
   survived, positive control produced 8 messages. The barriers were correct as written.
3. **A 3D LUT through the Vulkan OCIO path.** Implemented, and unreachable with the pinned
   config — no colour space in it emits one. It reuses the two helpers `MIXER LUT3D` already
   exercises, so it is inherited-proven rather than measured.
4. **Vulkan OCIO end to end** — A4f.

Also unverified: whether `(0,202,255)`-style agreement holds at **16-bit**.

## Gates — run these, in this order

```bash
cd d:\Github\CasparCG-TestRunner
python cli.py conformance   --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py grading       --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py ocio          --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py vk-validation --server d:\Github\CasparVP\build\shell\casparcg.exe
# then --mixer vulkan for the first three. Parity is required, not optional.
python -m pytest tests/ -q     # includes the OCIO oracle and vk-validation guards
```

**Redirect to a file and grep. Do not pipe a battery through `tail`** — it races the summary
line.

Build via a scratchpad `.bat` with the 14.50 BuildTools pin; `--target casparcg` does **not**
refresh deployed DLLs (see `BUILDING_WORKFLOW.md` #7).

## Traps that cost time here — all recorded in-tree

* **"Server did not become ready" is often the GPU running out of memory, not a port race.**
  This is a correction to what this document said earlier. `server_manager.py` reports the
  failure as `Cannot connect to localhost:<port>` and quotes a `caspar::user_error` taken
  from *another worker's* log, with that other log's **timestamp** — which is what makes the
  port explanation convincing. It was accepted as a port/readiness race five times in one
  session before anyone looked at the Windows event log:

  > The NVIDIA OpenGL driver has encountered an out of memory error. (pid=… casparcg.exe)

  **198 of those in the 30 minutes covering two failed `conformance` runs**, and 390 across
  the session, every cluster lining up with a battery. Seven parallel servers each hold an
  OpenGL context — and on the Vulkan mixer a *second*, dedicated OGL device for previz — on
  an 8 GB P4000 (adapter 0, where both mixers render; the 16 GB A4000 is not where the work
  goes). The server dies before it can answer AMCP, so the harness only sees a connect
  timeout.

  It is marginal rather than absolute: the same battery passed at 8 instances earlier in the
  same session. **`--instances 4` is the reliable setting on this box**, and re-running at
  the default is what makes this look intermittent.

  So the fix in `server_manager.py` is worth more than "stop quoting the wrong log": it
  should read the event log, or at least stop asserting a cause it has not established.
* **Do not run a second battery beside one**, and expect the same failure if you do — it is
  the same GPU. The registry protects pids; it does not make VRAM appear.
* **`CASPAR_VK_RUNTIME_SHADER=1` is incompatible with parallel battery runs** — use
  `--sequential`.
* **`glslangValidator -S frag` fails on the *Vulkan* shader** (`input_attachment_index`).
  Use `glslc`. `BUILDING_WORKFLOW.md` #6b.
* **OCIO's build is path-length sensitive.** `BUILDING_WORKFLOW.md` #6.
* **Never put backslash content through a Python heredoc.** Use Edit/Write.
* **The Vulkan loader's `Located json file` is not `Insert instance layer`.** It logs the
  first for every layer it can see, used or not. Matching it read as proof that validation
  was active when nothing had been loaded. Pinned by `tests/test_vk_validation.py`.
* **A header change does not trigger a rebuild in this tree, and the timestamp check does
  not notice.** This cost the most time of anything in this session and produced a bug report
  that was entirely fictitious: after A4f the server aborted with `0xC0000409` on the first
  composited frame under the validation layers, reproducibly, bisecting cleanly to the
  commit, and with **no OCIO in the path** — a stale object file calling through a shifted
  vtable slot after one virtual was added to `frame_context`. A full recompile made it
  vanish permanently. The tell was **zero validation messages before the abort**: the layers
  were not reporting a violation, so nothing was wrong with the Vulkan usage. Root cause and
  the touch-everything workaround are in `BUILDING_WORKFLOW.md`.
* **`cli.py vk-validation` reported that crash as a PASS**, because it counted VUIDs and
  never asked whether the server survived the scenario — 4 of 24 commands ran, exit 0. It
  now tracks refusals and the server's exit code and reports inconclusive. Every measurement
  tool in this pair of repos has now failed in the same direction at least once: silence
  reported as success.

## Open decisions

* **Frame-path compile stall.** A variant compiles on first draw: ~1.2 s, one dropped frame.
  **Pre-warming is the fix, not caching** — compile when `MIXER OCIO` is accepted, where the
  processor is already built for validation. A4e adds a second frame-path cost of the same
  shape (the LUT image creation does a `waitIdle`), and the same pre-warm removes both.
* **`exposure` and `gamut_compress` are unavailable on the OCIO path.** Both live in the
  `color_grade` struct inside the block OCIO replaces. Needs a decision and a doc line.
* **fp16 vs fp32** — half answered, 2026-08-12. Measured: fp16's quantum is 32.0 LSB16 near
  white, exactly 2× a 12-bit output step, and there is no `bit12` channel to have measured
  it on (12-bit is a consumer property). What remains is whether fp16's *shadow* precision
  is worth having, which a 16-bit unorm capture cannot see. `cli.py banding`.
* **A5's display transform: channel or consumer?** Decided channel-level for now; a channel
  feeding an LED processor *and* an SDI monitor needs two views from one composite, which
  argues for consumer-level. Settle before building A5.

## Cleanup

* **`d:\_ocioprobe` — 746 MB, needs a manual `Remove-Item -Recurse -Force`.** Redundant now
  that OCIO builds in-tree. The sandbox blocked me from deleting it.
* `ENABLE_OCIO` defaults **ON**. `-DENABLE_OCIO=OFF` skips it and its six bundled deps.

## Harness repo

`CasparCG-TestRunner`, unpushed. `7b18ac9` (the OCIO oracle), `70f6c6f` (the `ocio` battery +
tolerance doc), plus this session's `core/vk_validation.py`, its `cli.py` subcommand and
`tests/test_vk_validation.py`. That last one earned itself immediately: it caught a message
counter that double-counted one layer message as two.
