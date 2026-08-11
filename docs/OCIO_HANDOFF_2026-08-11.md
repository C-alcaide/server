# OCIO integration — handoff, 2026-08-11

Resume point for the OCIO work on branch **`feature/ocio-mixer`**, off `CasparVPV`. Plan of
record: [`OCIO_INTEGRATION_STUDY.md`](OCIO_INTEGRATION_STUDY.md).

**A4e and A4f are done: the OCIO input transform now works on both mixers, at parity.**
Updated in place rather than as a second file — four same-dated handoffs once existed in the
harness repo and which was current could only be recovered from git timestamps.

## Resume here: A5, channel-level display transform

**Decided: channel-level first, consumer-level added afterwards as an option.**

`build_display_transform()` exists and is measured. What is left is the render integration
and the operator surface. Read the next section first — three measurements shape the design
and one of them invalidates code A4e already shipped.

### What is done

`accelerator/ocio/ocio_config.{h,cpp}`: `build_display_transform(display, view, out, target)`
generates the working-space → display/view program for either backend, sharing
`fill_from_processor` with the input transform so the texture walk and the binding-index
read exist once.

### What is left, in order

1. **A second splice site.** `fragment_shader.frag` (both mixers) has markers for the input
   transform only. The display transform replaces the *output* half — the
   `working_to_output` matrix, the OETF and the tone map — so it needs its own marker at
   that block, and the uniform side must clear `F2_OUTPUT_CONVERT` the way A4f clears
   `F2_INPUT_CONVERT`. ⚠ Same no-swizzle rule on Vulkan.
2. **1D LUT images** — see below, this is a real gap.
3. **Channel-level state.** Follow the LED calibration LUT, which is already a channel-master
   setting applied over the composited frame (`ogl/image/image_mixer.cpp`,
   `set_calibration_lut`). Note it also has a render-fingerprint field, and a display
   transform needs one too or the still-frame cache will serve a stale look.
4. **AMCP.** `MIXER <ch> OCIO_DISPLAY "<display>" "<view>"` and `NONE`, validated at command
   time against `has_display_view()` — which already exists. Quote both arguments; every
   display and view name in this config contains spaces.
5. **A harness battery.** `cli.py ocio` compares an input transform against OCIO's CPU
   processor; the display half needs the same treatment and the oracle already has the
   pieces.

### Three measurements that shape it

**Display transforms emit `sampler1D`, and input transforms never did.** Two of the three
textures in an ACES 2.0 HDR view are 1D (the reach and gamut-cusp tables). **A4e's Vulkan
uploader treats every non-3D LUT as 2D** — `create_image_2d_wh` with an `e2D` view — so it
will produce an image whose view type does not match the `sampler1D` the shader declares.
That needs an `e1D` image and view before A5 renders anything, and the OpenGL side needs
`GL_TEXTURE_1D` for the same reason. It is unreached today only because no input transform
emits one.

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

### The architectural point, and why channel-level is built this way

A display transform must consume **working-space** pixels. The mixer converts to display
space **per layer, before blending** — deliberately, so blend modes operate on display
values — so by the time a composite exists it is already display-encoded and a
post-composite pass is too late.

So channel-level is implemented as a channel-scoped *setting* applied where the built-in
output conversion already runs, per layer. With one transform for the whole channel the
result is exactly a channel display transform, blend semantics are unchanged, and it needs
nothing from the float path.

**Consumer-level cannot be added on top of that.** One composite feeding two different views
requires the composite to still be in working space, which means turning off the per-layer
output conversion, compositing in float, and blending in linear — a blend-semantics change
and a hard dependency on the fp16 render target, which is still the "compiled but never
executed" item below. Treat consumer-level as that larger piece of work rather than an
increment on channel-level.

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

1. **The fp16 render target on either mixer.** No config selects it. `<render-format>fp16</render-format>`
   is parsed; nothing sets it. `config_generator` in the harness cannot emit the element
   either, so `vk-validation` cannot yet be aimed at it — teach it that first.
2. **The Vulkan fp16 resolve blit** (`renderpass::commit()`). Barriers written against the
   documented contracts and never exercised. `cli.py vk-validation` now exists precisely for
   this moment.
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

* **The harness misattributes startup failures.** `server_manager.py` prints a
  `caspar::user_error` from *another worker's* log as "the server's last relevant log line",
  with that other log's **timestamp**, which is what makes it convincing. Hit three times
  this session — `conformance --mixer ogl` (port 5250) and `grading --mixer vulkan` twice
  (5252, 5326). All three were port/readiness races: ogl re-ran clean at 100/100, and
  grading vulkan passed 48/48 once run `--sequential` on an idle machine. **Fixing this is
  still on the list**, and it is now the single most expensive thing left in the harness.
* **Do not run a second battery beside one.** Two of those three failures were provoked by
  overlapping runs — including one I caused by starting `vk-validation` alongside `grading`.
  The registry protects pids; it does not make the readiness timeout any longer.
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
* **fp16 vs fp32** for the float path — needs the B2 banding measurement on a `bit12`
  channel, not a calculation. Study §4.3.4: fp16 is ~2× coarser than a 12-bit output near
  white.
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
