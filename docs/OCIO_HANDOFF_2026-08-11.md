# OCIO integration — handoff, 2026-08-11

Resume point for the OCIO work on branch **`feature/ocio-mixer`**, off `CasparVPV`. Plan of
record: [`OCIO_INTEGRATION_STUDY.md`](OCIO_INTEGRATION_STUDY.md).

**A4e is done.** Updated in place rather than as a second file — four same-dated handoffs
once existed in the harness repo and which was current could only be recovered from git
timestamps.

## Resume here: A4f

**Splice OCIO's generated source into the Vulkan fragment shader and select the variant
pipeline per draw.**

Everything below A4f is built, run and measured. What is left is genuinely two things:

1. **The splice.** Markers are in place at `fragment_shader.frag` —
   `//__CASPAR_OCIO_DECLARATIONS__` before `main()` and `//__CASPAR_OCIO_TRANSFORM__` after
   the input block. ⚠ **It must NOT swizzle.** That shader carries true RGB
   (`col.rgb = ubo_mat3(...) * col.rgb`) where OGL carries BGR
   (`col.bgr = input_to_working * col.bgr`), so the OGL splice's `.bgr` is wrong here.
   Both markers carry this warning in the source. The generated text is spliceable verbatim:
   no `#version`, no `#extension`, no bare non-opaque uniform — verified across all six
   measured spaces.

2. **A per-layer pipeline, which the renderpass does not currently have.** This is the real
   structural work and it is not obvious from the task name. `renderpass` takes ONE pipeline
   at construction (`renderpass.cpp:57`, `_pipeline(ctx->get_pipeline())`) and uses it for
   every layer in `commit()`. Per-draw variant selection means `layer_info` needs its own
   `std::shared_ptr<pipeline>`, chosen in `renderpass::draw()` where the transform is known,
   and `commit()` must bind it per layer. `device::get_variant_pipeline` already exists and
   is keyed on (variant id, attachment format); the kernel just cannot reach it from where
   the pipeline is currently decided.

Also still to settle before A4f is finished: **the uniform half.** With OCIO producing the
working-space pixel, the shader's own input conversion must be switched off and the output
half left running — the OGL kernel does this at `image_kernel.cpp` (`do_input_convert=false`,
`working_to_output` from the channel's target). Nothing on the Vulkan side sets those yet, so
a splice alone would double-convert.

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
