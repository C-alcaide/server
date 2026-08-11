# OCIO integration — handoff, 2026-08-11

Resume point for the OCIO work on branch **`feature/ocio-mixer`** (21 commits, off
`CasparVPV`, clean tree, nothing pushed). Plan of record:
[`OCIO_INTEGRATION_STUDY.md`](OCIO_INTEGRATION_STUDY.md).

## Resume here: A4e

**Generate Vulkan shader source from OCIO and upload its LUTs into descriptor set 1.**

Everything it needs is already built and proven (A4a–A4d below). The two things that can
actually be wrong, and neither is answerable by reading:

1. **Do the generated declarations land at the bindings that were reserved?**
   `setDescriptorSetIndex(1, 1)` should put OCIO's UBO at set 1 binding 0 and its textures at
   bindings 1..N. Set 1 is declared with binding 0 as a uniform buffer and bindings 1..8 as
   combined image samplers. Generate the source and *read it* before wiring anything.
2. **Does OCIO emit `sampler3D` for anything in this config?**
   The descriptor type (`eCombinedImageSampler`) covers 2D and 3D alike, but the **image view
   type must match what the shader declares**. Every space measured so far needs either zero
   textures or one 2D (4096×17, single channel). A 3D LUT has not been seen and would need an
   `e3D` view.

Then **A4f**: splice and select per draw. ⚠ **The Vulkan splice must NOT swizzle.** That
shader carries true RGB (`col.rgb = ubo_mat3(...) * col.rgb`) where OGL carries BGR
(`col.bgr = input_to_working * col.bgr`). Markers are already in place at
`fragment_shader.frag` — `//__CASPAR_OCIO_DECLARATIONS__` before `main()` and
`//__CASPAR_OCIO_TRANSFORM__` after the input block — both commented with this warning.

## What is verified, and how

| | evidence |
| :--- | :--- |
| OGL OCIO input transform | **6/6 colour spaces within 1.0 LSB**, worst 0.55 (`cli.py ocio --mixer ogl`) |
| the numeric claim behind it | `#3080A0` → `MIXER OCIO ACEScct` rendered `(0,202,255)`; the model gives `(0,202,255)`, delta `[0 0 0]`. Red clamps because AP1→709 drives it to −0.509, blue clips at 2.718 — both correct, not artefacts |
| the 1 LSB gate itself | **measured**, not assumed: 0.39–0.55 clustered, no outliers. Basis: `CasparCG-TestRunner/docs/ocio_tolerance_2026-08-11.md` |
| LUT-texture upload (OGL) | proven via `ADX10` and `Rec.2100-PQ - Display`, the only spaces that emit a texture |
| base paths unregressed | conformance 100/100 and grading 48/48, **both mixers**, re-run after every shader-touching commit |
| runtime SPIR-V (Vulkan) | shaderc rebuilds the real 40 KB shader → 100/100 identical; logs confirm it was genuinely in use |
| two-set pipeline layout | validation layers **loaded** (confirmed via `VK_LOADER_DEBUG=layer`, not inferred from silence) and silent |

## What is NOT verified — compiled but never executed

Three items. All build, all look right, none has run:

1. **The fp16 render target on either mixer.** No config selects it. `<render-format>fp16</render-format>`
   exists and is parsed; nothing sets it.
2. **The Vulkan fp16 resolve blit** (`renderpass::commit()`). Barriers written against the
   documented contracts and never exercised. **Enable the Vulkan validation layers the first
   time fp16 is turned on** — a wrong barrier is exactly what they catch and exactly what
   reading will not.
3. **Vulkan OCIO** — the whole of A4e/A4f.

Also unverified: whether `(0,202,255)`-style agreement holds at **16-bit**. The tolerance doc
predicts the worst case stays in the shadows, where the same relative error lands on 256×
more code values.

## Gates — run these, in this order

```bash
cd d:\Github\CasparCG-TestRunner
python cli.py conformance --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py grading     --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py ocio        --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
# then --mixer vulkan for all three. Parity is required, not optional.
python -m pytest tests/test_ocio_reference.py -q      # 14 tests, the oracle + its guards
```

**Redirect to a file and grep. Do not pipe a battery through `tail`** — it races the summary
line, and that produced a number asserted-but-not-observed in this session.

Build via a scratchpad `.bat` with the 14.50 BuildTools pin; `--target casparcg` does **not**
refresh deployed DLLs (see `BUILDING_WORKFLOW.md` #7).

## Traps that cost time here — all recorded in-tree

* **The harness misattributes startup failures.** `server_manager.py` prints a
  `caspar::user_error` from *another worker's* log as "the server's last relevant log line".
  Cost time three times. It is a port/readiness race, not a server fault; re-run, or use
  `--sequential`. **Fixing this is on the list.**
* **`CASPAR_VK_RUNTIME_SHADER=1` is incompatible with parallel battery runs** — 8 workers each
  compiling at startup exceed the harness's fixed readiness timeout. Use `--sequential`.
* **`glslangValidator -S frag` fails on the *Vulkan* shader** (`input_attachment_index`).
  Use `glslc`. `BUILDING_WORKFLOW.md` #6b.
* **OCIO's build is path-length sensitive** — `OCIO_INSTALL_EXT_PACKAGES=ALL` nests deeply and
  blew Windows' 250-char object path from a long directory. `BUILDING_WORKFLOW.md` #6.
  *The first go/no-go probe failed for this reason and would have been reported as "OCIO cannot
  build under the pin" if taken at face value.*
* **Never put backslash content through a Python heredoc.** Four separate corruptions this
  session, including control characters committed into a Markdown file. Use Edit/Write, or
  build strings from `chr(92)` and assert no control characters survive.

## Open decisions

* **Frame-path compile stall.** A variant compiles on first draw: ~1.2 s, one dropped frame,
  logged as a warning. **Pre-warming is the fix, not caching** — compile when the `MIXER OCIO`
  command is accepted (the processor is already built there for validation), which removes the
  stall rather than shortening it. Caching only helps across restarts and adds invalidation to
  get wrong. Measured: `performance` optimisation beats `zero` on *total* time, because the
  optimiser front-loads work the driver would repeat — recorded in `glsl_compiler.cpp`.
* **`exposure` and `gamut_compress` are unavailable on the OCIO path.** Both live in the
  `color_grade` struct inside the block OCIO replaces. Neither belongs to an input transform's
  job, but operators may expect exposure to work. Needs a decision and a doc line.
* **fp16 vs fp32** for the float path — unresolved, and it needs the B2 banding measurement on
  a `bit12` channel, not a calculation. Study §4.3.4 has the numbers: fp16 is ~2× coarser than
  a 12-bit output near white.
* **A5's display transform: channel or consumer?** Decided channel-level for now; a channel
  feeding an LED processor *and* an SDI monitor needs two different views from one composite,
  which argues for consumer-level. Settle before building A5.

## Cleanup

* **`d:\_ocioprobe` — 746 MB, needs a manual `Remove-Item -Recurse -Force`.** The standalone
  OCIO go/no-go probe; redundant now that OCIO builds inside the tree. The sandbox blocked me
  from deleting it.
* `ENABLE_OCIO` defaults **ON**, so a clean build now builds OCIO plus its six bundled
  dependencies. `-DENABLE_OCIO=OFF` skips it.

## Harness repo

`CasparCG-TestRunner`, 2 commits, also unpushed: `7b18ac9` (the OCIO oracle,
`core/ocio_reference.py`) and `70f6c6f` (the `ocio` battery + tolerance doc). Three bugs were
fixed in that battery before it worked, **all in the measurement rather than the server**, and
all three reported total failure while the server was correct. Each is now a regression guard
in `tests/test_ocio_reference.py`, including one proving the gate still fails on a channel swap
by >20× — because with deviations clustered at half an LSB there is no absurdity left to catch
a subtly broken comparison.
