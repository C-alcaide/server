# CasparVP

Fork of CasparCG Server with virtual-production work: ACES colour management, a full
grading chain, a Vulkan mixer, GPU-direct paths, 360°/curved projection, DMX/Art-Net,
and extra consumers. Branch `CasparVPV`; upstream is `CasparCG/server`.

## Measuring a change — read this before writing a test script

**The measurement harness lives in a separate repo: `d:\Github\CasparCG-TestRunner`.**
It has a `CLAUDE.md` with a full capability index. Check it before writing anything
that starts a server, captures a frame or compares images — most of what you need
already exists there, and one-off scripts have repeatedly reinvented it.

The two that matter most for mixer work:

```bash
cd d:\Github\CasparCG-TestRunner
python cli.py conformance --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
python cli.py grading     --server d:\Github\CasparVP\build\shell\casparcg.exe --mixer ogl
```

Both gate at **1 LSB** — flat colour patches with no producer, no decode and no
resampling in the path, compared against a closed-form model of the shader. Run both
mixers (`--mixer ogl` and `--mixer vulkan`); parity is required, not optional. A
failure in either has exactly one possible cause, which is what makes every softer
threshold downstream defensible.

If the capability you need is missing, add it to the harness (`core/` + a `cli.py`
subcommand + tests) rather than writing a script in the scratchpad.

## Building

Full detail in `BUILDING_WORKFLOW.md`. The parts that are easy to get wrong:

- The toolset must be pinned to **14.50**, sourced from **BuildTools** (Community only
  has 14.51, and plain `vcvars64.bat` picks the wrong one). It is Visual Studio **18**,
  not 2022.
- Every cmake invocation must run inside one shell that started with `vcvars64.bat`.
  A `cmd /c "call vcvars64.bat && cmake ..."` one-liner invoked from the Bash tool
  silently does nothing: banner, exit 0, no compilation.
- What works is a two-line `.bat` in the scratchpad, run via PowerShell:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.50
cmake --build d:\Github\CasparVP\build --target casparcg
```

- Confirm the build happened: `build\shell\casparcg.exe`'s `LastWriteTime` must be
  newer than the newest file under `src\`. A no-op invocation leaves it older, and an
  inherited binary may not match the tree.
- **A header change does not trigger a rebuild, and the timestamp check above will not
  notice.** Ninja's localized `/showIncludes` prefix is stored with a broken encoding, so it
  records no header dependencies at all: only the `.cpp` files you also edited get
  recompiled, and everything else keeps objects built against the old header. Change a class
  layout or a vtable and the result is a binary whose halves disagree — which presented once
  as a reproducible `0xC0000409` abort in the Vulkan mixer that bisected cleanly to an
  innocent commit. **Touch every source before building whenever a header changed**;
  `BUILDING_WORKFLOW.md` has the one-liner and the full account.
- **And touching every source is not enough if the header is in a precompiled header.**
  The `.pch` files carry the same untracked dependency, so every translation unit
  recompiles and every one of them reads the *old* declaration. It shows up as a compile
  error that contradicts the source on screen — an `override` that "did not override" a
  base method identical to it. Delete `build/**/cmake_pch.*.pch` too;
  `BUILDING_WORKFLOW.md` has the sweep.
- A wrong `vcvars` path fails as `C1083: cannot open include file 'cstdint'` on every
  translation unit. Missing *standard library* headers means the environment was never
  initialised — don't go looking at the includes.

## Shaders

`src/accelerator/ogl/image/shader.frag` is embedded via `bin2c` into a generated
header; `src/accelerator/vulkan/image/fragment_shader.frag` is compiled to SPIR-V by
`glslc`. Both are build-time — but **a shader edit is a header change**, and so it hits the
missing-header-dependency trap above: ninja regenerates the header and does *not* recompile
the `.cpp` that embeds it, so the binary keeps the old shader while any `.cpp` you edited in
the same commit is up to date. The result is a binary whose halves disagree, and the
`casparcg.exe` timestamp check does not notice because the exe *did* relink.

Measured 2026-08-13: `ogl_image_fragment.h` regenerated at 13:47:10, `casparcg.exe` last
linked 13:45:20 — an exe two minutes *older* than its own shader. It presented as a kernel
change that had provably taken effect (a trace confirmed the uniform was set) driving a
shader that had not. **Touch the embedding sources before building whenever a `.frag`
changed:**

```
src/accelerator/ogl/image/image_shader.cpp
src/accelerator/vulkan/image/image_kernel.cpp
src/accelerator/vulkan/util/device.cpp
src/accelerator/vulkan/util/pipeline.cpp
```

Then confirm `build/shell/casparcg.exe` is newer than
`build/accelerator/ogl_image_fragment.h` and `vk_image_fragment.h`, not just newer than
`src/`.

**GLSL errors also do not surface at C++ compile time**. Syntax-check before building:

```
& "C:\VulkanSDK\1.4.341.1\Bin\glslangValidator.exe" -S frag <path to .frag>
```

### A new `image_transform` field is dead until you add it to the allowlist

`apply_transform_colour_values` (`src/accelerator/{ogl,vulkan}/util/transforms.cpp`) composes
a layer's `image_transform` onto the accumulated one, **field by hand-written field**. A
field that is not named there is silently dropped: the AMCP command sets it on the stage,
the query reads it back correctly, and the kernel sees the default on every frame.

It presents as a command that returns `202` and changes nothing — the same symptom as a
shader that never runs the code, and distinguishable from it only by a trace. Measured
2026-08-13: `MIXER EXPOSURE` accepted, uniform plumbed, shader correct, and
`[EXPTRACE] user=1` on every draw because the value never survived composition.

**Both mixers keep their own copy of this list.** Add a field to one and not the other and
the backends diverge in a way no single-backend test can see.

Choose the composition deliberately, because the function is doing real work rather than
copying: `opacity`, `brightness` and `exposure` **multiply**; `levels` take a min/max;
`chroma` takes a max; the flags OR or XOR; `layer_depth` adds. For a gain, the product is
the answer.

**Fields still absent from the list, checked 2026-08-13**: only the geometry ones (`anchor`,
`fill_translation`, `fill_scale`, `angle`, `geometry_override`), which the geometry half of
`combine_transform` handles and which are therefore fine.

`blend_mask` was the other one, and it was the same defect: `MIXER PROJECTION_BLEND_MASK`
returned 202, the query read the mask back at its right dimensions, and all four patches
rendered **byte-identical to no mask at all**, on both backends. Fixed by naming it here
(innermost wins, like the LUT — two masks cannot be composed without resampling one onto the
other's raster).

**And fixing it exposed a second defect underneath, which is the part worth remembering.**
Once the mask reached the shader the OpenGL picture came back multiplied by `(0.4, 0.6, 0.8)`
where `(0.8, 0.6, 0.4)` was asked for: `col.rgb *= texture(...).rgb` on a shader that carries
BGR. That code had never executed, so it had never been wrong before. A dead code path is
not a correct one — when you make one reachable, measure it with **asymmetric** values,
because a neutral mask is invariant under the exchange and would have passed.

### The YCbCr decode counted in 8-bit codes — fixed, and worth knowing how it hid

`ycbcra_to_rgba` did `vec3(Y,Cb,Cr) * 255 - vec3(16,128,128)`. Exact for an 8-bit texture,
where the sample IS `code/255`. For 10-bit video in a 16-bit texture the neutral chroma
normalises to `32768/65535`, so `* 255 - 128` leaves **-0.4981** and can never be zero: every
high-bit-depth YCbCr source arrived with a constant chroma offset, a green cast on greys.

The scale now comes from the texture depth — `65535/256` for 16-bit, which lands legal black
on 16, neutral chroma on 128 and legal white on 235; `255.0f` for 8-bit, the same literal as
before, so that path is bit-identical.

**What it cost to find, and why:**

* `conformance` and `grading` drive a **colour producer**, which is BGRA. The 1 LSB batteries
  never call the YCbCr decode at all — they *cannot* see a decode bug.
* The picture-based batteries do call it, but gate on PSNR over a decoded frame, where 1.5
  LSB of chroma bias vanishes.
* Both mixers were wrong identically, so every parity check passed.

It took a **flat decoded** fixture — neutral by construction, `u = v = 512`, so "neutral in,
neutral out" needs no model at all — and the harness had recorded such a fixture as
impossible. When a whole class of bug is invisible to every battery, the gap is usually in
what the fixtures can *be*, not in the gates.

Measured, both mixers: channel spread on a neutral source 1.48-2.95 LSB before, **0.00**
after; the auto path against the closed-form model, over 45 unclipped samples, mean 1.58 /
max 5.02 LSB before, **mean 0.009 / max 0.06** after.

### The channel-order trap

The OpenGL mixer carries the pixel through the whole grading chain in the mixer's
native **BGR** order, so `col.r` holds blue. The Vulkan mixer grades in **RGB**. The
OpenGL kernel compensates by reversing every per-channel uniform on upload; Vulkan
uploads straight through.

Consequences, all of which have bitten:

- Anything reading a specific channel needs `.bgr` on the OpenGL side — `rgb2hsv`,
  Rec.709 luma dot products, LUT lookups, per-channel level indices.
- **But a function's channel order is a property of its call site.**
  `ChromaOnCustomColor` is called as `ChromaOnCustomColor(c.bgra).bgra`, so it receives
  RGB and must *not* swizzle. Adding one there produced a double exchange that mirrored
  the hue wheel and made `MIXER CHROMA BLUE` key red. Check the caller.
- Greys are invariant under a red/blue exchange, so a grey ramp passes every one of
  these defects. Colour tests must use asymmetric per-channel values.

## Docs

`docs/` is extensive and kept current — `COLOR_GRADING.md`, `VIRTUAL_PRODUCTION_FEATURES.md`,
`HDR_GUIDE.md`, `VULKAN_MIXER_IMPLEMENTATION.md`, `GPU_INTEROP_ARCHITECTURE.md` and
others. Treat them as claims to verify against the source, not as ground truth: where
a doc and the shader disagree, the shader wins and the disagreement is a finding.

`CHANGELOG.md` leads with behaviour changes. Anything that alters rendered output for
an existing config belongs there with the measurement that established it.

### Update the doc in the same commit as the change

Not eventually. A doc that lags becomes a **claim that outlives the code**, and this tree
generates those faster than anyone re-reads them. Four were corrected in one sitting on
2026-08-14: a harness battery recorded as "still owed" that had existed for two days, a
render-fingerprint field described as "needed" that both mixers already had, an entry under
*What is NOT verified* that the same file's own header contradicted, and an open question
answered a session earlier. None was wrong when written.

So "which doc" is not left to judgment:

| what changed | where it goes |
| :--- | :--- |
| rendered output for an existing config | `CHANGELOG.md`, with the measurement |
| a new AMCP command or argument | the feature doc that owns it, and `docs/OPERATIONS_GUIDE.md` |
| a new `<configuration>` element | the feature doc that owns it |
| a new battery, or a new number for an existing claim | the doc carrying that claim, **and** the harness `CLAUDE.md` command table |
| a trap that cost more than an hour | this file |

**State what the measurement does not cover, in the same paragraph as the number.** A figure
without its limits reads as a stronger claim than it is. `LED_CALIBRATION.md` says 32/32 at
1 LSB and then says, immediately, that only the IMAGE consumer was captured and only one
layer was composited — so a reader takes it as "the LUT is applied correctly" rather than
"`CALIBRATION` works".

**Re-read a doc immediately before editing it.** Several sessions run against this tree at
once and docs are the likeliest file for two of them to collide in — editing from a stale
read put duplicate rows in the harness command table on 2026-08-16. If the file already
carries another session's uncommitted work, fix your part, leave it unstaged, and say so:
staging it sweeps their changes into your commit, which has happened twice.
