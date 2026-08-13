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
