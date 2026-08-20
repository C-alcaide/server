# CasparVP Build Workflow

Quick reference for AI agents and developers building CasparVP on this machine.

---

## Environment facts

| Item | Value |
|------|-------|
| Visual Studio | 2026 Community v18 |
| vcvars64.bat | `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat` |
| CMake | `C:\Program Files\CMake\bin\cmake.exe` (also on PATH after vcvars) |
| Build generator | **Ninja** (configured in CMakeCache) |
| Ninja | `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe` |
| Source root | `d:\Github\CasparVP\src` |
| Build dir | `d:\Github\CasparVP\build` |
| Main executable | `d:\Github\CasparVP\build\shell\casparcg.exe` |
| CUDA toolkit | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |

---

## THE ONLY WORKING BUILD METHOD

**Every cmake invocation must run inside a single `cmd.exe` session that starts with `vcvars64.bat`.**

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.50
cmake --build d:\Github\CasparVP\build --target <targets>
```

Use **BuildTools**, and **always** pass `-vcvars_ver=14.50`. Both halves matter —
see below.

> ### ⚠ Pin the toolset to 14.50, and source it from BuildTools
>
> **The whole tree must build on MSVC 14.50.** `nvcc` 12.9 cannot use 14.51:
> `cudafe++` dies with an access violation while parsing its STL, so CUDA
> constrains the entire project. `CMakeCache.txt` therefore pins `cl.exe`
> **14.50.35717** — and specifically the **BuildTools** copy.
>
> A VS update added toolset **14.51.36231**, and plain `vcvars64.bat` selects the
> newest one, pairing the 14.50 compiler with the 14.51 STL.
>
> **14.50 exists only in BuildTools on this machine.** The Community install has
> 14.51 alone, so `Community\...\vcvars64.bat -vcvars_ver=14.50` fails outright
> with `Toolset directory for version '14.50' was not found`. Worse is the
> version that *doesn't* fail: sourcing Community's vcvars **without** the pin
> succeeds and builds, because `CMakeCache.txt` names the BuildTools 14.50
> `cl.exe` by absolute path — so you get the 14.50 compiler with 14.51 headers on
> `INCLUDE` and no warning about it. Whether it compiles then depends on which
> translation units you happen to touch. (This is exactly how it went once: an
> edit confined to `ffmpeg_consumer.cpp` built and ran fine in that mismatched
> environment, and only a clean rebuild in the correct one proved the binary.)
>
> If in doubt which environment produced a binary, rebuild with
> `--clean-first` in the pinned one. 383 targets, and it is the only way to be
> certain.
>
> | Symptom | Cause |
> |---|---|
> | `include\variant(292): error C2988` / `C3855` in `accelerator/vulkan/util/device.cpp` | 14.50 `cl.exe` parsing the 14.51 STL — you forgot `-vcvars_ver=14.50`, or sourced Community's vcvars |
> | `[ERROR:vcvars.bat] Toolset directory for version '14.50' was not found.` | you used the **Community** vcvars. 14.50 is only in BuildTools |
> | `C1083: cannot open include file 'atlbase.h'` in `decklink`/`flash`, or `LNK1104: atls.lib` | the **ATL component for 14.50 is not installed**. Install it via the VS installer (Individual components → "MSVC v14.50 … ATL"). `decklink` and `flash` need ATL. |
> | `casparcg.exe` links but dies at load: `0xC0000005`, faulting module `unknown`, offset 0, no log output at all | 14.51 ATL headers compiled against the 14.50 STL, and 14.51's `atls.lib` linked in |
>
> **Do not** work around a missing 14.50 ATL by adding 14.51's `atlmfc` to
> `INCLUDE`/`LIB`. It links successfully and then crashes before `main`, which is
> far harder to diagnose than the missing-header error it replaces. Install the
> matching ATL component instead.
>
> Note: `run_build.py`, referenced below, is **not present** in the tree. Use the
> `cmd`/`.bat` route.

> ### ⚠ Precompiled headers do not track the headers they include
>
> Each target's PCH wraps its `StdAfx.h`, which pulls in `core/frame/*.h`,
> `common/*.h` and more — but ninja does **not** treat those as inputs to the PCH.
> Editing one of them leaves every module's PCH stale, and because the PCH is
> authoritative (`/Yu` plus include guards baked in), the edit is silently
> invisible. The usual symptom is nonsense like "no operator found" for an
> operator you just added, or worse, no symptom at all.
>
> After editing any header that appears in a `StdAfx.h`, force the PCHs to
> rebuild. Deleting the `.pch` alone is not enough — ninja only tracks the `.obj`:
>
> ```powershell
> Get-ChildItem d:\Github\CasparVP\build -Recurse -Include cmake_pch.cxx.pch,cmake_pch.cxx.obj | Remove-Item -Force
> ```

### Why this is the only correct method

`vcvars64.bat` is a thin stub that calls `vcvarsall.bat` through nested `call` chains. The INCLUDE, LIB, and LIBPATH environment variables are only fully set for the `cmd.exe` session that executed the bat file. Any approach that captures env vars in a subprocess and re-injects them into a *new* process (Python `os.environ`, PowerShell `$env:`, etc.) will silently miss variables set by nested `call` chains — resulting in `C1083: Cannot open include file: 'vector'` style fatal errors.

**Anti-patterns (DO NOT USE):**
```python
# BROKEN — env capture via subprocess misses nested call chain vars
result = subprocess.run('cmd /c vcvars64.bat && set', capture_output=True)
env = {k: v for line in result.stdout for k, v in [line.split('=', 1)]}
subprocess.run(['cmake', ...], env=env)   # ← fails with "Cannot open include file"
```

```powershell
# BROKEN — vcvars64 only affects the cmd.exe session, not the PowerShell session
& 'vcvars64.bat'
cmake --build ...   # ← cl.exe cannot find standard headers
```

---

## Build targets

| Target | What it builds |
|--------|---------------|
| `ffmpeg` | FFmpeg producer + consumer + av_util |
| `decklink` | DeckLink producer + consumer |
| `core` | Core mixer, frame pipeline |
| `casparcg` | Full server executable (links all modules) |
| `casparcg_copy_dependencies` | Copies DLLs next to the exe — **not implied by `casparcg`**, see pitfall #7 |
| `opencolorio` | OpenColorIO + its bundled deps, as an ExternalProject (slow, first build only) |

---

## Standard incremental builds (Ninja, fast)

### Using run_build.py (recommended for AI agents)

```powershell
# Default: rebuild ffmpeg + decklink modules
python d:\Github\CasparVP\run_build.py

# Specify targets explicitly
python d:\Github\CasparVP\run_build.py ffmpeg decklink
python d:\Github\CasparVP\run_build.py casparcg
```

`run_build.py` wraps the `cmd /c vcvars64 && cmake` pattern correctly and saves output to `build_out.txt`.

### Direct PowerShell one-liner

```powershell
cmd /c """C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build d:\Github\CasparVP\build --target ffmpeg decklink"
```

### Using build_hdr.bat (module-only)

```powershell
d:\Github\CasparVP\build_hdr.bat
```

Builds `ffmpeg` + `decklink` targets via the .bat route.

### Using build_now.bat (full casparcg exe)

```powershell
d:\Github\CasparVP\build_now.bat
```

Builds the `casparcg` target (links everything).

---

## CMake configure (first time / after CMakeLists.txt changes)

Run `build_ninja.bat` which calls cmake configure + build:

```powershell
d:\Github\CasparVP\build_ninja.bat
```

This re-runs cmake configure with Ninja, CUDA architectures, and the correct CUDA host compiler, then builds `casparcg` + dependencies.

---

## Verifying a successful build

Always check the **timestamp** of the output file:

```powershell
# Module libraries
Get-Item "d:\Github\CasparVP\build\modules\ffmpeg\ffmpeg.lib",
         "d:\Github\CasparVP\build\modules\decklink\decklink.lib" |
    Select-Object Name, LastWriteTime

# Full executable
Get-Item "d:\Github\CasparVP\build\shell\casparcg.exe" |
    Select-Object Name, LastWriteTime
```

A build was successful if:
- Exit code is `0`
- The `.lib` / `.exe` timestamp matches the time the build ran
- No `error C` or `error LNK` lines appear in the output

**None of that detects the failure below.** All three were true of a binary whose object
files disagreed with each other about a class layout.

---

## ⚠ Header changes do NOT trigger a rebuild in this tree

**Editing a `.h` and rebuilding produces a binary in which only the `.cpp` files you also
edited saw the change.** Every other translation unit keeps an object file compiled against
the old header. If the header changed a class layout, a vtable, or an enum, those objects
are now lying to each other and the program does undefined things at runtime.

### Why

Ninja learns header dependencies from MSVC's `/showIncludes` output by matching a prefix
string, which CMake probes at configure time and writes to
`build/CMakeFiles/rules.ninja`. On this machine `cl.exe` is **Spanish-localized**, and the
probe was stored with a broken encoding:

```
msvc_deps_prefix = Nota: inclusi<?> del archivo:      <-- the ó is mojibake
```

`cl.exe` emits `Nota: inclusión del archivo:` correctly at build time, the comparison never
matches, and ninja records **zero** header dependencies for every object in the tree.
`VSLANG=1033` does not fix it — only the Spanish resource pack is installed, so `cl` stays
localized regardless.

### What it cost

An afternoon, and it very nearly landed as a fabricated bug report. After a change that
added one virtual to `frame_context` and one member to `layer_info` (both in
`vulkan/util/renderpass.h`), the server began aborting with `0xC0000409` on the first
composited frame under the Vulkan validation layers. It reproduced perfectly, bisected
cleanly to the commit, and reproduced with **no OCIO in the path at all** — every sign of a
real defect in the mixer. It was a stale object file calling through a shifted vtable slot.
A full rebuild made it vanish and it has never returned.

The tell, in hindsight: **zero validation messages before the abort.** The layers were not
reporting a violation, so nothing was wrong with the Vulkan usage; something was jumping to
the wrong address.

### What to do

**Whenever you change a header, touch every source before building:**

```powershell
Get-ChildItem d:\Github\CasparVP\src -Recurse -Include *.cpp,*.h -File |
    ForEach-Object { $_.LastWriteTime = Get-Date }
```

Then build as usual. It is a full recompile of the project's own translation units (~286
targets, several minutes); it does **not** rebuild the external projects, so it is far
cheaper than a `--clean-first`.

If you only edited `.cpp` files, an ordinary incremental build is correct and fast.

### Touching every source is NOT enough when the header is in a precompiled header

Each target compiles through `/Yu…cmake_pch.hxx` with a prebuilt `cmake_pch.cxx.pch`, and
the PCH's own dependency on the headers it aggregates is not tracked either. So a change to
a header that the PCH pulls in — `core/mixer/image/image_mixer.h` is one — is invisible even
after the touch-everything sweep: every translation unit recompiles, and every one of them
reads the *old* declaration out of the stale `.pch`.

It presents as a compile error that contradicts the source in front of you. Measured
2026-08-12, after adding an 8th parameter to `set_target_color`:

```
image_mixer.h(97): error C3668: 'set_target_color': el método con el especificador
                   de invalidación 'override' no invalidó ningún método de clase base
```

— with the base and the override on screen, byte-for-byte identical in their parameter
lists. The tell is in the `/showIncludes` trail: the base class's header does **not** appear
in it, because it arrived through `/FIcmake_pch.hxx` instead.

**So when a header changes, delete the precompiled headers as well as touching the sources:**

```powershell
Get-ChildItem d:\Github\CasparVP\build -Recurse `
    -Include cmake_pch.cxx.pch,cmake_pch.c.pch,cmake_pch.cxx.obj,cmake_pch.c.obj -File |
    ForEach-Object { Remove-Item $_.FullName -Force }
```

Ninja regenerates them on the next build. Cheap — one PCH per target — and it is the
difference between a build that fails confusingly and one that succeeds while linking
halves that disagree, which is the far worse outcome documented above.

### A shader edit is a header change

`.frag` files are not compiled directly into anything. `src/accelerator/ogl/image/shader.frag`
goes through `bin2c` into `build/accelerator/ogl_image_fragment.h`, and
`src/accelerator/vulkan/image/fragment_shader.frag` through `glslc` into
`vk_image_fragment.h` / `vk_image_fragment_src.h`. Ninja *does* regenerate those headers —
that rule's dependency is declared — and then stops, because the `.cpp` files that `#include`
them have no recorded header dependency, for the same `/showIncludes` reason as above.

So the shader on disk, the generated header, and the binary can all disagree, and **the
`casparcg.exe` timestamp check will not catch it**: the exe relinks (some other object
changed), so it is newer than `src/`, while the object holding the shader is stale.

Measured 2026-08-13, moving one call in `shader.frag`:

```
build/accelerator/ogl_image_fragment.h   13:47:10   <- regenerated, contains the edit
build/shell/casparcg.exe                 13:45:20   <- older than its own shader
```

It presented as a change that half-worked: a trace proved the C++ kernel was setting
`gamut_compress_enable` true, and the shader ignored it, because the binary still held the
version where the call sat inside a block that path does not enter. Two measurement runs
were spent on the wrong binary.

**When a `.frag` changes, touch the four sources that embed the generated headers:**

```powershell
@("src\accelerator\ogl\image\image_shader.cpp",
  "src\accelerator\vulkan\image\image_kernel.cpp",
  "src\accelerator\vulkan\util\device.cpp",
  "src\accelerator\vulkan\util\pipeline.cpp") |
    ForEach-Object { (Get-Item "d:\Github\CasparVP\$_").LastWriteTime = Get-Date }
```

`grep -rl "ogl_image_fragment.h\|vk_image_fragment.h\|vk_image_fragment_src.h" src/` is what
produced that list; re-derive it rather than trusting it if the accelerator layout moves.
Then check the exe against the *headers*, not against `src/`:

```powershell
(Get-Item build\shell\casparcg.exe).LastWriteTime
(Get-Item build\accelerator\ogl_image_fragment.h).LastWriteTime
(Get-Item build\accelerator\vk_image_fragment.h).LastWriteTime
```

### Fixing it properly

Correct the prefix in `build/CMakeFiles/rules.ninja` to the exact bytes `cl.exe` emits, or
re-run the CMake configure in a console whose codepage matches what CMake reads. Either way
the fix is undone by the next configure until the root cause is addressed upstream, so the
touch-everything rule stays the safe default. Installing the English MSVC language pack
would make the prefix pure ASCII and end the problem outright.

---

## Forcing a recompile of specific files

Ninja only rebuilds changed files. To force-recompile a specific `.cpp` without touching it logically, `touch` it:

```powershell
(Get-Item "d:\Github\CasparVP\src\modules\ffmpeg\util\av_util.cpp").LastWriteTime = Get-Date
```

Or use Python:
```python
import pathlib
pathlib.Path(r'd:\Github\CasparVP\src\modules\ffmpeg\util\av_util.cpp').touch()
```

---

## CUDA module (cuda_prores)

The CUDA module is **not** included in the `ffmpeg` or `decklink` targets. It has its own CMakeLists.txt. To build it:

```powershell
cmd /c """C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build d:\Github\CasparVP\build --target cuda_prores"
```

CUDA requires PATH to include the CUDA bin directory before vcvars (already handled in `build_ninja.bat`). The cuda_prores `.cu` files are the authoritative sources — the matching `.cpp` copies exist for IntelliSense only and are not compiled.

---

## Pitfalls & Past Build Errors

Whenever a new build error is encountered and fixed, it is documented here so it is not repeated.

---

### #1 — `C1083: Cannot open include file: 'vector'` (and other STL/Win32 headers)

**Symptom:** Every TU fails with fatal `C1083: Cannot open include file` for basic headers (`vector`, `memory`, `cstddef`, `Windows.h`).

**Root cause:** `vcvars64.bat` is a thin stub (`@call vcvarsall.bat x64`). INCLUDE, LIB, and LIBPATH are set through nested `call` chains inside vcvarsall. When a build script runs `vcvars64.bat && set` in a subprocess and tries to re-inject the result into a new process, the nested-call-chain variables are not visible in the `set` output — so the injected environment is incomplete.

**Fix:** Run `vcvars64.bat` and `cmake` in the **same** `cmd.exe` invocation:
```
cmd /c ""<vcvars64.bat>"" && cmake --build ...
```
See the "Only Working Build Method" section above.

**Broken patterns to avoid:**
```python
# Subprocess env-capture — DO NOT USE
result = subprocess.run('cmd /c vcvars64.bat && set', capture_output=True)
env = parse(result.stdout)
subprocess.run(['cmake', ...], env=env)  # misses nested-call vars
```

---

### #2 — `pixel_format` cases inside a `color_space` switch (wrong enum type errors)

**File:** `src/modules/ffmpeg/util/av_util.cpp` — `make_av_video_frame()`

**Symptom:** 14 compile errors of the form:
```
error: this constant expression has type "caspar::core::pixel_format"
       instead of the required "caspar::core::color_space" type
```
All `pixel_format::bgr`, `pixel_format::rgba`, etc. case labels were flagged.

**Root cause:** When the `switch (pix_desc.color_space)` block was added to `make_av_video_frame()`, the `default:` case was left incomplete — it was missing `color_trc` assignment, `break;`, the closing `}` of the switch, and the closing `}` of the surrounding `if` block. As a result the compiler parsed the subsequent `switch (format)` cases as continuations of the `color_space` switch, producing enum-type mismatches for every `pixel_format` label.

**Fix:** Complete the `default:` case and close all open blocks before the `switch (format)`:
```cpp
        default: // bt709
            av_frame->color_primaries = AVCOL_PRI_BT709;
            av_frame->color_trc       = AVCOL_TRC_BT709;  // ← was missing
            break;                                          // ← was missing
    }                                                       // ← close switch
}                                                           // ← close if

switch (format) {   // ← was missing the switch statement
    case core::pixel_format::bgr:
    ...
```

**Prevention:** When adding a new `switch` block inside an existing function that already has a `switch`, always verify brace balance before committing. The IntelliSense "wrong enum type" error is a reliable signal that a `switch` block was left open.

---

### #3 — `C1083: Cannot open include file: 'memory'` when running `python run_build.py`

**Symptom:** Multiple TUs fail with `C1083: Cannot open include file` for basic STL headers (`<memory>`, `<chrono>`, `<exception>`, etc.) even though the toolchain exists at the configured path.

**Root cause:** `run_build.py` was using `cmd /c "vcvars64.bat" && cmake` as the `full_cmd` string, then launching it with `shell=True`. `shell=True` on Windows prepends `cmd.exe /c`, so the actual execution becomes:
```
cmd.exe /c  cmd /c "vcvars64.bat"  &&  cmake
```
The outer `cmd.exe /c` sees the inner `cmd /c "vcvars64.bat"` as a **child process**. vcvars64 sets INCLUDE/LIB inside that child, then the child exits — the vars are lost. The outer cmd then runs `cmake` without any INCLUDE set.

**Fix:** Use `call` instead of `cmd /c` so vcvars runs as a **subroutine** in the same cmd.exe session that `shell=True` spawned:
```python
full_cmd = f'call "{VCVARS}" && {cmake_cmd}'
# (shell=True already wraps this in cmd.exe /c)
```

**Also fixed:** `build_now.bat` had the wrong VS path (`\2026\` instead of `\18\`), causing it to fail silently. Updated to `\18\`.

---

### #4 — `STATUS_ACCESS_VIOLATION` (0xC0000005) when switching CUDA producers on the same layer

**Symptom:** CasparCG crashes with a fatal unhandled exception (code `3221225477` = `0xC0000005`) inside the NVIDIA GL driver (`nvoglv64.dll`) a few milliseconds after `LOAD OK` when a `LOAD 1-N CUDA_PRORES` (or `CUDA_NOTCHLC`) command is sent to a layer that already has the other producer playing:

```
[fatal] UNHANDLED EXCEPTION:
[fatal] Address:00007FFA3D5287B7
[fatal] Code:3221225477
```

**Root cause:** The stage executor runs asynchronously — it queues the producer swap and returns LOAD OK immediately. At that point:

- The **new** producer's `read_thread_` is **already running** (started at the end of its constructor, before LOAD OK was sent).
- The **old** producer's `read_thread_` is still alive and will call `cudaGraphicsUnregisterResource` during its exit cleanup a few ms later.

These two threads race on the NVIDIA interop driver:

```
new read_thread_: wglMakeCurrent + cudaGraphicsGLRegisterImage (setup)
old read_thread_: cudaGraphicsUnregisterResource + wglDeleteContext (teardown)
```

`cudaGraphicsGLRegisterImage` and `cudaGraphicsUnregisterResource` are **not thread-safe** when called concurrently from different threads, even for distinct GL textures. The driver dereferences a freed VA and crashes.

**Fix:** Added `src/modules/cuda_gl_interop_lock.h` — a single process-wide `std::mutex` that both `prores_producer.cpp` and `notchlc_producer.cpp` acquire around the `cudaGraphicsGLRegisterImage` burst at `read_loop()` startup and the `cudaGraphicsUnregisterResource` burst at `read_loop()` cleanup. The mutex is **not** held during normal frame decode (map/unmap/submit), so there is zero steady-state performance impact.

**Rule for future CUDA-GL producers:** Any new producer that calls `cudaGraphicsGLRegisterImage` or `cudaGraphicsUnregisterResource` must hold `caspar::cuda_gl_interop_mutex()` during those calls.

---

### #5 — Adding AMCP `ADD` support to an XML-only consumer

**Context:** ArtNet and sACN consumers only registered via `register_preconfigured_consumer_factory` (XML in caspar.config), not `register_consumer_factory` (AMCP ADD).

**Pattern:** Add a `create_consumer(params, ...)` function that:
1. Checks `params[0]` matches the keyword (return `frame_consumer::empty()` if not).
2. Parses keyword-value pairs with `get_param<T>(L"KEY", params, default)` and `contains_param(L"FLAG", params)` from `<common/param.h>`.
3. For structured sub-objects (fixtures), scan params for a sentinel keyword (e.g. `L"FIXTURE"`) then consume N following positional tokens via `boost::lexical_cast`.
4. Register BOTH factories in `init()`:
   ```cpp
   dependencies.consumer_registry->register_consumer_factory(L"ArtNet Consumer", create_consumer);
   dependencies.consumer_registry->register_preconfigured_consumer_factory(L"artnet", create_preconfigured_consumer);
   ```
5. In the header, add the new declaration alongside `create_preconfigured_consumer` and add `#include <core/fwd.h>` for `video_format_repository` and `video_channel`.

**AMCP syntax for ArtNet (example):**
```
ADD 1 ARTNET UNIVERSE 0 HOST 127.0.0.1 PORT 6454 REFRESH-RATE 10 FIXTURE RGB 1 3 3 0.0 0.0 1920.0 100.0 0.0
```

**AMCP syntax for sACN (example):**
```
ADD 1 SACN UNIVERSE 1 HOST 127.0.0.1 PORT 5568 PRIORITY 100 MULTICAST-TTL 10 REFRESH-RATE 10 FIXTURE RGBW 1 6 4 0.0 0.0 1920.0 200.0 0.0
```

**The nine FIXTURE arguments, in order** (`get_fixtures_params`, artnet_consumer.cpp):

```
FIXTURE <type> <start_addr> <count> <channels> <x> <y> <width> <height> <rotation>
```

Two things the examples above hide, both of which will silently produce wrong DMX
rather than an error:

- **`count` comes before `channels`.** The ArtNet example uses `3 3` for both, so it
  reads the same either way; the sACN one uses `6 4`, meaning 6 fixtures of 4
  channels. Getting these backwards builds a different number of fixtures at
  different addresses and still succeeds.
- **`start_addr` is 1-based on the wire and stored 0-based.** The consumer does
  `f.startAddress = startAddress - 1`, so `FIXTURE RGB 1 ...` writes DMX slots
  0,1,2 of the packet payload. A receiver indexing from the 1-based address reads
  one channel late.

Ten RGB fixtures starting at DMX address 1, across a 500×100 box at (960,540):

```
ADD 1 ARTNET UNIVERSE 0 HOST 127.0.0.1 PORT 6454 REFRESH-RATE 10 FIXTURE RGB 1 10 3 960.0 540.0 500.0 100.0 0.0
```

**Consumer indices (REMOVE command):** ArtNet = 1337, sACN = 1338 (hard-coded in `index()` overrides).

---

### #6 — A third-party sub-build fails and the error names nothing recognisable

`ENABLE_OCIO` (on by default) builds OpenColorIO as an ExternalProject with
`OCIO_INSTALL_EXT_PACKAGES=ALL`, which has OCIO fetch and build its own dependencies —
Imath, yaml-cpp, pystring, minizip-ng, expat, zlib — nesting each one under:

```
<build>/opencolorio-prefix/src/opencolorio-build/ext/build/<dep>/src/<dep>_install-build/...
```

That is deep before any of CMake's own scratch directories are appended. **Windows caps an
object file's full path at 250 characters** and CMake says so, then keeps going and fails
later somewhere unrelated-looking:

```
.../ext/build/libexpat/src/expat_install-build/CMakeFiles/CMakeScratch/TryCompile-jovtov/...
  has 237 characters.  The maximum full path to an object file is 250
  characters (see CMAKE_OBJECT_PATH_MAX).
...
ninja: build stopped: subcommand failed.
```

The failure mentions expat, not OCIO, and mentions a path without saying the path is the
problem. Measured: from `d:\Github\CasparVP\build` the deepest generated path lands near
**204** characters — it fits, with roughly 46 to spare. From a directory whose name carries
a long prefix (a temp dir with a UUID in it, say) it does not.

**So: keep the build directory shallow.** If OCIO's sub-build breaks after you move or
rename the build tree, suspect this before suspecting the toolset. Building with
`-DENABLE_OCIO=OFF` skips it entirely and is a fast way to confirm the diagnosis.

### #6b — `glslangValidator -S frag` rejects the *Vulkan* shader

The syntax check quoted in `CLAUDE.md` works for `ogl/image/shader.frag` and fails on
`vulkan/image/fragment_shader.frag`:

```
ERROR: 0:14: 'input_attachment_index' : only allowed when using GLSL for Vulkan
```

That is the checker's default target, not a defect in the shader. Either pass a Vulkan
target, or use the compiler the build actually uses:

```powershell
& "C:\VulkanSDK\1.4.341.1\Bin\glslc.exe" src\accelerator\vulkan\image\fragment_shader.frag -o out.spv
```

Worth knowing because following the documented command after editing that shader produces
two errors that look exactly like your edit broke something.

### #7 — Deployed DLLs are stale after `--target casparcg`

`--target casparcg` links the executable. It does **not** copy runtime DLLs: that is a
`POST_BUILD` step on the separate `casparcg_copy_dependencies` target. After adding or
updating a dependency (a new `casparcg_add_runtime_dependency`, a rebuilt
`OpenColorIO_2_5.dll`), the exe will link fine and then fail at load or run against the
previous DLL. Build `casparcg_copy_dependencies`, or the default target, to refresh them.

**And the copy step fails PARTWAY, quietly, leaving a mixed set.** Each dependency is its
own `add_custom_command` of the form `echo && copy && copy`, run in the order they were
registered — so one locked file aborts that command and everything registered after it,
while everything before it is already updated. Nobody building `--target casparcg` ever sees
the error, because that target does not run this step at all.

Measured 2026-08-21. `Bootstrap_Windows.cmake` registers FFmpeg as seven DLLs and then, last,
`ffmpeg.exe` and `ffprobe.exe`. In `build\shell`:

```
avcodec-62.dll   19/08/2026 21:59:40   <- all seven 8.1.2 DLLs, copied
...
swscale-9.dll    19/08/2026 21:59:40
ffmpeg.exe       16/08/2026 16:21:22   <- FFmpeg 7.0.2, the two LAST entries, not copied
ffprobe.exe      16/08/2026 16:21:22
```

That is not cosmetic: **`scanner.exe` shells out to the `ffprobe` beside it**, so the media
scanner was probing files with FFmpeg 7.0.2 while the server decoded them with 8.1.2. A
format 8.x reads and 7.x cannot probe shows up as a file missing its duration in the media
list while `PLAY` on the same file works — which reads as a scanner bug and is not one. It
also cost real time in a Vulkan-decode investigation, where a probe run with
`build\shell\ffmpeg.exe` could not possibly have used an FFmpeg 8 decoder.

**The copy step also never DELETES anything**, and FFmpeg 8 bumped every soname
(avcodec 61→62, avutil 59→60, avfilter 10→11, avformat 61→62, swresample 5→6, swscale 8→9,
libpostproc removed). So the entire 7.x set stayed on disk beside the new one. A running
server loads only the 8.x set — verified by enumerating its loaded modules — so they are
inert for the server, but not for everything: Windows resolves a DLL from the **application
directory first**, so a GStreamer `gstlibav.dll` loaded in-process would bind to the stale
`avcodec-61.dll` (61.3.100) instead of GStreamer's own 61.19.100. They are quarantined in
`build\shell\_stale_ffmpeg7\` with a README, and are safe to delete.

**So after any dependency-version change, check the tools and not just the DLLs:**

```powershell
cmake --build d:\Github\CasparVP\build --target casparcg_copy_dependencies
d:\Github\CasparVP\build\shell\ffmpeg.exe -version   # must match the pin in Bootstrap_Windows.cmake
Get-ChildItem d:\Github\CasparVP\build\shell\av*.dll, d:\Github\CasparVP\build\shell\ffmpeg.exe |
  Select-Object Name, LastWriteTime    # one timestamp cluster, not two
```

The exe-newer-than-`src\` check below cannot see any of this: `casparcg.exe` is freshly
linked and every DLL beside it may be from a previous pin.

---

The scripts are tracked in git. After modifying any build script, commit:

```powershell
cd d:\Github\CasparVP
git add run_build.py build_hdr.bat build_now.bat build_ninja.bat
git commit -m "build: <description of change>"
```
