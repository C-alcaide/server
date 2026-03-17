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

```
cmd /c ""C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build d:\Github\CasparVP\build --target <targets>
```

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
| `casparcg_copy_dependencies` | Copies DLLs next to the exe |

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

## Committing build scripts themselves

The scripts are tracked in git. After modifying any build script, commit:

```powershell
cd d:\Github\CasparVP
git add run_build.py build_hdr.bat build_now.bat build_ninja.bat
git commit -m "build: <description of change>"
```
