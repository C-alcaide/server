# Portability audit — Linux, Docker and cloud, 2026-09-01

**Question.** Upstream CasparCG builds and runs on Linux and in Docker. After ~1000
commits of fork work, does this tree still?

**Short answer.** Nothing structural has been broken, and **one flagship feature is
silently absent on Linux**: OpenColorIO. Beyond that, the honest status is *unverified* —
the fork has never been compiled for Linux, by CI or by hand.

Everything below is read from the source tree and from GitHub Actions history. Nothing
here was measured by running a build, and that distinction is the point of the last
section.

---

## 1. The fork has never been built for Linux

| branch | Linux CI runs |
| :--- | ---: |
| `preview-zero-copy` | **0** |
| `CasparVPV-GS` | **0** |
| `CasparVPV` | **0** |
| `alpha-fixtures` | **0** |

`.github/workflows/linux.yml` exists, is **untouched by the fork** (its last commits are
dependabot bumps), and passes — on `master`, `v2.5.x` and upstream PR branches. Those are
upstream commits. The VP branches are **1003 commits** ahead of the merge base and have
never been through it.

`on: push` has its branch filter commented out, so *any* pushed branch triggers the
workflow. The fork's branches have simply never been pushed to a remote that runs it.

**A green Linux badge on this repository currently certifies upstream, not the fork.**

## 2. A Linux/Docker path was written for one module and never run

The fork added `tools/linux/Dockerfile.vulkan-output` and
`src/modules/vulkan_output/LINUX_TESTING.md` in `fc2691acf`
("feat(vulkan_output): port module to Linux via EGL/VK_KHR_display").

`LINUX_TESTING.md`'s checklist is **entirely unticked**, including its first item:

```
## 1. Basic Compilation (CI — no GPU needed)
- [ ] Build completes with no errors
- [ ] libvulkan_output.a is produced
- [ ] No undefined symbols at link time
```

So the procedure exists and the build it describes has not been done. No document in the
tree claims a Linux build was ever completed.

## 3. What the audit found intact

These are the things that would break a Linux build outright, and they are all handled.

**Windows-only modules are gated.** `src/modules/CMakeLists.txt` puts `spout`, `flash`,
`bluefish` and `replay` inside `if (MSVC)`. `vulkan_output` is behind `ENABLE_VULKAN`, the
CUDA modules behind `BUILD_CUDA_MODULES`, `gstreamer` behind `ENABLE_GSTREAMER`.

**The headless path is intact, and it is the one that makes Docker viable.**
`accelerator/ogl/util/context.cpp` keeps upstream's `#ifndef _MSC_VER` EGL branch; the fork
only *added* a `#ifdef _MSC_VER` WGL branch beside it. EGL needs no X display, which is the
prerequisite for running in a container.

**Windows APIs in portable trees are `#ifdef`-wrapped.**
`accelerator/vulkan/util/d3d11_import_bridge.{h,cpp}` is compiled unconditionally — it sits
outside the `if (MSVC)` block in `accelerator/CMakeLists.txt` — but the whole file is inside
`#ifdef _WIN32`, so on Linux it compiles to nothing. `common/vulkan/platform_handles.h` is
guarded the same way. `cluster`'s `ptp_clock.cpp` and `command_relay.cpp` carry 8–10 guards
each for WinSock against POSIX sockets.

**Always-built fork modules are clean.** `tracking`, `keyframes` and `isf` contain no
Windows-only code at all. The five Windows-dependent `ofx` files are under
`src/modules/ofx/test/` and only `coregl_orientation_test.cpp` is referenced by
`Bootstrap_Linux.cmake`.

## 4. The material finding: no OpenColorIO on Linux

```
Bootstrap_Windows.cmake:224   option(ENABLE_OCIO "Enable OpenColorIO colour management" ON)
Bootstrap_Linux.cmake         (no such option, and no opencolorio project)
accelerator/CMakeLists.txt:213 IF(ENABLE_OCIO) → target_compile_definitions(... CASPAR_ENABLE_OCIO)
accelerator/ocio/ocio_config.cpp:25  #ifdef CASPAR_ENABLE_OCIO
```

`ENABLE_OCIO` is declared **only** in the Windows bootstrap. On Linux it is never defined,
so `CASPAR_ENABLE_OCIO` is not set and the OCIO sources compile to their stubs.

**A Linux build therefore has no OCIO**, which removes:

* `OCIO_DISPLAY` and `OCIO_LOOK`
* per-consumer OCIO views — the operator-monitor feature, on both `screen` and `spout`
* OCIO LUTs and the ACES view transforms

It **degrades rather than crashes**: `ocio_display_command` logs *"this server was built
without OCIO support"* and returns `501`. That is the right behaviour, and it also means the
absence is easy to miss — the server starts, plays and looks healthy.

Comparing the two bootstraps for declared dependencies, Windows declares and Linux does not:

| missing on Linux | consequence |
| :--- | :--- |
| **`opencolorio`** | **the whole OCIO feature set, as above** |
| `cudatoolkit` | `cuda_prores`, `cuda_notchlc`, `remotewall` unavailable — already behind `BUILD_CUDA_MODULES`, so expected |
| `ffmpeg-lib`, `zlib` | Linux uses system packages; not a gap |
| `flashtemplatehost` | Windows-only feature by nature |
| `cudatest_ofx`, `cudapassthrough_ofx`, `transitiontest_ofx`, `coreglpassthrough_ofx` | sample OFX plugins only |

The size of the asymmetry is the underlying story: the fork added **+329 lines** to
`Bootstrap_Windows.cmake` and **+116** to `Bootstrap_Linux.cmake`, and most of the Linux
ones arrived through upstream merges rather than fork work. Two fork commits touch it at
all (`chore: fedora build fixes`, `feat(ofx): add OpenFX host module`).

## 5. What is still unknown, and what it would cost to know

Reading source proves that nothing is *structurally* Windows-locked. It cannot prove the
build succeeds: a missing `find_package`, a header that only Windows happens to pull in
transitively, or a C++20 feature MSVC accepts and GCC does not would all show up only at
compile time.

**Cheapest real verification: push a VP branch to a remote whose Actions are enabled.**
The workflow already exists and triggers on any push, needs no GPU for the compile job, and
would answer the question in about eight minutes. That is an outward-facing action on a
shared remote, so it wants a decision rather than an assumption.

Local verification is not currently possible on this machine: no Docker, and WSL has no
distribution installed.

Beyond compiling, two runtime questions remain untested on Linux and are separate from each
other:

* **Headless.** The EGL path exists, but the `screen` consumer wants a display and is
  always built. A container would run with `vulkan_output`, `ffmpeg`, `decklink` or `ndi`
  and no `screen` — plausible, unmeasured.
* **Docker with a GPU.** Needs `nvidia-container-toolkit` and device passthrough. The
  fork's own `Dockerfile.vulkan-output` targets a build stage, not a runtime one.

## 6. Recommendation

1. **Push a VP branch and let the existing Linux CI compile it.** One push, eight minutes,
   and it converts every "unverified" above into a fact.
2. **Add `ENABLE_OCIO` to `Bootstrap_Linux.cmake`.** OCIO builds on Linux and is packaged
   by every major distribution; the option is simply absent. Without it a Linux deployment
   quietly loses the fork's colour management — which is most of what makes it a VP server.
3. **Tick or correct `LINUX_TESTING.md`.** An unticked checklist that reads as a plan is
   fine; one that gets quoted as status is not.
4. Only then consider a runtime container. Compiling is the gate everything else waits on.
