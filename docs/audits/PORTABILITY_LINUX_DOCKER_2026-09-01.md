# Portability audit — Linux, Docker and cloud, 2026-09-01

**Question.** Upstream CasparCG builds and runs on Linux and in Docker. After ~1000
commits of fork work, does this tree still?

**Short answer, as first written (2026-09-01).** Nothing structural has been broken, and
**one flagship feature is silently absent on Linux**: OpenColorIO. Beyond that, the honest
status is *unverified* — the fork has never been compiled for Linux, by CI or by hand.

**Short answer, after actually building it (2026-09-02).** It builds now, and it did not
before: **the fork did not compile on Linux at all**, and it took twelve commits' worth of
fixes to get from "unverified" to a 30.5 MB `shell/casparcg`. The first sentence above was
too kind and the second was the important one — see §7, added below.

The original audit is left as written from §1 to §6, because the interesting part of this
document is now the gap between what reading the source predicted and what compiling it
found. Everything in those sections was read from the tree and from GitHub Actions history,
and **not one line of it was measured**; §7 is the measurement. Two of the six sections
turned out to be wrong in ways that reading could not have caught, and one central claim was
wrong in the other direction — OCIO's absence was worse than "silent".

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
*(Corrected 2026-09-02 — see §7.5. `origin` is NOT such a remote: Actions is disabled at the
repository level on `C-alcaide/server`, so the push done that day triggered nothing.)*
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
   *(Corrected 2026-09-02: the push is done and it ran nothing — Actions is switched off for
   the repository. Enabling it is the actual first step, and it is a settings change rather
   than a code one. See §7.5.)*
2. **Add `ENABLE_OCIO` to `Bootstrap_Linux.cmake`.** OCIO builds on Linux and is packaged
   by every major distribution; the option is simply absent. Without it a Linux deployment
   quietly loses the fork's colour management — which is most of what makes it a VP server.
3. **Tick or correct `LINUX_TESTING.md`.** An unticked checklist that reads as a plan is
   fine; one that gets quoted as status is not.
4. Only then consider a runtime container. Compiling is the gate everything else waits on.

---

# 7. The measurement, 2026-09-02

§1–§6 above were read from the source. This section is what happened when the build was
actually run, and it is a different answer.

## 7.1 What was built, and where

No Docker on this machine, so: **WSL Ubuntu 24.04**, installed for this purpose. 24.04
deliberately, not the 26.04 that `wsl --install` offers by default — the CI job builds
`tools/linux/Dockerfile`, whose base image is `buildpack-deps:noble`, and a newer GCC is a
stricter gate whose failures would not be attributable to CI.

| | |
| :--- | :--- |
| distro | Ubuntu 24.04.4 LTS (noble), matching the CI image |
| toolchain | gcc 13.3.0, cmake 3.28.3, ninja 1.11.1 |
| dependencies | the repo's own `tools/linux/install-dependencies`, not a set invented here |
| flags | `-DUSE_STATIC_BOOST=ON -DUSE_SYSTEM_CEF=OFF -DENABLE_VULKAN=ON -DENABLE_OCIO=OFF`, from the fork's own Dockerfile |
| Vulkan SDK | LunarG 1.4.328.1, the version that Dockerfile pins |
| FFmpeg | the distribution's, via `find_package(FFmpeg REQUIRED)` — **6.1** (libavutil 58 / libavcodec 60) |

Two environment details that are not findings about the fork but that shaped the work:

* **`git` over HTTPS from this WSL instance fails about half the time**, with "could not read
  Username for 'https://github.com'". It is not the network — DNS resolves, TCP 443 connects,
  `curl` gets 200 from the git smart-http endpoint itself, and the TLS peer presents a genuine
  github.com certificate — and it is not the inherited Windows PATH either, which was the
  first plausible explanation and was **wrong**: six A/B runs had the full PATH fail then
  succeed, and the Linux-only PATH succeed then fail. FetchContent retries a clone three
  times, and three coin flips landing badly is what turns a flaky fetch into a hard configure
  error. Worked around by copying the eighteen already-populated dependency sources out of the
  Windows build tree and pointing `FETCHCONTENT_SOURCE_DIR_*` at them, which makes the run
  offline and deterministic without changing which sources are compiled.
* **`core.autocrlf=true` gives this Windows checkout CRLF shell scripts.** Running
  `tools/linux/install-dependencies` from `/mnt/d` therefore failed with `E: Unable to locate
  package` and *no package name*, because a backslash followed by CR is not a line
  continuation. The repo blob is LF (`git ls-files --eol` reports `i/lf`), so this is not a
  defect — but it does mean the mounted tree is not equivalent to a checkout, and the build
  was done from an `rsync`ed copy, which is also what the Dockerfile's `COPY ./src /source`
  does.

## 7.2 Result

From an empty build directory, `ENABLE_VULKAN=ON`:

```
530 targets, 0 failed, 0 warnings       (Bootstrap_Linux.cmake compiles with -Werror)
shell/casparcg      ELF 64-bit LSB pie executable, x86-64, 30.5 MB
libvulkan_output.a  3.47 MB
ldd shell/casparcg | grep "not found"   ->  none
```

The binary starts, initialises logging and reaches configuration parsing. **That is the whole
claim.** No channel has been opened, no frame rendered and no consumer instantiated on Linux;
that needs a GPU and is §7.5.

`ENABLE_VULKAN=OFF` builds too, and separately: 278 targets, 0 failed, a 17.4 MB binary. It
needed six more fixes of its own (`f82a22e05`), because that arm had never been compiled on
any platform — Windows defaults the option ON. Two of those six were an unguarded
*declaration* whose every *use* was already guarded, which is invisible until something
compiles the other arm: `av_producer`'s Vulkan importer, and the `unique_ptr` member in
`ffmpeg_consumer` whose destructor nothing defined, which broke the link rather than the
compile.

## 7.3 The fourteen defects, and what class each belonged to

Reading the source found **one** of these — the missing `ENABLE_OCIO` — and it turned out to
be the least of them.

| # | what | why Windows never saw it |
| --: | :--- | :--- |
| 1 | `display_blanker` added unconditionally, sourcing `../../tools/*` from **outside `src/`** and linking `user32 gdi32 dwmapi` | the Dockerfile does `COPY ./src /source`, so cmake failed at **generate**, before compiling anything |
| 2 | `-Wno-terminate` (C++-only) in the global options next to `-Werror` | upstream's Linux build has no C targets; the fork added `expat` and `portaudio` |
| 3 | `/WX-`, `/w` and `/W0` on tinyobjloader, lodepng and Snappy | gcc reads a slash-option as an input **file** — "linker input file not found", naming neither the flag nor the target |
| 4 | 31 lambdas capturing `this` through `[=]` | C++20 deprecates it; MSVC warns, and `-Werror` does not |
| 5 | `<cmath>` and `<GL/glew.h>` missing | MSVC supplies both transitively; `ogl/util/device.h` includes glew only under `#ifdef WIN32` |
| 6 | five `std::ifstream(std::wstring)` | a Microsoft extension. The same file already used `boost::filesystem::ifstream` five lines away |
| 7 | `std::max(0LL, int64_t)` ×3 | `int64_t` is `long long` on Windows and `long` on LP64 Linux |
| 8 | DeckLink `GUID` and `REFIID operator==` | LinuxCOM.h makes `REFIID` a by-value 16-byte struct with no comparison operator, and has no `GUID` at all |
| 9 | the screen consumer's VK→GL interop: WGL, `glImportMemoryWin32HandleEXT`, `DuplicateHandle`, `win32_gl_window::shared_` | guarded on `ENABLE_VULKAN` where it needed `ENABLE_VULKAN && _WIN32` |
| 10 | `vulkan_output`: `VkSurfaceFullScreenExclusiveInfoEXT` outside the guard that used it, three `TerminateProcess` calls | the module's Linux port had never been compiled — which is exactly what `LINUX_TESTING.md` §1 recorded |
| 11 | an exported Vulkan memory handle declared `void*` | it is `HANDLE` on Windows and an **fd** on Linux. The validity test was wrong too: invalid is `nullptr` there and `-1` here, so `if (!h)` accepted −1 and would have rejected fd 0 |
| 12 | five FFmpeg APIs newer than the distribution's, unguarded | Windows pins FFmpeg 8; Linux takes the distro's 6.1, with no pin and no minimum declared anywhere |
| 13 | two `#else`-arm stubs with signatures two parameters out of date | only the real arm is ever compiled on Windows |
| 14 | `ENABLE_VULKAN=OFF` did not compile or link — a guarded enumerator with unguarded comparisons, and two unguarded members | the option defaults ON on Windows, so this arm had never been built anywhere |

Thresholds for #12 were taken from FFmpeg's own `doc/APIchanges` rather than guessed —
`av_frame_side_data_new` at lavu 59.3.100, `AVCodecContext.decoded_side_data` at lavc
61.2.100, `av_buffersink_get_colorspace` at lavfi 9.16.100 — following this repo's rule that
a constant comes from the body that defines it.

## 7.4 Where this audit was wrong

Worth stating plainly, because the failure mode is the point.

* **§4 said a Linux build "quietly loses OCIO" and degrades to a `501`.** It does not: with
  `ENABLE_OCIO` undefined, the `build_display_transform` stub had four parameters where the
  declaration had five, so **the binary did not link**. The 501 path is real code and it was
  unreachable. Reading `#ifdef CASPAR_ENABLE_OCIO` and the stub block beneath it — which is
  what §4 did — cannot see an arity mismatch. A compiler can.
* **§3, "What the audit found intact", was right about everything it checked, and the checks
  were not the ones that mattered.** Windows-only *modules* are indeed gated, Windows APIs in
  portable trees are indeed `#ifdef`-wrapped, and the EGL path is indeed intact. Ten of the
  fourteen defects are in files that section correctly described as portable — the faults were
  in compiler flags, lambda captures, integer widths, transitive includes and library
  versions, none of which a structural read looks at.
* **§5 named three real classes and implied a handful of instances** — "a missing
  `find_package`, a header that only Windows pulls in transitively, or a C++20 feature MSVC
  accepts and GCC does not". The actual count was fourteen classes and roughly seventy
  individual sites. The direction was right; the magnitude was off by an order.

The one thing §5 got exactly right: *"Reading source proves that nothing is structurally
Windows-locked. It cannot prove the build succeeds."*

## 7.5 What is still unknown

Unchanged by this work, and now the whole of the remaining list:

* **Does it run?** Nothing beyond process start-up has been exercised. No channel, no
  producer, no consumer, no frame.
* **Headless.** The EGL path compiles. Whether a channel initialises with no display is
  untested, and WSL2's GPU support is not a fair test of a real EGL device.
* **`vulkan_output` on Linux.** `LINUX_TESTING.md` §1 is now ticked; §2 onwards is not.
  `VK_KHR_display` needs a directly attached output, which WSL does not have.
* **Docker.** Nothing has been built through the Dockerfile itself. Defect #1 means the Docker
  path was broken in a way a non-Docker build would *not* have found — so the container is now
  the more likely of the two to work, and it remains unrun.
* **CI — and this section had the wrong premise until 2026-09-02.** `CasparVPV-GS` HAS now
  been pushed, and **no workflow ran**, because **GitHub Actions is disabled at the repository
  level** on `C-alcaide/server`: `gh api repos/C-alcaide/server/actions/permissions` returns
  `{"enabled": false}`. So §5 and §6's "push a VP branch and let the existing Linux CI compile
  it — one push, eight minutes" was wrong about the eight minutes. A push alone achieves
  nothing here; **someone has to turn Actions on in the repository settings first.**

  Note also that `gh run list` in this working tree resolves to **`upstream`**, not to the
  fork, because the tree has three remotes. §1's table of green runs on `master` and `v2.5.x`
  is therefore CasparCG's CI, which §1 said, but it is very easy to read a `gh run list`
  output here as the fork's. The fork's own history is **15 runs, all in March 2026 on
  `feature/cuda-prores`, and every one of them failed** — Linux *and* Windows. This fork's CI
  has never been green, on any platform.

  Once Actions is enabled, there are **three** workflows, not the one this audit discussed:
  `linux.yml`, `linux-system.yml` ("Build Linux with system dependencies") and `windows.yml`.
  All three trigger on any push, their branch filters being commented out. `linux-system.yml`
  is the interesting extra one for the FFmpeg-version findings in §7.3 #12, since building
  against the distribution's libav* is exactly what it does.
* **`ENABLE_OCIO=ON` on Linux.** The bootstrap block added in `deeb168ba` **configures** — the
  `CHECKPOINT: Adding OpenColorIO` line prints — but its ExternalProject clones OpenColorIO at
  build time, which the flaky git in §7.1 makes unreliable here. The OCIO sub-build itself is
  unmeasured.
