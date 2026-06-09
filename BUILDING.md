# Building the CasparCG Server

The CasparCG Server source code uses the CMake build system in order to easily
generate build systems for multiple platforms. CMake is basically a build
system for generating build systems.

On Windows we can use CMake to generate a .sln file and .vcproj files. On
Linux CMake can generate make files or ninja files. Qt Creator has support for
loading CMakeLists.txt files directly.

# Dependency caching

CMake will automatically download some dependencies as part of the build process.
These are taken from https://github.com/CasparCG/dependencies/releases (make sure to expand the 'Assets' group under each release to see the files), most of which are direct copies of distributions from upstream.

During the build, you can specify the CMake option `CASPARCG_DOWNLOAD_MIRROR` to download from an alternate HTTP server (such as an internally hosted mirror), or `CASPARCG_DOWNLOAD_CACHE` to use a specific path on disk for the local cache of these files, by default a folder called `external` will be created inside the build directory to cache these files.

If you want to be able to build CasparCG offline, you may need to manually seed this cache. You can do so by placing the correct tar.gz or zip into a folder and using `CASPARCG_DOWNLOAD_CACHE` to tell CMake where to find it.
You can figure out which files you need by looking at each of the `ExternalProject_Add` function calls inside of [Bootstrap_Linux.cmake](./src/CMakeModules/Bootstrap_Linux.cmake) or [Bootstrap_Windows.cmake](./src/CMakeModules/Bootstrap_Windows.cmake). Some of the ones listed are optional, depending on other CMake flags.

# Windows

## Building distributable

1. Install Visual Studio 2022.

2. Install 7-zip (https://www.7-zip.org/).

3. `git clone --single-branch --branch master https://github.com/CasparCG/server casparcg-server-master`

4. `cd casparcg-server-master`

5. `.\tools\windows\build.bat`

6. Copy the `dist\casparcg_server.zip` file for distribution

## Development using Visual Studio

1. Install Visual Studio 2022.

2. `git clone --single-branch --branch master https://github.com/CasparCG/server casparcg-server-master`

3. Open the cloned folder in Visual Studio.

4. Build All and ensure it builds successfully

# Linux

## Building on your system

We only officially support Ubuntu LTS releases, other distros may work but often run into build issues. We are happy to accept PRs to resolve these issues, but are unlikely to write fixes ourselves.

We currently document two approaches to building CasparCG. The recommended way is to use the `deb` packaging we have in the repository, but we only provide that for Ubuntu LTS releases.
Other deb based distros can work with some tweaks to one of those, other distros will need something else which is not documented here.

We also provide a script to produce a build in docker, but this is not recommended unless absolutely necessary. The resulting builds are often rather brittle depending on where they are used.

To perform a custom build, follow the Development steps below, and you may need to do some extra packaging steps, or install steps on the target systems.

### Building inside Docker

1. `git clone --single-branch --branch master https://github.com/CasparCG/server casparcg-server-master`
2. `cd casparcg-server-master`
3. `./tools/linux/build-in-docker`

If all goes to plan, a docker image `casparcg/server` has been created containing CasparCG Server.

### Extracting CasparCG Server from Docker

1. `./tools/linux/extract-from-docker`

You will then find a folder called `casparcg_server` which should contain everything you need to run CasparCG Server.

_Note: if you ran docker with sudo, CasparCG server will not be able to run without sudo out of the box. For security reasons we do not recommend to run CasparCG with sudo. Instead you can use chown to change the ownership of the CasparCG Server folder._

## Development

Before beginning, check the build options section below, to decide if you want to use any to simplify or customise your build.

1. `git clone --single-branch --branch master https://github.com/CasparCG/server casparcg-server-master`
2. `cd casparcg-server-master`
3. Install dependencies, this can be done with `sudo ./tools/linux/install-dependencies`
4. If using system CEF (default & recommended), `sudo add-apt-repository ppa:casparcg/ppa` and `sudo apt-get install casparcg-cef-142-dev`
5. `mkdir build && cd build`
6. `cmake ../src` You can add any of the build options from below to this command
7. `cmake --build . --parallel`
8. `cmake --install . --prefix staging`

If all goes to plan, a folder called 'staging' has been created with everything you need to run CasparCG server.

## Build options

-DENABLE_HTML=OFF - useful if you lack CEF, and would like to build without that module.

-DUSE_STATIC_BOOST=ON - (Linux only, default OFF) statically link against Boost.

-DUSE_SYSTEM_CEF=OFF - (Linux only, default ON) use the version of CEF from your OS. This expects to be using builds from https://launchpad.net/~casparcg/+archive/ubuntu/ppa

-DENABLE_AVX2=ON (Linux only, default ON) Enable the AVX and AVX2 instruction sets (requires a CPU that supports it)

-DDIAG_FONT_PATH - Specify an alternate path/font to use for the DIAG window. On linux, this will often want to be set to an absolute path of a font

-DCASPARCG_BINARY_NAME=casparcg-server - (Linux only) generate the executable with the specified name. This also reconfigures the install target to be a bit more friendly with system package managers.

# CasparVPV Branch — Optional Dependencies

The CasparVPV branch adds several optional GPU-accelerated and virtual-production modules.
All optional dependencies are **auto-detected** at configure time — if they are not installed,
the corresponding modules are simply disabled. No manual flags are needed for a basic build.

## Vulkan SDK (optional)

Enables the Vulkan accelerator backend and the `vulkan_output` consumer (low-latency GPU output).

- **Install**: Download from https://vulkan.lunarg.com/ and run the installer.
- **Version**: 1.3+ (tested with 1.4.350)
- **Detection**: `find_package(Vulkan)` reads the `VULKAN_SDK` environment variable (set by the installer).
- **Override**: `-DENABLE_VULKAN=OFF` to force-disable.

## CUDA Toolkit (optional)

Enables GPU-accelerated ProRes encoding/decoding (`cuda_prores`) and NotchLC decoding (`cuda_notchlc`).
Both modules are fully supported on **Windows and Linux**.

- **Install**: Download from https://developer.nvidia.com/cuda-downloads
- **Version**: CUDA **12.8 or 12.9** recommended. Avoid 12.4–12.6 (nvcc template deduction bug). CUDA 13+ works but drops Maxwell/Pascal GPU support.
- **Detection**: `check_language(CUDA)` — auto-detected from PATH.
- **Override**: `-DBUILD_CUDA_MODULES=OFF` to force-disable all CUDA modules.

### Additional Linux dependencies for CUDA modules

The `cuda_prores` module requires:
- **liburing-dev** — io_uring async I/O for the high-throughput file writer (`sudo apt install liburing-dev`)
- **libEGL-dev** — EGL for CUDA-GL interop in the consumer (`sudo apt install libegl-dev`)
- **DeckLink SDK** — the DeckLink Linux headers are included in the repository (`src/modules/decklink/linux_interop/`)

## NVIDIA nvCOMP (optional, requires CUDA)

Required only by the `cuda_notchlc` module (NotchLC GPU decompression). If CUDA is enabled but nvCOMP is not found, `cuda_prores` still builds normally.

- **Install**: Download nvCOMP 5.x from https://developer.nvidia.com/nvcomp and extract/install it.
- **Detection**: Searches `NVCOMP_ROOT` cmake variable, `NVCOMP_ROOT` environment variable, then platform-specific default paths:
  - Windows: `C:/Program Files/NVIDIA nvCOMP/v5.2`
  - Linux: `/usr/local/lib/cmake/nvcomp` or `/opt/nvidia/nvcomp`
- **Override**: `-DBUILD_CUDA_NOTCHLC=OFF` to skip, or `-DNVCOMP_ROOT="/path/to/nvcomp"` to point to a custom location.

## NVIDIA NvAPI SDK (optional, with Vulkan)

Enables Quadro Sync II, EDID injection, and hardware HDR probing in the `vulkan_output` module. When absent, stubs are compiled and these features are simply unavailable at runtime.

- **Install**: Download from https://developer.nvidia.com/nvapi and extract.
- **Detection**: Searches `NVAPI_SDK_PATH` cmake variable, a sibling `nvapi-main/` directory, then the `NVAPI_SDK` environment variable.
- **Override**: `-DNVAPI_SDK_PATH="C:/path/to/nvapi"` to point to a custom location.

## Summary of CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_VULKAN` | Auto (ON if SDK found) | Vulkan accelerator + vulkan_output module |
| `BUILD_CUDA_MODULES` | Auto (ON if CUDA found) | cuda_prores and cuda_notchlc modules |
| `BUILD_CUDA_NOTCHLC` | Auto (ON if nvCOMP found) | cuda_notchlc only (subset of CUDA modules) |
| `BUILD_SPOUT` | Auto (ON on Windows) | Spout texture-sharing module |
| `BUILD_REPLAY` | ON | Replay module (Windows only) |
| `BUILD_TRACKING_VRPN` | OFF | Camera-tracking VRPN client support |
| `NVCOMP_ROOT` | Auto-detected | Path to nvCOMP installation |
| `NVAPI_SDK_PATH` | Auto-detected | Path to NvAPI SDK root |
