cmake_minimum_required(VERSION 3.16)

# Project settings
project("CasparCG Server")
set(CONFIG_VERSION_MAJOR 2)
set(CONFIG_VERSION_MINOR 6)
set(CONFIG_VERSION_BUG 0)
set(CONFIG_VERSION_TAG "Dev")

# support for language servers (eg, clangd or ccls)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CASPARCG_DOWNLOAD_MIRROR https://github.com/CasparCG/dependencies/releases/download/ CACHE STRING "Source/mirror to use for external dependencies")
set(CASPARCG_DOWNLOAD_CACHE ${CMAKE_CURRENT_BINARY_DIR}/external CACHE STRING "Download cache directory for cmake ExternalProjects")

option(ENABLE_HTML "Enable HTML module, require CEF" ON)

set(DIAG_FONT_PATH "LiberationMono-Regular.ttf" CACHE STRING
    "Path to font that will be used to load diag font at runtime. By default
    this loads the font distributed with the application from the working
    directory. It can be set to an absolute path to instead load a font from
    the system."
)
add_compile_definitions("DIAG_FONT_PATH=\"${DIAG_FONT_PATH}\"")

# Add custom cmake modules path
set(CASPARCG_PATCH_DIR ${CMAKE_SOURCE_DIR}/CMakeModules/patches)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)

# Determine the git hash
find_package(Git)
set(CONFIG_VERSION_GIT_HASH "N/A")
if(DEFINED ENV{GIT_HASH} AND NOT $ENV{GIT_HASH} STREQUAL "")
    set(CONFIG_VERSION_GIT_HASH "$ENV{GIT_HASH}")
elseif(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/../.git")
    message(STATUS "Getting git hash...")
    execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --verify --short HEAD
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../"
            OUTPUT_VARIABLE CONFIG_VERSION_GIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Git hash: ${CONFIG_VERSION_GIT_HASH}")
endif()

message(STATUS "Checking CPU count...")
cmake_host_system_information(RESULT CONFIG_CPU_COUNT QUERY NUMBER_OF_PHYSICAL_CORES)
message(STATUS "CPU count: ${CONFIG_CPU_COUNT}")

include(CasparCG_Util)
message(STATUS "Configuring version.h...")
configure_file("${PROJECT_SOURCE_DIR}/version.tmpl" "${CMAKE_BINARY_DIR}/generated/version.h")
message(STATUS "Configuring version.h done.")

if(MSVC)
    message(STATUS "Including Bootstrap_Windows...")
    include(Bootstrap_Windows)

else ()
    include(Bootstrap_Linux)

    # Output build settings
    message(STATUS "Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "CPU Count: ${CONFIG_CPU_COUNT}")
    message(STATUS "Target build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "Build Version: ${CONFIG_VERSION_MAJOR}.${CONFIG_VERSION_MINOR}.${CONFIG_VERSION_BUG} ${CONFIG_VERSION_TAG} (Revision: ${CONFIG_VERSION_GIT_HASH})")
endif ()

include_directories("${CMAKE_BINARY_DIR}/generated")

# ─── Auto-detect optional GPU features ───────────────────────────────────────

# Vulkan: detection handled in Bootstrap_Windows.cmake (where deps are fetched)
IF(ENABLE_VULKAN)
	add_definitions(-DENABLE_VULKAN)
ENDIF()

# CUDA: auto-detect from CUDA Toolkit if not explicitly set
# Workaround: MSBuild CUDA targets require CUDA_PATH_V<major>_<minor> env var.
# If CUDA_PATH is set but the versioned var is missing (e.g. terminal started
# before CUDA install), try to set it so check_language(CUDA) succeeds.
if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc.exe")
    execute_process(
        COMMAND "$ENV{CUDA_PATH}/bin/nvcc.exe" --version
        OUTPUT_VARIABLE _nvcc_version_output ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_nvcc_version_output MATCHES "release ([0-9]+)\\.([0-9]+)")
        set(_cuda_major "${CMAKE_MATCH_1}")
        set(_cuda_minor "${CMAKE_MATCH_2}")
        set(_cuda_path_var "CUDA_PATH_V${_cuda_major}_${_cuda_minor}")
        if(NOT DEFINED ENV{${_cuda_path_var}})
            set(ENV{${_cuda_path_var}} "$ENV{CUDA_PATH}")
            message(STATUS "Set ${_cuda_path_var}=$ENV{CUDA_PATH} (workaround for MSBuild)")
        endif()
        unset(_cuda_major)
        unset(_cuda_minor)
        unset(_cuda_path_var)
    endif()
    unset(_nvcc_version_output)
endif()

if(NOT DEFINED BUILD_CUDA_MODULES)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(BUILD_CUDA_MODULES ON CACHE BOOL "Build NVIDIA GPU-accelerated modules (ProRes, NotchLC)")
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER} (auto-enabled)")
    else()
        set(BUILD_CUDA_MODULES OFF CACHE BOOL "Build NVIDIA GPU-accelerated modules (ProRes, NotchLC)")
        message(STATUS "CUDA compiler not found -- CUDA modules disabled")
    endif()
elseif(BUILD_CUDA_MODULES)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "BUILD_CUDA_MODULES=ON but no CUDA compiler found -- disabling CUDA modules")
        set(BUILD_CUDA_MODULES OFF CACHE BOOL "" FORCE)
    endif()
endif()

# nvCOMP: needed only by cuda_notchlc (NotchLC GPU decompression)
if(BUILD_CUDA_MODULES)
    if(NOT DEFINED BUILD_CUDA_NOTCHLC)
        # Auto-detect nvCOMP installation
        if(DEFINED NVCOMP_ROOT)
            set(_nvcomp_test "${NVCOMP_ROOT}")
        elseif(DEFINED ENV{NVCOMP_ROOT})
            set(_nvcomp_test "$ENV{NVCOMP_ROOT}")
        elseif(EXISTS "C:/Program Files/NVIDIA nvCOMP/v5.2")
            set(_nvcomp_test "C:/Program Files/NVIDIA nvCOMP/v5.2")
        else()
            set(_nvcomp_test "")
        endif()

        if(_nvcomp_test AND EXISTS "${_nvcomp_test}/include/nvcomp.h")
            set(BUILD_CUDA_NOTCHLC ON CACHE BOOL "Build cuda_notchlc module (requires nvCOMP)")
            message(STATUS "nvCOMP found: ${_nvcomp_test} -- cuda_notchlc enabled")
        else()
            set(BUILD_CUDA_NOTCHLC OFF CACHE BOOL "Build cuda_notchlc module (requires nvCOMP)")
            message(STATUS "nvCOMP not found -- cuda_notchlc disabled (set NVCOMP_ROOT to enable)")
        endif()
        unset(_nvcomp_test)
    endif()
else()
    set(BUILD_CUDA_NOTCHLC OFF CACHE BOOL "Build cuda_notchlc module (requires nvCOMP)")
endif()

# Fetch shared header-only libraries before subdirectories that use them
include(FetchContent)

# tinygltf — header-only glTF 2.0 / GLB loader (used by core + accelerator)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinygltf
	GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
	GIT_TAG        v2.9.5
)
FetchContent_MakeAvailable(tinygltf)

# tinyobjloader — header-only OBJ loader (used by core + accelerator)
set(TINYOBJLOADER_WITH_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_Declare(tinyobjloader
	GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
	GIT_TAG        v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobjloader)

add_subdirectory(tools)
add_subdirectory(accelerator)
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(modules)
add_subdirectory(protocol)
add_subdirectory(shell)
