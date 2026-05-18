cmake_minimum_required(VERSION 3.28)
project(cuda_notchlc LANGUAGES CXX CUDA)

# ─── CUDA configuration ─────────────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES "52;61;75;80;86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Wno-deprecated-gpu-targets")

# ─── Source lists ────────────────────────────────────────────────────────────
set(CUDA_SOURCES
    cuda/notchlc_decode.cu
)

set(CPU_SOURCES
    producer/notchlc_demuxer.cpp
    producer/notchlc_producer.cpp
    cuda_notchlc.cpp
)

set(HEADER_SOURCES
    cuda_notchlc.h
    util/cuda_gl_texture.h
    cuda/notchlc_decode.h
    cuda/notchlc_y_decode.cuh
    cuda/notchlc_uv_decode.cuh
    cuda/notchlc_a_decode.cuh
    cuda/notchlc_ycocg_to_bgra16.cuh
    producer/notchlc_demuxer.h
    producer/notchlc_producer.h
)

set(ALL_SOURCES ${CUDA_SOURCES} ${CPU_SOURCES} ${HEADER_SOURCES})

# ─── CasparCG module target ──────────────────────────────────────────────────
casparcg_add_module_project(cuda_notchlc
    SOURCES ${ALL_SOURCES}
    INIT_FUNCTION "cuda_notchlc::init"
)

set_property(TARGET cuda_notchlc PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_notchlc PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET cuda_notchlc PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ─── nvcomp ─────────────────────────────────────────────────────────────────
# nvcomp v5.x ships separate cmake configs for CUDA 12 and CUDA 13.
# Detect which CUDA toolkit is active and point to the matching cmake dir.
#
# Search order for nvCOMP root:
#   1. NVCOMP_ROOT cmake variable or environment variable
#   2. Default install path: C:/Program Files/NVIDIA nvCOMP/v5.2
set(_NVCOMP_ROOT "")
if(DEFINED NVCOMP_ROOT)
    set(_NVCOMP_ROOT "${NVCOMP_ROOT}")
elseif(DEFINED ENV{NVCOMP_ROOT})
    set(_NVCOMP_ROOT "$ENV{NVCOMP_ROOT}")
elseif(EXISTS "C:/Program Files/NVIDIA nvCOMP/v5.2")
    set(_NVCOMP_ROOT "C:/Program Files/NVIDIA nvCOMP/v5.2")
endif()

if(NOT _NVCOMP_ROOT OR NOT EXISTS "${_NVCOMP_ROOT}")
    message(FATAL_ERROR "\n*** nvCOMP 5.x not found. ***\n"
                        "Set NVCOMP_ROOT to the nvCOMP install directory, e.g.:\n"
                        "  cmake -DNVCOMP_ROOT=\"C:/Program Files/NVIDIA nvCOMP/v5.2\" ..\n"
                        "Or set the NVCOMP_ROOT environment variable.\n")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
    set(_NVCOMP_CMAKE_DIR "${_NVCOMP_ROOT}/lib/13/cmake/nvcomp")
else()
    set(_NVCOMP_CMAKE_DIR "${_NVCOMP_ROOT}/lib/12/cmake/nvcomp")
endif()

if(NOT EXISTS "${_NVCOMP_CMAKE_DIR}")
    message(FATAL_ERROR "\n*** nvCOMP cmake config not found at: ${_NVCOMP_CMAKE_DIR} ***\n"
                        "Ensure nvCOMP is installed and NVCOMP_ROOT points to the correct directory.\n")
endif()

find_package(nvcomp REQUIRED
    PATHS "${_NVCOMP_CMAKE_DIR}"
    NO_DEFAULT_PATH
)

# ─── Include paths ───────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

target_include_directories(cuda_notchlc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    "${_NVCOMP_ROOT}/include"
    ${FFMPEG_INCLUDE_PATH}
)

# nvcomp_export.h defaults to __declspec(dllimport) unless this is defined.
# Required when linking the static nvcomp library.
target_compile_definitions(cuda_notchlc PRIVATE NVCOMP_STATIC_DEFINE)

# ─── Link libraries ──────────────────────────────────────────────────────────
target_link_libraries(cuda_notchlc PRIVATE
    CUDA::cudart_static
    nvcomp::nvcomp_static       # static link — no nvcomp DLL needed on target machine
    ffmpeg
    GLEW::glew
)

IF(ENABLE_VULKAN)
    target_link_libraries(cuda_notchlc PRIVATE Vulkan::Headers)
ENDIF()

