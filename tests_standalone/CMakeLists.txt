cmake_minimum_required(VERSION 3.28)
project(cuda_prores_tests LANGUAGES CXX CUDA)

# ─── CUDA configuration ──────────────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES "52;61;80;86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ─── Paths ───────────────────────────────────────────────────────────────────
set(MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/modules/cuda_prores")
set(CUDA_DIR   "${MODULE_DIR}/cuda")
set(TEST_DIR   "${MODULE_DIR}/test")

# ─── FFmpeg (shared pre-built from CasparCG dependency mirror) ───────────────
include(FetchContent)
set(CASPARCG_DOWNLOAD_MIRROR
    "https://github.com/CasparCG/dependencies/releases/download/"
    CACHE STRING "CasparCG dependency mirror")
set(CASPARCG_DOWNLOAD_CACHE "${CMAKE_CURRENT_BINARY_DIR}/dep_cache"
    CACHE STRING "Download cache dir")

FetchContent_Declare(ffmpeg_prebuilt
    URL      "${CASPARCG_DOWNLOAD_MIRROR}ffmpeg/ffmpeg-7.0.2-full_build-shared.7z"
    URL_HASH MD5=c5127aeed36a9a86dd3b84346be182f8
    DOWNLOAD_DIR "${CASPARCG_DOWNLOAD_CACHE}"
)
FetchContent_MakeAvailable(ffmpeg_prebuilt)

set(FFMPEG_ROOT "${ffmpeg_prebuilt_SOURCE_DIR}")
set(FFMPEG_INCLUDE "${FFMPEG_ROOT}/include")
set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

# Create an imported interface target so test targets can just link "ffmpeg"
add_library(ffmpeg INTERFACE IMPORTED)
target_include_directories(ffmpeg INTERFACE "${FFMPEG_INCLUDE}")
target_link_directories(ffmpeg INTERFACE "${FFMPEG_LIB_DIR}")
target_link_libraries(ffmpeg INTERFACE
    avcodec avformat avutil avdevice avfilter swresample swscale)

# ─── CUDA source objects (shared by GPU tests) ───────────────────────────────
# NOTE: cuda_prores_frame.cu #includes cuda_prores_entropy.cu textually,
# so compiling both would produce duplicate symbol definitions at link time.
set(CUDA_SOURCES
    "${CUDA_DIR}/cuda_prores_frame.cu"
)

# ─── test_rice_entropy ───────────────────────────────────────────────────────
# Pure CPU test — zero dependencies
add_executable(test_rice_entropy "${TEST_DIR}/test_rice_entropy.cpp")
target_include_directories(test_rice_entropy PRIVATE "${MODULE_DIR}")
target_compile_features(test_rice_entropy PRIVATE cxx_std_17)

# ─── test_bgra_convert ───────────────────────────────────────────────────────
# Tests the BGRA→V210 CUDA kernel only; no FFmpeg needed
# The .cpp includes a .cuh with __global__ kernels and <<<>>> syntax, so it
# must be compiled by nvcc — we override the language for this source file.
set_source_files_properties("${TEST_DIR}/test_bgra_convert.cpp" PROPERTIES LANGUAGE CUDA)
add_executable(test_bgra_convert "${TEST_DIR}/test_bgra_convert.cpp")
target_include_directories(test_bgra_convert PRIVATE
    "${MODULE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(test_bgra_convert PRIVATE CUDA::cudart_static)
set_property(TARGET test_bgra_convert PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ─── test_perf_benchmark ─────────────────────────────────────────────────────
# Includes cuda_prores_tables.cuh which has __constant__ symbols — must be nvcc.
set_source_files_properties("${TEST_DIR}/test_perf_benchmark.cpp" PROPERTIES LANGUAGE CUDA)
add_executable(test_perf_benchmark
    "${TEST_DIR}/test_perf_benchmark.cpp"
    ${CUDA_SOURCES})
target_include_directories(test_perf_benchmark PRIVATE
    "${MODULE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(test_perf_benchmark PRIVATE
    CUDA::cudart_static
    CUDA::nppc
    CUDA::nppist)
set_property(TARGET test_perf_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_perf_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET test_perf_benchmark PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ─── test_prores_encode ───────────────────────────────────────────────────────
# Includes cuda_prores_tables.cuh which has __constant__ symbols — must be nvcc.
set_source_files_properties("${TEST_DIR}/test_prores_encode.cpp" PROPERTIES LANGUAGE CUDA)
add_executable(test_prores_encode
    "${TEST_DIR}/test_prores_encode.cpp"
    ${CUDA_SOURCES}
    "${MODULE_DIR}/muxer/mov_muxer.cpp"
    "${MODULE_DIR}/storage/async_file_writer.cpp")
target_include_directories(test_prores_encode PRIVATE
    "${MODULE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${FFMPEG_INCLUDE}")
target_link_libraries(test_prores_encode PRIVATE
    CUDA::cudart_static
    CUDA::nppc
    CUDA::nppist
    ffmpeg)
set_property(TARGET test_prores_encode PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_prores_encode PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET test_prores_encode PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ─── test_timecode_roundtrip ──────────────────────────────────────────────────
add_executable(test_timecode_roundtrip
    "${TEST_DIR}/test_timecode_roundtrip.cpp"
    "${MODULE_DIR}/muxer/mov_muxer.cpp"
    "${MODULE_DIR}/storage/async_file_writer.cpp")
target_include_directories(test_timecode_roundtrip PRIVATE
    "${MODULE_DIR}"
    "${FFMPEG_INCLUDE}")
target_link_libraries(test_timecode_roundtrip PRIVATE ffmpeg)
