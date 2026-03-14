cmake_minimum_required(VERSION 3.28)
project(cuda_prores LANGUAGES CXX CUDA)

# ─── CUDA configuration ─────────────────────────────────────────────────────
# sm_52 = Maxwell (Quadro M4000), sm_61 = Pascal (P2000/P5000),
# sm_80 = Ampere A100, sm_86 = RTX A5000/A4000, sm_89 = Ada L40/RTX 4000.
# CUDA 12.9 supports all of these (Maxwell/Pascal dropped in CUDA 13.0).
set(CMAKE_CUDA_ARCHITECTURES "52;61;80;86;89" CACHE STRING "CUDA architectures")

# Separable compilation: required so device functions defined in .cuh files
# can be called across translation units (e.g. rice/entropy/DCT).
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# CUDA 12.9 formally supports VS 2017-2022.  VS 2026 (MSVC 19.50) triggers a
# version check in host_config.h; the flag below suppresses it.  The CUDA
# constructs used here (basic kernels, CUB, __clz) are well within what the
# newer compiler handles correctly.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Wno-deprecated-gpu-targets")

# ─── Source lists ────────────────────────────────────────────────────────────
# Pure CUDA kernel TUs — no CasparCG core/common dependency.
# Used by standalone test targets as well as the main module.
set(CUDA_KERNEL_SOURCES
    # cuda_prores_entropy.cu is #included by cuda_prores_frame.cu — do NOT list
    # it as a separate compilation unit (would cause nvlink multiple-definition errors
    # with separable compilation since the same device functions would be in two objects).
    cuda/cuda_prores_frame.cu
)

# CasparCG consumer TUs — depend on core/common (Boost, monitor, etc.).
# Must NOT be included in standalone test targets.
set(CUDA_CONSUMER_SOURCES
    consumer/prores_consumer.cu
    consumer/prores_bypass_consumer.cu
)

set(CUDA_SOURCES ${CUDA_KERNEL_SOURCES} ${CUDA_CONSUMER_SOURCES})

set(CPU_SOURCES
    # Muxer
    muxer/mov_muxer.cpp
    muxer/mxf_muxer.cpp

    # Storage
    storage/async_file_writer.cpp

    # Input (DeckLink capture + allocator)
    input/cuda_pinned_allocator.cpp
    input/decklink_capture.cpp

    # Module entry point
    cuda_prores.cpp
)

set(HEADER_SOURCES
    # CUDA headers
    cuda/cuda_prores_rice.cuh
    cuda/cuda_prores_v210_unpack.cuh
    cuda/cuda_prores_tables.cuh
    cuda/cuda_prores_dct_quant.cuh
    cuda/cuda_prores_frame.h

    # CPU headers
    muxer/mov_muxer.h
    muxer/mxf_muxer.h
    storage/async_file_writer.h
    input/cuda_pinned_allocator.h
    input/decklink_capture.h

    # Module header
    cuda_prores.h

    # Consumer
    consumer/prores_consumer.h
    consumer/prores_bypass_consumer.h

    # Timecode
    timecode.h

    # BGRA->V210 CUDA kernel
    cuda/cuda_bgra_to_v210.cuh
    cuda/cuda_bgra_to_field422p10.cuh
)

set(ALL_SOURCES ${CUDA_SOURCES} ${CPU_SOURCES} ${HEADER_SOURCES})

# ─── CasparCG module target ──────────────────────────────────────────────────
casparcg_add_module_project(cuda_prores
    SOURCES ${ALL_SOURCES}
    INIT_FUNCTION "cuda_prores::init"
)

# CUDA separable compilation for the module target
set_property(TARGET cuda_prores PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_prores PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET cuda_prores PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ─── Include paths ───────────────────────────────────────────────────────────
# CUB (ships with CUDA Toolkit 11+)
find_package(CUDAToolkit REQUIRED)

target_include_directories(cuda_prores PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_PATH}
    # DeckLink SDK headers — shared with the decklink module
    ../decklink
)

# ─── Link libraries ──────────────────────────────────────────────────────────
target_link_libraries(cuda_prores PRIVATE
    # CUDA runtime (static preferred for redistribution)
    CUDA::cudart_static
    # NVIDIA Performance Primitives (for optional DCT path)
    CUDA::nppc
    CUDA::nppist
    # FFmpeg (via the CasparCG ffmpeg target which re-exports the libs)
    ffmpeg
)

if (MSVC)
    # Nothing extra on Windows — DeckLink interfaces come in via the DeckLink
    # module's interop/ headers (COM, oleaut32, etc via the SDK).
    # No explicit DeckLink .lib: COM late-binding via CoCreateInstance.
else()
    target_link_libraries(cuda_prores PRIVATE dl)
endif()

# ─── Standalone test target ──────────────────────────────────────────────────
# Built separately; does NOT depend on CasparCG core/common.
if (MSVC)
    add_executable(test_prores_encode EXCLUDE_FROM_ALL
        test/test_prores_encode.cpp
        ${CUDA_KERNEL_SOURCES}
        muxer/mov_muxer.cpp
        storage/async_file_writer.cpp
    )
    target_include_directories(test_prores_encode PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_PATH}
        ../decklink
    )
    target_link_libraries(test_prores_encode PRIVATE
        CUDA::cudart_static
        CUDA::nppc
        CUDA::nppist
        ffmpeg
    )
    set_property(TARGET test_prores_encode PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET test_prores_encode PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(test_prores_encode PROPERTIES FOLDER "modules/tests")

    # ── test_bgra_convert ─────────────────────────────────────────────────────
    # Validates the BGRA→V210 CUDA kernel with solid-colour golden values.
    add_executable(test_bgra_convert EXCLUDE_FROM_ALL
        test/test_bgra_convert.cpp
        ${CUDA_KERNEL_SOURCES}
    )
    target_include_directories(test_bgra_convert PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(test_bgra_convert PRIVATE
        CUDA::cudart_static
    )
    set_property(TARGET test_bgra_convert PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET test_bgra_convert PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(test_bgra_convert PROPERTIES FOLDER "modules/tests")

    # ── test_timecode_roundtrip ───────────────────────────────────────────────
    # CPU-only: writes a MOV with tmcd track and verifies with ffprobe.
    add_executable(test_timecode_roundtrip EXCLUDE_FROM_ALL
        test/test_timecode_roundtrip.cpp
        muxer/mov_muxer.cpp
        storage/async_file_writer.cpp
    )
    target_include_directories(test_timecode_roundtrip PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_PATH}
    )
    target_link_libraries(test_timecode_roundtrip PRIVATE
        ffmpeg
    )
    set_target_properties(test_timecode_roundtrip PROPERTIES FOLDER "modules/tests")

    # ── test_rice_entropy ─────────────────────────────────────────────────────
    # CPU-only: validates Rice coding algorithm and FIRST_DC_CB parameters.
    add_executable(test_rice_entropy EXCLUDE_FROM_ALL
        test/test_rice_entropy.cpp
    )
    target_include_directories(test_rice_entropy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    # No CUDA runtime needed — pure CPU test
    set_target_properties(test_rice_entropy PROPERTIES FOLDER "modules/tests")

    # ── test_perf_benchmark ───────────────────────────────────────────────────
    # GPU throughput benchmark: encodes N frames per profile and reports fps/Mbps.
    add_executable(test_perf_benchmark EXCLUDE_FROM_ALL
        test/test_perf_benchmark.cpp
        ${CUDA_KERNEL_SOURCES}
    )
    target_include_directories(test_perf_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(test_perf_benchmark PRIVATE
        CUDA::cudart_static
        CUDA::nppc
        CUDA::nppist
    )
    set_property(TARGET test_perf_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET test_perf_benchmark PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set_target_properties(test_perf_benchmark PROPERTIES FOLDER "modules/tests")
endif()

# ─── IDE organisation ────────────────────────────────────────────────────────
set_target_properties(cuda_prores PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\cuda       cuda/*)
source_group(sources\\muxer      muxer/*)
source_group(sources\\storage    storage/*)
source_group(sources\\input      input/*)
source_group(sources\\consumer   consumer/*)
source_group(sources\\test       test/*)
