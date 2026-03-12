cmake_minimum_required(VERSION 3.28)
project(cuda_prores LANGUAGES CXX CUDA)

# ─── CUDA configuration ─────────────────────────────────────────────────────
# RTX A5000 = sm_86 (Ampere).  Add sm_80 (A100) and sm_89 (Ada) as extras.
set(CMAKE_CUDA_ARCHITECTURES "86;80;89" CACHE STRING "CUDA architectures")

# Separable compilation: required so device functions defined in .cuh files
# can be called across translation units (e.g. rice/entropy/DCT).
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ─── Source lists ────────────────────────────────────────────────────────────
set(CUDA_SOURCES
    cuda/cuda_prores_entropy.cu
    cuda/cuda_prores_frame.cu
)

set(CPU_SOURCES
    # Muxer
    muxer/mov_muxer.cpp
    muxer/mxf_muxer.cpp

    # Storage
    storage/async_file_writer.cpp

    # Input (DeckLink capture + allocator)
    input/cuda_pinned_allocator.cpp
    input/decklink_capture.cpp

    # Consumer (CasparCG integration)
    consumer/prores_consumer.cpp

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
    add_executable(test_prores_encode
        test/test_prores_encode.cpp
        ${CUDA_SOURCES}
        muxer/mov_muxer.cpp
        storage/async_file_writer.cpp
    )
    target_include_directories(test_prores_encode PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
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
