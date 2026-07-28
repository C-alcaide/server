cmake_minimum_required(VERSION 3.28)
project(remotewall LANGUAGES CXX CUDA)

# Native cloudXR tile-wall producer.
#   - Vendored (permissive): the recv_wall receiver + cloudXR wire/SEI headers.
#   - Referenced (not vendored): the NVIDIA Video Codec SDK (own EULA) via a cache
#     var; set REMOTEWALL_NVCODEC_SDK to your SDK root if it is not the default.
set(REMOTEWALL_NVCODEC_SDK "d:/Github/cloudXR-main/ref/Video_Codec_SDK_13.0.37"
    CACHE PATH "NVIDIA Video Codec SDK root (contains Samples/ and Interface/)")

set(CMAKE_CUDA_ARCHITECTURES "52;61;80;86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Wno-deprecated-gpu-targets")

set(_NVC "${REMOTEWALL_NVCODEC_SDK}/Samples")

set(SOURCES
    remotewall.cpp
    producer/remotewall_producer.cpp
    # Vendored receiver (permissive)
    vendored/recvwall/RecvWall.cpp
    # Referenced NVIDIA Video Codec SDK sources (compiled in place; not vendored)
    "${_NVC}/NvCodec/NvDecoder/NvDecoder.cpp"
    "${_NVC}/Utils/ColorSpace.cu"
)

set(HEADERS
    remotewall.h
    producer/remotewall_producer.h
    vendored/recvwall/RecvWall.h
)

casparcg_add_module_project(remotewall
    SOURCES ${SOURCES} ${HEADERS}
    INIT_FUNCTION "remotewall::init"
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(remotewall PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendored/recvwall
    ${CMAKE_CURRENT_SOURCE_DIR}/vendored/cloudxr        # net/ + sync/
    "${_NVC}/NvCodec"                                    # "NvDecoder/NvDecoder.h"
    "${_NVC}/Utils"                                      # ColorSpace.h, NvCodecUtils.h
    "${REMOTEWALL_NVCODEC_SDK}/Interface"                # nvcuvid.h, cuviddec.h
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(remotewall PRIVATE
    CUDA::cudart_static
    CUDA::cuda_driver
    "${REMOTEWALL_NVCODEC_SDK}/Lib/win/x64/nvcuvid.lib"
    d3d11
    dxgi
)

if (ENABLE_VULKAN)
    target_link_libraries(remotewall PRIVATE Vulkan::Headers)
endif()

# The vendored receiver + referenced SDK sources are not warning-clean; relax /WX
# for this module only (the CasparCG default is /W4 /WX). Scope to C++ so nvcc,
# which misparses a raw /WX-, is left alone.
if (MSVC)
    target_compile_options(remotewall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX->)
endif()

set_property(TARGET remotewall PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

set_target_properties(remotewall PROPERTIES FOLDER modules)
source_group(sources\\producer producer/*)
source_group(sources\\vendored vendored/*)
source_group(sources ./*)
