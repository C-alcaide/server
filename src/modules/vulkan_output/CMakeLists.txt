cmake_minimum_required(VERSION 3.28)
project(vulkan_output)

# ─── Find Vulkan SDK ─────────────────────────────────────────────────────────
# Use VULKAN_SDK env var or standard install location
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
    string(STRIP "${VULKAN_SDK_PATH}" VULKAN_SDK_PATH)
else()
    # Default Windows install location
    file(GLOB VULKAN_SDK_CANDIDATES "C:/VulkanSDK/*")
    list(SORT VULKAN_SDK_CANDIDATES ORDER DESCENDING)
    if(VULKAN_SDK_CANDIDATES)
        list(GET VULKAN_SDK_CANDIDATES 0 VULKAN_SDK_PATH)
    endif()
endif()

if(VULKAN_SDK_PATH)
    set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_PATH}/Include")
    set(Vulkan_LIBRARY "${VULKAN_SDK_PATH}/Lib/vulkan-1.lib")
    message(STATUS "Vulkan SDK: ${VULKAN_SDK_PATH}")
endif()

find_package(Vulkan REQUIRED)

# ─── Find NvAPI SDK ──────────────────────────────────────────────────────────
set(NVAPI_SDK_PATH "" CACHE PATH "Path to NvAPI SDK root")
if(NOT NVAPI_SDK_PATH)
    # Try common locations: sibling of source, sibling of repo root, env var
    foreach(_path "${CMAKE_SOURCE_DIR}/../nvapi-main" "${CMAKE_SOURCE_DIR}/../../nvapi-main" "${CMAKE_SOURCE_DIR}/../../../nvapi-main" "$ENV{NVAPI_SDK}")
        if(EXISTS "${_path}/nvapi.h")
            set(NVAPI_SDK_PATH "${_path}")
            break()
        endif()
    endforeach()
endif()

if(NVAPI_SDK_PATH AND EXISTS "${NVAPI_SDK_PATH}/nvapi.h")
    set(NVAPI_FOUND TRUE)
    set(NVAPI_INCLUDE_DIR "${NVAPI_SDK_PATH}")
    set(NVAPI_LIBRARY "${NVAPI_SDK_PATH}/amd64/nvapi64.lib")
    message(STATUS "NvAPI SDK: ${NVAPI_SDK_PATH}")
else()
    set(NVAPI_FOUND FALSE)
    message(STATUS "NvAPI SDK not found - Quadro Sync and EDID features disabled")
endif()

# ─── Find CUDA Toolkit (optional, for GPU peer transfer) ────────────────────
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(CUDA_PEER_FOUND TRUE)
    message(STATUS "CUDA Toolkit found - GPU peer transfer enabled")
else()
    set(CUDA_PEER_FOUND FALSE)
    message(STATUS "CUDA Toolkit not found - GPU peer transfer disabled (PBO fallback)")
endif()

# ─── Sources ─────────────────────────────────────────────────────────────────
set(SOURCES
    consumer/vulkan_output_consumer.cpp
    consumer/vulkan_output_consumer.h
    consumer/config.cpp
    consumer/config.h

    util/vulkan_device.cpp
    util/vulkan_device.h
    util/vk_device_manager.cpp
    util/vk_device_manager.h
    util/gpu_frame_cache.cpp
    util/gpu_frame_cache.h
    util/vulkan_interop.cpp
    util/vulkan_interop.h
    util/shared_texture_pool.cpp
    util/shared_texture_pool.h
    util/interop_context.cpp
    util/interop_context.h
    util/gpu_affinity_context.cpp
    util/gpu_affinity_context.h
    util/cuda_peer_transfer.cpp
    util/cuda_peer_transfer.h
    util/nvapi_helpers.cpp
    util/nvapi_helpers.h
    util/color_convert_pipeline.cpp
    util/color_convert_pipeline.h

    vulkan_output.cpp
    vulkan_output.h
)

# ─── Compile color_convert compute shader ────────────────────────────────────
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "${Vulkan_INCLUDE_DIRS}/../Bin")
if(GLSLC)
    set(CC_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/color_convert.comp")
    set(CC_SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/color_convert_comp.spv")
    set(CC_SHADER_HDR "${CMAKE_CURRENT_BINARY_DIR}/color_convert_spv.h")

    add_custom_command(
        OUTPUT "${CC_SHADER_SPV}"
        COMMAND "${GLSLC}" "${CC_SHADER_SRC}" -o "${CC_SHADER_SPV}"
        DEPENDS "${CC_SHADER_SRC}"
        COMMENT "Compiling color_convert.comp to SPIR-V"
    )

    add_custom_command(
        OUTPUT "${CC_SHADER_HDR}"
        COMMAND bin2c "caspar::vulkan_output" "color_convert_comp" "${CC_SHADER_SPV}" > "${CC_SHADER_HDR}"
        DEPENDS bin2c "${CC_SHADER_SPV}"
        COMMENT "Embedding color_convert SPIR-V as C++ header"
    )

    list(APPEND SOURCES "${CC_SHADER_HDR}")
else()
    message(WARNING "glslc not found - vulkan_output color conversion shader will not be compiled")
endif()

# ─── Module registration ─────────────────────────────────────────────────────
casparcg_add_module_project(vulkan_output
    SOURCES ${SOURCES}
    INIT_FUNCTION "vulkan_output::init"
)

target_include_directories(vulkan_output PRIVATE
    ..
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Required because texture_wrapper.h → texture.h → vulkan.hpp
target_compile_definitions(vulkan_output PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

if(NVAPI_FOUND)
    target_include_directories(vulkan_output SYSTEM PRIVATE ${NVAPI_INCLUDE_DIR})
    target_compile_definitions(vulkan_output PRIVATE CASPAR_NVAPI_ENABLED)
endif()

if(CUDA_PEER_FOUND)
    target_include_directories(vulkan_output SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(vulkan_output PRIVATE CASPAR_CUDA_PEER_ENABLED)
endif()

set_target_properties(vulkan_output PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\consumer consumer/*)
source_group(sources\\util util/*)
source_group(sources\\shaders shaders/*)

# ─── Link libraries ─────────────────────────────────────────────────────────
target_link_libraries(vulkan_output PRIVATE
    Vulkan::Vulkan
    accelerator
    GLEW::glew
    d3d11
    dxgi
    dwmapi
)

if(NVAPI_FOUND)
    target_link_libraries(vulkan_output PRIVATE ${NVAPI_LIBRARY})
endif()

if(CUDA_PEER_FOUND)
    target_link_libraries(vulkan_output PRIVATE CUDA::cudart_static)
endif()
