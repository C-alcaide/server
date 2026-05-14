cmake_minimum_required (VERSION 3.28)

# Enable CUDA when Vulkan accelerator is active — needed for the CUDA-VK
# direct GPU decklink strategy (cuda_vk_strategy.cu).
if(ENABLE_VULKAN)
	find_package(CUDAToolkit QUIET)
	if(CUDAToolkit_FOUND)
		project(decklink LANGUAGES CXX CUDA)
		set(CMAKE_CUDA_ARCHITECTURES "52;61;80;86;89" CACHE STRING "CUDA architectures (decklink)")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Wno-deprecated-gpu-targets")
		set(DECKLINK_CUDA_VK TRUE)
	else()
		project(decklink)
		set(DECKLINK_CUDA_VK FALSE)
	endif()
else()
	project(decklink)
	set(DECKLINK_CUDA_VK FALSE)
endif()

set(SOURCES
		consumer/decklink_consumer.cpp
		consumer/decklink_consumer.h
		consumer/v210_strategies.cpp
		consumer/sdr_bgra_strategy.cpp
		consumer/format_strategy.h
		consumer/config.cpp
		consumer/config.h
		consumer/monitor.cpp
		consumer/monitor.h
		consumer/vanc.cpp
		consumer/vanc.h
		consumer/vanc_scte104_strategy.cpp
		consumer/vanc_op47_strategy.cpp

		producer/decklink_producer.cpp
		producer/decklink_producer.h

		util/util.h

		decklink.cpp
		decklink.h

		decklink_api.h
		StdAfx.h
)

if (MSVC)
	list(APPEND SOURCES
			interop/DeckLinkAPI_i.c
			interop/DeckLinkAPI.h
			interop/DeckLinkAPIVersion.h
	)
else()
	list(APPEND SOURCES
			linux_interop/DeckLinkAPIConfiguration.h
			linux_interop/DeckLinkAPIDeckControl.h
			linux_interop/DeckLinkAPIDispatch.cpp
			linux_interop/DeckLinkAPIModes.h
			linux_interop/DeckLinkAPI_v10_2.h
			linux_interop/DeckLinkAPI_v7_3.h
			linux_interop/DeckLinkAPI_v7_9.h
			linux_interop/DeckLinkAPI_v8_1.h
			linux_interop/DeckLinkAPI_v9_9.h
			linux_interop/LinuxCOM.h
			linux_interop/DeckLinkAPIConfiguration_v10_2.h
			linux_interop/DeckLinkAPIDiscovery.h
			linux_interop/DeckLinkAPI.h
			linux_interop/DeckLinkAPITypes.h
			linux_interop/DeckLinkAPI_v7_1.h
			linux_interop/DeckLinkAPI_v7_6.h
			linux_interop/DeckLinkAPI_v8_0.h
			linux_interop/DeckLinkAPI_v9_2.h
			linux_interop/DeckLinkAPIVersion.h
	)
endif ()

# CUDA-VK direct GPU strategy sources
if(DECKLINK_CUDA_VK)
	list(APPEND SOURCES
		consumer/cuda_vk_strategy.cpp
		consumer/cuda_vk_strategy.h
		consumer/cuda_vk_kernels.cu
		consumer/cuda_vk_v210.cuh
	)
endif()

# Pure-Vulkan readback strategy sources (no CUDA dependency)
if(ENABLE_VULKAN)
	list(APPEND SOURCES
		consumer/vk_readback_strategy.cpp
		consumer/vk_readback_strategy.h
		consumer/vk_readback_v210_spv.h
		consumer/vk_readback_bgra_spv.h
	)
endif()

casparcg_add_module_project(decklink
	SOURCES ${SOURCES}
	INIT_FUNCTION "decklink::init"
)
target_include_directories(decklink PRIVATE
    ..
    ${FFMPEG_INCLUDE_PATH}
    )
target_precompile_headers(decklink PRIVATE "StdAfx.h")

# Exclude CUDA files from precompiled headers (nvcc is not compatible with them)
if(DECKLINK_CUDA_VK)
	set_source_files_properties(
		consumer/cuda_vk_kernels.cu
		PROPERTIES SKIP_PRECOMPILE_HEADERS ON
	)
endif()

set_target_properties(decklink PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\consumer consumer/*)
source_group(sources\\interop interop/*)
source_group(sources\\producer producer/*)
source_group(sources\\util util/*)

if (MSVC)
	target_link_libraries(decklink PRIVATE
			ffmpeg
	)
else()
	target_link_libraries(decklink PRIVATE
			ffmpeg

			dl
	)
endif ()

# CUDA-VK direct GPU strategy dependencies
if(DECKLINK_CUDA_VK)
	target_compile_definitions(decklink PRIVATE DECKLINK_CUDA_VK_ENABLED)
	target_include_directories(decklink SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(decklink PRIVATE CUDA::cudart_static)
	set_property(TARGET decklink PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
	set_property(TARGET decklink PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	set_property(TARGET decklink PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
	IF(ENABLE_VULKAN)
		target_link_libraries(decklink PRIVATE Vulkan::Headers)
	ENDIF()
endif()

# Pure-Vulkan readback strategy dependencies (Vulkan loader for API calls)
if(ENABLE_VULKAN)
	find_package(Vulkan REQUIRED)
	target_link_libraries(decklink PRIVATE Vulkan::Vulkan)
	set_source_files_properties(
		consumer/vk_readback_strategy.cpp
		PROPERTIES SKIP_PRECOMPILE_HEADERS ON
	)
endif()

