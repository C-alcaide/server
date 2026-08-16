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
		consumer/gpu_output_buffer_pool.cpp
		consumer/gpu_output_buffer_pool.h
		consumer/ogl_gl_strategy.cpp
		consumer/ogl_gl_strategy.h
		consumer/dvp_support.cpp
		consumer/dvp_support.h
		consumer/config.cpp
		consumer/config.h
		consumer/monitor.cpp
		consumer/monitor.h
		consumer/vanc.cpp
		consumer/vanc.h
		consumer/vanc_scte104_strategy.cpp
		consumer/vanc_op47_strategy.cpp
		consumer/vanc_hdr_strategy.cpp
		consumer/color_primaries.h

		producer/decklink_producer.cpp
		producer/sdi_signalling.cpp
		producer/decklink_producer.h
		producer/sdi_signalling.h

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

# OpenGL-mixer GPU pack (GL compute v210/BGRA) needs GLEW for GL entry points.
# The calls run on the ogl accelerator's GL thread (context current there).
target_link_libraries(decklink PRIVATE GLEW::glew)
set_source_files_properties(
	consumer/ogl_gl_strategy.cpp
	PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

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

	# NVIDIA GPUDirect for Video (DVP) — Tier-2 hardware-synced GPU->card transfer.
	# Referenced from the DeckLink SDK's bundled NVIDIA_GPUDirect (own NVIDIA EULA),
	# not vendored. Uses the DLL import lib (dvp.lib + dvp.dll): the static lib is
	# built against the static CRT (/MT) and conflicts with CasparCG's dynamic CRT.
	set(DECKLINK_GPUDIRECT_SDK "d:/Github/Blackmagic DeckLink SDK 15.3/Win/Samples/NVIDIA_GPUDirect"
		CACHE PATH "NVIDIA GPUDirect for Video (DVP) root inside the DeckLink SDK")
	if(MSVC AND EXISTS "${DECKLINK_GPUDIRECT_SDK}/lib/x64/dvp.lib")
		message(STATUS "decklink: NVIDIA DVP (GPUDirect for Video) enabled from ${DECKLINK_GPUDIRECT_SDK}")
		target_compile_definitions(decklink PRIVATE DECKLINK_CUDA_DVP_ENABLED)
		target_include_directories(decklink SYSTEM PRIVATE "${DECKLINK_GPUDIRECT_SDK}/include")
		target_link_libraries(decklink PRIVATE
			"${DECKLINK_GPUDIRECT_SDK}/lib/x64/dvp.lib"
			CUDA::cuda_driver)
		# NOTE: dvp.dll from ${DECKLINK_GPUDIRECT_SDK}/bin/x64 must be staged next to
		# casparcg.exe (done by the packaging step / copied manually for dev runs).
	endif()
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


# The SDI signalling parsers, checked against EBU Tech 3375's published payload-identifier
# bytes. EXCLUDE_FROM_ALL because this fork has no test infrastructure and a normal build
# must not change shape; build it explicitly:
#
#   cmake --build build --target decklink_sdi_signalling_test
#   build/modules/decklink/decklink_sdi_signalling_test.exe
#
# It is worth the target rather than a scratchpad script because the vectors are the only
# third-party ground truth available for this code -- the loopback on this rig carries no
# ancillary data at all, so the decode cannot be exercised end to end here.
add_executable(decklink_sdi_signalling_test EXCLUDE_FROM_ALL
	producer/test_sdi_signalling.cpp
	producer/sdi_signalling.cpp
)
target_include_directories(decklink_sdi_signalling_test PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../..
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${CMAKE_CURRENT_SOURCE_DIR}/producer
)
# Deliberately links nothing. `common` would pull GLEW and TBB in behind it, and the point of
# keeping these parsers free of every dependency but the colour enums is that the test then
# needs no graphics stack, no DeckLink card and no server -- it is bytes in, enums out.
set_target_properties(decklink_sdi_signalling_test PROPERTIES
	DISABLE_PRECOMPILE_HEADERS ON
)
