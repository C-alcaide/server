cmake_minimum_required (VERSION 3.28)
project (accelerator)

set(SOURCES
	ogl/image/image_kernel.cpp
	ogl/image/image_mixer.cpp
	ogl/image/image_shader.cpp
	ogl/image/previz_renderer.cpp

	ogl/util/buffer.cpp
	ogl/util/context.cpp
	ogl/util/device.cpp
	ogl/util/shader.cpp
	ogl/util/texture.cpp
	ogl/util/matrix.cpp
	ogl/util/transforms.cpp

	accelerator.cpp
)
set(HEADERS
	ogl/image/image_kernel.h
	ogl/image/image_mixer.h
	ogl/image/image_shader.h
	ogl/image/previz_renderer.h
	ogl/image/previz_scene.h

	ogl/util/buffer.h
	ogl/util/context.h
	ogl/util/device.h
	ogl/util/shader.h
	ogl/util/texture.h
	ogl/util/matrix.h
	ogl/util/transforms.h

	ogl_image_vertex.h
	ogl_image_fragment.h
	ogl_previz_vertex.h
	ogl_previz_fragment.h

	accelerator.h
	StdAfx.h
)

IF(ENABLE_VULKAN)
	set(VULKAN_SOURCES
		vulkan/image/image_mixer.cpp
		vulkan/image/image_kernel.cpp
		vulkan/image/previz_texture_bridge.cpp
		vulkan/util/device.cpp
		vulkan/util/buffer.cpp
		vulkan/util/texture.cpp
		vulkan/util/pipeline.cpp
		vulkan/util/renderpass.cpp
		vulkan/util/matrix.cpp
		vulkan/util/transforms.cpp
	)
	set(VULKAN_HEADERS
		vulkan/image/image_mixer.h
		vulkan/image/image_kernel.h
		vulkan/image/previz_texture_bridge.h
		vulkan/util/device.h
		vulkan/util/buffer.h
		vulkan/util/texture.h
		vulkan/util/pipeline.h
		vulkan/util/renderpass.h
		vulkan/util/matrix.h
		vulkan/util/transforms.h
		vulkan/util/draw_params.h
		vulkan/util/uniform_block.h
		vulkan/util/texture_wrapper.h

		vk_image_vertex.h
		vk_image_fragment.h
	)

	find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

	set(VK_VERT_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/image/vertex_shader.vert")
	set(VK_FRAG_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/image/fragment_shader.frag")
	set(VK_VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/vertex_shader.spv")
	set(VK_FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/fragment_shader.spv")

	add_custom_command(
		OUTPUT "${VK_VERT_SPV}"
		COMMAND "${GLSLC}" "${VK_VERT_SHADER}" -o "${VK_VERT_SPV}"
		DEPENDS "${VK_VERT_SHADER}"
		COMMENT "Compiling vertex_shader.vert to SPIR-V"
	)
	add_custom_command(
		OUTPUT "${VK_FRAG_SPV}"
		COMMAND "${GLSLC}" "${VK_FRAG_SHADER}" -o "${VK_FRAG_SPV}"
		DEPENDS "${VK_FRAG_SHADER}"
		COMMENT "Compiling fragment_shader.frag to SPIR-V"
	)

	bin2c_file("${VK_VERT_SPV}" "vk_image_vertex.h" "caspar::accelerator::vulkan" "vertex_shader")
	bin2c_file("${VK_FRAG_SPV}" "vk_image_fragment.h" "caspar::accelerator::vulkan" "fragment_shader")

	list(APPEND SOURCES ${VULKAN_SOURCES})
	list(APPEND HEADERS ${VULKAN_HEADERS})
ENDIF()

bin2c("ogl/image/shader.vert" "ogl_image_vertex.h" "caspar::accelerator::ogl" "vertex_shader")
bin2c("ogl/image/shader.frag" "ogl_image_fragment.h" "caspar::accelerator::ogl" "fragment_shader")
bin2c("ogl/image/previz.vert" "ogl_previz_vertex.h" "caspar::accelerator::ogl" "previz_vertex_shader")
bin2c("ogl/image/previz.frag" "ogl_previz_fragment.h" "caspar::accelerator::ogl" "previz_fragment_shader")

casparcg_add_library(accelerator SOURCES ${SOURCES} ${HEADERS})
target_include_directories(accelerator PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(accelerator PRIVATE ${tinygltf_SOURCE_DIR} ${tinyobjloader_SOURCE_DIR})
target_precompile_headers(accelerator PRIVATE StdAfx.h)
target_link_libraries(accelerator PRIVATE common core GLEW::glew sfml-window)

IF(ENABLE_VULKAN)
	target_link_libraries(accelerator PRIVATE vk-bootstrap::vk-bootstrap VulkanMemoryAllocator Vulkan::Headers)
	target_compile_definitions(accelerator PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
ENDIF()

source_group(sources ./.*)
source_group(sources\\cpu\\image cpu/image/.*)
source_group(sources\\cpu\\util cpu/util/.*)
source_group(sources\\ogl\\image ogl/image/.*)
source_group(sources\\ogl\\util ogl/util/.*)
IF(ENABLE_VULKAN)
	source_group(sources\\vulkan\\image vulkan/image/.*)
	source_group(sources\\vulkan\\util vulkan/util/.*)
ENDIF()
