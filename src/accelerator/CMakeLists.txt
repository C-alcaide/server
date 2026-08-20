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
	ogl/util/dx_interop.cpp
	ogl/util/shader.cpp
	ogl/util/texture.cpp
	ogl/util/matrix.cpp
	ogl/util/transforms.cpp

	ocio/ocio_config.cpp

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
	ogl/util/dx_interop.h
	ogl/util/shader.h
	ogl/util/texture.h
	ogl/util/matrix.h
	ogl/util/transforms.h

	ogl_image_vertex.h
	ogl_image_fragment.h
	ogl_previz_vertex.h
	ogl_previz_fragment.h

	ocio/ocio_config.h

	accelerator.h
	StdAfx.h
)

# Upstream's D3D11 device wrappers (2427604fa). Compiled, but their FRAME-IMPORT path is
# not adopted: `frame_factory::import_d3d_texture` stays undeclared because this fork has
# its own GPU-direct HTML path (html_gpu_bridge + ogl/util/dx_interop). These three files
# are needed anyway because html.cpp's is_gpu_shared_texture_enabled() probes the D3D
# device to decide whether CEF can hand over a shared texture at all, and without them that
# is an unresolved symbol at link time. See docs/UPSTREAM_SYNC_2026-08-18.md.
if (MSVC)
	list(APPEND SOURCES
		d3d/d3d_device.cpp
		d3d/d3d_device_context.cpp
		d3d/d3d_texture2d.cpp
	)
	list(APPEND HEADERS
		d3d/d3d_device.h
		d3d/d3d_device_context.h
		d3d/d3d_texture2d.h
	)
endif ()

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
		vulkan/util/texture_wrapper.cpp
		vulkan/util/d3d11_import_bridge.cpp
		vulkan/util/shared_device_info.cpp
		vulkan/util/shared_device_info.h
		vulkan/util/gl_export_bridge.cpp
		vulkan/util/glsl_compiler.cpp
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
		vulkan/util/glsl_compiler.h
		vulkan/util/d3d11_import_bridge.h
		vulkan/util/gl_export_bridge.h

		vk_image_vertex.h
		vk_image_fragment.h
		vk_image_fragment_src.h
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

	# The GLSL SOURCE as well as the compiled SPIR-V.
	#
	# The SPIR-V above is what the mixer normally uses, built by glslc at configure time and
	# unchanged. But a generated colour transform arrives as GLSL text and has to be spliced
	# into this shader's source, then compiled at runtime -- and SPIR-V cannot be spliced. The
	# OGL backend never needed this step because bin2c already embeds shader.frag as text:
	# OpenGL compiles GLSL itself, so the source was there all along. Vulkan embeds only the
	# binary, so without this there is nothing at runtime to splice INTO.
	#
	# Costs ~30 KB in the binary and nothing at run time until a variant is requested.
	bin2c("vulkan/image/fragment_shader.frag" "vk_image_fragment_src.h" "caspar::accelerator::vulkan" "fragment_shader_src")

	list(APPEND SOURCES ${VULKAN_SOURCES})
	list(APPEND HEADERS ${VULKAN_HEADERS})
ENDIF()

IF (ENABLE_VULKAN)
	find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
	set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/image)
	set(SHADER_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/image)

	file(GLOB SHADERS
		${SHADER_SOURCE_DIR}/vertex_shader.vert
		${SHADER_SOURCE_DIR}/fragment_shader.frag)

	foreach(source IN LISTS SHADERS)
		get_filename_component(FILENAME ${source} NAME_WLE)
		add_custom_command(
			COMMAND
				${glslc_executable}
				#      -MD -MF ${SHADER_BINARY_DIR}/${FILENAME}.d
				-o ${SHADER_BINARY_DIR}/${FILENAME}.spv
				${source}
			OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
			DEPENDS ${source} ${SHADER_BINARY_DIR}
			COMMENT "Compiling ${FILENAME}"
		)
		list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
	endforeach()

	add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})

	bin2c("vulkan/image/vertex_shader.spv" "vulkan_image_vertex.h" "caspar::accelerator::vulkan" "vertex_shader")
	bin2c("vulkan/image/fragment_shader.spv" "vulkan_image_fragment.h" "caspar::accelerator::vulkan" "fragment_shader")
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

	# shaderc, for compiling a generated colour transform to SPIR-V at runtime. The mixer's
	# own shader is still compiled at build time by glslc and embedded; this is only for
	# source that does not exist until a transform is chosen.
	#
	# Vulkan::shaderc_combined rather than the shared library: it is the imported target
	# CMake's own FindVulkan provides, so no SDK path is hardcoded, and being static there
	# is no additional DLL to deploy alongside the server.
	target_link_libraries(accelerator PRIVATE Vulkan::shaderc_combined)
ENDIF()

# OpenColorIO lives in the mixer, so the accelerator is where it links. The link directory
# comes from Bootstrap_Windows.cmake's link_directories() on the ExternalProject install
# tree, the same way ffmpeg and zlib are handled.
IF(ENABLE_OCIO)
	target_include_directories(accelerator PRIVATE ${OCIO_INCLUDE_PATH})
	target_link_libraries(accelerator PRIVATE OpenColorIO)
	# accelerator/ocio/ocio_config.cpp is the only translation unit that includes OCIO
	# headers; its facade is std-only, so nothing downstream needs the include path.
	target_compile_definitions(accelerator PRIVATE CASPAR_ENABLE_OCIO)
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
