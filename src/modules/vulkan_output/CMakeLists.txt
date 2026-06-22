cmake_minimum_required (VERSION 3.28)
project (vulkan_output)

# Compile GLSL shaders to SPIR-V
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

file(GLOB SHADERS ${SHADER_SOURCE_DIR}/color_convert.comp)

foreach(source IN LISTS SHADERS)
	get_filename_component(FILENAME ${source} NAME)
	add_custom_command(
		COMMAND
			${glslc_executable}
			-o ${SHADER_BINARY_DIR}/${FILENAME}.spv
			${source}
		OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
		DEPENDS ${source} ${SHADER_BINARY_DIR}
		COMMENT "Compiling ${FILENAME}"
	)
	list(APPEND SPV_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach()

add_custom_target(vulkan_output_shaders ALL DEPENDS ${SPV_SHADERS})

bin2c("shaders/color_convert.comp.spv" "color_convert_comp.h" "caspar::vulkan_output" "color_convert_comp_spv")

set(SOURCES
	consumer/config.cpp
	consumer/config.h
	consumer/vulkan_output_consumer.cpp
	consumer/vulkan_output_consumer.h

	util/color_convert_pipeline.cpp
	util/color_convert_pipeline.h
	util/display_enum.cpp
	util/display_enum.h
	util/frame_transfer.cpp
	util/frame_transfer.h
	util/nvapi_helpers.cpp
	util/nvapi_helpers.h
	util/output_device.cpp
	util/output_device.h
	util/output_window.cpp
	util/output_window.h
	util/present_barrier.h
	util/presentation.cpp
	util/presentation.h
	util/startup_gate.h

	vulkan_output.cpp
	vulkan_output.h

	color_convert_comp.h
)

casparcg_add_module_project(vulkan_output
	SOURCES ${SOURCES}
	INIT_FUNCTION "vulkan_output::init"
	VULKAN_REQUIREMENTS_FUNCTION "vulkan_output::register_vulkan_requirements"
)

target_include_directories(vulkan_output PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(vulkan_output vulkan_output_shaders)
target_link_libraries(vulkan_output PRIVATE accelerator)
target_link_libraries(vulkan_output PRIVATE Vulkan::Headers vk-bootstrap::vk-bootstrap)
target_compile_definitions(vulkan_output PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

if (MSVC)
	target_link_libraries(vulkan_output PRIVATE dxgi.lib)
endif ()

# Optional NvAPI support (Quadro Sync, EDID emulation, hardware HDR)
if (MSVC)
	find_path(NVAPI_INCLUDE_DIR nvapi.h HINTS "${CMAKE_SOURCE_DIR}/../nvapi" "${CMAKE_SOURCE_DIR}/external/nvapi")
	find_library(NVAPI_LIBRARY NAMES nvapi64 HINTS "${CMAKE_SOURCE_DIR}/../nvapi/amd64" "${CMAKE_SOURCE_DIR}/external/nvapi/amd64")

	if (NVAPI_INCLUDE_DIR AND NVAPI_LIBRARY)
		message(STATUS "NvAPI found: ${NVAPI_INCLUDE_DIR}")
		target_include_directories(vulkan_output PRIVATE ${NVAPI_INCLUDE_DIR})
		target_link_libraries(vulkan_output PRIVATE ${NVAPI_LIBRARY})
		target_compile_definitions(vulkan_output PRIVATE CASPAR_NVAPI_ENABLED=1)
	else ()
		message(STATUS "NvAPI not found — building without Quadro Sync / EDID emulation / hardware HDR.")
	endif ()
endif ()
