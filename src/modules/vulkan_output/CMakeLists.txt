cmake_minimum_required (VERSION 3.28)
project (vulkan_output)

set(SOURCES
	consumer/config.cpp
	consumer/config.h
	consumer/vulkan_output_consumer.cpp
	consumer/vulkan_output_consumer.h

	util/display_enum.cpp
	util/display_enum.h
	util/output_window.cpp
	util/output_window.h
	util/presentation.cpp
	util/presentation.h

	vulkan_output.cpp
	vulkan_output.h
)

casparcg_add_module_project(vulkan_output
	SOURCES ${SOURCES}
	INIT_FUNCTION "vulkan_output::init"
	VULKAN_REQUIREMENTS_FUNCTION "vulkan_output::register_vulkan_requirements"
)

target_link_libraries(vulkan_output PRIVATE accelerator)
target_link_libraries(vulkan_output PRIVATE Vulkan::Headers vk-bootstrap::vk-bootstrap)
target_compile_definitions(vulkan_output PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

if (MSVC)
	target_link_libraries(vulkan_output PRIVATE dxgi.lib)
endif ()
