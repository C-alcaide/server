cmake_minimum_required (VERSION 3.28)
project (gpu_test)

set(SOURCES
		gpu_test.cpp
		gpu_test.h
)

casparcg_add_module_project(gpu_test
	SOURCES ${SOURCES}
	INIT_FUNCTION "gpu_test::init"
)
target_include_directories(gpu_test PRIVATE
	..
    )

set_target_properties(gpu_test PROPERTIES FOLDER modules)
source_group(sources ./*)

target_link_libraries(gpu_test PRIVATE
	accelerator
)

if (ENABLE_VULKAN)
	target_link_libraries(gpu_test PRIVATE Vulkan::Headers vk-bootstrap::vk-bootstrap)
	target_compile_definitions(gpu_test PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
endif ()
