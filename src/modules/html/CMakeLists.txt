cmake_minimum_required (VERSION 3.28)
project (html)

set(SOURCES
		producer/html_cg_proxy.cpp
		producer/html_cg_proxy.h
		producer/html_gpu_bridge.cpp
		producer/html_gpu_bridge.h
		producer/html_producer.cpp
		producer/html_producer.h

		html.cpp
		html.h
		util.h
)

casparcg_add_module_project(html
	SOURCES ${SOURCES}
	INIT_FUNCTION "html::init"
	UNINIT_FUNCTION "html::uninit"
	CLI_INTERCEPTOR "html::intercept_command_line"
)
target_link_libraries(html PRIVATE ffmpeg CEF::CEF)

if (MSVC)
	# GPU-direct path: CEF's OnAcceleratedPaint hands back a DXGI shared NT handle,
	# opened with ID3D11Device1::OpenSharedResource1 on the mixer's adapter and then
	# handed to whichever accelerator the mixer is using -- Vulkan's
	# d3d11_import_bridge or OpenGL's dx_interop.
	target_link_libraries(html PRIVATE d3d11 dxgi accelerator)
endif()

target_include_directories(html PRIVATE
    ..
	${FFMPEG_INCLUDE_PATH}
)

set_target_properties(html PROPERTIES FOLDER modules)
source_group(sources\\producer producer/*)
source_group(sources ./*)
