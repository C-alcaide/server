cmake_minimum_required (VERSION 3.28)
project (isf)

set(SOURCES
	isf.cpp
	isf.h
	isf_gl_context.cpp
	isf_gl_context.h
	isf_image_load.cpp
	isf_image_load.h
	isf_producer.cpp
	isf_producer.h
	isf_shader.cpp
	isf_shader.h
)

casparcg_add_module_project(isf
	SOURCES ${SOURCES}
	INIT_FUNCTION "isf::init"
)

# GL rendering (fullscreen ISF pass on the mixer's OGL device or a self-contained context);
# image module + FFmpeg for IMPORTED; SFML for the self-contained context (non-OGL mixers).
target_link_libraries(isf PRIVATE GLEW::glew sfml-window sfml-system image)
# Vulkan mixer: render into a shared VK texture instead of reading back (gl_export_bridge).
IF(ENABLE_VULKAN)
	target_link_libraries(isf PRIVATE Vulkan::Headers)
ENDIF()
target_include_directories(isf PRIVATE .. ${FFMPEG_INCLUDE_PATH})

# Prototype: GL + boost-json code is relaxed from the global /W4 /WX to avoid warning-as-error churn.
if (MSVC)
	target_compile_options(isf PRIVATE /W0 /WX-)
endif()

set_target_properties(isf PROPERTIES FOLDER modules)
source_group(sources ./*)
