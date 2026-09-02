cmake_minimum_required (VERSION 3.28)
project (core)

# tinygltf and tinyobjloader are fetched in the top-level CMakeLists.txt

# tinyobjloader compiles sources and inherits the global warnings-as-errors and /FI — this
# relaxes them for third-party code.
#
# PER COMPILER, because a bare `/WX-` reaches gcc as a FILENAME. gcc does not reject an
# unknown slash-option; it treats it as a second input and fails with "cannot specify '-o'
# with '-c', '-S' or '-E' with multiple files" -- a message that names neither the flag nor
# this file. It was the first target to build on the first Linux build of this fork, and it
# failed for a Windows flag.
#
# `-w` is the Linux equivalent and is needed on its own account: Bootstrap_Linux.cmake sets
# a global `-Werror`, so this third-party source is warning-gated there too. `modules/ltc`
# already carries exactly this pattern; `modules/isf` and `modules/remotewall` guard theirs
# with `if (MSVC)`. This site was the only unguarded one.
if(TARGET tinyobjloader)
    target_compile_options(tinyobjloader PRIVATE
        "$<$<CXX_COMPILER_ID:MSVC>:/WX->$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>")
    # Global /FI requires the source root in include path to find disable_silly_warnings.h
    target_include_directories(tinyobjloader PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

set(SOURCES
		consumer/frame_consumer.cpp
		consumer/frame_consumer_registry.cpp
		consumer/output.cpp

		diagnostics/call_context.cpp
		diagnostics/log_graph.cpp
		diagnostics/osd_graph.cpp

		frame/draw_frame.cpp
		frame/blend_mask_loader.cpp
		frame/frame.cpp
		frame/frame_transform.cpp
		frame/geometry.cpp
		frame/mesh_loader.cpp
		frame/write_frame.cpp

		mixer/audio/audio_mixer.cpp
		mixer/image/blend_modes.cpp
		mixer/mixer.cpp

		producer/color/color_producer.cpp
		producer/separated/separated_producer.cpp
		producer/transition/transition_producer.cpp
		producer/transition/sting_producer.cpp
		producer/route/route_producer.cpp

		producer/cg_proxy.cpp
		producer/frame_producer.cpp
		producer/frame_producer_registry.cpp
		producer/layer.cpp
		producer/stage.cpp

		video_channel.cpp
		video_format.cpp
)
set(HEADERS
		consumer/frame_consumer.h
		consumer/frame_consumer_registry.h
		consumer/output.h

		diagnostics/call_context.h
		diagnostics/log_graph.h
		diagnostics/osd_graph.h

		frame/alpha_mode.h
		frame/draw_frame.h
		frame/frame.h
		frame/frame_metadata.h
		frame/frame_factory.h
		frame/frame_transform.h
		frame/frame_visitor.h
		frame/geometry.h
		frame/blend_mask_loader.h
		frame/mesh_loader.h
		frame/pixel_format.h
		frame/write_frame.h

		mixer/audio/audio_mixer.h

		mixer/image/blend_modes.h
		mixer/image/image_mixer.h

		mixer/mixer.h

		monitor/monitor.h

		producer/color/color_producer.h
		producer/separated/separated_producer.h
		producer/transition/transition_producer.h
		producer/transition/sting_producer.h
		producer/route/route_producer.h

		producer/cg_proxy.h
		producer/frame_producer.h
		producer/frame_producer_registry.h
		producer/layer.h
		producer/stage.h

		fwd.h
		module_dependencies.h
		StdAfx.h
		video_channel.h
		video_format.h
		consumer/channel_info.h
)

casparcg_add_library(core SOURCES ${SOURCES} ${HEADERS}
	${tinygltf_SOURCE_DIR}/examples/common/lodepng.cpp
)
# `/w` is MSVC's "no warnings"; gcc's is `-w`. COMPILE_FLAGS is a plain string, so it
# cannot carry a generator expression the way COMPILE_OPTIONS above can -- hence the `if`.
#
# Unguarded, gcc read `/w` as a LINKER INPUT and failed with "/w: linker input file not
# found", naming no flag and no cause. Same class as the `/WX-` on tinyobjloader a few
# lines up, and it is the reason a flag sweep has to include the lowercase form.
if (MSVC)
	set(LODEPNG_NO_WARN "/w")
else ()
	set(LODEPNG_NO_WARN "-w")
endif ()
set_source_files_properties(${tinygltf_SOURCE_DIR}/examples/common/lodepng.cpp PROPERTIES
	COMPILE_FLAGS "${LODEPNG_NO_WARN}"
	SKIP_PRECOMPILE_HEADERS ON
)
target_include_directories(core PRIVATE ..)
target_precompile_headers(core PRIVATE StdAfx.h)
target_link_libraries(core PRIVATE common GLEW::glew sfml-graphics)
target_include_directories(core PRIVATE
	${tinygltf_SOURCE_DIR}
	${tinyobjloader_SOURCE_DIR}
	${tinygltf_SOURCE_DIR}/examples/common
)

source_group(sources ./*)
source_group(sources\\consumer consumer/*)
source_group(sources\\diagnostics diagnostics/*)
source_group(sources\\producer producer/*)
source_group(sources\\monitor monitor/*)
source_group(sources\\frame frame/*)
source_group(sources\\mixer mixer/*)
source_group(sources\\mixer\\audio mixer/audio/*)
source_group(sources\\mixer\\image mixer/image/*)
source_group(sources\\producer\\color producer/color/*)
source_group(sources\\producer\\route producer/route/*)
source_group(sources\\producer\\transition producer/transition/*)
source_group(sources\\producer\\separated producer/separated/*)
