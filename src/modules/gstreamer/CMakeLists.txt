cmake_minimum_required (VERSION 3.28)
project (gstreamer)

# The GStreamer development files. The Windows installer only sets
# GSTREAMER_1_0_ROOT_MSVC_X86_64 for an all-users install, so a per-user install has to name
# the root here. This is a build-time path only; at runtime the module resolves the
# installation again from <configuration><gstreamer><path>, which may differ.
set(GSTREAMER_ROOT "$ENV{GSTREAMER_1_0_ROOT_MSVC_X86_64}" CACHE PATH "GStreamer 1.0 installation root")

if (NOT GSTREAMER_ROOT OR NOT EXISTS "${GSTREAMER_ROOT}/include/gstreamer-1.0/gst/gst.h")
	message(FATAL_ERROR
		"ENABLE_GSTREAMER is on but no GStreamer installation was found. Set GSTREAMER_ROOT to "
		"the installation root — the directory holding include/, lib/, bin/ and libexec/.")
endif()

set(SOURCES
		producer/gst_producer.cpp
		producer/gst_producer.h

		util/gst_frame.cpp
		util/gst_frame.h

		util/gst_runtime.cpp
		util/gst_runtime.h

		gstreamer.cpp
		gstreamer.h

		StdAfx.h
)

casparcg_add_module_project(gstreamer
	SOURCES ${SOURCES}
	INIT_FUNCTION "gstreamer::init"
)

target_include_directories(gstreamer PRIVATE
	..
	${FFMPEG_INCLUDE_PATH}
	${GSTREAMER_ROOT}/include/gstreamer-1.0
	${GSTREAMER_ROOT}/include/glib-2.0
	${GSTREAMER_ROOT}/lib/glib-2.0/include
)

target_link_libraries(gstreamer PRIVATE
	ffmpeg
	${GSTREAMER_ROOT}/lib/gstreamer-1.0.lib
	${GSTREAMER_ROOT}/lib/gstapp-1.0.lib
	${GSTREAMER_ROOT}/lib/gstvideo-1.0.lib
	${GSTREAMER_ROOT}/lib/gobject-2.0.lib
	${GSTREAMER_ROOT}/lib/glib-2.0.lib
)

if (MSVC)
	# Delay-loaded, and deliberately: nothing from GStreamer is shipped next to casparcg.exe,
	# so a server on a machine without it must still start. gst_runtime loads the libraries by
	# explicit path before the first call, which is also what keeps GStreamer's own FFmpeg —
	# same base names as ours in 7.x, different in 8.x — off every other process's search path.
	target_link_libraries(gstreamer PRIVATE delayimp.lib)
	target_link_options(gstreamer INTERFACE
		"/DELAYLOAD:gstreamer-1.0-0.dll"
		"/DELAYLOAD:gstapp-1.0-0.dll"
		"/DELAYLOAD:gstvideo-1.0-0.dll"
		"/DELAYLOAD:gobject-2.0-0.dll"
		"/DELAYLOAD:glib-2.0-0.dll"
	)
endif()

target_precompile_headers(gstreamer PRIVATE "StdAfx.h")

set_target_properties(gstreamer PROPERTIES FOLDER modules)
source_group(sources\\producer producer/*)
source_group(sources\\util util/*)
source_group(sources ./*)
