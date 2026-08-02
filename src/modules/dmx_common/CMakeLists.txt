cmake_minimum_required (VERSION 3.28)
project (dmx_common)

# Shared by the artnet and sacn modules. A plain static library rather than a
# casparcg_add_module_project: this has no init function and registers nothing --
# it is library code, not a module. Listing its sources in both modules' SOURCES
# instead would compile it into two archives linked into the same binary.
set(SOURCES
	fixture_geometry.cpp
	fixture_geometry.h
)

add_library(dmx_common STATIC ${SOURCES})
target_compile_features(dmx_common PRIVATE cxx_std_17)
target_link_libraries(dmx_common PRIVATE common core)
target_include_directories(dmx_common
	PRIVATE
		../..
	INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_include_directories(dmx_common SYSTEM PRIVATE
	${BOOST_INCLUDE_PATH}
)

casparcg_add_build_dependencies(dmx_common)

set_target_properties(dmx_common PROPERTIES FOLDER modules)
source_group(sources ./*)
