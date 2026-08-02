cmake_minimum_required (VERSION 3.28)
project (sacn)

set(SOURCES
	consumer/sacn_consumer.cpp
	consumer/sacn_consumer.h

	util/fixture_calculation.h

	sacn.cpp
	sacn.h
)

casparcg_add_module_project(sacn
	SOURCES ${SOURCES}
	INIT_FUNCTION "sacn::init"
)

# The rasteriser and fixture geometry now live in dmx_common, shared with artnet.
target_link_libraries(sacn PRIVATE dmx_common)

set_target_properties(sacn PROPERTIES FOLDER modules)
source_group(sources\\consumer consumer/*)
source_group(sources\\util util/*)
source_group(sources ./*)
