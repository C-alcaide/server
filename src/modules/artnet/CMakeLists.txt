cmake_minimum_required (VERSION 3.28)
project (artnet)

set(SOURCES
	consumer/artnet_consumer.cpp
	consumer/artnet_consumer.h

	producer/dmx_producer.cpp
	producer/dmx_producer.h

	util/fixture_calculation.h

	artnet.cpp
	artnet.h
)

casparcg_add_module_project(artnet
	SOURCES ${SOURCES}
	INIT_FUNCTION "artnet::init"
)

# The rasteriser and fixture geometry now live in dmx_common, shared with sacn.
target_link_libraries(artnet PRIVATE dmx_common)

set_target_properties(artnet PROPERTIES FOLDER modules)
source_group(sources\\consumer consumer/*)
source_group(sources\\producer producer/*)
source_group(sources\\util util/*)
source_group(sources ./*)

