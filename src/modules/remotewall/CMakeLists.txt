cmake_minimum_required(VERSION 3.28)
project(remotewall)

# Native cloudXR tile-wall producer. Phase 0 is plain C++ (module scaffold + a
# placeholder producer). Phases 1+ add the vendored recv_wall receiver (CUDA +
# NVIDIA Video Codec SDK) and the CUDA<->VK/GL zero-copy hand-off.

set(SOURCES
    remotewall.cpp
    producer/remotewall_producer.cpp
)

set(HEADERS
    remotewall.h
    producer/remotewall_producer.h
)

casparcg_add_module_project(remotewall
    SOURCES ${SOURCES} ${HEADERS}
    INIT_FUNCTION "remotewall::init"
)

target_include_directories(remotewall PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(remotewall PROPERTIES FOLDER modules)
source_group(sources\\producer producer/*)
source_group(sources ./*)
