cmake_minimum_required(VERSION 3.28)
project(cluster)

# ─── Sources ─────────────────────────────────────────────────────────────────
set(SOURCES
    ptp/ptp_clock.cpp
    ptp/ptp_clock.h
    ptp/ptp_messages.h

    sync/frame_clock.cpp
    sync/frame_clock.h
    sync/command_scheduler.cpp
    sync/command_scheduler.h
    sync/content_sync.cpp
    sync/content_sync.h

    relay/command_relay.cpp
    relay/command_relay.h
    relay/virtual_channel_map.cpp
    relay/virtual_channel_map.h

    cluster_config.cpp
    cluster_config.h
    cluster.cpp
    cluster.h
)

# ─── Module registration ─────────────────────────────────────────────────────
casparcg_add_module_project(cluster
    SOURCES ${SOURCES}
    INIT_FUNCTION "cluster::init"
)

target_include_directories(cluster PRIVATE
    ..
)

set_target_properties(cluster PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\ptp ptp/*)
source_group(sources\\sync sync/*)
source_group(sources\\relay relay/*)

# ─── Link libraries ─────────────────────────────────────────────────────────
target_link_libraries(cluster PRIVATE
    protocol
    ws2_32
)
