cmake_minimum_required(VERSION 3.28)
project(tracking)

# ---------------------------------------------------------------------------
# Optional VRPN dependency
# ---------------------------------------------------------------------------
option(BUILD_TRACKING_VRPN "Build camera-tracking with VRPN client support" OFF)

if (BUILD_TRACKING_VRPN)
    include(FetchContent)
    # VRPN is licensed under the Boost Software License 1.0 (BSL-1.0),
    # which is compatible with GPL-3. See https://github.com/vrpn/vrpn/blob/master/LICENSE
    FetchContent_Declare(
        vrpn
        GIT_REPOSITORY https://github.com/vrpn/vrpn.git
        GIT_TAG        v07.35
    )
    set(VRPN_BUILD_CLIENTS   ON  CACHE BOOL "" FORCE)
    set(VRPN_BUILD_SERVERS   OFF CACHE BOOL "" FORCE)
    set(VRPN_BUILD_JAVA      OFF CACHE BOOL "" FORCE)
    set(VRPN_BUILD_PYTHON    OFF CACHE BOOL "" FORCE)
    set(VRPN_USE_HID         OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(vrpn)
    message(STATUS "Tracking: VRPN support enabled")
else()
    message(STATUS "Tracking: VRPN support disabled (rebuild with -DBUILD_TRACKING_VRPN=ON)")
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SOURCES
    tracking.cpp
    tracking.h

    camera_data.h
    tracker_binding.h

    tracker_registry.cpp
    tracker_registry.h

    receiver_manager.cpp
    receiver_manager.h

    tracking_commands.cpp
    tracking_commands.h

    protocol/receiver_base.h
    protocol/freed_receiver.cpp
    protocol/freed_receiver.h
    protocol/freed_plus_receiver.cpp
    protocol/freed_plus_receiver.h
    protocol/osc_receiver.cpp
    protocol/osc_receiver.h
    protocol/vrpn_receiver.cpp
    protocol/vrpn_receiver.h
)

casparcg_add_module_project(tracking
    SOURCES   ${SOURCES}
    INIT_FUNCTION   "tracking::init"
    UNINIT_FUNCTION "tracking::uninit"
)

# ---------------------------------------------------------------------------
# VRPN linkage
# ---------------------------------------------------------------------------
if (BUILD_TRACKING_VRPN)
    target_compile_definitions(tracking PRIVATE CASPAR_TRACKING_WITH_VRPN)
    target_link_libraries(tracking PRIVATE vrpn)
    target_include_directories(tracking PRIVATE "${vrpn_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# IDE organisation
# ---------------------------------------------------------------------------
set_target_properties(tracking PROPERTIES FOLDER modules)
source_group(sources\\protocol protocol/*)
source_group(sources ./*)
