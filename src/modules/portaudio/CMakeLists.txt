cmake_minimum_required(VERSION 3.28)
project(portaudio_module)

include(FetchContent)

# Fetch PortAudio
FetchContent_Declare(
    portaudio
    GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
    GIT_TAG        v19.7.0
)

# Configure PortAudio options before making available
set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Only build static library to avoid target name clash with our module
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# PortAudio v19.7.0 uses cmake_minimum_required(VERSION 3.0) which CMake 4.x rejects.
# Allow it to configure with a minimum policy version of 3.5.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# On Windows, enable WASAPI and DirectSound; ASIO requires the SDK
if(WIN32)
    set(PA_USE_WASAPI ON CACHE BOOL "" FORCE)
    set(PA_USE_DS ON CACHE BOOL "" FORCE)
    set(PA_USE_WMME ON CACHE BOOL "" FORCE)

    # ASIO SDK: set PA_USE_ASIO=ON and ASIOSDK_ROOT_DIR to the Steinberg ASIO SDK path
    # Users must download the ASIO SDK separately from Steinberg.
    if(DEFINED ASIOSDK_ROOT_DIR AND EXISTS "${ASIOSDK_ROOT_DIR}")
        set(PA_USE_ASIO ON CACHE BOOL "" FORCE)
        set(PA_ASIO_SDK_PATH "${ASIOSDK_ROOT_DIR}" CACHE PATH "" FORCE)
        message(STATUS "PortAudio: ASIO SDK found at ${ASIOSDK_ROOT_DIR}")
    else()
        set(PA_USE_ASIO OFF CACHE BOOL "" FORCE)
        message(STATUS "PortAudio: ASIO SDK not found (set ASIOSDK_ROOT_DIR to enable)")
    endif()
endif()

FetchContent_MakeAvailable(portaudio)

set(SOURCES
    portaudio_module.cpp
    portaudio_module.h

    consumer/portaudio_consumer.cpp
    consumer/portaudio_consumer.h

    producer/portaudio_producer.cpp
    producer/portaudio_producer.h

    util/portaudio_device.cpp
    util/portaudio_device.h
    util/spsc_ring_buffer.h
)

casparcg_add_module_project(portaudio_io
    NAME "portaudio"
    HEADER_FILE "modules/portaudio/portaudio_module.h"
    SOURCES ${SOURCES}
    INIT_FUNCTION "portaudio::init"
    UNINIT_FUNCTION "portaudio::uninit"
)

target_include_directories(portaudio_io PRIVATE
    ${portaudio_SOURCE_DIR}/include
)

target_link_libraries(portaudio_io PRIVATE portaudio_static)

set_target_properties(portaudio_io PROPERTIES FOLDER modules)
source_group(sources\\consumer consumer/*)
source_group(sources\\producer producer/*)
source_group(sources\\util util/*)
source_group(sources ./*)
