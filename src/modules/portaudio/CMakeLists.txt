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
    set(PA_USE_WDMKS OFF CACHE BOOL "" FORCE)  # Avoid GUID_NULL conflict with uuid.lib

    # ASIO. Steinberg dual-licensed the SDK under GPLv3 on 2025-10-15 (SDK 2.3.4), so it
    # can now be fetched and linked by a GPLv3 project like this one; before that it was
    # proprietary-only and no GPL build could ship it. The SDK's own LICENSE.txt carries
    # the dual grant. Set -DENABLE_ASIO=OFF to build without it, or point ASIOSDK_ROOT_DIR
    # at a local copy to skip the download.
    option(ENABLE_ASIO "Build PortAudio with ASIO support (Steinberg SDK, GPLv3 option)" ON)

    if(ENABLE_ASIO AND NOT (DEFINED ASIOSDK_ROOT_DIR AND EXISTS "${ASIOSDK_ROOT_DIR}"))
        # Pinned by version and hash rather than the /asiosdk redirect, which always points
        # at the newest release and would change the build under us.
        FetchContent_Declare(
            asiosdk
            URL      https://download.steinberg.net/sdk_downloads/ASIO-SDK_2.3.4_2025-10-15.zip
            URL_HASH SHA256=d5ebf0c20dd2c5f43771fd0c1418f4b361bf52434ee670097cfa6b3a335e2eca
        )
        # Populate only -- the archive ships no CMakeLists.txt, so this unpacks and stops.
        FetchContent_MakeAvailable(asiosdk)
        set(ASIOSDK_ROOT_DIR "${asiosdk_SOURCE_DIR}" CACHE PATH "" FORCE)
    endif()

    if(ENABLE_ASIO AND EXISTS "${ASIOSDK_ROOT_DIR}/common/asio.h")
        set(PA_USE_ASIO ON CACHE BOOL "" FORCE)
        set(PA_ASIO_SDK_PATH "${ASIOSDK_ROOT_DIR}" CACHE PATH "" FORCE)
        message(STATUS "PortAudio: ASIO SDK at ${ASIOSDK_ROOT_DIR}")
    else()
        set(PA_USE_ASIO OFF CACHE BOOL "" FORCE)
        message(STATUS "PortAudio: building without ASIO (ENABLE_ASIO=${ENABLE_ASIO})")
    endif()
endif()

FetchContent_MakeAvailable(portaudio)

if(WIN32 AND TARGET portaudio_static)
    # The global CMAKE_CXX_FLAGS force-includes common/compiler/vs/disable_silly_warnings.h
    # by a path relative to the source root, so an out-of-tree target cannot resolve it.
    # Same fix as openfx_host and tinyobjloader.
    target_include_directories(portaudio_static PRIVATE ${CMAKE_SOURCE_DIR})
    # PortAudio's ASIO host API and Steinberg's SDK are third-party and predate /W4 /WX by
    # about two decades. Scoped to this target so the fork's own code keeps the strict flags.
    #
    # /UUNICODE /U_UNICODE for the same reason openfx_host needs it: asiolist.cpp calls the
    # TCHAR Win32 APIs (RegOpenKey, CharLowerBuff) with char* and assumes an ANSI build, so
    # the global UNICODE resolves them to the W variants and every call fails to convert.
    target_compile_options(portaudio_static PRIVATE /W0 /WX- /UUNICODE /U_UNICODE)
endif()

set(SOURCES
    portaudio_module.cpp
    portaudio_module.h

    consumer/portaudio_consumer.cpp
    consumer/portaudio_consumer.h

    producer/portaudio_producer.cpp
    producer/portaudio_producer.h

    util/portaudio_device.cpp
    util/portaudio_device.h
    util/shared_capture.h
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
