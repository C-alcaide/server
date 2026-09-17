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
    # ── THE LIBRARY, NOT THE EXAMPLE PROGRAMS ──────────────────────────────────────────
    #
    # `VRPN_BUILD_CLIENTS` builds VRPN's example client APPS and tests; the thing this module
    # links is `VRPN_BUILD_CLIENT_LIBRARY`. Setting only the first made VRPN's own CMake stop
    # with *"Cannot compile clients without either the client or full library"*, so
    # `-DBUILD_TRACKING_VRPN=ON` did not configure AT ALL -- measured 2026-09-17, the first time
    # anyone turned it on.
    #
    # It went unnoticed because the option defaults OFF and nothing in the tree or the harness
    # ever set it: the receiver's `#ifdef CASPAR_TRACKING_WITH_VRPN` branch had never been
    # compiled either. **An option nobody exercises is not a supported option, it is a claim** --
    # and this one was being quoted in a log line that told operators to rebuild with it.
    set(VRPN_BUILD_CLIENT_LIBRARY ON  CACHE BOOL "" FORCE)
    set(VRPN_BUILD_SERVER_LIBRARY OFF CACHE BOOL "" FORCE)
    # The example apps and servers are not wanted -- only the library this module links.
    set(VRPN_BUILD_CLIENTS   OFF CACHE BOOL "" FORCE)
    set(VRPN_BUILD_SERVERS   OFF CACHE BOOL "" FORCE)
    set(VRPN_BUILD_JAVA      OFF CACHE BOOL "" FORCE)
    set(VRPN_BUILD_PYTHON    OFF CACHE BOOL "" FORCE)
    set(VRPN_USE_HID         OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(vrpn)

    # ── THE SAME TREATMENT `openfx_host` GETS, AND FOR THE SAME TWO REASONS ────────────
    #
    # `Bootstrap_Windows.cmake` puts `/FIcommon/compiler/vs/disable_silly_warnings.h` in the
    # GLOBAL `CMAKE_CXX_FLAGS` as a RELATIVE path, so every out-of-tree target inherits a forced
    # include it cannot resolve. VRPN's own sources then failed with `C1083` on a CasparVP
    # header they have never heard of -- measured 2026-09-17, and it is why `-DBUILD_TRACKING_
    # VRPN=ON` had never produced a binary even once the option configured.
    #
    # And VRPN is third-party C++98-era code that does not compile clean under this tree's
    # `/W4 /WX`, exactly as OpenFX's HostSupport does not. `Bootstrap_Windows.cmake` already
    # solves both for that dependency; this is the same solution, not a new one.
    foreach(_vrpn_tgt vrpn quat vrpnserver)
        if (TARGET ${_vrpn_tgt})
            target_include_directories(${_vrpn_tgt} PRIVATE ${CMAKE_SOURCE_DIR})
            # `/UUNICODE /U_UNICODE` for the same reason `openfx_host` needs them: VRPN uses the
            # TCHAR Win32 APIs and assumes an ANSI build, while this tree defines UNICODE
            # globally, so `LPTSTR` stops converting (`C2440`). Third instance of the same
            # third-party integration shape, and the third time the answer is the one
            # `Bootstrap_Windows.cmake` already wrote down.
            target_compile_options(${_vrpn_tgt} PRIVATE /W0 /WX- /UUNICODE /U_UNICODE)
            # VRPN's own CMake adds a Vista-era `_WIN32_WINNT`/`NTDDI_VERSION` PAIR while this
            # tree's global flags say Windows 10, and `sdkddkver.h` refuses the mismatch
            # (`C1189: NTDDI_VERSION setting conflicts with _WIN32_WINNT setting`). Target-level
            # definitions land last on the command line, so restating BOTH here is what settles
            # it -- setting only one leaves the pair inconsistent, which is the error itself.
            target_compile_definitions(${_vrpn_tgt} PRIVATE _WIN32_WINNT=0x0A00 NTDDI_VERSION=0x0A000000)
        endif()
    endforeach()

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

    lens_profile.cpp
    lens_profile.h

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
    protocol/opentrackio_receiver.cpp
    protocol/opentrackio_receiver.h
    protocol/psn_receiver.cpp
    protocol/psn_receiver.h
    protocol/vrpn_receiver.cpp
    protocol/vrpn_receiver.h
)

casparcg_add_module_project(tracking
    SOURCES   ${SOURCES}
    INIT_FUNCTION   "tracking::init"
    UNINIT_FUNCTION "tracking::uninit"
)

# ---------------------------------------------------------------------------
# Accelerator linkage (for previz camera integration)
# ---------------------------------------------------------------------------
target_link_libraries(tracking PRIVATE accelerator)

# ---------------------------------------------------------------------------
# VRPN linkage
# ---------------------------------------------------------------------------
if (BUILD_TRACKING_VRPN)
    target_compile_definitions(tracking PRIVATE CASPAR_TRACKING_WITH_VRPN)
    target_link_libraries(tracking PRIVATE vrpn)
    target_include_directories(tracking PRIVATE "${vrpn_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# PSN (PosiStageNet) — header-only, MIT licensed, vendored in vendor/psn/
# ---------------------------------------------------------------------------
target_include_directories(tracking PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/psn")

# ---------------------------------------------------------------------------
# IDE organisation
# ---------------------------------------------------------------------------
set_target_properties(tracking PROPERTIES FOLDER modules)
source_group(sources\\protocol protocol/*)
source_group(sources ./*)
