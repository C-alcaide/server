cmake_minimum_required(VERSION 3.20)
project(ltc)

include(FetchContent)
FetchContent_Declare(
    libltc
    GIT_REPOSITORY https://github.com/x42/libltc.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(libltc)

if(NOT miniaudio_POPULATED)
    FetchContent_Declare(
        miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(miniaudio)
endif()

set(LIBLTC_ROOT "${libltc_SOURCE_DIR}")

# Add libltc sources directly to avoid external build steps
set(LIBLTC_SOURCES
    ${LIBLTC_ROOT}/src/ltc.c
    ${LIBLTC_ROOT}/src/timecode.c
    ${LIBLTC_ROOT}/src/decoder.c
    ${LIBLTC_ROOT}/src/encoder.c
)

casparcg_add_module_project(ltc
    SOURCES 
        ltc_input.cpp
        ${LIBLTC_SOURCES}
    INIT_FUNCTION "ltc::init"
)

target_include_directories(ltc PRIVATE 
    ${LIBLTC_ROOT}/src
    "${miniaudio_SOURCE_DIR}"
)
