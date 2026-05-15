cmake_minimum_required(VERSION 3.20)
project(ltc)

include(FetchContent)
FetchContent_Declare(
    libltc
    GIT_REPOSITORY https://github.com/x42/libltc.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(libltc)

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
    UNINIT_FUNCTION "ltc::uninit"
)

# Suppress warnings-as-errors for third-party C sources
set_source_files_properties(
    ${LIBLTC_SOURCES}
    PROPERTIES COMPILE_OPTIONS "/W3;/WX-"
)

target_include_directories(ltc PRIVATE 
    ${LIBLTC_ROOT}/src
)

# PortAudio for audio capture (portaudio_static target from portaudio module)
target_link_libraries(ltc PRIVATE portaudio_static)
