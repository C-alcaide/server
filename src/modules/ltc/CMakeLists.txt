cmake_minimum_required(VERSION 3.20)
project(ltc)

# Point this to your libltc path
set(LIBLTC_ROOT "../../../../../libltc")

# Add libltc sources directly to avoid external build steps
set(LIBLTC_SOURCES
    ${LIBLTC_ROOT}/src/ltc.c
    ${LIBLTC_ROOT}/src/timecode.c
    ${LIBLTC_ROOT}/src/decoder.c
)

casparcg_add_module_project(ltc
    SOURCES
        ltc_input.cpp
        ${LIBLTC_SOURCES}
    INIT_FUNCTION "ltc::init"
    HEADER_FILE "modules/ltc/ltc_input.h"
)

target_include_directories(ltc PRIVATE ${LIBLTC_ROOT}/src)
target_compile_options(ltc PRIVATE /wd4244 /wd4267 /wd4146 /wd4101 /wd4828 /wd4018 /wd4996 /wd4389 /wd4756 /wd4245 /wd4018)

