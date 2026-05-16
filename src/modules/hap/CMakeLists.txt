cmake_minimum_required(VERSION 3.28)
project(hap LANGUAGES CXX)

# ─── Snappy 1.2.1 (embedded source build) ───────────────────────────────────
# We compile Snappy's 4 source files directly into the hap module target to
# avoid FetchContent complications (CasparCG's global /FI and /WX flags leak
# into FetchContent targets and break the build).
#
# The snappy/ subdirectory contains pre-generated config.h and
# snappy-stubs-public.h for MSVC x64.  The actual .cc/.h source files are
# fetched once by FetchContent (download only, no build) and compiled as part
# of this target with appropriate flag overrides.
include(FetchContent)
FetchContent_Declare(snappy_src
    GIT_REPOSITORY https://github.com/google/snappy.git
    GIT_TAG        1.2.1
)
FetchContent_Populate(snappy_src)

set(SNAPPY_SOURCES
    ${snappy_src_SOURCE_DIR}/snappy.cc
    ${snappy_src_SOURCE_DIR}/snappy-c.cc
    ${snappy_src_SOURCE_DIR}/snappy-sinksource.cc
    ${snappy_src_SOURCE_DIR}/snappy-stubs-internal.cc
)

# Disable warnings-as-errors and the CasparCG forced-include for Snappy sources.
set_source_files_properties(${SNAPPY_SOURCES} PROPERTIES
    COMPILE_FLAGS "/W0"
    SKIP_PRECOMPILE_HEADERS ON
)

# ─── bcdec (header-only BCn decompression library, MIT/Unlicense) ────────────
FetchContent_Declare(bcdec_src
    GIT_REPOSITORY https://github.com/iOrange/bcdec.git
    GIT_TAG        main
)
FetchContent_Populate(bcdec_src)

# ─── Source lists ────────────────────────────────────────────────────────────
set(CPU_SOURCES
    producer/hap_demuxer.cpp
    producer/hap_producer.cpp
    gl/hap_gl_decode.cpp
    cpu/hap_cpu_decode.cpp
    hap.cpp
)

set(HEADER_SOURCES
    hap.h
    producer/hap_demuxer.h
    producer/hap_producer.h
    gl/hap_gl_decode.h
    cpu/hap_cpu_decode.h
    util/hap_frame_parser.h
)

set(ALL_SOURCES ${CPU_SOURCES} ${HEADER_SOURCES} ${SNAPPY_SOURCES})

# ─── CasparCG module target ──────────────────────────────────────────────────
casparcg_add_module_project(hap
    SOURCES ${ALL_SOURCES}
    INIT_FUNCTION "hap::init"
)

# ─── Include paths ───────────────────────────────────────────────────────────
# Snappy's own headers #include "config.h" and "snappy-stubs-public.h";
# point at our pre-generated copies first, then the snappy source dir.
target_include_directories(hap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/snappy   # pre-generated config.h + stubs
    ${snappy_src_SOURCE_DIR}             # snappy.h, snappy-sinksource.h, etc.
    ${bcdec_src_SOURCE_DIR}              # bcdec.h (header-only)
    ${FFMPEG_INCLUDE_PATH}
)

# ─── Link libraries ──────────────────────────────────────────────────────────
target_link_libraries(hap PRIVATE
    ffmpeg
    GLEW::glew
)
IF(ENABLE_VULKAN)
    target_link_libraries(hap PRIVATE Vulkan::Headers)
ENDIF()
