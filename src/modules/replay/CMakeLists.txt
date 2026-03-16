cmake_minimum_required(VERSION 3.28)
project(replay)

include(FetchContent)
FetchContent_Declare(
    libvmx_ext
    GIT_REPOSITORY https://github.com/openmediatransport/libvmx.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(libvmx_ext)
set(LIBVMX_ROOT "${libvmx_ext_SOURCE_DIR}")

# Include sources to compile libvmx natively
add_library(libvmx STATIC
    ${LIBVMX_ROOT}/src/vmxcodec.cpp
    ${LIBVMX_ROOT}/src/vmxcodec_x86.cpp
    ${LIBVMX_ROOT}/src/vmxcodec_avx2.cpp
)
target_include_directories(libvmx PUBLIC ${LIBVMX_ROOT}/src)
target_include_directories(libvmx PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(libvmx PRIVATE "/w" "/WX-")

if(MSVC)
    set_source_files_properties(${LIBVMX_ROOT}/src/vmxcodec_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
endif()

set(SOURCES
    replay_extended_index.h
    replay_segmented_storage.cpp
    replay_segmented_storage.h
    replay_file_operations.cpp
    replay_file_operations.h
    replay_consumer.cpp
    replay_consumer.h
    replay_producer.cpp
    replay_producer.h
    replay.cpp
    replay.h
)

casparcg_add_module_project(replay
    SOURCES ${SOURCES}
    INIT_FUNCTION "replay::init"
    UNINIT_FUNCTION "replay::uninit"
)

target_include_directories(replay PRIVATE
    ${CMAKE_SOURCE_DIR}/../../
    ${LIBVMX_ROOT}/src
)

target_link_libraries(replay PRIVATE
    core
    common
    libvmx
)

set_target_properties(replay PROPERTIES FOLDER modules)
