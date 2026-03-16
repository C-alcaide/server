cmake_minimum_required(VERSION 3.28)
project(vmx)

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
    vmx_segmented_storage.cpp
    vmx_segmented_storage.h
    vmx_extended_index.h
    vmx_file_operations.cpp
    vmx_file_operations.h
    vmx_consumer.cpp
    vmx_consumer.h
    vmx_producer.cpp
    vmx_producer.h
    vmx.cpp
    vmx.h
)

casparcg_add_module_project(vmx
    SOURCES ${SOURCES}
    INIT_FUNCTION "vmx::init"
    UNINIT_FUNCTION "vmx::uninit"
)

target_include_directories(vmx PRIVATE
    ${CMAKE_SOURCE_DIR}/../../
    ${LIBVMX_ROOT}/src
)

target_link_libraries(vmx PRIVATE
    core
    common
    libvmx
)

set_target_properties(vmx PROPERTIES FOLDER modules)
