cmake_minimum_required(VERSION 3.28)
project(vmx)

set(LIBVMX_ROOT "D:/Github/libvmx" CACHE PATH "Path to the libvmx root directory")

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

add_library(libvmx STATIC IMPORTED GLOBAL)
set_target_properties(libvmx PROPERTIES
    IMPORTED_LOCATION         "${LIBVMX_ROOT}/build/libvmx.lib"
    IMPORTED_LOCATION_RELEASE "${LIBVMX_ROOT}/build/Release/libvmx.lib"
    IMPORTED_LOCATION_DEBUG   "${LIBVMX_ROOT}/build/libvmx.lib"
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
