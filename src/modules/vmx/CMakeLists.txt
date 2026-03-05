cmake_minimum_required(VERSION 3.28)
project(vmx)

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

include_directories(d:/Github/libvmx/src)


casparcg_add_module_project(vmx
    SOURCES ${SOURCES}
    INIT_FUNCTION "vmx::init"
    UNINIT_FUNCTION "vmx::uninit"
)


add_library(libvmx STATIC IMPORTED GLOBAL)
set_target_properties(libvmx PROPERTIES IMPORTED_LOCATION "D:/Github/libvmx/build/libvmx.lib")

target_link_libraries(vmx PRIVATE libvmx)


target_include_directories(vmx PRIVATE
    ${CMAKE_SOURCE_DIR}/../../
    D:/Github/libvmx/src
)

target_link_libraries(vmx PRIVATE
    core
    common
    D:/Github/libvmx/build/Release/libvmx.lib
)

set_target_properties(vmx PROPERTIES FOLDER modules)
