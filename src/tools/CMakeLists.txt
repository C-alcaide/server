cmake_minimum_required (VERSION 3.28)

add_executable(bin2c bin2c.cpp)
target_compile_features(bin2c PRIVATE cxx_std_17)

target_include_directories(bin2c PRIVATE ..)

# Display blanker companion tool — black windows on output monitors
# Needs the parent include dir for the forced /FI header from CMAKE_CXX_FLAGS
add_executable(display_blanker WIN32
    ../../tools/display_blanker.cpp
    ../../tools/display_blanker.rc
)
target_compile_features(display_blanker PRIVATE cxx_std_17)
target_link_libraries(display_blanker PRIVATE user32 gdi32 shell32 comctl32 advapi32 dwmapi)
target_include_directories(display_blanker PRIVATE ..)
set_target_properties(display_blanker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shell"
)

function(bin2c source_file dest_file namespace obj_name)
    ADD_CUSTOM_COMMAND(
        OUTPUT ${dest_file}
        COMMAND bin2c ${namespace} ${obj_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} > ${CMAKE_CURRENT_BINARY_DIR}/${dest_file}
        DEPENDS bin2c ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
    )
endfunction()
