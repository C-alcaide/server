cmake_minimum_required (VERSION 3.28)

add_executable(bin2c bin2c.cpp)
target_compile_features(bin2c PRIVATE cxx_std_20)

target_include_directories(bin2c PRIVATE ..)

# Display blanker companion tool — black windows on output monitors
#
# WINDOWS ONLY, and guarded for two independent reasons — either one alone breaks a Linux
# build, and the second is not fixable by adding a file:
#
#  1. The sources sit OUTSIDE this directory, at the repository root rather than under
#     `src/`. `tools/linux/Dockerfile` — which is what `tools/linux/build-in-docker` and
#     therefore the Linux CI job run — does `COPY ./src /source`, so `../../tools/` does
#     not exist in the image at all. Unguarded, this failed at cmake's GENERATE step with
#     "Cannot find source file: ../../tools/display_blanker.cpp", before a single
#     translation unit was compiled.
#  2. It is a `WIN32` executable linking `user32 gdi32 shell32 comctl32 advapi32 dwmapi`.
#     Even with the sources present it could not link on Linux, so the guard is what this
#     target actually means rather than a way around fault 1.
#
# This is the same gate `src/modules/CMakeLists.txt` puts around spout, flash, bluefish and
# replay. It went unnoticed because no VP branch has ever been through the Linux CI job.
if (MSVC)
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
endif ()

function(bin2c source_file dest_file namespace obj_name)
    ADD_CUSTOM_COMMAND(
        OUTPUT ${dest_file}
        COMMAND bin2c ${namespace} ${obj_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} > ${CMAKE_CURRENT_BINARY_DIR}/${dest_file}
        DEPENDS bin2c ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
    )
endfunction()

function(bin2c_file source_file dest_file namespace obj_name)
    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${dest_file}
        COMMAND bin2c ${namespace} ${obj_name} ${source_file} > ${CMAKE_CURRENT_BINARY_DIR}/${dest_file}
        DEPENDS bin2c ${source_file}
    )
endfunction()
