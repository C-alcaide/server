cmake_minimum_required(VERSION 3.28)
project(keyframes)

set(SOURCES
    keyframes.cpp
    keyframes.h

    keyframe_data.cpp
    keyframe_data.h

    keyframe_registry.cpp
    keyframe_registry.h

    keyframe_json.cpp
    keyframe_json.h

    keyframe_commands.cpp
    keyframe_commands.h
)

casparcg_add_module_project(keyframes
    SOURCES         ${SOURCES}
    INIT_FUNCTION   "keyframes::init"
    UNINIT_FUNCTION "keyframes::uninit"
)

set_target_properties(keyframes PROPERTIES FOLDER modules)
source_group(sources ./)
