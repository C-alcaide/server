cmake_minimum_required(VERSION 3.20)
project(system_audio)

include(FetchContent)
if(NOT miniaudio_POPULATED)
    FetchContent_Declare(
        miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(miniaudio)
endif()

casparcg_add_module_project(system_audio
    SOURCES
        system_audio.cpp
        producer/system_audio_producer.cpp
    INIT_FUNCTION "system_audio::init"
)

target_include_directories(system_audio PRIVATE "${miniaudio_SOURCE_DIR}")
