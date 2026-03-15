cmake_minimum_required(VERSION 3.20)
project(system_audio)

casparcg_add_module_project(system_audio
    SOURCES 
        system_audio.cpp
        producer/system_audio_producer.cpp
    INIT_FUNCTION "system_audio::init"
)