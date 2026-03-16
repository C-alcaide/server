cmake_minimum_required(VERSION 3.28)
project(spout)

if(NOT DEFINED SPOUT_SDK_DIR)
    include(FetchContent)
    FetchContent_Declare(
        spout_sdk
        GIT_REPOSITORY https://github.com/leadedge/Spout2.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(spout_sdk)
    set(SPOUT_SDK_DIR "${spout_sdk_SOURCE_DIR}/SPOUTSDK")
endif()

set(SOURCES
    spout.cpp
    spout.h
    consumer/spout_consumer.cpp
    consumer/spout_consumer.h
    producer/spout_producer.cpp
    producer/spout_producer.h
    ${SPOUT_SDK_DIR}/SpoutGL/Spout.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutGL.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutDirectX.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutSenderNames.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutSender.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutReceiver.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutFrameCount.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutUtils.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutSharedMemory.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutGLextensions.cpp
    ${SPOUT_SDK_DIR}/SpoutGL/SpoutCopy.cpp
)

casparcg_add_module_project(spout
    SOURCES ${SOURCES}
    INIT_FUNCTION "spout::init"
)


target_include_directories(spout PRIVATE
    ${SPOUT_SDK_DIR}
    ${SPOUT_SDK_DIR}/SpoutGL
    ${FFMPEG_INCLUDE_PATH}
)

target_link_libraries(spout PRIVATE
    opengl32
    glu32
    core
    common
    ffmpeg
)
