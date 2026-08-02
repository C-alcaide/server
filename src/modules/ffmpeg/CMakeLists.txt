cmake_minimum_required (VERSION 3.28)
project (ffmpeg)

set(SOURCES
	producer/av_producer.cpp
	producer/av_producer.h
	producer/av_input.cpp
	producer/av_input.h
	producer/ffmpeg_producer.cpp
	producer/ffmpeg_producer.h
	consumer/ffmpeg_consumer.cpp
	consumer/ffmpeg_consumer.h
	consumer/cuda_gl_upload.cpp
	consumer/cuda_gl_upload.h
	consumer/cuda_vk_upload.cpp
	consumer/cuda_vk_upload.h

	util/av_util.cpp
	util/av_util.h
	util/audio_resampler.cpp
	util/audio_resampler.h
	util/av_assert.h

	ffmpeg.cpp
	ffmpeg.h
	StdAfx.h
)

casparcg_add_module_project(ffmpeg
	SOURCES ${SOURCES}
	INIT_FUNCTION "ffmpeg::init"
	UNINIT_FUNCTION "ffmpeg::uninit"
)

target_include_directories(ffmpeg PRIVATE
    ${FFMPEG_INCLUDE_PATH}
    )
target_precompile_headers(ffmpeg PRIVATE "StdAfx.h")

set_target_properties(ffmpeg PROPERTIES FOLDER modules)
source_group(sources ./*)
source_group(sources\\consumer ./consumer/.*)
source_group(sources\\producer ./producer/.*)
source_group(sources\\util ./util/.*)

if (MSVC)
	target_link_libraries(ffmpeg PRIVATE
		# GLEW + accelerator for D3D11→GL GPU-direct producer path
		GLEW::glew
		accelerator
		opengl32
		# D3DCompile, for the NV12 plane-extraction shaders in that path
		d3dcompiler
		# Definitions of the D3D11 decoder-profile GUIDs (d3d11.h only declares
		# them), used to ask the adapter whether it decodes AV1.
		dxguid
	)

	# CUDA runtime only -- no kernels, so no CUDA language and no nvcc. The
	# consumer's GPU-direct NVENC path copies the mixer's texture into a CUDA
	# frame with the runtime API; NVENC accepts RGB input, so no colour
	# conversion is needed and there is nothing to write a kernel for.
	find_package(CUDAToolkit QUIET)
	if (CUDAToolkit_FOUND)
		target_include_directories(ffmpeg PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
		# cuda_driver too: the copies push the encoder's CUcontext, which is a
		# driver-API call.
		target_link_libraries(ffmpeg PRIVATE CUDA::cudart_static CUDA::cuda_driver)
		target_compile_definitions(ffmpeg PRIVATE CASPAR_FFMPEG_HAS_CUDA=1)

	else()
		message(STATUS "ffmpeg: CUDA toolkit not found - NVENC GPU-direct recording disabled")
	endif()
else()
	target_link_libraries(ffmpeg PRIVATE
			${FFMPEG_LIBRARIES}
	)
endif()
