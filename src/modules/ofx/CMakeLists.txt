cmake_minimum_required (VERSION 3.28)
project (ofx)

set(SOURCES
	bridge/ofx_image_bridge.cpp
	bridge/ofx_image_bridge.h

	host/ofx_clip_instance.cpp
	host/ofx_clip_instance.h
	host/ofx_cuda_render.cpp
	host/ofx_cuda_render.h
	host/ofx_effect_instance.cpp
	host/ofx_effect_instance.h
	host/ofx_gl_render.cpp
	host/ofx_gl_render.h
	host/ofx_host.cpp
	host/ofx_host.h
	host/ofx_includes.h
	host/ofx_param_instance.cpp
	host/ofx_param_instance.h

	producer/ofx_producer.cpp
	producer/ofx_producer.h

	ofx.cpp
	ofx.h
)

casparcg_add_module_project(ofx
	SOURCES ${SOURCES}
	INIT_FUNCTION "ofx::init"
)

if (TARGET openfx_host)
	target_link_libraries(ofx PRIVATE openfx_host)
	target_compile_definitions(ofx PRIVATE CASPAR_OFX_ENABLED)
	# OpenGL render backend (self-contained offscreen context via SFML + GLEW).
	target_link_libraries(ofx PRIVATE GLEW::glew sfml-window sfml-system)
	# CUDA render backend (Windows) — runtime host API only (cudaMalloc/cudaMemcpy), no nvcc.
	find_package(CUDAToolkit QUIET)
	if (CUDAToolkit_FOUND)
		target_compile_definitions(ofx PRIVATE CASPAR_OFX_CUDA)
		target_link_libraries(ofx PRIVATE CUDA::cudart_static)
		# NPP for on-device source/output convert (channel swap, vertical mirror, alpha premultiply).
		target_link_libraries(ofx PRIVATE CUDA::nppc CUDA::nppidei CUDA::nppig CUDA::nppial)
		message(STATUS "ofx: CUDA render backend enabled")
		# CUDA<->Vulkan zero-copy interop (VkImage exported to CUDA) for the Vulkan mixer.
		if (ENABLE_VULKAN)
			target_compile_definitions(ofx PRIVATE CASPAR_OFX_VULKAN_CUDA)
			target_link_libraries(ofx PRIVATE Vulkan::Headers)
			message(STATUS "ofx: CUDA-Vulkan zero-copy interop enabled")
		endif ()
	else ()
		message(STATUS "ofx: CUDA toolkit not found; CUDA render backend disabled")
	endif ()
endif ()

set_target_properties(ofx PROPERTIES FOLDER modules)
source_group(sources\\bridge bridge/*)
source_group(sources\\host host/*)
source_group(sources\\producer producer/*)
source_group(sources ./*)
