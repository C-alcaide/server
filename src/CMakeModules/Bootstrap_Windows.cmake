cmake_minimum_required (VERSION 3.28)

include(ExternalProject)
include(FetchContent)

if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()
# Prefer the new boost helper
if(POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()

message(STATUS "CHECKPOINT: Bootstrap start")

set(BOOST_USE_PRECOMPILED ON CACHE BOOL "Use precompiled boost")
set(ENABLE_VULKAN OFF CACHE BOOL "Enable Vulkan support")

set(CASPARCG_RUNTIME_DEPENDENCIES_RELEASE "" CACHE INTERNAL "")
set(CASPARCG_RUNTIME_DEPENDENCIES_DEBUG "" CACHE INTERNAL "")
set(CASPARCG_RUNTIME_DEPENDENCIES_RELEASE_DIRS "" CACHE INTERNAL "")
set(CASPARCG_RUNTIME_DEPENDENCIES_DEBUG_DIRS "" CACHE INTERNAL "")

message(STATUS "CHECKPOINT: Bootstrap functions defined")

function(casparcg_add_runtime_dependency FILE_TO_COPY)
	if ("${ARGV1}" STREQUAL "Release" OR NOT ARGV1)
		set(CASPARCG_RUNTIME_DEPENDENCIES_RELEASE "${CASPARCG_RUNTIME_DEPENDENCIES_RELEASE}" "${FILE_TO_COPY}" CACHE INTERNAL "")
	endif()
	if ("${ARGV1}" STREQUAL "Debug" OR NOT ARGV1)
		set(CASPARCG_RUNTIME_DEPENDENCIES_DEBUG "${CASPARCG_RUNTIME_DEPENDENCIES_DEBUG}" "${FILE_TO_COPY}" CACHE INTERNAL "")
	endif()
endfunction()
function(casparcg_add_runtime_dependency_dir FILE_TO_COPY)
	if ("${ARGV1}" STREQUAL "Release" OR NOT ARGV1)
		set(CASPARCG_RUNTIME_DEPENDENCIES_RELEASE_DIRS "${CASPARCG_RUNTIME_DEPENDENCIES_RELEASE_DIRS}" "${FILE_TO_COPY}" CACHE INTERNAL "")
	endif()
	if ("${ARGV1}" STREQUAL "Debug" OR NOT ARGV1)
		set(CASPARCG_RUNTIME_DEPENDENCIES_DEBUG_DIRS "${CASPARCG_RUNTIME_DEPENDENCIES_DEBUG_DIRS}" "${FILE_TO_COPY}" CACHE INTERNAL "")
	endif()
endfunction()
function(casparcg_add_runtime_dependency_from_target TARGET)
	get_target_property(_runtime_lib_name ${TARGET} IMPORTED_LOCATION_RELEASE)
	if (NOT "${_runtime_lib_name}" STREQUAL "")
		set(CASPARCG_RUNTIME_DEPENDENCIES_RELEASE "${CASPARCG_RUNTIME_DEPENDENCIES_RELEASE}" "${_runtime_lib_name}" CACHE INTERNAL "")
	endif()

	get_target_property(_runtime_lib_name ${TARGET} IMPORTED_LOCATION_DEBUG)
	if (NOT "${_runtime_lib_name}" STREQUAL "")
		set(CASPARCG_RUNTIME_DEPENDENCIES_DEBUG "${CASPARCG_RUNTIME_DEPENDENCIES_DEBUG}" "${_runtime_lib_name}" CACHE INTERNAL "")
	endif()
endfunction()

casparcg_add_runtime_dependency("${PROJECT_SOURCE_DIR}/shell/casparcg.config")

# BOOST
message(STATUS "CHECKPOINT: Adding Boost")
casparcg_add_external_project(boost)
if (BOOST_USE_PRECOMPILED)
	ExternalProject_Add(boost
	URL ${CASPARCG_DOWNLOAD_MIRROR}/boost/boost_1_83_0-win32-x64-debug-release.zip
	URL_HASH MD5=0b9990a24259867c8c04ae30c423f86b
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	)
	ExternalProject_Get_Property(boost SOURCE_DIR)
	set(BOOST_INCLUDE_PATH "${SOURCE_DIR}/include/boost-1_83")
	link_directories("${SOURCE_DIR}/lib")
else ()
    # ...
endif ()
add_definitions( -DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE )
add_definitions( -DBOOST_COROUTINES_NO_DEPRECATION_WARNING )
add_definitions( -DBOOST_LOCALE_HIDE_AUTO_PTR )

# FFMPEG
message(STATUS "CHECKPOINT: Adding FFmpeg")
casparcg_add_external_project(ffmpeg-lib)
ExternalProject_Add(ffmpeg-lib
	URL ${CASPARCG_DOWNLOAD_MIRROR}/ffmpeg/ffmpeg-8.1.2-full_build-shared.7z
	URL_HASH SHA256=cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
)
ExternalProject_Get_Property(ffmpeg-lib SOURCE_DIR)
set(FFMPEG_INCLUDE_PATH "${SOURCE_DIR}/include")
set(FFMPEG_BIN_PATH "${SOURCE_DIR}/bin")
link_directories("${SOURCE_DIR}/lib")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/avcodec-62.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/avdevice-62.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/avfilter-11.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/avformat-62.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/avutil-60.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/swresample-6.dll")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/swscale-9.dll")
# for scanner:
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/ffmpeg.exe")
casparcg_add_runtime_dependency("${FFMPEG_BIN_PATH}/ffprobe.exe")

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

set(EXTERNAL_CMAKE_ARGS "")
if (is_multi_config)
	set(EXTERNAL_CMAKE_ARGS "-DCMAKE_BUILD_TYPE:STRING=$<CONFIG>")
else()
	set(EXTERNAL_CMAKE_ARGS "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif ()
list(APPEND EXTERNAL_CMAKE_ARGS
	"-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
	"-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
	"-DCMAKE_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
	"-DCMAKE_RC_COMPILER:FILEPATH=${CMAKE_RC_COMPILER}"
	"-DCMAKE_MT:FILEPATH=${CMAKE_MT}"
	"-DCMAKE_LINKER:FILEPATH=${CMAKE_LINKER}"
	"-DCMAKE_AR:FILEPATH=${CMAKE_AR}"
)

# TBB
FetchContent_Declare(tbb
	URL ${CASPARCG_DOWNLOAD_MIRROR}/tbb/oneapi-tbb-2022.3.0-win.zip
	URL_HASH SHA256=e1b2373f25558bf47d16b4c89cf0a31e6689aaf7221400d209e8527afc7c9eee
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
)
FetchContent_MakeAvailable(tbb)

list(APPEND CMAKE_PREFIX_PATH ${tbb_SOURCE_DIR}/lib/cmake/tbb)
find_package(tbb REQUIRED)

casparcg_add_runtime_dependency_from_target(TBB::tbb)
casparcg_add_runtime_dependency_from_target(TBB::tbbmalloc)
casparcg_add_runtime_dependency_from_target(TBB::tbbmalloc_proxy)

# GLEW
FetchContent_Declare(glew
	URL ${CASPARCG_DOWNLOAD_MIRROR}/glew/glew-2.2.0-win32.zip
	URL_HASH MD5=1feddfe8696c192fa46a0df8eac7d4bf
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
)
FetchContent_MakeAvailable(glew)

add_library(GLEW::glew INTERFACE IMPORTED)
target_include_directories(GLEW::glew INTERFACE ${glew_SOURCE_DIR}/include)
target_link_directories(GLEW::glew INTERFACE ${glew_SOURCE_DIR}/lib/Release/x64)
target_link_libraries(GLEW::glew INTERFACE glew32)
casparcg_add_runtime_dependency("${glew_SOURCE_DIR}/bin/Release/x64/glew32.dll")

IF(ENABLE_VULKAN)
	find_package(Vulkan REQUIRED)

	FetchContent_Declare(vk_bootstrap
			URL ${CASPARCG_DOWNLOAD_MIRROR}/vk-bootstrap/vk-bootstrap-1.4.328.zip
			URL_HASH SHA256=10f257c30a0a49d30b28a72cf3a7942d93a61f977adaa04bee29304c6506dc12
			DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
			)
	FetchContent_MakeAvailable(vk_bootstrap)

	FetchContent_Declare(vma
			URL ${CASPARCG_DOWNLOAD_MIRROR}/VulkanMemoryAllocator/VulkanMemoryAllocator-3.3.0.zip
			URL_HASH SHA256=81755d8fcb411b97292c6682e828501315db319374c7c34ba6e1226452c6c392
			DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
	)
	FetchContent_MakeAvailable(vma)
ENDIF()

# SFML
FetchContent_Declare(sfml
	URL ${CASPARCG_DOWNLOAD_MIRROR}/sfml/SFML-2.6.2-windows-vc17-64-bit.zip
	URL_HASH MD5=dee0602d6f94d1843eef4d7568d2c23d
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
)
FetchContent_MakeAvailable(sfml)

list(APPEND CMAKE_PREFIX_PATH ${sfml_SOURCE_DIR}/lib/cmake/SFML)
# set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2 COMPONENTS graphics system window REQUIRED)

# Force RelWithDebInfo to use the optimized release DLLs instead of SFML's debug variants
foreach(_sfml_target sfml-graphics sfml-system sfml-window)
    set_target_properties(${_sfml_target} PROPERTIES
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
endforeach()

casparcg_add_runtime_dependency_from_target(sfml-graphics)
casparcg_add_runtime_dependency_from_target(sfml-system)
casparcg_add_runtime_dependency_from_target(sfml-window)

#ZLIB
casparcg_add_external_project(zlib)
ExternalProject_Add(zlib
	URL ${CASPARCG_DOWNLOAD_MIRROR}/zlib/zlib-1.3.tar.gz
	URL_HASH MD5=60373b133d630f74f4a1f94c1185a53f
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
	CMAKE_ARGS ${EXTERNAL_CMAKE_ARGS}
	INSTALL_COMMAND ""
)
ExternalProject_Get_Property(zlib SOURCE_DIR)
ExternalProject_Get_Property(zlib BINARY_DIR)
set(ZLIB_INCLUDE_PATH "${SOURCE_DIR};${BINARY_DIR}")
link_directories(${BINARY_DIR})

# OPENCOLORIO — colour management for the mixer.
#
# Built from source as an ExternalProject rather than taken from a package manager,
# because the whole tree is pinned to MSVC 14.50 (nvcc 12.9 cannot use 14.51) and
# EXTERNAL_CMAKE_ARGS forwards that pinned compiler/linker/ar/mt down, exactly as zlib
# above relies on. A previously abandoned attempt (b304665b8) instead pulled MSYS2/MinGW64
# binaries into this MSVC process -- libOpenColorIO plus libstdc++-6, libgcc_s_seh-1 and
# libwinpthread-1 -- which is two C++ runtimes and two heaps in one address space. Do not
# do that again; see docs/OCIO_INTEGRATION_STUDY.md section 1.3.
#
# OCIO_INSTALL_EXT_PACKAGES=ALL has OCIO download and statically link its own dependencies
# (Imath, yaml-cpp, pystring, minizip-ng, expat, zlib) into the one OpenColorIO DLL, so
# this adds a single runtime dependency rather than six.
#
# ⚠ PATH LENGTH. That option nests each dependency under
# <build>/ext/build/<dep>/src/<dep>_install-build/..., which is deep. A probe run from a
# directory with a long name blew Windows' 250-character object-path limit inside expat's
# TryCompile, and the failure names neither OCIO nor the path -- it surfaces as a broken
# sub-build. Keep the build directory shallow; verified working from d:/Github/CasparVP/build.
option(ENABLE_OCIO "Enable OpenColorIO colour management" ON)
if (ENABLE_OCIO)
	message(STATUS "CHECKPOINT: Adding OpenColorIO")
	casparcg_add_external_project(opencolorio)
	ExternalProject_Add(opencolorio
		GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/OpenColorIO.git
		# 2.5.2 is a floor, not a preference: 2.5.1 reworked the Vulkan texture binding
		# indices and broke ABI, and 2.5.2 fixes CVE-2026-42450, stack buffer overflows in
		# the .spi3d/.spi1d/.cube/.lut parsers. This server already ingests
		# operator-supplied LUT files.
		GIT_TAG        v2.5.2
		GIT_SHALLOW    TRUE
		CMAKE_ARGS     ${EXTERNAL_CMAKE_ARGS}
		               "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
		               -DOCIO_INSTALL_EXT_PACKAGES=ALL
		               -DOCIO_BUILD_APPS=OFF
		               -DOCIO_BUILD_PYTHON=OFF
		               -DOCIO_BUILD_TESTS=OFF
		               -DOCIO_BUILD_GPU_TESTS=OFF
		               -DOCIO_BUILD_DOCS=OFF
		               -DOCIO_BUILD_NUKE=OFF
		               # 14.50 emits warnings OCIO's CI has not seen; a new warning must
		               # not fail this build.
		               -DOCIO_WARNING_AS_ERROR=OFF
	)
	ExternalProject_Get_Property(opencolorio INSTALL_DIR)
	set(OCIO_INCLUDE_PATH "${INSTALL_DIR}/include")
	link_directories("${INSTALL_DIR}/lib")
	casparcg_add_runtime_dependency("${INSTALL_DIR}/bin/OpenColorIO_2_5.dll")
	# CASPAR_ENABLE_OCIO is set on the accelerator target rather than globally: only
	# accelerator/ocio/ocio_config.cpp includes OCIO headers, and its facade is std-only,
	# so no other target needs either the macro or the include path.
endif ()

# OpenFX (host) — used by the ofx module to load OFX plug-ins.
# We vendor the OpenFX C API headers + the BSD-3 HostSupport C++ library and build
# HostSupport as an internal static lib (openfx_host). HostSupport uses expat for its
# persistent plug-in cache, so we also fetch libexpat and build it statically.
option(ENABLE_OFX "Enable the OpenFX host module" ON)
if (ENABLE_OFX)
	# --- expat (XML, required by HostSupport) ---
	set(EXPAT_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
	set(EXPAT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(EXPAT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
	set(EXPAT_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
	set(EXPAT_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
	set(EXPAT_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
	FetchContent_Declare(expat
		GIT_REPOSITORY https://github.com/libexpat/libexpat.git
		GIT_TAG        R_2_6_4
		GIT_SHALLOW    TRUE
		SOURCE_SUBDIR  expat
	)
	FetchContent_MakeAvailable(expat)

	# --- OpenFX source (headers + HostSupport). SOURCE_SUBDIR points at a dir with no
	#     CMakeLists.txt so MakeAvailable only populates the source (no add_subdirectory). ---
	FetchContent_Declare(openfx
		GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openfx.git
		GIT_TAG        OFX_Release_1.5.1
		GIT_SHALLOW    TRUE
		SOURCE_SUBDIR  include
	)
	FetchContent_MakeAvailable(openfx)

	set(OPENFX_HOSTSUPPORT_SOURCES
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhBinary.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhClip.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhHost.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhImageEffect.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhImageEffectAPI.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhInteract.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhMemory.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhParam.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhPluginAPICache.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhPluginCache.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhPropertySuite.cpp
		${openfx_SOURCE_DIR}/HostSupport/src/ofxhUtilities.cpp
	)

	add_library(openfx_host STATIC ${OPENFX_HOSTSUPPORT_SOURCES})
	target_include_directories(openfx_host PUBLIC
		${openfx_SOURCE_DIR}/include
		${openfx_SOURCE_DIR}/HostSupport/include
	)
	# The global CMAKE_CXX_FLAGS force-includes common/compiler/vs/disable_silly_warnings.h
	# (a relative path); add the source root so it resolves for this out-of-tree target.
	target_include_directories(openfx_host PRIVATE ${CMAKE_SOURCE_DIR})
	target_link_libraries(openfx_host PUBLIC expat)
	# HostSupport is third-party BSD code and does not compile clean under /W4 /WX.
	# It also assumes an ANSI (non-UNICODE) build for TCHAR Win32 APIs, so undefine the
	# global UNICODE/_UNICODE for this target only.
	target_compile_options(openfx_host PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
	target_compile_definitions(openfx_host PRIVATE _CRT_SECURE_NO_WARNINGS)
	# Enable the OFX OpenGL render suite in HostSupport. Must match the definition on the
	# ofx module (aligns the class vtables across the two static libs).
	target_compile_definitions(openfx_host PUBLIC OFX_SUPPORTS_OPENGLRENDER)
	# Host provides a real (parallel) OfxMultiThreadSuite: HostSupport forwards the suite to the
	# host's multiThread*/mutex* virtuals, which caspar_ofx_host implements.
	target_compile_definitions(openfx_host PUBLIC OFX_SUPPORTS_MULTITHREAD)
	set_target_properties(openfx_host expat PROPERTIES FOLDER external)

	# Optional: build a couple of OpenFX sample plug-ins (from the fetched OpenFX repo) into
	# .ofx bundles under ${CMAKE_BINARY_DIR}/ofx-plugins for end-to-end testing of the host.
	option(BUILD_OFX_SAMPLE_PLUGINS "Build OpenFX sample plug-ins (Invert, Basic) for testing" OFF)
	if (BUILD_OFX_SAMPLE_PLUGINS)
		# OFX C++ plug-in Support library.
		file(GLOB OFX_SUPPORT_SOURCES ${openfx_SOURCE_DIR}/Support/Library/*.cpp)
		add_library(ofxsupport STATIC ${OFX_SUPPORT_SOURCES})
		target_include_directories(ofxsupport PUBLIC
			${openfx_SOURCE_DIR}/include
			${openfx_SOURCE_DIR}/Support/include
		)
		target_include_directories(ofxsupport PRIVATE ${CMAKE_SOURCE_DIR})
		target_compile_options(ofxsupport PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
		target_compile_definitions(ofxsupport PRIVATE _CRT_SECURE_NO_WARNINGS)
		set_target_properties(ofxsupport PROPERTIES FOLDER external)

		# Helper to build one example .ofx bundle: <name>.ofx.bundle/Contents/Win64/<name>.ofx
		function(casparcg_add_ofx_sample_plugin NAME SOURCE)
			add_library(${NAME} MODULE ${SOURCE})
			target_link_libraries(${NAME} PRIVATE ofxsupport)
			target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR})
			target_compile_options(${NAME} PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
			target_compile_definitions(${NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
			set_target_properties(${NAME} PROPERTIES
				PREFIX ""
				SUFFIX ".ofx"
				FOLDER external
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/${NAME}.ofx.bundle/Contents/Win64"
			)
		endfunction()

		casparcg_add_ofx_sample_plugin(Invert "${openfx_SOURCE_DIR}/Examples/Invert/invert.cpp")
		casparcg_add_ofx_sample_plugin(Basic  "${openfx_SOURCE_DIR}/Examples/Basic/basic.cpp")

		# The OpenGL example uses the raw C API (defines its own OFX entry points) and calls GL
		# directly, so it must NOT link ofxsupport (duplicate symbols) and needs opengl32.
		add_library(OpenGLExample_ofx MODULE "${openfx_SOURCE_DIR}/Examples/OpenGL/opengl.cpp")
		target_include_directories(OpenGLExample_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(OpenGLExample_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
		target_compile_definitions(OpenGLExample_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
		target_link_libraries(OpenGLExample_ofx PRIVATE opengl32)
		set_target_properties(OpenGLExample_ofx PROPERTIES
			PREFIX ""
			SUFFIX ".ofx"
			OUTPUT_NAME "OpenGL"
			FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/OpenGL.ofx.bundle/Contents/Win64"
		)

		# Core-profile GL test plug-in (glScissor/glClear only) — renders a deterministic
		# top/bottom colour pattern to verify the zero-copy OpenGL render path + orientation.
		add_library(CoreGLTest_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/coregl_orientation_test.cpp")
		target_include_directories(CoreGLTest_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(CoreGLTest_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
		target_compile_definitions(CoreGLTest_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
		target_link_libraries(CoreGLTest_ofx PRIVATE opengl32)
		set_target_properties(CoreGLTest_ofx PROPERTIES
			PREFIX ""
			SUFFIX ".ofx"
			OUTPUT_NAME "CoreGLTest"
			FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/CoreGLTest.ofx.bundle/Contents/Win64"
		)

		# Core-profile GL source-sampling passthrough test plug-in (validates the zero-copy source
		# path) — raw C API + GLEW for the GL 3.3 entry points.
		add_library(CoreGLPassthrough_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/coregl_passthrough_test.cpp")
		target_include_directories(CoreGLPassthrough_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(CoreGLPassthrough_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
		target_compile_definitions(CoreGLPassthrough_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
		target_link_libraries(CoreGLPassthrough_ofx PRIVATE opengl32 GLEW::glew)
		set_target_properties(CoreGLPassthrough_ofx PROPERTIES
			PREFIX ""
			SUFFIX ".ofx"
			OUTPUT_NAME "CoreGLPassthrough"
			FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/CoreGLPassthrough.ofx.bundle/Contents/Win64"
		)

		# CPU transition test plug-in (blends SourceFrom/SourceTo by the Transition param) — raw C API.
		add_library(TransitionTest_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/transition_mix_test.cpp")
		target_include_directories(TransitionTest_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(TransitionTest_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
		target_compile_definitions(TransitionTest_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
		set_target_properties(TransitionTest_ofx PROPERTIES
			PREFIX ""
			SUFFIX ".ofx"
			OUTPUT_NAME "TransitionTest"
			FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/TransitionTest.ofx.bundle/Contents/Win64"
		)

		# CUDA test plug-in (cudaMemset the output device buffer) — validates the host CUDA render
		# path. Runtime host API only (cudaMemset), so no nvcc; just links cudart.
		find_package(CUDAToolkit QUIET)
		if (CUDAToolkit_FOUND)
			add_library(CudaTest_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/cuda_fill_test.cpp")
			target_include_directories(CudaTest_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
			target_compile_options(CudaTest_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
			target_compile_definitions(CudaTest_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
			target_link_libraries(CudaTest_ofx PRIVATE CUDA::cudart_static)
			set_target_properties(CudaTest_ofx PROPERTIES
				PREFIX ""
				SUFFIX ".ofx"
				OUTPUT_NAME "CudaTest"
				FOLDER external
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/CudaTest.ofx.bundle/Contents/Win64"
			)

			# CUDA source-sampling passthrough test plug-in (validates the CUDA source path /
			# orientation): copies source device buffer -> output device buffer. Runtime API only.
			add_library(CudaPassthrough_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/cuda_passthrough_test.cpp")
			target_include_directories(CudaPassthrough_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
			target_compile_options(CudaPassthrough_ofx PRIVATE /W0 /WX- /EHsc /UUNICODE /U_UNICODE)
			target_compile_definitions(CudaPassthrough_ofx PRIVATE _CRT_SECURE_NO_WARNINGS)
			target_link_libraries(CudaPassthrough_ofx PRIVATE CUDA::cudart_static)
			set_target_properties(CudaPassthrough_ofx PROPERTIES
				PREFIX ""
				SUFFIX ".ofx"
				OUTPUT_NAME "CudaPassthrough"
				FOLDER external
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/CudaPassthrough.ofx.bundle/Contents/Win64"
			)
		endif ()
	endif ()
endif ()

# OpenAL
FetchContent_Declare(openal
	URL ${CASPARCG_DOWNLOAD_MIRROR}/openal/openal-soft-1.19.1-bin.zip
	URL_HASH MD5=b78ef1ba26f7108e763f92df6bbc3fa5
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
)
FetchContent_MakeAvailable(openal)
file(COPY_FILE ${openal_SOURCE_DIR}/bin/Win64/soft_oal.dll ${openal_SOURCE_DIR}/bin/Win64/OpenAL32.dll)

add_library(OpenAL::OpenAL INTERFACE IMPORTED)
target_include_directories(OpenAL::OpenAL INTERFACE ${openal_SOURCE_DIR}/include)
target_link_directories(OpenAL::OpenAL INTERFACE ${openal_SOURCE_DIR}/libs/Win64)
target_link_libraries(OpenAL::OpenAL INTERFACE OpenAL32)
casparcg_add_runtime_dependency("${openal_SOURCE_DIR}/bin/Win64/OpenAL32.dll")

# Vulkan: auto-detect from SDK if not explicitly set via -DENABLE_VULKAN=ON/OFF
if(NOT DEFINED ENABLE_VULKAN)
	# shaderc_combined is requested as a component so a missing one is a clear
	# configure-time error rather than an unresolved symbol at link time. It is needed
	# because a generated colour transform arrives as GLSL text and has to be compiled to
	# SPIR-V at runtime; the mixer's own shader is still built by glslc and embedded.
	find_package(Vulkan QUIET COMPONENTS shaderc_combined)
	if(Vulkan_FOUND)
		set(ENABLE_VULKAN ON CACHE BOOL "Enable Vulkan accelerator backend")
		message(STATUS "Vulkan SDK found: ${Vulkan_INCLUDE_DIR} (auto-enabled)")
	else()
		set(ENABLE_VULKAN OFF CACHE BOOL "Enable Vulkan accelerator backend")
		message(STATUS "Vulkan SDK not found -- Vulkan modules disabled")
	endif()
else()
	if(ENABLE_VULKAN)
		find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
		message(STATUS "Vulkan SDK: ${Vulkan_INCLUDE_DIR} (explicitly enabled)")
	endif()
endif()

IF(ENABLE_VULKAN)

	FetchContent_Declare(vk_bootstrap
		GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
		GIT_TAG        v1.4.328
	)
	FetchContent_MakeAvailable(vk_bootstrap)

	FetchContent_Declare(vma
		GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
		GIT_TAG        v3.3.0
	)
	set(VMA_STATIC_VULKAN_FUNCTIONS OFF)
	set(VMA_DYNAMIC_VULKAN_FUNCTIONS ON)
	FetchContent_MakeAvailable(vma)
ENDIF()

# flash template host
casparcg_add_external_project(flashtemplatehost)
ExternalProject_Add(flashtemplatehost
	URL ${CASPARCG_DOWNLOAD_MIRROR}/flash-template-host/flash-template-host-files.zip
	URL_HASH MD5=360184ce21e34d585d1d898fdd7a6bd8
	DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
)
ExternalProject_Get_Property(flashtemplatehost SOURCE_DIR)
set(TEMPLATE_HOST_PATH "${SOURCE_DIR}")
# casparcg_add_runtime_dependency_dir("${TEMPLATE_HOST_PATH}")

# LIBERATION_FONTS
set(LIBERATION_FONTS_BIN_PATH "${PROJECT_SOURCE_DIR}/shell/liberation-fonts")
casparcg_add_runtime_dependency("${LIBERATION_FONTS_BIN_PATH}/LiberationMono-Regular.ttf")

# CEF
if (ENABLE_HTML)
	casparcg_add_external_project(cef)
	ExternalProject_Add(cef
		URL ${CASPARCG_DOWNLOAD_MIRROR}/cef/cef_binary_142.0.17+g60aac24+chromium-142.0.7444.176_windows64_minimal.tar.bz2
		URL_HASH SHA256=16c072a44484fe521037c74d03a339a77573b1fc0146cf44cc71e79fd0cc0198
		DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
		CMAKE_ARGS -DUSE_SANDBOX=Off -DCEF_RUNTIME_LIBRARY_FLAG=/MD ${EXTERNAL_CMAKE_ARGS}
		INSTALL_COMMAND ""
	)
	ExternalProject_Get_Property(cef SOURCE_DIR)
	ExternalProject_Get_Property(cef BINARY_DIR)

    add_library(CEF::CEF INTERFACE IMPORTED)
	add_dependencies(CEF::CEF cef)
    target_include_directories(CEF::CEF INTERFACE
        "${SOURCE_DIR}"
    )

	set(CEF_RESOURCE_PATH ${SOURCE_DIR}/Resources)
	set(CEF_BIN_PATH ${SOURCE_DIR}/Release)

	if (is_multi_config)
	    target_link_libraries(CEF::CEF INTERFACE
			${SOURCE_DIR}/Release/libcef.lib
			optimized ${BINARY_DIR}/libcef_dll_wrapper/Release/libcef_dll_wrapper.lib
			debug ${BINARY_DIR}/libcef_dll_wrapper/Debug/libcef_dll_wrapper.lib)
	else()
		link_directories(${SOURCE_DIR}/Release ${BINARY_DIR}/libcef_dll_wrapper)
		target_link_libraries(CEF::CEF INTERFACE
			libcef.lib
			libcef_dll_wrapper.lib)
	endif()

	casparcg_add_runtime_dependency_dir("${CEF_RESOURCE_PATH}/locales")
	casparcg_add_runtime_dependency("${CEF_RESOURCE_PATH}/chrome_100_percent.pak")
	casparcg_add_runtime_dependency("${CEF_RESOURCE_PATH}/chrome_200_percent.pak")
	casparcg_add_runtime_dependency("${CEF_RESOURCE_PATH}/resources.pak")
	casparcg_add_runtime_dependency("${CEF_RESOURCE_PATH}/icudtl.dat")

	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/v8_context_snapshot.bin")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/libcef.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/chrome_elf.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/d3dcompiler_47.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/libEGL.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/libGLESv2.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/vk_swiftshader.dll")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/vk_swiftshader_icd.json")
	casparcg_add_runtime_dependency("${CEF_BIN_PATH}/vulkan-1.dll")
endif ()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT casparcg)

add_definitions(-DUNICODE)
add_definitions(-D_UNICODE)
add_definitions(-DCASPAR_SOURCE_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-D_WIN32_WINNT=0x0A00) # Minimum windows 10

# TODO: recompile boost to avoid this
add_compile_definitions(BOOST_USE_WINAPI_VERSION=0x0601)  # Boost ABI: must match prebuilt deps

# ignore boost deprecated headers, as these are often reported inside boost
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")

# Ensure /EHsc is not defined as it clashes with EHa below
string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa /Zi /W4 /WX /MP /fp:fast /Zm192 /FIcommon/compiler/vs/disable_silly_warnings.h")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}	/D TBB_USE_ASSERT=1 /D TBB_USE_DEBUG /bigobj")
# Deliberate divergence from upstream, which dropped /arch:AVX2 in 2b97ac61d. Kept at AVX2
# here for two reasons: `decklink/consumer/v210_strategies.cpp` uses AVX2 intrinsics with no
# runtime CPU dispatch in either tree, so lowering the baseline buys this fork no portability
# it can actually use; and /arch: changes the auto-vectorisation baseline for every
# translation unit, which can move floating-point results in the mixer. Revisit only together
# with a runtime dispatch, and measure the mixer when you do.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}	/Oi /arch:AVX2 /Ot /Gy /bigobj")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Oi /arch:AVX2 /Ot /Gy /bigobj")
