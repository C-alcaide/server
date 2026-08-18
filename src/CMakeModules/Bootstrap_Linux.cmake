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

set(ENABLE_HTML ON CACHE BOOL "Enable CEF and HTML producer")
set(USE_STATIC_BOOST OFF CACHE BOOL "Use shared library version of Boost")
set(USE_SYSTEM_CEF ON CACHE BOOL "Use the version of cef from your OS (only tested with Ubuntu)")
set(CASPARCG_BINARY_NAME "casparcg" CACHE STRING "Custom name of the binary to build (this disables some install files)")
set(ENABLE_AVX2 OFF CACHE BOOL "Enable the AVX2 instruction set (requires a CPU that supports it)")

# Determine build (target) platform
SET (PLATFORM_FOLDER_NAME "linux")

IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE (STATUS "Setting build type to 'Release' as none was specified.")
	SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
	SET_PROPERTY (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF ()
MARK_AS_ADVANCED (CMAKE_INSTALL_PREFIX)

if (USE_STATIC_BOOST)
	SET (Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost 1.83.0 COMPONENTS thread filesystem log_setup log locale regex date_time coroutine REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL GLX EGL)
find_package(GLEW REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenAL REQUIRED)
find_package(SFML 3 COMPONENTS Graphics System Window QUIET)
if(NOT SFML_FOUND)
    find_package(SFML 2 COMPONENTS graphics system window REQUIRED)
endif()
find_package(X11 REQUIRED)

if (ENABLE_HTML)
    if (USE_SYSTEM_CEF)
        set(CEF_LIB_PATH "/usr/lib/casparcg-cef-142")

        add_library(CEF::CEF INTERFACE IMPORTED)
        target_include_directories(CEF::CEF INTERFACE
            "/usr/include/casparcg-cef-142"
        )
        target_link_libraries(CEF::CEF INTERFACE
            "-Wl,-rpath,${CEF_LIB_PATH} ${CEF_LIB_PATH}/libcef.so"
            "${CEF_LIB_PATH}/libcef_dll_wrapper.a"
        )
    else()
        casparcg_add_external_project(cef)
        ExternalProject_Add(cef
            URL ${CASPARCG_DOWNLOAD_MIRROR}/cef/cef_binary_142.0.17+g60aac24+chromium-142.0.7444.176_linux64_minimal.tar.bz2
            URL_HASH SHA256=1d89e19b2f446105f9a1fe6fdc96bced86249b5884241dcc4013b7c94dabf424
            DOWNLOAD_DIR ${CASPARCG_DOWNLOAD_CACHE}
            CMAKE_ARGS -DUSE_SANDBOX=Off
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS
                "<SOURCE_DIR>/Release/libcef.so"
                "<BINARY_DIR>/libcef_dll_wrapper/libcef_dll_wrapper.a"
        )
        ExternalProject_Get_Property(cef SOURCE_DIR)
        ExternalProject_Get_Property(cef BINARY_DIR)

        add_library(CEF::CEF INTERFACE IMPORTED)
        target_include_directories(CEF::CEF INTERFACE
            "${SOURCE_DIR}"
        )
        target_link_libraries(CEF::CEF INTERFACE
            # Note: All of these must be referenced in the BUILD_BYPRODUCTS above, to satisfy ninja
            "${SOURCE_DIR}/Release/libcef.so"
            "${BINARY_DIR}/libcef_dll_wrapper/libcef_dll_wrapper.a"
        )

        install(DIRECTORY ${SOURCE_DIR}/Resources/locales TYPE LIB)
        install(FILES ${SOURCE_DIR}/Resources/chrome_100_percent.pak TYPE LIB)
        install(FILES ${SOURCE_DIR}/Resources/chrome_200_percent.pak TYPE LIB)
        install(FILES ${SOURCE_DIR}/Resources/icudtl.dat TYPE LIB)
        install(FILES ${SOURCE_DIR}/Resources/resources.pak TYPE LIB)

        install(FILES ${SOURCE_DIR}/Release/chrome-sandbox TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/libcef.so TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/libEGL.so TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/libGLESv2.so TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/libvk_swiftshader.so TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/libvulkan.so.1 TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/v8_context_snapshot.bin TYPE LIB)
        install(FILES ${SOURCE_DIR}/Release/vk_swiftshader_icd.json TYPE LIB)
    endif()
endif ()

SET (BOOST_INCLUDE_PATH "${Boost_INCLUDE_DIRS}")
SET (FFMPEG_INCLUDE_PATH "${FFMPEG_INCLUDE_DIRS}")

LINK_DIRECTORIES("${FFMPEG_LIBRARY_DIRS}")

SET_PROPERTY (GLOBAL PROPERTY USE_FOLDERS ON)

ADD_DEFINITIONS (-DSFML_STATIC)
ADD_DEFINITIONS (-DUNICODE)
ADD_DEFINITIONS (-D_UNICODE)
ADD_DEFINITIONS (-DGLEW_NO_GLU)
ADD_DEFINITIONS (-DGLEW_EGL)
ADD_DEFINITIONS (-D__NO_INLINE__) # Needed for precompiled headers to work
ADD_DEFINITIONS (-DBOOST_NO_SWPRINTF) # swprintf on Linux seems to always use , as decimal point regardless of C-locale or C++-locale
ADD_DEFINITIONS (-DTBB_USE_CAPTURED_EXCEPTION=1)
ADD_DEFINITIONS (-DNDEBUG) # Needed for precompiled headers to work
ADD_DEFINITIONS (-DBOOST_LOCALE_HIDE_AUTO_PTR) # Needed for C++17 in boost 1.67+


if (NOT USE_STATIC_BOOST)
	ADD_DEFINITIONS (-DBOOST_ALL_DYN_LINK)
endif()

IF (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
	ADD_COMPILE_OPTIONS (-O3) # Needed for precompiled headers to work
endif()
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "(i[3-6]86|x64|x86_64|amd64|e2k)")
    ADD_COMPILE_OPTIONS (-msse3)
    ADD_COMPILE_OPTIONS (-mssse3)
    ADD_COMPILE_OPTIONS (-msse4.1)
    IF (ENABLE_AVX2)
        ADD_COMPILE_OPTIONS (-mfma)
        ADD_COMPILE_OPTIONS (-mavx)
        ADD_COMPILE_OPTIONS (-mavx2)
    ENDIF ()
ELSE ()
    ADD_COMPILE_DEFINITIONS (USE_SIMDE) # Enable OpenMP support in simde
    ADD_COMPILE_DEFINITIONS (SIMDE_ENABLE_OPENMP) # Enable OpenMP support in simde
    ADD_COMPILE_OPTIONS (-fopenmp-simd) # Enable OpenMP SIMD support
ENDIF ()

ADD_COMPILE_OPTIONS (-fnon-call-exceptions) # Allow signal handler to throw exception

ADD_COMPILE_OPTIONS (-Wno-deprecated-declarations -Wno-write-strings -Wno-multichar -Wno-cpp -Werror)
IF (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    ADD_COMPILE_OPTIONS (-Wno-terminate)
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Help TBB figure out what compiler support for c++11 features
    # https://github.com/01org/tbb/issues/22
    string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    message(STATUS "ADDING: -DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION}")
    add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
ENDIF ()

# OpenFX (host) — Linux mirror of the Windows OFX bootstrap block. Builds the BSD-3 HostSupport
# C++ library as an internal static lib (openfx_host) plus libexpat (its XML dependency), so the
# ofx module can load OFX plug-ins. Compiler flags are Linux/GCC-Clang appropriate.
include(FetchContent)
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
	target_include_directories(openfx_host PRIVATE ${CMAKE_SOURCE_DIR})
	target_link_libraries(openfx_host PUBLIC expat ${CMAKE_DL_LIBS})
	# Third-party BSD code: silence warnings and neutralise the global -Werror for this target.
	target_compile_options(openfx_host PRIVATE -w)
	# Match the definitions used on the ofx module so the class vtables align across the two libs.
	target_compile_definitions(openfx_host PUBLIC OFX_SUPPORTS_OPENGLRENDER)
	target_compile_definitions(openfx_host PUBLIC OFX_SUPPORTS_MULTITHREAD)
	set_target_properties(openfx_host PROPERTIES FOLDER external)

	# Optional OFX sample plug-ins for end-to-end host testing (bundled .ofx modules).
	option(BUILD_OFX_SAMPLE_PLUGINS "Build OpenFX sample plug-ins (Invert, Basic) for testing" OFF)
	if (BUILD_OFX_SAMPLE_PLUGINS)
		file(GLOB OFX_SUPPORT_SOURCES ${openfx_SOURCE_DIR}/Support/Library/*.cpp)
		add_library(ofxsupport STATIC ${OFX_SUPPORT_SOURCES})
		target_include_directories(ofxsupport PUBLIC
			${openfx_SOURCE_DIR}/include
			${openfx_SOURCE_DIR}/Support/include
		)
		target_include_directories(ofxsupport PRIVATE ${CMAKE_SOURCE_DIR})
		target_compile_options(ofxsupport PRIVATE -w)
		set_target_properties(ofxsupport PROPERTIES FOLDER external)

		function(casparcg_add_ofx_sample_plugin NAME SOURCE)
			add_library(${NAME} MODULE ${SOURCE})
			target_link_libraries(${NAME} PRIVATE ofxsupport)
			target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR})
			target_compile_options(${NAME} PRIVATE -w)
			set_target_properties(${NAME} PROPERTIES
				PREFIX ""
				SUFFIX ".ofx"
				FOLDER external
				LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/${NAME}.ofx.bundle/Contents/Linux-x86-64"
			)
		endfunction()

		casparcg_add_ofx_sample_plugin(Invert "${openfx_SOURCE_DIR}/Examples/Invert/invert.cpp")
		casparcg_add_ofx_sample_plugin(Basic  "${openfx_SOURCE_DIR}/Examples/Basic/basic.cpp")

		# GL example + core-GL test plug-in (raw C API, link GL directly, no ofxsupport).
		find_package(OpenGL REQUIRED)
		add_library(OpenGLExample_ofx MODULE "${openfx_SOURCE_DIR}/Examples/OpenGL/opengl.cpp")
		target_include_directories(OpenGLExample_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(OpenGLExample_ofx PRIVATE -w)
		target_link_libraries(OpenGLExample_ofx PRIVATE OpenGL::GL)
		set_target_properties(OpenGLExample_ofx PROPERTIES
			PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "OpenGL" FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/OpenGL.ofx.bundle/Contents/Linux-x86-64"
		)

		add_library(CoreGLTest_ofx MODULE "${CMAKE_SOURCE_DIR}/modules/ofx/test/coregl_orientation_test.cpp")
		target_include_directories(CoreGLTest_ofx PRIVATE ${openfx_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
		target_compile_options(CoreGLTest_ofx PRIVATE -w)
		target_link_libraries(CoreGLTest_ofx PRIVATE OpenGL::GL)
		set_target_properties(CoreGLTest_ofx PROPERTIES
			PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "CoreGLTest" FOLDER external
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ofx-plugins/CoreGLTest.ofx.bundle/Contents/Linux-x86-64"
		)
	endif ()
endif ()
