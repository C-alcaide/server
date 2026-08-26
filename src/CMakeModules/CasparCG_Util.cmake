# Collect all ExternalProjects that have been defined
set(CASPARCG_EXTERNAL_PROJECTS "" CACHE INTERNAL "")
FUNCTION (casparcg_add_external_project NAME)
	SET (CASPARCG_EXTERNAL_PROJECTS "${CASPARCG_EXTERNAL_PROJECTS}" "${NAME}" CACHE INTERNAL "")
ENDFUNCTION()

# Mark a project as depending on all of the ExternalProjects, to ensure build order
FUNCTION(casparcg_add_build_dependencies TARGET)
	if (CASPARCG_EXTERNAL_PROJECTS)
        foreach(_dep ${CASPARCG_EXTERNAL_PROJECTS})
            if (TARGET ${_dep})
		        ADD_DEPENDENCIES (${TARGET} ${_dep})
            endif()
        endforeach()
	endif()
ENDFUNCTION()

SET (CASPARCG_MODULE_INCLUDE_STATEMENTS "" CACHE INTERNAL "")
SET (CASPARCG_MODULE_INIT_STATEMENTS "" CACHE INTERNAL "")
SET (CASPARCG_MODULE_UNINIT_STATEMENTS "" CACHE INTERNAL "")
SET (CASPARCG_MODULE_COMMAND_LINE_ARG_INTERCEPTORS_STATEMENTS "" CACHE INTERNAL "")
SET (CASPARCG_MODULE_TARGETS "" CACHE INTERNAL "")

# CasparCG version of CMake `add_library`
FUNCTION (casparcg_add_library TARGET)
	cmake_parse_arguments(
        PARSED_ARGS # prefix of output variables
        "" # list of names of the boolean arguments (only defined ones will be true)
        "" # list of names of mono-valued arguments
        "SOURCES" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )

	if(NOT TARGET)
        message(FATAL_ERROR "You must provide a target name")
	endif()

	# Setup the library and some default config
	ADD_LIBRARY (${TARGET} ${PARSED_ARGS_SOURCES})
	target_compile_features (${TARGET} PRIVATE cxx_std_20)

	# CUDA stays at C++17 while C++ moved to C++20 (upstream f9fa5c342).
	#
	# CUDA 12.9's nvcc cannot parse MSVC 14.50's C++20 <chrono>, which the fork's .cu files
	# pull in transitively through common/log.h:
	#   include/chrono(5125): error C2760: syntax error: '}' unexpected here
	# It hits cuda_prores and cuda_notchlc; decklink and remotewall also carry .cu sources,
	# so this is set for every module rather than the two that happened to fail first.
	#
	# Keeping CUDA at 17 leaves those translation units on exactly the standard the whole
	# tree used before the sync, i.e. the configuration they were last known to build under.
	# KNOWN RISK, not yet verified: this makes the link boundary mixed-standard, and std
	# types (std::wstring, spl::shared_ptr, boost::property_tree) do cross it -- every CUDA
	# module registers a consumer or producer factory. MSVC's library is ABI-compatible
	# across /std:c++17 and /std:c++20 in practice but does not promise to be. Revisit when
	# CUDA gains a C++20 host parser; see docs/audits/UPSTREAM_SYNC_2026-08-18.md.
	set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
	target_include_directories(${TARGET} SYSTEM PRIVATE
		${BOOST_INCLUDE_PATH}
	)
	target_link_libraries(${TARGET} PRIVATE TBB::tbb)

	if (CASPARCG_EXTERNAL_PROJECTS)
		# Setup dependency on ExternalProject
        foreach(_dep ${CASPARCG_EXTERNAL_PROJECTS})
            if (TARGET ${_dep})
		        ADD_DEPENDENCIES (${TARGET} ${_dep})
            endif()
        endforeach()
	endif()

ENDFUNCTION ()

# CasparCG version of CMake `add_library` specifically for modules
SET (CASPARCG_MODULE_TARGETS "" CACHE INTERNAL "")
FUNCTION (casparcg_add_module_project TARGET)
	cmake_parse_arguments(
        PARSED_ARGS # prefix of output variables
        "" # list of names of the boolean arguments (only defined ones will be true)
        "NAME;HEADER_FILE;INIT_FUNCTION;UNINIT_FUNCTION;CLI_INTERCEPTOR" # list of names of mono-valued arguments
        "SOURCES" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )

	# Use target if name is missing
	if (NOT PARSED_ARGS_NAME)
		set (PARSED_ARGS_NAME ${TARGET})
	endif()
	# Use default path if header not defined
	if (NOT PARSED_ARGS_HEADER_FILE)
		set (PARSED_ARGS_HEADER_FILE "modules/${TARGET}/${TARGET}.h")
	endif()
	# Use default init name if not d
	if (NOT PARSED_ARGS_INIT_FUNCTION)
        message(FATAL_ERROR "You must provide an INIT_FUNCTION")
	endif()

	# Setup the library and some default config
	casparcg_add_library (${TARGET} SOURCES ${PARSED_ARGS_SOURCES})
	target_link_libraries(${TARGET} PRIVATE common core)
	target_include_directories(${TARGET} PRIVATE
			# TODO: This should be replaced by the linked libraries eventually
			../..
	)

	# Setup linker and code loading
	SET (CASPARCG_MODULE_TARGETS "${CASPARCG_MODULE_TARGETS}" "${TARGET}" CACHE INTERNAL "")
	SET (CASPARCG_MODULE_INCLUDE_STATEMENTS "${CASPARCG_MODULE_INCLUDE_STATEMENTS}"
		"#include <${PARSED_ARGS_HEADER_FILE}>"
		CACHE INTERNAL ""
	)
	SET (CASPARCG_MODULE_INIT_STATEMENTS "${CASPARCG_MODULE_INIT_STATEMENTS}"
		"	${PARSED_ARGS_INIT_FUNCTION}(dependencies)\;"
		"	CASPAR_LOG(info) << L\"Initialized ${PARSED_ARGS_NAME} module.\"\;"
		""
		CACHE INTERNAL ""
	)

	IF (PARSED_ARGS_UNINIT_FUNCTION)
		SET (CASPARCG_MODULE_UNINIT_STATEMENTS
			"	${PARSED_ARGS_UNINIT_FUNCTION}()\;"
			"${CASPARCG_MODULE_UNINIT_STATEMENTS}"
			CACHE INTERNAL ""
		)
	ENDIF ()

	IF (PARSED_ARGS_CLI_INTERCEPTOR)
		SET (CASPARCG_MODULE_COMMAND_LINE_ARG_INTERCEPTORS_STATEMENTS "${CASPARCG_MODULE_COMMAND_LINE_ARG_INTERCEPTORS_STATEMENTS}"
			"	if (${PARSED_ARGS_CLI_INTERCEPTOR}(argc, argv))"
			"		return true\;"
			""
			CACHE INTERNAL ""
		)
	ENDIF ()

ENDFUNCTION ()

# http://stackoverflow.com/questions/7172670/best-shortest-way-to-join-a-list-in-cmake
FUNCTION (join_list VALUES GLUE OUTPUT)
	STRING (REGEX REPLACE "([^\\]|^);" "\\1${GLUE}" _TMP_STR "${VALUES}")
	STRING (REGEX REPLACE "[\\](.)" "\\1" _TMP_STR "${_TMP_STR}") #fixes escaping
	SET (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
ENDFUNCTION ()
