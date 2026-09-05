cmake_minimum_required (VERSION 3.28)
project (protocol_http)

# A separate library from `protocol`, for two reasons that are not stylistic:
#
#   * `protocol` force-includes a precompiled header into every translation unit. Beast in
#     a PCH costs seconds of template instantiation in every file of that library -- and in
#     this tree a header edit does not invalidate the PCH, so it would need the full manual
#     sweep from CLAUDE.md on every change. This target has no PCH.
#   * Boost.JSON is compiled from source here (see boost_json_src.cpp), and confining that
#     to one small target keeps the rest of the tree unaffected either way.

set(SOURCES
		api_events.cpp
		api_status.cpp
		api_tree.cpp
		api_value.cpp
		boost_json_src.cpp
		http_server.cpp
		json_state.cpp
		state_hub.cpp
)

set(HEADERS
		api_events.h
		api_status.h
		api_context.h
		api_tree.h
		api_value.h
		boost_prelude.h
		http_config.h
		http_server.h
		json_state.h
		state_hub.h
)

casparcg_add_library(protocol_http SOURCES ${SOURCES} ${HEADERS})
target_include_directories(protocol_http PRIVATE
    ../..
    )

# Header-only Boost.JSON: `boost_json_src.cpp` provides the whole implementation, so MSVC
# must not autolink the prebuilt library and the Linux bootstrap needs no `json` component.
target_compile_definitions(protocol_http PRIVATE BOOST_JSON_NO_LIB)

source_group(sources ./*)

target_link_libraries(protocol_http PRIVATE common core)
