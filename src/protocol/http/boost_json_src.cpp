/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Boost.JSON built from source into exactly one translation unit.
//
// The alternative is linking the prebuilt library, and that costs a change in BOTH
// bootstraps for no benefit: Windows takes Boost as a precompiled zip and MSVC would
// autolink it, but `Bootstrap_Linux.cmake`'s `find_package(Boost COMPONENTS ...)` list has
// no `json` and `${Boost_LIBRARIES}` is linked only into the executable, not into the
// protocol libraries. Compiling the source here makes the dependency identical on both
// platforms and adds one file to one target.
//
// `BOOST_JSON_NO_LIB` (set on this target in CMakeLists.txt) stops MSVC's auto-linker
// looking for the library this file replaces.

#include "boost_prelude.h"

#include <boost/json/src.hpp>
