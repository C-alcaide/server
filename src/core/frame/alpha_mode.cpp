/*
 * Copyright (c) contributors to the CasparCG project
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#include "alpha_mode.h"

#include <common/env.h>
#include <common/log.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ptree.hpp>

namespace caspar { namespace core {

bool configured_default_straight()
{
    // Read ONCE. The config cannot change under a running server, and a per-clip read would
    // repeat the warning below for every PLAY of every layer.
    static const bool value = [] {
        const auto mode = env::properties().get<std::wstring>(L"configuration.decode-alpha-mode", L"default");

        if (boost::iequals(mode, L"straight"))
            return true;
        if (boost::iequals(mode, L"default") || boost::iequals(mode, L"premultiplied"))
            return false;

        // An unrecognised value must not silently select one of the two, which is this
        // tree's most repeated defect -- see the AMCP handler sweep in CLAUDE.md.
        CASPAR_LOG(warning) << L"Unknown decode-alpha-mode '" << mode
                            << L"'. Expected default, straight or premultiplied. Using premultiplied.";
        return false;
    }();

    return value;
}

}} // namespace caspar::core
