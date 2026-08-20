/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include "gpu_extension_requests.h"

#include <algorithm>

namespace caspar { namespace common {

namespace {
// Function-local rather than a namespace-scope object: callers register from static
// initialisers, and a function-local static is constructed on first use rather than in an
// order the linker chooses.
std::vector<std::string>& store()
{
    static std::vector<std::string> names;
    return names;
}
} // namespace

void register_vulkan_device_extension_request(std::vector<std::string> names)
{
    auto& s = store();
    for (auto& n : names) {
        if (n.empty())
            continue;
        if (std::find(s.begin(), s.end(), n) == s.end())
            s.push_back(std::move(n));
    }
}

const std::vector<std::string>& requested_vulkan_device_extensions() { return store(); }

}} // namespace caspar::common
