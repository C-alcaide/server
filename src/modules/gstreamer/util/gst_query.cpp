/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#include "../StdAfx.h"

#include "gst_query.h"

#include "gst_runtime.h"

#include <common/utf.h>

#include <boost/algorithm/string.hpp>

#include <gst/gst.h>

#include <algorithm>
#include <sstream>
#include <vector>

namespace caspar { namespace gstreamer {

namespace {

/// How many factories a LIST will print. A substring like "src" matches hundreds, and an
/// AMCP client reading a reply of unbounded length is a worse outcome than a truncated
/// answer that says it was truncated.
constexpr std::size_t max_listed = 100;

struct registry_counts
{
    int plugins     = 0;
    int features    = 0;
    int blacklisted = 0;
};

registry_counts count_registry()
{
    registry_counts counts;

    auto* registry = gst_registry_get();
    if (registry == nullptr)
        return counts;

    auto* plugins = gst_registry_get_plugin_list(registry);
    for (auto* p = plugins; p != nullptr; p = p->next) {
        auto* plugin = GST_PLUGIN(p->data);
        ++counts.plugins;

        // The flag rather than gst_plugin_is_blacklisted(): that function is internal to
        // GStreamer and not in the public headers, while the flag it reads is.
        if (GST_OBJECT_FLAG_IS_SET(plugin, GST_PLUGIN_FLAG_BLACKLISTED)) {
            ++counts.blacklisted;
            continue;
        }

        auto* features =
            gst_registry_get_feature_list_by_plugin(registry, gst_plugin_get_name(plugin));
        for (auto* f = features; f != nullptr; f = f->next)
            ++counts.features;
        gst_plugin_feature_list_free(features);
    }
    gst_plugin_list_free(plugins);

    return counts;
}

} // namespace

std::wstring info_command(protocol::amcp::command_context& ctx)
{
    if (!runtime::is_initialized()) {
        // Deliberately not initialising it here. INFO is a question about the running server,
        // and a query that changes what it is asked about is a bad query — a `GST INFO` that
        // loaded GStreamer would make "was it loaded?" unanswerable.
        std::wstringstream reply;
        reply << L"201 GST INFO OK\r\n";
        reply << L"not initialised\r\n";
        reply << L"path " << (runtime::root_path().empty() ? L"(none configured)" : runtime::root_path()) << L"\r\n";
        reply << L"\r\n";
        return reply.str();
    }

    const auto counts = count_registry();

    std::wstringstream reply;
    reply << L"201 GST INFO OK\r\n";
    reply << L"version " << runtime::version() << L"\r\n";
    reply << L"path " << runtime::root_path() << L"\r\n";
    reply << L"plugins " << counts.plugins << L"\r\n";
    reply << L"features " << counts.features << L"\r\n";
    reply << L"blacklisted " << counts.blacklisted << L"\r\n";
    reply << L"\r\n";
    return reply.str();
}

std::wstring list_command(protocol::amcp::command_context& ctx)
{
    if (ctx.parameters.empty())
        return L"402 GST LIST FAILED\r\n";

    if (!runtime::is_initialized())
        return L"501 GST LIST FAILED\r\n";

    const auto needle = boost::algorithm::to_lower_copy(u8(ctx.parameters.at(0)));

    std::vector<std::wstring> matches;

    auto* registry = gst_registry_get();
    auto* features = registry != nullptr
                         ? gst_registry_get_feature_list(registry, GST_TYPE_ELEMENT_FACTORY)
                         : nullptr;

    for (auto* f = features; f != nullptr; f = f->next) {
        auto* feature = GST_PLUGIN_FEATURE(f->data);
        const std::string name = gst_plugin_feature_get_name(feature);

        if (boost::algorithm::to_lower_copy(name).find(needle) == std::string::npos)
            continue;

        const auto* description = gst_element_factory_get_metadata(GST_ELEMENT_FACTORY(feature),
                                                                   GST_ELEMENT_METADATA_LONGNAME);
        matches.push_back(u16(name) + L" \"" + (description ? u16(std::string(description)) : L"") + L"\"");
    }
    gst_plugin_feature_list_free(features);

    std::sort(matches.begin(), matches.end());

    std::wstringstream reply;
    reply << L"200 GST LIST OK\r\n";
    for (std::size_t n = 0; n < matches.size() && n < max_listed; ++n)
        reply << matches[n] << L"\r\n";

    if (matches.size() > max_listed)
        reply << L"... " << (matches.size() - max_listed) << L" more not listed\r\n";

    reply << L"\r\n";
    return reply.str();
}

}} // namespace caspar::gstreamer
