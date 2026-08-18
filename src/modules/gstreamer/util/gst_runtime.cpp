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

#include "gst_runtime.h"

#include <common/env.h>
#include <common/except.h>
#include <common/log.h>
#include <common/utf.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>

#include <gst/gst.h>

#include <iterator>
#include <mutex>
#include <set>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace caspar { namespace gstreamer { namespace runtime {

namespace {

std::wstring configured_root()
{
    auto configured = env::properties().get(L"configuration.gstreamer.path", L"");

    if (configured.empty()) {
#ifdef _WIN32
        // The Windows installer sets this only when installed for all users; the per-user
        // install leaves the environment untouched, which is why the config element exists.
        if (const char* from_env = std::getenv("GSTREAMER_1_0_ROOT_MSVC_X86_64"))
            configured = u16(std::string(from_env));
#else
        if (const char* from_env = std::getenv("GSTREAMER_1_0_ROOT"))
            configured = u16(std::string(from_env));
#endif
    }

    if (configured.empty())
        return L"";

    boost::filesystem::path root(configured);
    boost::system::error_code ec;
    if (!boost::filesystem::is_directory(root, ec))
        return L"";

    return root.wstring();
}

#ifdef _WIN32
/// Loads one library by its full path. LOAD_WITH_ALTERED_SEARCH_PATH makes the library's own
/// directory the first place its dependencies are looked for, so GStreamer's ~140 DLLs resolve
/// among themselves without any of them reaching PATH — which is what keeps GStreamer's FFmpeg
/// out of every other process on the machine.
bool load_by_path(const boost::filesystem::path& dll)
{
    if (GetModuleHandleW(dll.filename().wstring().c_str()) != nullptr)
        return true; // already in the process, by base name; a second load would be the same module

    return LoadLibraryExW(dll.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH) != nullptr;
}

std::set<std::wstring> dll_names(const boost::filesystem::path& directory)
{
    std::set<std::wstring>    names;
    boost::system::error_code ec;

    for (boost::filesystem::directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
        auto name = it->path().filename().wstring();
        if (boost::algorithm::iends_with(name, L".dll"))
            names.insert(boost::algorithm::to_lower_copy(name));
    }

    return names;
}

/// GStreamer's plugins are loaded by GStreamer, with a plain LoadLibrary that knows nothing of
/// the explicit paths above — so a plugin needing gstaudio-1.0-0.dll or orc-0.4-0.dll finds
/// nothing and is dropped, which is how audioresample went missing. Its bin/ has to be on the
/// process search path.
///
/// SetDllDirectory rather than PATH: this process only, and it is searched *after* the
/// application directory, so nothing shipped next to casparcg.exe can be displaced by a name
/// in GStreamer's bin. That is the safety argument, and it is checked rather than asserted —
/// the two directories are compared and every shared base name is reported. On FFmpeg 8 the
/// set is empty; on FFmpeg 7 it was six, every one of them an av* library, which is why this
/// module could not have existed before the migration.
void add_to_search_path(const boost::filesystem::path& bin)
{
    const auto ours   = dll_names(boost::filesystem::path(env::initial_folder()));
    const auto theirs = dll_names(bin);

    std::vector<std::wstring> shared;
    std::set_intersection(ours.begin(), ours.end(), theirs.begin(), theirs.end(), std::back_inserter(shared));

    if (!shared.empty()) {
        CASPAR_LOG(warning) << L"[gstreamer] " << shared.size()
                            << L" DLL name(s) exist both next to casparcg.exe and in GStreamer's bin. Ours win, "
                               L"because the application directory is searched first, but a GStreamer plugin "
                               L"resolving against ours may then fail to load:";
        for (const auto& name : shared)
            CASPAR_LOG(warning) << L"[gstreamer]     " << name;
    }

    SetDllDirectoryW(bin.wstring().c_str());
}
#endif

void install_log_handler()
{
    // GStreamer's own categories, routed into our log rather than stderr. WARNING is the
    // default threshold: a failing pipeline says why, and a working one stays quiet.
    gst_debug_set_default_threshold(GST_LEVEL_WARNING);
    gst_debug_remove_log_function(gst_debug_log_default);
    gst_debug_add_log_function(
        [](GstDebugCategory* category,
           GstDebugLevel     level,
           const gchar*      file,
           const gchar*      function,
           gint              line,
           GObject*          object,
           GstDebugMessage*  message,
           gpointer          user_data) {
            const auto text = std::string(gst_debug_category_get_name(category)) + ": " +
                              std::string(gst_debug_message_get(message));

            switch (level) {
                case GST_LEVEL_ERROR:
                    CASPAR_LOG(error) << L"[gstreamer] " << u16(text);
                    break;
                case GST_LEVEL_WARNING:
                    CASPAR_LOG(warning) << L"[gstreamer] " << u16(text);
                    break;
                case GST_LEVEL_FIXME:
                case GST_LEVEL_INFO:
                    CASPAR_LOG(info) << L"[gstreamer] " << u16(text);
                    break;
                default:
                    CASPAR_LOG(debug) << L"[gstreamer] " << u16(text);
                    break;
            }
        },
        nullptr,
        nullptr);
}

struct state
{
    std::once_flag once;
    bool           initialized = false;
    std::wstring   error;
    std::wstring   version;
};

state& get_state()
{
    static state s;
    return s;
}

void initialize_once(state& s)
{
    const auto root = root_path();

    if (root.empty()) {
        s.error = L"No GStreamer installation found. Set <configuration><gstreamer><path> to the "
                  L"installation root (the directory holding bin/, lib/ and libexec/).";
        return;
    }

    const boost::filesystem::path base(root);
    const auto                    bin     = base / L"bin";
    const auto                    plugins = base / L"lib" / L"gstreamer-1.0";
    const auto                    scanner = base / L"libexec" / L"gstreamer-1.0" / L"gst-plugin-scanner.exe";

#ifdef _WIN32
    // Order matters only in that each library's dependencies must be resolvable from bin/,
    // which LOAD_WITH_ALTERED_SEARCH_PATH guarantees. These five are what this module's
    // delay-load stubs need; everything else arrives as a dependency of one of them.
    for (const auto* name : {L"glib-2.0-0.dll",
                             L"gobject-2.0-0.dll",
                             L"gstreamer-1.0-0.dll",
                             L"gstbase-1.0-0.dll",
                             L"gstvideo-1.0-0.dll",
                             L"gstapp-1.0-0.dll"}) {
        if (!load_by_path(bin / name)) {
            s.error = L"Failed to load " + std::wstring(name) + L" from " + bin.wstring() +
                      L" (error " + std::to_wstring(GetLastError()) + L").";
            return;
        }
    }

    add_to_search_path(bin);
#endif

    // Everything below this line is a delay-loaded call, and is only reachable because the
    // loads above succeeded.

    // A private registry, so that scanning this install cannot disturb — or be disturbed by —
    // a registry any other GStreamer application on the machine shares.
    const auto registry = boost::filesystem::path(env::data_folder()) / L"gstreamer-registry.bin";

    auto set_env = [](const char* name, const std::wstring& value) {
        g_setenv(name, u8(value).c_str(), TRUE);
    };
    set_env("GST_PLUGIN_SYSTEM_PATH_1_0", plugins.wstring());
    set_env("GST_PLUGIN_PATH_1_0", L"");
    set_env("GST_REGISTRY", registry.wstring());
    if (boost::filesystem::exists(scanner))
        set_env("GST_PLUGIN_SCANNER_1_0", scanner.wstring());

    GError* error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &error)) {
        s.error = L"gst_init failed: " + (error ? u16(std::string(error->message)) : L"unknown error");
        if (error)
            g_error_free(error);
        return;
    }

    install_log_handler();

    s.version     = u16(std::string(gst_version_string()));
    s.initialized = true;

    CASPAR_LOG(info) << L"[gstreamer] Initialized " << s.version << L" from " << root;
    CASPAR_LOG(debug) << L"[gstreamer] Plugins " << plugins.wstring() << L", registry " << registry.wstring();
}

} // namespace

const std::wstring& root_path()
{
    static const std::wstring root = configured_root();
    return root;
}

void ensure_initialized()
{
    auto& s = get_state();
    std::call_once(s.once, [&] { initialize_once(s); });

    if (!s.initialized) {
        // Logged here rather than left to the handler: the producer registry rethrows user_error
        // immediately — which is what stops a missing installation being reported as
        // "404 File not found" by the next factory in line — but AMCP answers a user_error with
        // "Check syntax" and discards the message. Without this line the operator gets nothing
        // to act on.
        CASPAR_LOG(error) << L"[gstreamer] " << s.error;
        CASPAR_THROW_EXCEPTION(not_supported() << msg_info(u8(s.error)));
    }
}

bool is_initialized() { return get_state().initialized; }

const std::wstring& version() { return get_state().version; }

}}} // namespace caspar::gstreamer::runtime
