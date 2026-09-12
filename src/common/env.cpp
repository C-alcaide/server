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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */
#include "env.h"

#include "version.h"

#include "except.h"
#include "log.h"
#include "os/filesystem.h"
#include "ptree.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <boost/algorithm/string.hpp>
#include <atomic>
#include <fstream>
#include <functional>
#include <thread>

namespace caspar { namespace env {

std::wstring                 initial;
std::wstring                 media;
std::wstring                 log;
bool                         log_enabled = true;
std::wstring                 ftemplate;
std::wstring                 data;
boost::property_tree::wptree pt;

void check_is_configured()
{
    if (pt.empty())
        CASPAR_THROW_EXCEPTION(invalid_operation() << msg_info(L"Environment properties has not been configured"));
}

std::wstring resolve_or_create(const std::wstring& folder)
{
    auto found_path = find_case_insensitive(folder);

    if (found_path)
        return *found_path;
    boost::system::error_code ec;
    boost::filesystem::create_directories(folder, ec);

    if (ec)
        CASPAR_THROW_EXCEPTION(user_error()
                               << msg_info("Failed to create directory " + u8(folder) + " (" + ec.message() + ")"));

    return folder;
}

void ensure_writable(const std::wstring& folder)
{
    // UNIQUE PER PROCESS AND PER CALL, and that is a bug fix rather than tidiness.
    //
    // This probe used to be the fixed name `casparcg_test_writable.empty`. Two servers starting
    // against the same folder -- which is every parallel test run, and any rig where two
    // instances share a data or template directory -- then race on one filename: each creates
    // it and each removes it, so one server's `remove` lands between the other's `ofstream` and
    // its `fail()` check, or its own create fails outright. The loser dies at start-up with
    // "Directory <x> is not writable.", a message that names a PERMISSION problem and means
    // CONCURRENCY.
    //
    // What that cost is out of proportion to the line: `conformance` and `grading` both carry a
    // `--sequential` flag that exists ONLY for this, and their headline is computed over the
    // conversions that SURVIVED -- so a run whose servers lost the race printed a plausible
    // `81/81 conversions within 1.0 LSB` and read exactly like a clean run of a smaller
    // battery. It is recorded in three places as a trap to work around; this removes it.
    //
    // The probe's question is "can this process write here", and a unique name answers it
    // identically while making the answer independent of anyone else asking at the same moment.
    static std::atomic<std::uint64_t> probe_counter{0};

    // The THREAD id rather than a process id: it is std, needs no extra header, and is just as
    // unique between two servers as a pid is -- nothing here needs the number to mean anything.
    const auto unique = std::to_wstring(std::hash<std::thread::id>{}(std::this_thread::get_id())) +
                        L"_" + std::to_wstring(probe_counter.fetch_add(1));

    boost::system::error_code   ec;
    boost::filesystem::path     test_file(folder + L"/casparcg_writable_" + unique + L".tmp");
    boost::filesystem::ofstream out(test_file);

    if (out.fail()) {
        boost::filesystem::remove(test_file, ec);
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Directory " + folder + L" is not writable."));
    }

    out.close();
    boost::filesystem::remove(test_file, ec);
}

void configure(const std::wstring& filename)
{
    initial = clean_path(boost::filesystem::initial_path().wstring());

    std::wstring fullpath = filename;
    if (!boost::filesystem::exists(fullpath)) {
        fullpath = initial + L"/" + filename;
    }

    if (!boost::filesystem::exists(fullpath)) {
        CASPAR_LOG(fatal) << L"### Configuration file " + filename + L" was not found. ###";
        CASPAR_THROW_EXCEPTION(expected_user_error()
                               << msg_info(L"Configuration file " + fullpath + L" was not found."));
    }

    try {
        boost::filesystem::wifstream file(fullpath);
        boost::property_tree::read_xml(file,
                                       pt,
                                       boost::property_tree::xml_parser::trim_whitespace |
                                           boost::property_tree::xml_parser::no_comments);

        auto paths = ptree_get_child(pt, L"configuration.paths");
        media      = clean_path(paths.get(L"media-path", initial + L"/media/"));

        auto log_path_node = paths.get_child(L"log-path");
        log_enabled        = !log_path_node.get(L"<xmlattr>.disabled", false);
        if (log_enabled) {
            log = clean_path(log_path_node.get_value(initial + L"/log/"));
        }

        ftemplate =
            clean_path(boost::filesystem::absolute(paths.get(L"template-path", initial + L"/template/")).wstring());
        data = clean_path(paths.get(L"data-path", initial + L"/data/"));
    } catch (...) {
        CASPAR_LOG(error) << L" ### Invalid configuration file. ###";
        throw;
    }

    media     = ensure_trailing_slash(resolve_or_create(media));
    ftemplate = ensure_trailing_slash(resolve_or_create(ftemplate));
    data      = ensure_trailing_slash(resolve_or_create(data));

    if (log_enabled) {
        log = ensure_trailing_slash(resolve_or_create(log));
        ensure_writable(log);
    }

    ensure_writable(ftemplate);
    ensure_writable(data);
}

const std::wstring& initial_folder()
{
    check_is_configured();
    return initial;
}

const std::wstring& media_folder()
{
    check_is_configured();
    return media;
}

const std::wstring& log_folder()
{
    check_is_configured();
    return log;
}

bool log_to_file()
{
    check_is_configured();
    return log_enabled;
}

const std::wstring& template_folder()
{
    check_is_configured();
    return ftemplate;
}

const std::wstring& data_folder()
{
    check_is_configured();
    return data;
}

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

const std::wstring& version()
{
    static std::wstring ver = u16(EXPAND_AND_QUOTE(CASPAR_GEN) "." EXPAND_AND_QUOTE(CASPAR_MAJOR) "." EXPAND_AND_QUOTE(
        CASPAR_MINOR) " " CASPAR_HASH " " CASPAR_TAG);
    return ver;
}

const boost::property_tree::wptree& properties()
{
    check_is_configured();
    return pt;
}

void log_configuration_warnings()
{
    if (pt.empty())
        return;
}

}} // namespace caspar::env
