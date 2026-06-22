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

#include "icd_filter.h"

#include <common/log.h>

#ifdef _WIN32

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace caspar { namespace vulkan_common {

void filter_stale_nvidia_icds()
{
    static std::once_flag once;
    std::call_once(once, [] {
        namespace fs = std::filesystem;
        try {
            fs::path driver_store = L"C:\\Windows\\System32\\DriverStore\\FileRepository";
            if (!fs::exists(driver_store))
                return;

            // Collect all nv-vk64.json files with their parent directory timestamps
            struct icd_entry
            {
                fs::path             json_path;
                fs::file_time_type   dir_time;
            };
            std::vector<icd_entry> icd_files;

            for (auto& dir : fs::directory_iterator(driver_store)) {
                if (!dir.is_directory())
                    continue;
                auto name = dir.path().filename().wstring();
                if (name.find(L"nv_disp") == std::wstring::npos && name.find(L"nv_lh") == std::wstring::npos)
                    continue;
                auto json = dir.path() / L"nv-vk64.json";
                if (fs::exists(json)) {
                    icd_files.push_back({json, dir.last_write_time()});
                }
            }

            if (icd_files.size() <= 1)
                return; // Only one ICD — no filtering needed

            // Sort by directory timestamp, newest first
            std::sort(icd_files.begin(), icd_files.end(),
                      [](const icd_entry& a, const icd_entry& b) { return a.dir_time > b.dir_time; });

            // Use only the newest ICD
            auto&        newest  = icd_files[0].json_path;
            std::wstring env_val = newest.wstring();
            _wputenv_s(L"VK_DRIVER_FILES", env_val.c_str());

            CASPAR_LOG(info) << L"[vulkan] Filtered " << icd_files.size() << L" NVIDIA ICD(s) - using newest: "
                             << newest.filename().wstring() << L" from "
                             << newest.parent_path().filename().wstring();

            for (size_t i = 1; i < icd_files.size(); ++i) {
                CASPAR_LOG(warning) << L"[vulkan] Skipping stale ICD: "
                                    << icd_files[i].json_path.parent_path().filename().wstring();
            }
        } catch (const std::exception& e) {
            CASPAR_LOG(debug) << L"[vulkan] ICD filtering skipped: " << e.what();
        }
    });
}

}} // namespace caspar::vulkan_common

#else // non-Windows

namespace caspar { namespace vulkan_common {

void filter_stale_nvidia_icds()
{
    // No-op on Linux/Mac — the ICD problem is Windows-specific.
}

}} // namespace caspar::vulkan_common

#endif
