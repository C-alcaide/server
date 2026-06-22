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

#include "display_enum.h"

#include <common/log.h>

#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#endif

namespace caspar { namespace vulkan_output {

#ifdef _WIN32

std::vector<display_info> enumerate_displays()
{
    std::vector<display_info> results;

    // Use DXGI for reliable GPU-to-display mapping
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory)))) {
        CASPAR_LOG(warning) << L"[vulkan_output] Failed to create DXGI factory for display enumeration.";
        return results;
    }

    IDXGIAdapter* adapter = nullptr;
    for (UINT adapter_idx = 0; factory->EnumAdapters(adapter_idx, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapter_idx) {
        DXGI_ADAPTER_DESC adapter_desc{};
        adapter->GetDesc(&adapter_desc);

        std::wstring gpu_name(adapter_desc.Description);

        IDXGIOutput* output = nullptr;
        for (UINT output_idx = 0; adapter->EnumOutputs(output_idx, &output) != DXGI_ERROR_NOT_FOUND; ++output_idx) {
            DXGI_OUTPUT_DESC output_desc{};
            output->GetDesc(&output_desc);

            display_info info;
            info.gpu_index    = static_cast<int>(adapter_idx);
            info.output_index = static_cast<int>(results.size()) + 1; // 1-based global index
            info.gpu_name     = gpu_name;
            info.display_name = output_desc.DeviceName;
            info.pos_x        = output_desc.DesktopCoordinates.left;
            info.pos_y        = output_desc.DesktopCoordinates.top;
            info.width        = output_desc.DesktopCoordinates.right - output_desc.DesktopCoordinates.left;
            info.height       = output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top;

            results.push_back(std::move(info));
            output->Release();
        }
        adapter->Release();
    }
    factory->Release();

    return results;
}

#else

std::vector<display_info> enumerate_displays()
{
    // Linux/macOS: minimal fallback using primary display
    display_info primary;
    primary.gpu_index    = 0;
    primary.output_index = 1;
    primary.gpu_name     = L"GPU 0";
    primary.display_name = L"Primary";
    primary.width        = 1920;
    primary.height       = 1080;
    return {primary};
}

#endif

const display_info* find_display(const std::vector<display_info>& displays,
                                 int                              output_index,
                                 const std::wstring&              display_name)
{
    // If display_name is set, match by substring first
    if (!display_name.empty()) {
        for (const auto& d : displays) {
            if (d.display_name.find(display_name) != std::wstring::npos)
                return &d;
        }
        CASPAR_LOG(warning) << L"[vulkan_output] No display matching name '" << display_name << L"' found.";
    }

    // Fall back to index match
    for (const auto& d : displays) {
        if (d.output_index == output_index)
            return &d;
    }

    return nullptr;
}

}} // namespace caspar::vulkan_output
