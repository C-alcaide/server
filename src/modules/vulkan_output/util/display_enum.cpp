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

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>

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

#else // Linux: enumerate via VK_KHR_display

std::vector<display_info> enumerate_displays()
{
    std::vector<display_info> results;

    // Create temporary VkInstance with KHR_display to enumerate
    VkApplicationInfo app_info{};
    app_info.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_1;

    const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = 2;
    ci.ppEnabledExtensionNames = extensions;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
        CASPAR_LOG(warning) << L"[vulkan_output] Failed to create VkInstance for display enumeration.";
        return results;
    }

    auto vkEnumPhysDevs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    auto vkGetDispProps = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    auto vkGetPhysDevProps = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));

    if (!vkEnumPhysDevs || !vkGetDispProps || !vkGetPhysDevProps) {
        vkDestroyInstance(instance, nullptr);
        CASPAR_LOG(warning) << L"[vulkan_output] VK_KHR_display functions not available.";
        return results;
    }

    uint32_t dev_count = 0;
    vkEnumPhysDevs(instance, &dev_count, nullptr);
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumPhysDevs(instance, &dev_count, devices.data());

    for (uint32_t gpu_idx = 0; gpu_idx < dev_count; ++gpu_idx) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysDevProps(devices[gpu_idx], &props);

        std::wstring gpu_name(props.deviceName, props.deviceName + strlen(props.deviceName));

        uint32_t display_count = 0;
        vkGetDispProps(devices[gpu_idx], &display_count, nullptr);
        if (display_count == 0)
            continue;

        std::vector<VkDisplayPropertiesKHR> display_props(display_count);
        vkGetDispProps(devices[gpu_idx], &display_count, display_props.data());

        for (uint32_t d = 0; d < display_count; ++d) {
            const auto& dp = display_props[d];

            display_info info;
            info.gpu_index    = static_cast<int>(gpu_idx);
            info.output_index = static_cast<int>(results.size()) + 1;
            info.gpu_name     = gpu_name;
            info.width        = dp.physicalResolution.width;
            info.height       = dp.physicalResolution.height;
            info.pos_x        = 0;
            info.pos_y        = 0;

            if (dp.displayName)
                info.display_name = std::wstring(dp.displayName, dp.displayName + strlen(dp.displayName));
            else
                info.display_name = L"Display " + std::to_wstring(d + 1);

            results.push_back(std::move(info));
        }
    }

    vkDestroyInstance(instance, nullptr);

    if (results.empty()) {
        CASPAR_LOG(warning) << L"[vulkan_output] No VK_KHR_display outputs found. "
                               L"Ensure displays are not managed by X11/Wayland compositor.";
    }

    return results;
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
