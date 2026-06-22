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

#include "display_control.h"

#include <common/log.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>

#include <vulkan/vulkan.h>
#endif

#include <filesystem>
#include <thread>

namespace caspar { namespace vulkan_output {

#ifdef _WIN32

bool detach_display_from_desktop(const std::wstring& device_name)
{
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);

    // Query current settings
    if (!EnumDisplaySettingsW(device_name.c_str(), ENUM_CURRENT_SETTINGS, &dm)) {
        CASPAR_LOG(warning) << L"[display_control] Failed to query display settings for " << device_name;
        return false;
    }

    // Detach by setting resolution to 0x0
    dm.dmPelsWidth  = 0;
    dm.dmPelsHeight = 0;
    dm.dmFields     = DM_PELSWIDTH | DM_PELSHEIGHT | DM_POSITION;

    LONG result = ChangeDisplaySettingsExW(device_name.c_str(), &dm, nullptr,
                                           CDS_UPDATEREGISTRY | CDS_NORESET, nullptr);
    if (result != DISP_CHANGE_SUCCESSFUL) {
        CASPAR_LOG(warning) << L"[display_control] ChangeDisplaySettingsExW(detach) failed: " << result;
        return false;
    }

    // Apply the change
    ChangeDisplaySettingsExW(nullptr, nullptr, nullptr, 0, nullptr);

    CASPAR_LOG(info) << L"[display_control] Detached " << device_name << L" from Windows desktop.";
    return true;
}

bool reattach_display_to_desktop(const std::wstring& device_name)
{
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);

    // Get registry settings (the last attached mode)
    if (!EnumDisplaySettingsW(device_name.c_str(), ENUM_REGISTRY_SETTINGS, &dm)) {
        CASPAR_LOG(warning) << L"[display_control] Failed to query registry settings for " << device_name;
        return false;
    }

    // Restore by setting fields back
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL | DM_POSITION |
                  DM_DISPLAYFREQUENCY | DM_DISPLAYFLAGS;

    LONG result = ChangeDisplaySettingsExW(device_name.c_str(), &dm, nullptr,
                                           CDS_UPDATEREGISTRY | CDS_NORESET, nullptr);
    if (result != DISP_CHANGE_SUCCESSFUL) {
        CASPAR_LOG(warning) << L"[display_control] ChangeDisplaySettingsExW(reattach) failed: " << result;
        return false;
    }

    // Apply
    ChangeDisplaySettingsExW(nullptr, nullptr, nullptr, 0, nullptr);

    CASPAR_LOG(info) << L"[display_control] Reattached " << device_name << L" to Windows desktop.";
    return true;
}

bool ensure_vk_khr_display_exported()
{
    // Check if any KHR_display outputs are already available
    auto vkGetInstanceProcAddr_ = vkGetInstanceProcAddr;
    (void)vkGetInstanceProcAddr_;

    // Create a temporary instance to probe KHR_display
    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion         = VK_API_VERSION_1_1;

    const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = 2;
    ci.ppEnabledExtensionNames = extensions;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        return false;

    auto vkEnumPhysDevs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    auto vkGetDispProps = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));

    if (!vkEnumPhysDevs || !vkGetDispProps) {
        vkDestroyInstance(instance, nullptr);
        return false;
    }

    uint32_t dev_count = 0;
    vkEnumPhysDevs(instance, &dev_count, nullptr);
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumPhysDevs(instance, &dev_count, devices.data());

    uint32_t total_displays = 0;
    for (auto& dev : devices) {
        uint32_t count = 0;
        vkGetDispProps(dev, &count, nullptr);
        total_displays += count;
    }

    vkDestroyInstance(instance, nullptr);

    if (total_displays > 0) {
        CASPAR_LOG(debug) << L"[display_control] " << total_displays << L" KHR_display output(s) found.";
        return true;
    }

    // No displays — try to run configureDriver.exe
    CASPAR_LOG(info) << L"[display_control] No KHR_display outputs found. "
                        L"Attempting configureDriver.exe --set 6 (requires admin)...";

    // Look for configureDriver.exe next to our executable or in PATH
    std::wstring exe_path = L"configureDriver.exe";

    wchar_t module_path[MAX_PATH]{};
    if (GetModuleFileNameW(nullptr, module_path, MAX_PATH)) {
        auto dir = std::filesystem::path(module_path).parent_path();
        auto candidate = dir / L"configureDriver.exe";
        if (std::filesystem::exists(candidate))
            exe_path = candidate.wstring();
    }

    SHELLEXECUTEINFOW sei{};
    sei.cbSize       = sizeof(sei);
    sei.fMask        = SEE_MASK_NOCLOSEPROCESS;
    sei.lpVerb       = L"runas";  // UAC elevation
    sei.lpFile       = exe_path.c_str();
    sei.lpParameters = L"--set 6";
    sei.nShow        = SW_HIDE;

    if (!ShellExecuteExW(&sei)) {
        DWORD err = GetLastError();
        if (err == ERROR_CANCELLED) {
            CASPAR_LOG(warning) << L"[display_control] UAC elevation declined by user.";
        } else {
            CASPAR_LOG(warning) << L"[display_control] Failed to launch configureDriver.exe (error " << err << L").";
        }
        return false;
    }

    // Wait up to 10 seconds for it to complete
    if (sei.hProcess) {
        WaitForSingleObject(sei.hProcess, 10000);
        DWORD exit_code = 0;
        GetExitCodeProcess(sei.hProcess, &exit_code);
        CloseHandle(sei.hProcess);
        CASPAR_LOG(info) << L"[display_control] configureDriver.exe exited with code " << exit_code;
    }

    // Give driver time to enumerate new displays
    std::this_thread::sleep_for(std::chrono::seconds(2));

    CASPAR_LOG(info) << L"[display_control] Driver reconfiguration complete. "
                        L"KHR_display outputs should now be available.";
    return true;
}

#else // Linux stubs

bool detach_display_from_desktop(const std::wstring& /*device_name*/)
{
    // Not applicable on Linux (no DWM to detach from)
    return true;
}

bool reattach_display_to_desktop(const std::wstring& /*device_name*/)
{
    return true;
}

bool ensure_vk_khr_display_exported()
{
    // On Linux, KHR_display is generally available without reconfiguration
    return true;
}

#endif

}} // namespace caspar::vulkan_output
