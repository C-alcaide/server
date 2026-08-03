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

#pragma once

// Host-OS capability probes for the vulkan_output tier decision.
//
// The only OS gate that matters today: VK_KHR_display direct scanout needs the
// "Specialized Monitors" / "Remove display from desktop" feature, which exists
// only on Windows 11 (build 22000+).  On Windows 10, `configureDriver.exe
// --set 6` may exit 0 but the extension never yields a display, so attempting
// the recovery sequence costs a UAC prompt and a desktop detach/reattach cycle
// that cannot succeed.
//
// This is a *suppressor*, not the authority: even on Windows 11 the real test
// is whether vkGetPhysicalDeviceDisplayPropertiesKHR returns a display (the
// driver only picks up --set 6 after a reboot).

#include <cstdint>
#include <string>

#ifdef _WIN32
#include <common/os/windows/windows.h>
#endif

namespace caspar { namespace vulkan_output { namespace os_support {

struct os_version_info
{
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t build = 0;
    bool     valid = false;
};

// First retail Windows 11 build.  Both Win10 and Win11 report major.minor
// 10.0 — the build number is the only discriminator.  Declared unconditionally
// so diagnostic messages referencing it compile on every platform.
inline constexpr uint32_t kWindows11FirstBuild = 22000;

#ifdef _WIN32

// GetVersionExW and VerifyVersionInfo are shimmed by the application manifest:
// without a supportedOS GUID for Win10+ they report 6.2 regardless of the real
// OS.  RtlGetVersion bypasses the shim and always reports the true version.
inline os_version_info query_os_version()
{
    static const os_version_info info = [] {
        os_version_info v;

        using rtl_get_version_t = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);

        auto ntdll = GetModuleHandleW(L"ntdll.dll");
        if (!ntdll)
            return v;

        auto rtl_get_version = reinterpret_cast<rtl_get_version_t>(GetProcAddress(ntdll, "RtlGetVersion"));
        if (!rtl_get_version)
            return v;

        RTL_OSVERSIONINFOW vi{};
        vi.dwOSVersionInfoSize = sizeof(vi);
        if (rtl_get_version(&vi) != 0) // 0 == STATUS_SUCCESS
            return v;

        v.major = vi.dwMajorVersion;
        v.minor = vi.dwMinorVersion;
        v.build = vi.dwBuildNumber;
        v.valid = true;
        return v;
    }();
    return info;
}

// True only when the OS can support VK_KHR_display direct scanout.
// If the version cannot be determined we answer false: the cost of a wrong
// "true" is a UAC prompt plus a desktop topology change on a playout machine,
// while a wrong "false" only forgoes a path that was probably unavailable.
inline bool khr_display_supported_by_os()
{
    const auto v = query_os_version();
    if (!v.valid)
        return false;
    return v.major > 10 || (v.major == 10 && v.build >= kWindows11FirstBuild);
}

// Product-type values for the workstation SKUs.  The Windows SDK shipped with
// VS 18 (winnt.h) does not define PRODUCT_PRO_WORKSTATION at all, so relying on
// the symbol silently drops the case — spell the values out instead.
// Verified by GetProductInfo on Windows 10 Pro for Workstations: 0xA1.
inline constexpr DWORD kProductProWorkstation  = 0x000000A1; // PRODUCT_PRO_WORKSTATION
inline constexpr DWORD kProductProWorkstationN = 0x000000A2; // PRODUCT_PRO_WORKSTATION_N
inline constexpr DWORD kProductIotEnterprise   = 0x000000BC; // PRODUCT_IOTENTERPRISE

// Editions documented to expose the "Remove display from desktop" API.
// Advisory only — used to explain a failure, never to block the attempt, since
// the authoritative test is display enumeration itself.
inline bool edition_documented_for_specialized_monitors()
{
    DWORD product_type = 0;
    if (!GetProductInfo(10, 0, 0, 0, &product_type))
        return false;

    switch (product_type) {
        case PRODUCT_ENTERPRISE:
        case PRODUCT_ENTERPRISE_E:
        case PRODUCT_ENTERPRISE_N:
#ifdef PRODUCT_ENTERPRISE_S
        case PRODUCT_ENTERPRISE_S:
#endif
        case kProductProWorkstation:
        case kProductProWorkstationN:
        case kProductIotEnterprise:
            return true;
        default:
            return false;
    }
}

inline std::wstring os_version_string()
{
    const auto v = query_os_version();
    if (!v.valid)
        return L"Windows (version unknown)";

    // 10.0 covers both; the build number decides which name is honest.
    std::wstring name = (v.major == 10 && v.build < kWindows11FirstBuild) ? L"Windows 10"
                        : (v.major >= 10) ? L"Windows 11"
                                          : L"Windows";

    return name + L" build " + std::to_wstring(v.build);
}

#else // !_WIN32

// Linux: VK_KHR_display is always available with the proprietary NVIDIA driver;
// no OS-level gate exists.
inline os_version_info query_os_version() { return {}; }
inline bool            khr_display_supported_by_os() { return true; }
inline bool            edition_documented_for_specialized_monitors() { return true; }
inline std::wstring    os_version_string() { return L"Linux"; }

#endif

}}} // namespace caspar::vulkan_output::os_support
