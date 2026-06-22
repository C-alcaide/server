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

#include "presentation.h"
#include "output_window.h"

#include <common/log.h>

#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#include <windows.h>
#endif

#include <vector>

namespace caspar { namespace vulkan_output {

// ─── Tier Detection ─────────────────────────────────────────────────────────

static bool has_khr_display(vk::Instance instance)
{
    auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    return fn != nullptr;
}

static bool has_fse([[maybe_unused]] vk::PhysicalDevice physical)
{
#ifdef _WIN32
    auto props = physical.enumerateDeviceExtensionProperties();
    for (const auto& p : props) {
        if (std::string(p.extensionName.data()) == "VK_EXT_full_screen_exclusive")
            return true;
    }
#endif
    return false;
}

presentation_tier detect_presentation_tier(vk::Instance       instance,
                                           vk::PhysicalDevice physical,
                                           const display_info& /*display*/)
{
    if (has_khr_display(instance)) {
        // Probe whether the physical device actually reports any displays
        auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
        uint32_t count = 0;
        fn(physical, &count, nullptr);
        if (count > 0)
            return presentation_tier::khr_display;
    }

    if (has_fse(physical))
        return presentation_tier::full_screen_exclusive;

    return presentation_tier::borderless;
}

std::wstring tier_name(presentation_tier tier)
{
    switch (tier) {
    case presentation_tier::khr_display:           return L"Direct Display (VK_KHR_display)";
    case presentation_tier::full_screen_exclusive: return L"Full-Screen Exclusive";
    case presentation_tier::borderless:            return L"Borderless Fullscreen";
    }
    return L"Unknown";
}

// ─── VK_KHR_display Surface Creation ────────────────────────────────────────

static surface_result try_khr_display_surface(vk::Instance        instance,
                                              vk::PhysicalDevice  physical,
                                              const display_info& display,
                                              uint32_t            target_refresh_mhz)
{
    auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    auto get_modes = reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetDisplayModePropertiesKHR"));
    auto create_surface = reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(
        vkGetInstanceProcAddr(instance, "vkCreateDisplayPlaneSurfaceKHR"));

    if (!get_props || !get_modes || !create_surface)
        return {};

    // Enumerate displays on this physical device
    uint32_t display_count = 0;
    get_props(physical, &display_count, nullptr);
    if (display_count == 0)
        return {};

    std::vector<VkDisplayPropertiesKHR> displays(display_count);
    get_props(physical, &display_count, displays.data());

    // Match display by resolution (best we can do without LUID/EDID matching)
    // output_index is 1-based; use index-1 if in range, else first matching resolution
    int target_idx = display.output_index - 1;
    VkDisplayKHR target_display = VK_NULL_HANDLE;

    if (target_idx >= 0 && target_idx < static_cast<int>(display_count)) {
        target_display = displays[target_idx].display;
    } else {
        // Fallback: match by resolution
        for (const auto& d : displays) {
            if (d.physicalResolution.width == static_cast<uint32_t>(display.width) &&
                d.physicalResolution.height == static_cast<uint32_t>(display.height)) {
                target_display = d.display;
                break;
            }
        }
    }

    if (target_display == VK_NULL_HANDLE && !displays.empty())
        target_display = displays[0].display;

    if (target_display == VK_NULL_HANDLE)
        return {};

    // Get display modes and pick the best match
    uint32_t mode_count = 0;
    get_modes(physical, target_display, &mode_count, nullptr);
    if (mode_count == 0)
        return {};

    std::vector<VkDisplayModePropertiesKHR> modes(mode_count);
    get_modes(physical, target_display, &mode_count, modes.data());

    VkDisplayModeKHR selected_mode = modes[0].displayMode;
    uint32_t         sel_width     = modes[0].parameters.visibleRegion.width;
    uint32_t         sel_height    = modes[0].parameters.visibleRegion.height;

    for (const auto& mode : modes) {
        bool res_match = (mode.parameters.visibleRegion.width == static_cast<uint32_t>(display.width) &&
                          mode.parameters.visibleRegion.height == static_cast<uint32_t>(display.height));
        bool refresh_match = (target_refresh_mhz > 0 &&
                              mode.parameters.refreshRate == target_refresh_mhz);

        if (res_match && refresh_match) {
            selected_mode = mode.displayMode;
            sel_width     = mode.parameters.visibleRegion.width;
            sel_height    = mode.parameters.visibleRegion.height;
            break;
        } else if (res_match) {
            selected_mode = mode.displayMode;
            sel_width     = mode.parameters.visibleRegion.width;
            sel_height    = mode.parameters.visibleRegion.height;
        }
    }

    // Create display surface
    VkDisplaySurfaceCreateInfoKHR ci{};
    ci.sType           = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR;
    ci.displayMode     = selected_mode;
    ci.planeIndex      = 0;
    ci.transform       = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.alphaMode       = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
    ci.imageExtent     = {sel_width, sel_height};

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = create_surface(instance, &ci, nullptr, &surface);
    if (result != VK_SUCCESS) {
        CASPAR_LOG(warning) << L"[vulkan_output] VK_KHR_display surface creation failed: " << static_cast<int>(result);
        return {};
    }

    CASPAR_LOG(info) << L"[vulkan_output] Direct display surface created: " << sel_width << L"x" << sel_height;

    return {vk::SurfaceKHR(surface), presentation_tier::khr_display, vk::DisplayKHR(target_display), sel_width, sel_height};
}

// ─── Win32 Surface with optional FSE ────────────────────────────────────────

#ifdef _WIN32
static surface_result try_win32_surface(vk::Instance        instance,
                                        const display_info& display,
                                        presentation_tier   tier)
{
    // Create the borderless window (used for both FSE and borderless paths)
    // The window is created on the stack here — caller must keep the output_window alive.
    // For this tiered function we just create the surface; the window lifetime is
    // managed by the consumer.

    VkWin32SurfaceCreateInfoKHR ci{};
    ci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hinstance = GetModuleHandleW(nullptr);
    ci.hwnd      = nullptr; // Caller provides HWND via output_window

    // We can't create the surface without a window. Return empty to signal
    // that the caller should create an output_window and use it directly.
    // The tier information is enough for the consumer to know how to proceed.
    surface_result r{};
    r.tier   = tier;
    r.width  = static_cast<uint32_t>(display.width);
    r.height = static_cast<uint32_t>(display.height);
    return r;
}
#endif

// ─── Tiered Surface Creation ────────────────────────────────────────────────

surface_result create_tiered_surface(vk::Instance        instance,
                                     vk::PhysicalDevice  physical,
                                     vk::Device          /*device*/,
                                     const display_info& display,
                                     uint32_t            target_refresh_mhz)
{
    auto tier = detect_presentation_tier(instance, physical, display);
    CASPAR_LOG(info) << L"[vulkan_output] Detected presentation tier: " << tier_name(tier);

    // Tier 1: Try VK_KHR_display (direct-to-hardware)
    if (tier == presentation_tier::khr_display) {
        auto result = try_khr_display_surface(instance, physical, display, target_refresh_mhz);
        if (result.surface) {
            return result;
        }
        CASPAR_LOG(warning) << L"[vulkan_output] VK_KHR_display failed, falling back to FSE.";
        tier = has_fse(physical) ? presentation_tier::full_screen_exclusive : presentation_tier::borderless;
    }

    // Tier 2/3: FSE or borderless — both need a Win32 window (surface created by consumer)
#ifdef _WIN32
    return try_win32_surface(instance, display, tier);
#else
    // Linux: if KHR_display failed, borderless is the only remaining path
    surface_result r{};
    r.tier   = presentation_tier::borderless;
    r.width  = static_cast<uint32_t>(display.width);
    r.height = static_cast<uint32_t>(display.height);
    return r;
#endif
}

// ─── Present Mode Selection ─────────────────────────────────────────────────

vk::PresentModeKHR pick_present_mode(vk::PhysicalDevice physical,
                                      vk::SurfaceKHR    surface,
                                      bool              force_fifo)
{
    if (force_fifo)
        return vk::PresentModeKHR::eFifo;

    auto modes = physical.getSurfacePresentModesKHR(surface);

    // MAILBOX ideal for externally-paced playout:
    // - Channel clock controls submission rate (e.g. 50fps from send())
    // - MAILBOX lets GPU process immediately (no vsync queue stall)
    // - Display refreshes at vsync, picking latest submitted frame
    for (auto m : modes) {
        if (m == vk::PresentModeKHR::eMailbox) {
            CASPAR_LOG(info) << L"[vulkan_output] Using MAILBOX present mode (low-latency).";
            return vk::PresentModeKHR::eMailbox;
        }
    }
    for (auto m : modes) {
        if (m == vk::PresentModeKHR::eFifoRelaxed) {
            CASPAR_LOG(info) << L"[vulkan_output] Using FIFO_RELAXED present mode.";
            return vk::PresentModeKHR::eFifoRelaxed;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

}} // namespace caspar::vulkan_output
