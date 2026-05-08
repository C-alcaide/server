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

#include "vulkan_device.h"

#include <common/except.h>
#include <common/log.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace caspar { namespace vulkan_output {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult result_ = (call);                                                                                     \
        if (result_ != VK_SUCCESS) {                                                                                   \
            CASPAR_THROW_EXCEPTION(caspar_exception()                                                                  \
                                   << msg_info("Vulkan call failed: " #call " = " + std::to_string(result_)));         \
        }                                                                                                              \
    } while (0)

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
                                               VkDebugUtilsMessageTypeFlagsEXT             type,
                                               const VkDebugUtilsMessengerCallbackDataEXT* data,
                                               void*                                       user)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        CASPAR_LOG(warning) << "[vulkan] " << data->pMessage;
    }
    return VK_FALSE;
}

// Detect professional GPU by device name pattern.
// NVIDIA's professional GPUs: Quadro (legacy), RTX A-series (Ampere/Ada workstation),
// RTX 4000/5000/6000 Ada (current workstation line).
bool is_professional_gpu(const char* name)
{
    std::string n(name);
    // Case-insensitive substring matching
    std::string lower;
    lower.resize(n.size());
    std::transform(n.begin(), n.end(), lower.begin(), ::tolower);

    if (lower.find("quadro") != std::string::npos)
        return true;
    // RTX A-series: "RTX A2000", "RTX A4000", "RTX A5000", "RTX A6000"
    if (lower.find("rtx a") != std::string::npos)
        return true;
    // Ada workstation: "RTX 4000 Ada", "RTX 5000 Ada", "RTX 6000 Ada"
    if (lower.find("ada") != std::string::npos)
        return true;
    // Tesla / data center
    if (lower.find("tesla") != std::string::npos)
        return true;
    return false;
}

} // namespace

vulkan_device::vulkan_device(int gpu_index, int output_index)
{
    create_instance();
    select_physical_device(gpu_index);
    create_logical_device();
}

vulkan_device::~vulkan_device()
{
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);

    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (func)
            func(instance_, debug_messenger_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

void vulkan_device::create_instance()
{
    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "CasparCG";
    app_info.applicationVersion = VK_MAKE_VERSION(2, 5, 0);
    app_info.pEngineName        = "CasparCG";
    app_info.engineVersion      = VK_MAKE_VERSION(2, 5, 0);
    app_info.apiVersion         = VK_API_VERSION_1_3;

    std::vector<const char*> desired_extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_KHR_DISPLAY_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    // Enumerate available instance extensions and filter
    uint32_t avail_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, nullptr);
    std::vector<VkExtensionProperties> avail_exts(avail_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, avail_exts.data());

    std::vector<const char*> extensions;
    for (auto* desired : desired_extensions) {
        bool found = false;
        for (const auto& avail : avail_exts) {
            if (strcmp(avail.extensionName, desired) == 0) {
                found = true;
                break;
            }
        }
        if (found) {
            extensions.push_back(desired);
        } else {
            CASPAR_LOG(debug) << L"[vulkan] Instance extension not available: " << desired;
        }
    }

    std::vector<const char*> layers;
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    VkInstanceCreateInfo create_info{};
    create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo        = &app_info;
    create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames     = layers.data();

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));

    // Setup debug messenger
    auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create_messenger) {
        VkDebugUtilsMessengerCreateInfoEXT dbg_info{};
        dbg_info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg_info.pfnUserCallback = debug_callback;
        create_messenger(instance_, &dbg_info, nullptr, &debug_messenger_);
    }
}

void vulkan_device::select_physical_device(int gpu_index)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No Vulkan-capable GPUs found"));

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    if (gpu_index < 0 || gpu_index >= static_cast<int>(count))
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("Invalid GPU index: " + std::to_string(gpu_index)));

    physical_device_ = devices[gpu_index];

    // Determine tier: check for VK_KHR_display support
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> ext_props(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, ext_props.data());

    bool has_display   = false;
    bool has_fse       = false;
    bool has_ext_mem   = false;
    bool has_present_barrier = false;

    for (const auto& ext : ext_props) {
        if (strcmp(ext.extensionName, VK_KHR_DISPLAY_EXTENSION_NAME) == 0)
            has_display = true;
        if (strcmp(ext.extensionName, VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME) == 0)
            has_fse = true;
        if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0)
            has_ext_mem = true;
        if (strcmp(ext.extensionName, VK_NV_PRESENT_BARRIER_EXTENSION_NAME) == 0)
            has_present_barrier = true;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);

    if (has_display)
        tier_ = gpu_tier::pro;
    else if (has_present_barrier || is_professional_gpu(props.deviceName))
        tier_ = gpu_tier::pro;
    else if (has_fse)
        tier_ = gpu_tier::consumer;
    else
        tier_ = gpu_tier::none;

    CASPAR_LOG(info) << L"[vulkan_output] Selected GPU " << gpu_index << L": " << props.deviceName
                     << L" (tier=" << (tier_ == gpu_tier::pro ? L"pro" : tier_ == gpu_tier::consumer ? L"consumer" : L"none")
                     << L", ext_mem=" << has_ext_mem
                     << L", present_barrier=" << has_present_barrier << L")";

    gpu_index_ = gpu_index;

    // Query device LUID for cross-API GPU matching
    VkPhysicalDeviceIDProperties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);
    if (id_props.deviceLUIDValid) {
        memcpy(device_luid_, id_props.deviceLUID, 8);
        device_luid_valid_ = true;
    }
}

void vulkan_device::create_logical_device()
{
    // Find a queue family that supports graphics + present
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    present_queue_family_ = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            present_queue_family_ = i;
            break;
        }
    }

    if (present_queue_family_ == UINT32_MAX)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No graphics queue family found"));

    float                   queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = present_queue_family_;
    queue_info.queueCount       = 1;
    queue_info.pQueuePriorities = &queue_priority;

    // Request high GPU scheduling priority so Vulkan present work preempts
    // other GPU clients (e.g. OpenGL screen consumer, DWM compositor).
    VkDeviceQueueGlobalPriorityCreateInfoKHR priority_info{};
    priority_info.sType          = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
    priority_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
    bool use_global_priority     = false;

    // Core required extension (must have)
    std::vector<const char*> required_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // Extensions we want but can survive without
    std::vector<const char*> desired_device_exts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_EXT_HDR_METADATA_EXTENSION_NAME,
        VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME,
    };

    if (tier_ == gpu_tier::consumer) {
        desired_device_exts.push_back(VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME);
    }

    // Check and add optional extensions if available
    {
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> ext_props(ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, ext_props.data());

        auto is_available = [&](const char* name) {
            for (const auto& ext : ext_props) {
                if (strcmp(ext.extensionName, name) == 0)
                    return true;
            }
            return false;
        };

        // Start with required
        std::vector<const char*> device_extensions = required_extensions;

        // Add desired ones that are available
        for (auto* desired : desired_device_exts) {
            if (is_available(desired)) {
                device_extensions.push_back(desired);
            } else {
                CASPAR_LOG(debug) << L"[vulkan] Device extension not available: " << desired;
            }
        }

        // Add optional NV extensions
        if (is_available(VK_NV_PRESENT_BARRIER_EXTENSION_NAME))
            device_extensions.push_back(VK_NV_PRESENT_BARRIER_EXTENSION_NAME);
        if (is_available(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME) && tier_ == gpu_tier::pro)
            device_extensions.push_back(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME);

        // Store enabled extensions for runtime queries
        for (const auto* ext : device_extensions)
            enabled_extensions_.emplace_back(ext);

        // Chain global priority if the extension was enabled
        if (std::find_if(device_extensions.begin(), device_extensions.end(), [](const char* e) {
                return strcmp(e, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME) == 0;
            }) != device_extensions.end()) {
            queue_info.pNext = &priority_info;
            use_global_priority = true;
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo device_info{};
        device_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount    = 1;
        device_info.pQueueCreateInfos       = &queue_info;
        device_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();
        device_info.pEnabledFeatures        = &features;

        VkResult create_result = VK_ERROR_UNKNOWN;

        // Try creating device with HIGH priority first, fall back to MEDIUM, then no priority
        if (use_global_priority) {
            priority_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
            create_result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
            if (create_result == VK_SUCCESS) {
                CASPAR_LOG(info) << L"[vulkan] Queue created with HIGH global priority.";
            } else {
                CASPAR_LOG(debug) << L"[vulkan] HIGH priority rejected (code " << create_result
                                  << L"), trying MEDIUM...";
                priority_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
                create_result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
                if (create_result == VK_SUCCESS) {
                    CASPAR_LOG(info) << L"[vulkan] Queue created with MEDIUM global priority.";
                } else {
                    CASPAR_LOG(debug) << L"[vulkan] MEDIUM priority rejected, creating without priority.";
                    queue_info.pNext = nullptr;
                    create_result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
                }
            }
        } else {
            create_result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
        }

        if (create_result != VK_SUCCESS) {
            CASPAR_THROW_EXCEPTION(caspar_exception()
                                   << msg_info("vkCreateDevice failed: " + std::to_string(create_result)));
        }
    }

    vkGetDeviceQueue(device_, present_queue_family_, 0, &present_queue_);
}

VkSurfaceKHR vulkan_device::create_display_surface(const display_info& info, uint32_t target_refresh_mhz)
{
    // For Pro tier: direct display surface using VK_KHR_display
    auto get_display_mode_props = reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(
        vkGetInstanceProcAddr(instance_, "vkGetDisplayModePropertiesKHR"));

    uint32_t mode_count = 0;
    get_display_mode_props(physical_device_, info.display_handle, &mode_count, nullptr);
    std::vector<VkDisplayModePropertiesKHR> modes(mode_count);
    get_display_mode_props(physical_device_, info.display_handle, &mode_count, modes.data());

    // Pick the best mode: prefer exact resolution + refresh match
    VkDisplayModeKHR selected_mode   = modes[0].displayMode;
    bool             exact_match     = false;
    bool             resolution_only = false;

    for (const auto& mode : modes) {
        bool res_match = (mode.parameters.visibleRegion.width == info.width &&
                          mode.parameters.visibleRegion.height == info.height);
        bool refresh_match = (target_refresh_mhz > 0 && mode.parameters.refreshRate == target_refresh_mhz);

        if (res_match && refresh_match && !exact_match) {
            selected_mode = mode.displayMode;
            exact_match   = true;
            CASPAR_LOG(info) << L"[vulkan_output] Mode match: " << mode.parameters.visibleRegion.width
                             << L"x" << mode.parameters.visibleRegion.height
                             << L"@" << (mode.parameters.refreshRate / 1000) << L"Hz (exact)";
        } else if (res_match && !exact_match && !resolution_only) {
            selected_mode   = mode.displayMode;
            resolution_only = true;
        }
    }

    if (!exact_match && !resolution_only && mode_count > 0) {
        CASPAR_LOG(warning) << L"[vulkan_output] No matching display mode for "
                            << info.width << L"x" << info.height
                            << (target_refresh_mhz > 0
                                    ? (L"@" + std::to_wstring(target_refresh_mhz / 1000) + L"Hz")
                                    : L"")
                            << L". Using first available mode.";
    }

    VkDisplaySurfaceCreateInfoKHR surface_info{};
    surface_info.sType           = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR;
    surface_info.displayMode     = selected_mode;
    surface_info.planeIndex      = 0;
    surface_info.planeStackIndex = 0;
    surface_info.transform       = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    surface_info.globalAlpha     = 1.0f;
    surface_info.alphaMode       = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
    surface_info.imageExtent     = {info.width, info.height};

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    auto         create  = reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(
        vkGetInstanceProcAddr(instance_, "vkCreateDisplayPlaneSurfaceKHR"));
    VK_CHECK(create(instance_, &surface_info, nullptr, &surface));

    return surface;
}

VkSurfaceKHR vulkan_device::create_win32_surface(HWND hwnd)
{
    VkWin32SurfaceCreateInfoKHR surface_info{};
    surface_info.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surface_info.hinstance = GetModuleHandle(nullptr);
    surface_info.hwnd      = hwnd;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(vkCreateWin32SurfaceKHR(instance_, &surface_info, nullptr, &surface));

    return surface;
}

bool vulkan_device::has_extension(const char* name) const
{
    for (const auto& ext : enabled_extensions_) {
        if (ext == name)
            return true;
    }
    return false;
}

VkFence vulkan_device::create_vblank_fence(VkDisplayKHR display)
{
    if (!has_extension(VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME) || display == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    auto vkRegisterDisplayEventEXT_ = reinterpret_cast<PFN_vkRegisterDisplayEventEXT>(
        vkGetDeviceProcAddr(device_, "vkRegisterDisplayEventEXT"));
    if (!vkRegisterDisplayEventEXT_)
        return VK_NULL_HANDLE;

    VkDisplayEventInfoEXT event_info{};
    event_info.sType        = VK_STRUCTURE_TYPE_DISPLAY_EVENT_INFO_EXT;
    event_info.displayEvent = VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT;

    VkFence fence = VK_NULL_HANDLE;
    auto    result = vkRegisterDisplayEventEXT_(device_, display, &event_info, nullptr, &fence);
    if (result != VK_SUCCESS)
        return VK_NULL_HANDLE;

    return fence;
}

std::vector<display_info> vulkan_device::enumerate_displays_on_device() const
{
    std::vector<display_info> results;

    if (tier_ != gpu_tier::pro)
        return results;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);

    auto get_display_props = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    if (!get_display_props)
        return results;

    uint32_t display_count = 0;
    get_display_props(physical_device_, &display_count, nullptr);
    std::vector<VkDisplayPropertiesKHR> displays(display_count);
    get_display_props(physical_device_, &display_count, displays.data());

    for (uint32_t di = 0; di < display_count; ++di) {
        display_info info{};
        info.gpu_index      = 0; // Caller should set this
        info.output_index   = static_cast<int>(di + 1);
        info.gpu_name       = std::wstring(props.deviceName, props.deviceName + strlen(props.deviceName));
        info.display_name   = std::wstring(displays[di].displayName,
                                           displays[di].displayName + strlen(displays[di].displayName));
        info.width          = displays[di].physicalResolution.width;
        info.height         = displays[di].physicalResolution.height;
        info.refresh_rate   = 0;
        info.tier           = tier_;
        info.display_handle = displays[di].display;
        results.push_back(info);
    }

    return results;
}

std::vector<display_info> vulkan_device::enumerate_displays()
{
    std::vector<display_info> results;

    // Create a temporary instance for enumeration
    VkApplicationInfo app_info{};
    app_info.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CasparCG_Enum";
    app_info.apiVersion       = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_DISPLAY_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    // Filter extensions to only those available
    uint32_t avail_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, nullptr);
    std::vector<VkExtensionProperties> avail_exts(avail_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, avail_exts.data());

    extensions.erase(
        std::remove_if(extensions.begin(), extensions.end(),
                       [&](const char* name) {
                           return std::none_of(avail_exts.begin(), avail_exts.end(),
                                               [&](const VkExtensionProperties& p) {
                                                   return strcmp(p.extensionName, name) == 0;
                                               });
                       }),
        extensions.end());

    VkInstanceCreateInfo create_info{};
    create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo        = &app_info;
    create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        return results;

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());

    for (uint32_t gi = 0; gi < gpu_count; ++gi) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpus[gi], &props);

        // Check device extensions for tier detection
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(gpus[gi], nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> ext_props(ext_count);
        vkEnumerateDeviceExtensionProperties(gpus[gi], nullptr, &ext_count, ext_props.data());

        bool has_display = false;
        bool has_fse     = false;
        bool has_present_barrier = false;
        for (const auto& ext : ext_props) {
            if (strcmp(ext.extensionName, VK_KHR_DISPLAY_EXTENSION_NAME) == 0)
                has_display = true;
            if (strcmp(ext.extensionName, VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME) == 0)
                has_fse = true;
            if (strcmp(ext.extensionName, VK_NV_PRESENT_BARRIER_EXTENSION_NAME) == 0)
                has_present_barrier = true;
        }

        gpu_tier tier = has_display ? gpu_tier::pro
                      : (has_present_barrier || is_professional_gpu(props.deviceName)) ? gpu_tier::pro
                      : has_fse ? gpu_tier::consumer
                      : gpu_tier::none;

        if (has_display) {
            // Enumerate physical displays via VK_KHR_display
            auto get_display_props = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceDisplayPropertiesKHR"));

            uint32_t display_count = 0;
            get_display_props(gpus[gi], &display_count, nullptr);
            std::vector<VkDisplayPropertiesKHR> displays(display_count);
            get_display_props(gpus[gi], &display_count, displays.data());

            for (uint32_t di = 0; di < display_count; ++di) {
                display_info info{};
                info.gpu_index      = static_cast<int>(gi);
                info.output_index   = static_cast<int>(di + 1);
                info.gpu_name       = std::wstring(props.deviceName, props.deviceName + strlen(props.deviceName));
                info.display_name   = std::wstring(displays[di].displayName,
                                                   displays[di].displayName + strlen(displays[di].displayName));
                info.width          = displays[di].physicalResolution.width;
                info.height         = displays[di].physicalResolution.height;
                info.refresh_rate   = 0; // Would need mode enumeration for exact rate
                info.tier           = tier;
                info.display_handle = displays[di].display;
                results.push_back(info);
            }
        } else if (has_fse) {
            // For consumer GPUs, enumerate via Win32 DXGI or report a generic entry
            display_info info{};
            info.gpu_index      = static_cast<int>(gi);
            info.output_index   = 1;
            info.gpu_name       = std::wstring(props.deviceName, props.deviceName + strlen(props.deviceName));
            info.display_name   = L"Fullscreen Exclusive";
            info.width          = 0;
            info.height         = 0;
            info.refresh_rate   = 0;
            info.tier           = tier;
            info.display_handle = VK_NULL_HANDLE;
            results.push_back(info);
        }
    }

    vkDestroyInstance(instance, nullptr);
    return results;
}

}} // namespace caspar::vulkan_output
