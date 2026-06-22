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

#include "output_device.h"

#include <common/except.h>
#include <common/log.h>

#include <algorithm>
#include <cstring>

namespace caspar { namespace vulkan_output {

namespace {

#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult result_ = (call);                                                                                     \
        if (result_ != VK_SUCCESS)                                                                                     \
            CASPAR_THROW_EXCEPTION(caspar_exception()                                                                  \
                                   << msg_info("Vulkan call failed: " #call " = " + std::to_string(result_)));         \
    } while (0)

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
                                               VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
                                               const VkDebugUtilsMessengerCallbackDataEXT* data,
                                               void*                                       /*user*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        CASPAR_LOG(warning) << L"[vulkan_output_device] " << data->pMessage;
    return VK_FALSE;
}

} // namespace

// ─── Construction / Destruction ─────────────────────────────────────────────

output_device::output_device(int gpu_index)
{
    create_instance_();
    select_physical_device_(gpu_index);
    create_device_();
}

output_device::~output_device()
{
    if (device_ != VK_NULL_HANDLE)
        vkDestroyDevice(device_, nullptr);

    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn)
            fn(instance_, debug_messenger_, nullptr);
    }

    if (instance_ != VK_NULL_HANDLE)
        vkDestroyInstance(instance_, nullptr);
}

// ─── Public API ─────────────────────────────────────────────────────────────

uint32_t output_device::acquire_queue_index()
{
    return next_queue_.fetch_add(1, std::memory_order_relaxed) % queue_count_;
}

#ifdef _WIN32
VkSurfaceKHR output_device::create_win32_surface(HWND hwnd)
{
    VkWin32SurfaceCreateInfoKHR ci{};
    ci.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    ci.hinstance = GetModuleHandle(nullptr);
    ci.hwnd      = hwnd;

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(vkCreateWin32SurfaceKHR(instance_, &ci, nullptr, &surface));
    return surface;
}
#endif

VkSurfaceKHR output_device::create_display_surface(VkDisplayKHR display, uint32_t width, uint32_t height,
                                                    uint32_t target_refresh_mhz)
{
    // Enumerate display modes and pick the best match
    auto fn = reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(
        vkGetInstanceProcAddr(instance_, "vkGetDisplayModePropertiesKHR"));
    if (!fn)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("vkGetDisplayModePropertiesKHR not available"));

    uint32_t mode_count = 0;
    fn(physical_device_, display, &mode_count, nullptr);
    std::vector<VkDisplayModePropertiesKHR> modes(mode_count);
    fn(physical_device_, display, &mode_count, modes.data());

    VkDisplayModeKHR selected_mode = modes.empty() ? VK_NULL_HANDLE : modes[0].displayMode;
    for (const auto& m : modes) {
        bool res_ok = (m.parameters.visibleRegion.width == width && m.parameters.visibleRegion.height == height);
        bool ref_ok = (target_refresh_mhz > 0 && m.parameters.refreshRate == target_refresh_mhz);
        if (res_ok && ref_ok) {
            selected_mode = m.displayMode;
            break;
        }
        if (res_ok && selected_mode == modes[0].displayMode)
            selected_mode = m.displayMode;
    }

    VkDisplaySurfaceCreateInfoKHR ci{};
    ci.sType           = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR;
    ci.displayMode     = selected_mode;
    ci.planeIndex      = 0;
    ci.planeStackIndex = 0;
    ci.transform       = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.globalAlpha     = 1.0f;
    ci.alphaMode       = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
    ci.imageExtent     = {width, height};

    auto create_fn = reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(
        vkGetInstanceProcAddr(instance_, "vkCreateDisplayPlaneSurfaceKHR"));
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VK_CHECK(create_fn(instance_, &ci, nullptr, &surface));
    return surface;
}

void output_device::destroy_surface(VkSurfaceKHR surface)
{
    if (surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(instance_, surface, nullptr);
}

bool output_device::has_extension(const char* name) const
{
    for (const auto& ext : enabled_extensions_)
        if (ext == name)
            return true;
    return false;
}

// ─── Private: Instance Creation ─────────────────────────────────────────────

void output_device::create_instance_()
{
    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "CasparCG Output";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName        = "CasparCG";
    app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_API_VERSION_1_3;

    // Request all presentation-related instance extensions
    std::vector<const char*> desired = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
        VK_KHR_DISPLAY_EXTENSION_NAME,
        "VK_KHR_get_display_properties2",
        "VK_EXT_direct_mode_display",
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        "VK_KHR_get_surface_capabilities2",
        VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    // Filter to available extensions
    uint32_t avail_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, nullptr);
    std::vector<VkExtensionProperties> avail(avail_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_count, avail.data());

    std::vector<const char*> extensions;
    for (auto* d : desired) {
        for (const auto& a : avail) {
            if (strcmp(a.extensionName, d) == 0) {
                extensions.push_back(d);
                break;
            }
        }
    }

    std::vector<const char*> layers;
#ifndef NDEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app_info;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.data();

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    // Debug messenger
    auto create_dbg = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (create_dbg) {
        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        dbg.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbg.pfnUserCallback = debug_callback;
        create_dbg(instance_, &dbg, nullptr, &debug_messenger_);
    }

    CASPAR_LOG(info) << L"[vulkan_output] Separate VkInstance created with " << extensions.size() << L" extensions.";
}

// ─── Private: Physical Device Selection ─────────────────────────────────────

void output_device::select_physical_device_(int gpu_index)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No Vulkan-capable GPUs found"));

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    if (gpu_index < 0 || gpu_index >= static_cast<int>(count))
        CASPAR_THROW_EXCEPTION(caspar_exception()
                               << msg_info("GPU index " + std::to_string(gpu_index) + " out of range (" +
                                           std::to_string(count) + " devices)"));

    physical_device_ = devices[gpu_index];

    // Query LUID
    VkPhysicalDeviceIDProperties id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(physical_device_, &props2);

    if (id_props.deviceLUIDValid) {
        memcpy(luid_, id_props.deviceLUID, 8);
        luid_valid_ = true;
    }

    CASPAR_LOG(info) << L"[vulkan_output] Selected GPU " << gpu_index << L": "
                     << props2.properties.deviceName
                     << L" (LUID " << (luid_valid_ ? L"valid" : L"unavailable") << L")";
}

// ─── Private: Logical Device Creation ───────────────────────────────────────

void output_device::create_device_()
{
    // Find graphics queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qf_count, qf_props.data());

    queue_family_ = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == UINT32_MAX)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("No graphics queue family found"));

    // Request all available queues (NVIDIA exposes 16, AMD typically 1-2).
    queue_count_ = std::min(qf_props[queue_family_].queueCount, 16u);
    std::vector<float> priorities(queue_count_, 1.0f);

    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = queue_family_;
    queue_ci.queueCount       = queue_count_;
    queue_ci.pQueuePriorities = priorities.data();

    // Try HIGH global priority for presentation preemption
    VkDeviceQueueGlobalPriorityCreateInfoKHR prio_ci{};
    prio_ci.sType          = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
    prio_ci.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
    bool use_priority = false;

    // Enumerate available device extensions
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> ext_props(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count, ext_props.data());

    auto has_ext = [&](const char* name) {
        return std::any_of(ext_props.begin(), ext_props.end(),
                           [&](const VkExtensionProperties& p) { return strcmp(p.extensionName, name) == 0; });
    };

    // Build extension list
    std::vector<const char*> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    std::vector<const char*> optional = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME,
#endif
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_EXT_HDR_METADATA_EXTENSION_NAME,
        VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME,
        "VK_KHR_display_swapchain",
        "VK_NV_present_barrier",
        "VK_EXT_display_control",
    };

    for (auto* ext : optional) {
        if (has_ext(ext))
            device_extensions.push_back(ext);
    }

    // Check if global priority is available
    if (has_ext(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)) {
        queue_ci.pNext = &prio_ci;
        use_priority = true;
    }

    // Store enabled extensions
    for (const auto* ext : device_extensions)
        enabled_extensions_.emplace_back(ext);

    // Timeline semaphores (Vulkan 1.2 core)
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo dev_ci{};
    dev_ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.pNext                   = &features12;
    dev_ci.queueCreateInfoCount    = 1;
    dev_ci.pQueueCreateInfos       = &queue_ci;
    dev_ci.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
    dev_ci.ppEnabledExtensionNames = device_extensions.data();
    dev_ci.pEnabledFeatures        = &features;

    VkResult result = vkCreateDevice(physical_device_, &dev_ci, nullptr, &device_);

    // Retry without priority if it failed
    if (result != VK_SUCCESS && use_priority) {
        CASPAR_LOG(warning) << L"[vulkan_output] Device creation with HIGH priority failed. Retrying without.";
        queue_ci.pNext = nullptr;
        result = vkCreateDevice(physical_device_, &dev_ci, nullptr, &device_);
    }

    if (result != VK_SUCCESS)
        CASPAR_THROW_EXCEPTION(caspar_exception()
                               << msg_info("vkCreateDevice failed: " + std::to_string(result)));

    // Retrieve queues
    queues_.resize(queue_count_);
    for (uint32_t i = 0; i < queue_count_; ++i)
        vkGetDeviceQueue(device_, queue_family_, i, &queues_[i]);

    queue_mutexes_ = std::vector<std::mutex>(queue_count_);

    CASPAR_LOG(info) << L"[vulkan_output] Separate VkDevice created: " << queue_count_ << L" queues"
                     << (use_priority ? L" (HIGH priority)" : L"") << L".";
}

}} // namespace caspar::vulkan_output
