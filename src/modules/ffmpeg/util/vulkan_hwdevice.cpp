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

#include "vulkan_hwdevice.h"

#include <common/log.h>

#ifdef ENABLE_VULKAN
// The pure-std descriptor, NOT device.h: that header pulls in vulkan.hpp, which this
// module has no include path for (C1083). Same boundary as dxgi_adapter_for_vk_device.
#include <accelerator/vulkan/util/shared_device_info.h>
#endif

extern "C" {
#include <libavutil/hwcontext.h>
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8: the compute decoders this exists for
#include <libavutil/hwcontext_vulkan.h>
#endif
}

#include <vector>

namespace caspar { namespace ffmpeg {

#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60

AVBufferRef* make_vulkan_hwdevice_from_mixer(void* vk_device_handle)
{
    const auto info = accelerator::vulkan::describe_shared_device(vk_device_handle);
    if (!info.valid) {
        CASPAR_LOG(debug) << L"[vk_hwdevice] the mixer exposes no Vulkan device";
        return nullptr;
    }

    // Refuse rather than share the graphics queue. See the header: two submitters on one
    // VkQueue is undefined behaviour, and FFmpeg's own queue mutexes do not cover the
    // mixer's submissions. A GPU with no separate compute family is a reason to use the
    // existing path, not a reason to race.
    if (!info.decode_qf_isolated) {
        CASPAR_LOG(info) << L"[vk_hwdevice] this GPU has no compute queue family separate "
                            L"from graphics, so an FFmpeg decoder would have to share the "
                            L"mixer's queue; declining";
        return nullptr;
    }

    AVBufferRef* ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!ref) {
        CASPAR_LOG(warning) << L"[vk_hwdevice] av_hwdevice_ctx_alloc failed";
        return nullptr;
    }

    auto* hwctx = static_cast<AVVulkanDeviceContext*>(
        reinterpret_cast<AVHWDeviceContext*>(ref->data)->hwctx);

    hwctx->get_proc_addr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(info.get_proc_addr);
    hwctx->inst     = static_cast<VkInstance>(info.instance);
    hwctx->phys_dev = static_cast<VkPhysicalDevice>(info.physical_device);
    hwctx->act_dev  = static_cast<VkDevice>(info.device);

    // What the application enabled, which is what FFmpeg is allowed to rely on. This is a
    // DECLARATION, not something FFmpeg can query -- if the list understates what the
    // device has, FFmpeg simply uses less; if it overstates, FFmpeg will call into an
    // extension that was never enabled.
    //
    // The strings must outlive the context, so they are kept in a static store rather
    // than pointing at the device's vector, whose lifetime is the mixer's and whose
    // reallocation would dangle these pointers.
    static std::vector<std::string> ext_store;
    static std::vector<const char*> ext_ptrs;
    ext_store = info.enabled_device_extensions;
    ext_ptrs.clear();
    ext_ptrs.reserve(ext_store.size());
    for (const auto& e : ext_store)
        ext_ptrs.push_back(e.c_str());
    hwctx->enabled_dev_extensions    = ext_ptrs.empty() ? nullptr : ext_ptrs.data();
    hwctx->nb_enabled_dev_extensions = static_cast<int>(ext_ptrs.size());
    hwctx->enabled_inst_extensions    = nullptr;
    hwctx->nb_enabled_inst_extensions = 0;

    // Exactly one family, and deliberately not the graphics one. `flags` must be
    // non-zero; COMPUTE plus TRANSFER is what NVIDIA's compute family carries, and it is
    // what a compute decoder needs (`prores_vulkan` declares VK_QUEUE_COMPUTE_BIT).
    hwctx->nb_qf     = 1;
    hwctx->qf[0].idx = static_cast<int>(info.decode_qf);
    hwctx->qf[0].num = 1;
    hwctx->qf[0].flags =
        static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);

    const int err = av_hwdevice_ctx_init(ref);
    if (err < 0) {
        char msg[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(err, msg, sizeof(msg));
        CASPAR_LOG(warning) << L"[vk_hwdevice] av_hwdevice_ctx_init rejected the mixer's "
                               L"device: " << u16(msg)
                            << L" (declared " << hwctx->nb_enabled_dev_extensions
                            << L" device extensions, queue family " << hwctx->qf[0].idx << L")";
        av_buffer_unref(&ref);
        return nullptr;
    }

    CASPAR_LOG(info) << L"[vk_hwdevice] FFmpeg is using the mixer's Vulkan device on queue "
                        L"family " << hwctx->qf[0].idx << L" ("
                     << hwctx->nb_enabled_dev_extensions << L" extensions declared); "
                        L"decoded frames need no copy to reach the mixer";
    return ref;
}

#else // no Vulkan accelerator, or FFmpeg older than 8

AVBufferRef* make_vulkan_hwdevice_from_mixer(void*)
{
    return nullptr;
}

#endif

}} // namespace caspar::ffmpeg
