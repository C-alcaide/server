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

#include <common/gpu_extension_requests.h>
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

#include <map>
#include <mutex>
#include <vector>

namespace caspar { namespace ffmpeg {

#if defined(ENABLE_VULKAN) && LIBAVUTIL_VERSION_MAJOR >= 60

namespace {
/// Publish FFmpeg's own list of wanted device extensions before any Vulkan device exists.
///
/// `av_vk_get_optional_device_extensions()` is the list FFmpeg enables when it creates its
/// own device, and matching that configuration is not optional: with only the mixer's four
/// extensions declared, `prores_vulkan` initialised cleanly and then jumped to address 0 on
/// its first dispatch -- a null device function pointer, because vkGetDeviceProcAddr returns
/// null for anything whose extension the logical device never enabled, and FFmpeg's guards
/// are written against what the DEVICE has rather than what it was told.
///
/// A static initialiser rather than module init, because the mixer's device is created
/// before the ffmpeg module is initialised. See common/gpu_extension_requests.h.
struct extension_registrar
{
    extension_registrar()
    {
        int          count = 0;
        const char** names = av_vk_get_optional_device_extensions(&count);
        if (!names || count <= 0)
            return;
        std::vector<std::string> list;
        list.reserve(count);
        for (int i = 0; i < count; ++i)
            if (names[i])
                list.emplace_back(names[i]);
        common::register_vulkan_device_extension_request(std::move(list));
    }
};
const extension_registrar g_extension_registrar;
} // namespace

//: The mixer's queue lock, taken from `shared_device_info` once. Same pair for every device --
//: they only cast their argument to `accelerator::vulkan::device*` and call a member -- so a
//: single copy avoids rebuilding the descriptor on every FFmpeg submission.
static void (*g_lock_queue)(void*)   = nullptr;
static void (*g_unlock_queue)(void*) = nullptr;

AVBufferRef* make_vulkan_hwdevice_from_mixer(void* vk_device_handle)
{
    // ONE DEVICE CONTEXT PER VkDevice, SHARED BY EVERY PRODUCER. This is not an
    // optimisation; a context per producer corrupts the GPU.
    //
    // FFmpeg serialises its own submissions with a mutex that lives on the
    // AVHWDeviceContext (`vulkan_device_init` installs `lock_queue` when the application
    // leaves it null). Two producers each building their own context therefore get two
    // independent mutexes guarding THE SAME VkQueue -- ours, from the queue family we hand
    // over -- and FFmpeg's decode threads then submit to it concurrently.
    //
    // Measured 2026-08-21 with the Vulkan validation layer
    // (`CASPARVP_VK_VALIDATION=1`): `UNASSIGNED-Threading-MultipleThreads-Write,
    // vkQueueSubmit2(): THREADING ERROR : object of type VkQueue is simultaneously used in
    // current thread` -- reported until it hit the layer's duplicate limit -- followed by
    // `VK_ERROR_DEVICE_LOST` and an `nvlddmkm` TDR. One producer survived indefinitely; two
    // failed in 6 of 6 interleaved 25-second runs.
    //
    // It also explains the result that had ruled FFmpeg out: four concurrent Vulkan ProRes
    // decoders inside one `ffmpeg.exe` sharing one `-init_hw_device vulkan` complete cleanly,
    // because that is ONE context and therefore one mutex. The difference was never the
    // decoder; it was how many contexts wrapped the device.
    //
    // The cache is keyed on the device handle and holds its own reference for the life of the
    // process. The mixer's device outlives every producer, so that reference is deliberately
    // never released -- one AVBufferRef per GPU, not a leak that grows.
    //
    // The assumption that buys is that a `device*` is stable for the process. It holds today:
    // the Vulkan device is created once per GPU at channel setup and lives until shutdown. If
    // a mixer device is ever destroyed and another allocated at the same address, this cache
    // would hand out a context wrapping a dead VkDevice -- so anything that makes device
    // lifetime dynamic has to invalidate this entry with it.
    static std::mutex                       cache_mutex;
    static std::map<void*, AVBufferRef*>    cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto                  it = cache.find(vk_device_handle);
        if (it != cache.end() && it->second != nullptr)
            return av_buffer_ref(it->second);
    }

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

    // TWO families, and the second one is the whole reason this works.
    //
    // `qf[0]` is where FFmpeg actually submits: the compute family, deliberately not the
    // graphics one. `flags` must be non-zero; COMPUTE plus TRANSFER is what NVIDIA's
    // compute family carries and what a compute decoder needs (`prores_vulkan` declares
    // VK_QUEUE_COMPUTE_BIT).
    //
    // `qf[1]` is the graphics family, and FFmpeg will never submit on it. This list is
    // also what `hwcontext_vulkan.c` builds `pQueueFamilyIndices` from, and with a single
    // entry it creates every decoded image VK_SHARING_MODE_EXCLUSIVE, owned by the compute
    // family (`AVVkFrame::queue_family[i]`). The mixer's copy runs on its GRAPHICS queue,
    // and reading an exclusively-owned image from another family without a queue family
    // ownership transfer gives undefined contents -- a transfer this side cannot perform,
    // because the release half must be submitted on FFmpeg's queue, which FFmpeg owns.
    // Naming a second family makes the images CONCURRENT and `queue_family` IGNORED, so
    // the question does not arise.
    //
    // It is inert for selection: `ff_vk_qf_find` returns the FIRST entry sharing ANY
    // requested bit, so COMPUTE and TRANSFER both resolve to `qf[0]`, and no caller in
    // lavu/lavc/lavfi asks for VK_QUEUE_GRAPHICS_BIT at all. Declaring GRAPHICS alone here
    // understates what the family can do, which is the safe direction: FFmpeg uses less
    // than the device offers rather than calling into something that was never enabled.
    hwctx->nb_qf     = 2;
    hwctx->qf[0].idx = static_cast<int>(info.decode_qf);
    hwctx->qf[0].num = 1;
    hwctx->qf[0].flags =
        static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    hwctx->qf[1].idx   = static_cast<int>(info.graphics_qf);
    hwctx->qf[1].num   = 1;
    hwctx->qf[1].flags = VK_QUEUE_GRAPHICS_BIT;

    // `qf[2]`, THE VIDEO ENCODE FAMILY, and unlike `qf[1]` it is NOT inert: `h264_vulkan` and
    // `hevc_vulkan` are `VK_KHR_video_encode` codecs and ask for VK_QUEUE_VIDEO_ENCODE_BIT_KHR.
    // `ff_vk_qf_find` returns the first entry sharing any requested bit AND carrying the
    // requested `video_caps`, so an encoder skips qf[0] (compute|transfer) and qf[1] (graphics)
    // and lands here. The decode-side lesson applies in reverse: handing an encode codec the
    // COMPUTE family would be handing it a queue that cannot do the work, which is how VP9
    // came to fault instead of declining.
    //
    // `video_caps` names only H.264 and H.265 because those are the encode extensions this
    // device reports. AV1 is deliberately absent, and nothing here needs to special-case it:
    // FFmpeg queries the device itself and `av1_vulkan` refuses with "Device does not support
    // encoding av1!" -- measured 2026-08-22 on an RTX A4000, which is Ampere and has no AV1
    // encoder at all.
    //
    // Omitted entirely when the GPU has no such family, so an encoder gets no queue to find
    // and declines, rather than being pointed at a family that cannot serve it.
    if (info.encode_qf_present) {
        hwctx->qf[2].idx   = static_cast<int>(info.encode_qf);
        hwctx->qf[2].num   = 1;
        hwctx->qf[2].flags = static_cast<VkQueueFlagBits>(VK_QUEUE_VIDEO_ENCODE_BIT_KHR);
        hwctx->qf[2].video_caps =
            static_cast<VkVideoCodecOperationFlagBitsKHR>(VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR |
                                                          VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR);
        hwctx->nb_qf = 3;
    }

    // DECLARE THE ENABLED FEATURES. `device_features` is documented as "the set of features
    // that present and enabled during device creation" -- an application declaration, exactly
    // like `enabled_dev_extensions`, and not something the sharing library can query.
    //
    // Leaving it zeroed cost a day: `libplacebo` refused the device with "Missing device
    // feature: hostQueryReset" and "Failed importing Vulkan device!", while our own startup log
    // said "39 core/1.1/1.2/1.3 features" enabled -- including that one. The feature was on; we
    // simply never said so. Measured 2026-08-22.
    //
    // The chain points at structs the mixer's device owns, and that device outlives every
    // consumer, so no copy or lifetime dance is needed.
    if (info.features10 && info.features12) {
        auto* f11 = const_cast<VkPhysicalDeviceVulkan11Features*>(
            static_cast<const VkPhysicalDeviceVulkan11Features*>(info.features11));
        auto* f12 = const_cast<VkPhysicalDeviceVulkan12Features*>(
            static_cast<const VkPhysicalDeviceVulkan12Features*>(info.features12));
        auto* f13 = const_cast<VkPhysicalDeviceVulkan13Features*>(
            static_cast<const VkPhysicalDeviceVulkan13Features*>(info.features13));

        // Chain them, since they are stored separately rather than pre-linked.
        if (f11 && f12) {
            f11->pNext = f12;
            if (f13)
                f12->pNext = f13;
        }

        hwctx->device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        hwctx->device_features.pNext = f11 ? static_cast<void*>(f11) : static_cast<void*>(f12);
        hwctx->device_features.features =
            *static_cast<const VkPhysicalDeviceFeatures*>(info.features10);
    }

    // ── HAND FFMPEG THE MIXER'S QUEUE LOCK ─────────────────────────────────────────
    //
    // Left null, `vulkan_device_init` installs its own per-(family, index) mutex. That
    // serialises FFmpeg's submissions and nothing else -- so FFmpeg's mutex and the mixer's
    // `_queue_mutex` guard the SAME `VkQueue` independently. On this GPU family 0 carries
    // graphics, compute and transfer, so FFmpeg's "compute queue" and the mixer's queue are one
    // object, and `vkQueueSubmit` requires external synchronisation that neither party can
    // provide alone.
    //
    // Measured 2026-08-22, recording through `h264_vulkan` under the validation layer:
    // `UNASSIGNED-Threading-MultipleThreads-Write, vkQueueSubmit(): THREADING ERROR : object of
    // type VkQueue is simultaneously used in current thread 77836 and thread 71140`. The decode
    // path had not shown it because its own fix -- one shared context, therefore one FFmpeg
    // mutex -- happens to cover producer-against-producer, and only the encode path adds a
    // second submitter that is the MIXER.
    //
    // `user_opaque` carries the `device*` because the callback is handed an
    // `AVHWDeviceContext*` and nothing else. This context is ours, so the field is free.
    if (info.lock_queue && info.unlock_queue && info.mixer_device) {
        auto* device_ctx        = reinterpret_cast<AVHWDeviceContext*>(ref->data);
        device_ctx->user_opaque = info.mixer_device;
        // The thunks are held in file scope rather than re-derived per call: these fire on
        // every FFmpeg submission, and `describe_shared_device` builds the whole descriptor.
        // They are the same two functions for every device, so one copy is correct.
        g_lock_queue            = info.lock_queue;
        g_unlock_queue          = info.unlock_queue;
        hwctx->lock_queue       = [](AVHWDeviceContext* ctx, uint32_t, uint32_t) {
            g_lock_queue(ctx->user_opaque);
        };
        hwctx->unlock_queue = [](AVHWDeviceContext* ctx, uint32_t, uint32_t) {
            g_unlock_queue(ctx->user_opaque);
        };
    }

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

    // Publish it before handing out the first reference, so every later producer shares this
    // context -- and with it FFmpeg's queue mutex -- rather than building a rival one.
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto                  it = cache.find(vk_device_handle);
        if (it != cache.end() && it->second != nullptr) {
            // Another thread won the race; keep theirs and drop ours so there is exactly one.
            av_buffer_unref(&ref);
            return av_buffer_ref(it->second);
        }
        cache[vk_device_handle] = ref;
    }
    return av_buffer_ref(ref);
}

#else // no Vulkan accelerator, or FFmpeg older than 8

AVBufferRef* make_vulkan_hwdevice_from_mixer(void*)
{
    return nullptr;
}

#endif

}} // namespace caspar::ffmpeg
