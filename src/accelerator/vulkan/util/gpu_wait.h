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

// Waiting for the GPU, where a TIMEOUT IS NOT AN OUTCOME YOU CAN PROCEED FROM.
//
// Every wait in this backend used to be `waitForFences(f, true, 1s)` followed by a warning
// and then the work the wait was protecting. That reads as a safety net and is the opposite
// of one: a fence that has not signalled is still owned by a submission that is still
// executing, so recycling it is undefined behaviour by the letter of the specification --
//
//   * `vkResetFences` "must not be used on a fence that is in use by a queue submission"
//   * `vkResetCommandBuffer` requires the buffer not be in the *pending* state
//
// and on this driver the observed consequence is not a wrong picture but a GPU HANG, which
// the watchdog then turns into a TDR and every context on the device into
// `VK_ERROR_DEVICE_LOST`.
//
// MEASURED, 2026-08-21. Four ProRes producers on `<vulkan-decode>`, one round in ten:
//
//   14:03:08.682  [Vulkan image_kernel] Timeout waiting for render completion
//   14:03:08.691  [vk::av_import] waiting for the previous plane copy timed out
//   14:03:09.696  [vk::av_import] waiting for the previous plane copy timed out
//   14:03:10.072  Unable to submit command buffer: VK_ERROR_DEVICE_LOST
//   14:03:11      Windows event 4101: nvlddmkm stopped responding and recovered
//
// The timeouts come 1.4 s BEFORE the device loss, so they are the cause and not a symptom:
// one frame ran long, the code below it reset resources that were still in flight, and the
// GPU wedged. A slow frame is survivable; recycling a pending submission is not.
//
// So the only correct response to a timeout is to KEEP WAITING. `eTimeout` means slow, not
// dead -- a genuinely lost device throws `DeviceLostError` from the wait itself, which is
// diagnosable and already handled by the callers. If the budget really does expire, throwing
// is still strictly better than proceeding: an exception loses a frame, undefined behaviour
// loses the GPU.
//
// The one wait that already did this correctly is `av_vulkan_importer`'s wait on the
// *decoder's* semaphore, which declines the frame instead of copying from it. Declining is
// available there because nothing has been recycled yet; once resources are about to be
// reused, waiting is the only option left.

#include <common/except.h>
#include <common/log.h>

#include <chrono>

#include <vulkan/vulkan.hpp>

namespace caspar { namespace accelerator { namespace vulkan {

/// How long to keep waiting before declaring the device unusable. Long enough that no
/// legitimately slow frame reaches it -- a 12K NotchLC decode is ~0.3 ms and the slowest
/// thing measured in this tree is under 30 ms -- and short enough to fail before an operator
/// concludes the server has hung. The driver's own watchdog fires at ~2 s, so in practice a
/// real hang is reported by the throw from `waitForFences` long before this.
inline constexpr auto gpu_wait_budget = std::chrono::seconds(10);

/// One second per attempt, so a slow frame still produces the warning that used to be the
/// only signal here. Losing that signal would trade a visible symptom for a silent stall.
inline constexpr uint64_t gpu_wait_slice_ns = 1'000'000'000ull;

/// Wait for `fence`, and NEVER return on a timeout. `what` names the caller in the log.
inline void wait_for_fence(vk::Device device, vk::Fence fence, const wchar_t* what)
{
    const auto deadline = std::chrono::steady_clock::now() + gpu_wait_budget;
    int        slices    = 0;
    for (;;) {
        // Throws DeviceLostError if the device is gone, which is the outcome we WANT to
        // propagate: the caller can decline a frame, and nothing has been recycled.
        if (device.waitForFences(fence, VK_TRUE, gpu_wait_slice_ns) == vk::Result::eSuccess) {
            if (slices > 0)
                CASPAR_LOG(warning) << what << L" recovered after " << slices
                                    << L"s waiting for the GPU; nothing was recycled while it "
                                       L"was still in flight";
            return;
        }
        ++slices;
        CASPAR_LOG(warning) << what << L" still waiting for the GPU after " << slices
                            << L"s -- continuing to wait rather than reusing a submission "
                               L"that is still executing";
        if (std::chrono::steady_clock::now() >= deadline)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                                       "the GPU did not complete a submission within the wait "
                                       "budget; refusing to recycle resources that are still in "
                                       "flight"));
    }
}

/// The same contract for a timeline-semaphore wait the caller is about to build on.
inline void wait_for_semaphores(vk::Device device, const vk::SemaphoreWaitInfo& info, const wchar_t* what)
{
    const auto deadline = std::chrono::steady_clock::now() + gpu_wait_budget;
    int        slices   = 0;
    for (;;) {
        if (device.waitSemaphores(info, gpu_wait_slice_ns) == vk::Result::eSuccess) {
            if (slices > 0)
                CASPAR_LOG(warning) << what << L" recovered after " << slices << L"s waiting for the GPU";
            return;
        }
        ++slices;
        CASPAR_LOG(warning) << what << L" still waiting for a GPU semaphore after " << slices << L"s";
        if (std::chrono::steady_clock::now() >= deadline)
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                                       "a GPU semaphore did not signal within the wait budget"));
    }
}

}}} // namespace caspar::accelerator::vulkan
