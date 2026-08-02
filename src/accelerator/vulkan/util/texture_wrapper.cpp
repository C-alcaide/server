#include "texture_wrapper.h"

#include "device.h"

#include <common/log.h>

namespace caspar { namespace accelerator { namespace vulkan {

std::vector<std::uint8_t> texture_wrapper::read_pixels() const
{
    if (!vk_device_ || !tex_)
        return {};

    // The caller may be on any thread and asynchronous to the channel's render.
    ensure_render_complete();

    auto future = vk_device_->copy_async(tex_);
    auto arr    = future.get();

    return std::vector<std::uint8_t>(arr.data(), arr.data() + arr.size());
}

std::vector<std::uint8_t> texture_wrapper::read_pixels_reduced(int levels, int& out_width, int& out_height) const
{
    out_width = out_height = 0;

    if (!vk_device_ || !tex_)
        return {};

    // The caller is on its own thread and asynchronous to the channel's render, so
    // the mixer may still be writing this attachment. This is what orders the two.
    ensure_render_complete();

    try {
        // Not named `small`: <rpcndr.h>, which the Windows headers pull in,
        // #defines small as char, so `auto small = ...` becomes `auto char = ...`.
        auto reduced = vk_device_->reduce_texture(tex_, levels);
        if (!reduced)
            return {};

        // reduce_texture leaves the result in eColorAttachmentOptimal precisely so
        // this composes with the existing, tested readback rather than open-coding
        // a second one.
        auto arr = vk_device_->copy_async(reduced).get();
        if (!arr.data() || arr.size() == 0)
            return {};

        out_width  = reduced->width();
        out_height = reduced->height();
        return std::vector<std::uint8_t>(arr.data(), arr.data() + arr.size());
    } catch (...) {
        CASPAR_LOG(warning) << L"[vulkan::texture_wrapper] reduced readback failed; the caller will fall back.";
        return {};
    }
}

}}} // namespace caspar::accelerator::vulkan
