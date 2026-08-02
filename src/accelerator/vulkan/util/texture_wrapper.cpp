#include "texture_wrapper.h"

#include "device.h"

#include <common/log.h>

#include <algorithm>
#include <cstddef>

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

        std::vector<std::uint8_t> out(arr.data(), arr.data() + arr.size());

        // core::texture::read_pixels_reduced promises packed 8-bit BGRA whatever the
        // texture's own depth. reduce_texture() delivers the depth half -- it always
        // runs at least one pass into an 8-bit target -- but not the order.
        //
        // A blit maps components, not bytes: R goes to R. An 8-bit mixer attachment
        // already holds BGRA bytes, because the shader writes col.bgra into an
        // eR8G8B8A8Unorm image, so component R *is* blue and the bytes come out in the
        // promised order. A 16-bit attachment does not -- there is no
        // eB16G16R16A16Unorm, so the shader writes RGBA directly (see
        // image_kernel.cpp, which sets F2_OUTPUT_BGRA only at bit8) -- and the blit
        // faithfully carries that RGBA order into the 8-bit result.
        //
        // So swap it here rather than at the call site. The one caller today indexes
        // [2]=R,[0]=B (dmx::average_color) and would have read red and blue swapped on
        // a 16-bit Vulkan channel, silently. The image is the reduced one -- 240x135 at
        // levels=3 -- so this costs nothing measurable, and it keeps the contract true
        // in one place instead of asking every future caller to know this.
        if (tex_->depth() != common::bit_depth::bit8) {
            for (std::size_t i = 0; i + 3 < out.size(); i += 4)
                std::swap(out[i], out[i + 2]);
        }

        return out;
    } catch (...) {
        CASPAR_LOG(warning) << L"[vulkan::texture_wrapper] reduced readback failed; the caller will fall back.";
        return {};
    }
}

}}} // namespace caspar::accelerator::vulkan
