#include "texture_wrapper.h"

#include "device.h"

namespace caspar { namespace accelerator { namespace vulkan {

std::vector<std::uint8_t> VkReadableTextureWrapper::read_pixels() const
{
    if (!vk_device_ || !tex_)
        return {};

    auto future = vk_device_->copy_async(tex_);
    auto arr    = future.get();

    return std::vector<std::uint8_t>(arr.data(), arr.data() + arr.size());
}

}}} // namespace caspar::accelerator::vulkan
