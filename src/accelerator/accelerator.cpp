#include "accelerator.h"

#include "ogl/image/image_mixer.h"
#include "ogl/image/previz_scene.h"
#include "ogl/util/device.h"

#include <boost/property_tree/ptree.hpp>

#include <common/bit_depth.h>

#include <core/mixer/image/image_mixer.h>

#include <memory>
#include <mutex>
#include <utility>

namespace caspar { namespace accelerator {

struct accelerator::impl
{
    std::shared_ptr<ogl::device>                 ogl_device_;
    std::shared_ptr<ogl::channel_texture_store>  channel_tex_store_;
    std::once_flag                               tex_store_init_flag_;
    const core::video_format_repository          format_repository_;

    impl(const core::video_format_repository format_repository)
        : format_repository_(format_repository)
    {
    }

    std::unique_ptr<core::image_mixer> create_image_mixer(int channel_id, common::bit_depth depth)
    {
        auto mixer = std::make_unique<ogl::image_mixer>(
            spl::make_shared_ptr(get_device()), channel_id, format_repository_.get_max_video_format_size(), depth);
        mixer->set_channel_texture_store(get_channel_texture_store());
        return mixer;
    }

    std::shared_ptr<ogl::device> get_device()
    {
        if (!ogl_device_) {
            ogl_device_ = std::make_shared<ogl::device>();
        }

        return ogl_device_;
    }

    std::shared_ptr<ogl::channel_texture_store> get_channel_texture_store()
    {
        std::call_once(tex_store_init_flag_, [this] {
            channel_tex_store_ = std::make_shared<ogl::channel_texture_store>();
        });
        return channel_tex_store_;
    }
};

accelerator::accelerator(const core::video_format_repository format_repository)
    : impl_(std::make_unique<impl>(format_repository))
{
}

accelerator::~accelerator() {}

std::unique_ptr<core::image_mixer> accelerator::create_image_mixer(const int channel_id, common::bit_depth depth)
{
    return impl_->create_image_mixer(channel_id, depth);
}

std::shared_ptr<accelerator_device> accelerator::get_device() const
{
    return std::dynamic_pointer_cast<accelerator_device>(impl_->get_device());
}

std::shared_ptr<ogl::channel_texture_store> accelerator::get_channel_texture_store()
{
    return impl_->get_channel_texture_store();
}

}} // namespace caspar::accelerator
