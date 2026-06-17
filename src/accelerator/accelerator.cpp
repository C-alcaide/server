#include "accelerator.h"

#include "ogl/image/image_mixer.h"
#include "ogl/image/previz_scene.h"
#include "ogl/util/device.h"

#ifdef ENABLE_VULKAN
#include "vulkan/image/image_mixer.h"
#include "vulkan/image/previz_texture_bridge.h"
#include "vulkan/util/device.h"
#endif

#include <boost/property_tree/ptree.hpp>

#include <common/bit_depth.h>
#include <common/except.h>

#include <core/mixer/image/image_mixer.h>

#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace caspar { namespace accelerator {

struct accelerator::impl
{
    // One accelerator_device per physical GPU index. Multiple channel mixers on
    // the same GPU share a single device; cross-GPU channels get their own.
    std::map<int, std::shared_ptr<accelerator_device>> devices_;
    std::mutex                                   devices_mutex_;
    std::shared_ptr<ogl::device>                 previz_ogl_device_;
    std::shared_ptr<ogl::channel_texture_store>  channel_tex_store_;
#ifdef ENABLE_VULKAN
    std::shared_ptr<vulkan::previz_texture_bridge> previz_bridge_;
    std::once_flag                               bridge_init_flag_;
#endif
    std::once_flag                               tex_store_init_flag_;
    std::once_flag                               previz_ogl_init_flag_;
    const core::video_format_repository          format_repository_;
    accelerator_backend                          backend_;

    impl(const core::video_format_repository format_repository)
        : format_repository_(format_repository)
        , backend_(accelerator_backend::invalid)
    {
    }

    void set_backend(accelerator_backend backend)
    {
        if (backend_ != accelerator_backend::invalid) {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Accelerator backend already set"));
        }

        backend_ = backend;
    }

    std::unique_ptr<core::image_mixer> create_image_mixer(int channel_id, common::bit_depth depth, int gpu_index)
    {
#ifdef ENABLE_VULKAN
        if (backend_ == accelerator_backend::vulkan) {
            auto mixer = std::make_unique<vulkan::image_mixer>(
                spl::make_shared_ptr(std::dynamic_pointer_cast<vulkan::device>(get_device(gpu_index))),
                channel_id,
                format_repository_.get_max_video_format_size(),
                depth);
            // Previz (3D monitoring) is bridged to a dedicated OGL device on GPU 0.
            // Only wire it up for mixers that also run on GPU 0; mixers on other
            // GPUs would require a cross-device bridge that is not yet implemented.
            if (gpu_index == 0) {
                mixer->set_previz_ogl_device(get_previz_ogl_device());
                mixer->set_previz_bridge(get_previz_bridge());
            } else {
                CASPAR_LOG(info) << L"[accelerator] Channel " << channel_id << L" mixer runs on GPU "
                                 << gpu_index << L"; previz 3D bridge disabled (previz device is on GPU 0).";
            }
            mixer->set_channel_texture_store(get_channel_texture_store());
            return mixer;
        }
#endif
        auto mixer = std::make_unique<ogl::image_mixer>(
            spl::make_shared_ptr(std::dynamic_pointer_cast<ogl::device>(get_device(gpu_index))),
            channel_id,
            format_repository_.get_max_video_format_size(),
            depth);
        mixer->set_channel_texture_store(get_channel_texture_store());
        return mixer;
    }

    std::shared_ptr<accelerator_device> get_device(int gpu_index = 0)
    {
        if (backend_ == accelerator_backend::invalid) {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Accelerator backend not set"));
        }

#ifndef ENABLE_VULKAN
        // OGL backend: GPU affinity is not yet implemented — all channels share a
        // single default OGL device regardless of gpu_index.
        if (gpu_index != 0) {
            CASPAR_LOG(warning) << L"[accelerator] OGL backend ignores gpu_index " << gpu_index
                                << L" (GPU affinity not implemented); using default GPU.";
            gpu_index = 0;
        }
#else
        if (backend_ != accelerator_backend::vulkan && gpu_index != 0) {
            CASPAR_LOG(warning) << L"[accelerator] OGL backend ignores gpu_index " << gpu_index
                                << L" (GPU affinity not implemented); using default GPU.";
            gpu_index = 0;
        }
#endif

        std::lock_guard<std::mutex> lock(devices_mutex_);
        auto it = devices_.find(gpu_index);
        if (it != devices_.end())
            return it->second;

        std::shared_ptr<accelerator_device> dev;
#ifdef ENABLE_VULKAN
        if (backend_ == accelerator_backend::vulkan) {
            dev = std::dynamic_pointer_cast<accelerator_device>(std::make_shared<vulkan::device>(gpu_index));
        } else
#endif
        {
            dev = std::dynamic_pointer_cast<accelerator_device>(std::make_shared<ogl::device>());
        }
        devices_[gpu_index] = dev;
        return dev;
    }

    std::shared_ptr<ogl::channel_texture_store> get_channel_texture_store()
    {
        std::call_once(tex_store_init_flag_, [this] {
            channel_tex_store_ = std::make_shared<ogl::channel_texture_store>();
        });
        return channel_tex_store_;
    }

    std::shared_ptr<ogl::device> get_previz_ogl_device()
    {
        std::call_once(previz_ogl_init_flag_, [this] {
            previz_ogl_device_ = std::make_shared<ogl::device>();
            CASPAR_LOG(info) << L"[accelerator] Created dedicated OGL device for previz (VK backend).";
        });
        return previz_ogl_device_;
    }

#ifdef ENABLE_VULKAN
    std::shared_ptr<vulkan::previz_texture_bridge> get_previz_bridge()
    {
        std::call_once(bridge_init_flag_, [this] {
            // Previz bridge always couples GPU 0's vulkan device with the previz OGL device.
            auto vk_dev = std::dynamic_pointer_cast<vulkan::device>(get_device(0));
            previz_bridge_ = std::make_shared<vulkan::previz_texture_bridge>(
                spl::make_shared_ptr(vk_dev), get_previz_ogl_device());
            CASPAR_LOG(info) << L"[accelerator] Created shared VK->GL previz texture bridge.";
        });
        return previz_bridge_;
    }
#endif
};

accelerator::accelerator(const core::video_format_repository format_repository)
    : impl_(std::make_unique<impl>(format_repository))
{
}

accelerator::~accelerator() {}

void accelerator::set_backend(accelerator_backend backend) { impl_->set_backend(backend); }

std::unique_ptr<core::image_mixer>
accelerator::create_image_mixer(const int channel_id, common::bit_depth depth, int gpu_index)
{
    return impl_->create_image_mixer(channel_id, depth, gpu_index);
}

std::shared_ptr<accelerator_device> accelerator::get_device() const
{
    return impl_->get_device();
}

std::shared_ptr<ogl::channel_texture_store> accelerator::get_channel_texture_store()
{
    return impl_->get_channel_texture_store();
}

std::shared_ptr<ogl::device> accelerator::get_previz_ogl_device()
{
    return impl_->get_previz_ogl_device();
}

}} // namespace caspar::accelerator
