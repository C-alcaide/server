/*
 * Copyright 2025
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
 *
 * Author: Niklas Andersson, niklas@niklaspandersson.se
 */

#include "gpu_test.h"

#include <core/frame/draw_frame.h>
#include <core/frame/frame.h>
#include <core/frame/frame_factory.h>
#include <core/frame/pixel_format.h>
#include <core/producer/frame_producer.h>
#include <core/producer/frame_producer_registry.h>
#include <core/video_format.h>

#include <common/log.h>
#include <common/memory.h>

#include <boost/algorithm/string.hpp>

#include <array>
#include <cstdint>
#include <cstring>

#ifdef ENABLE_VULKAN
#include <accelerator/vulkan/util/gpu_producer.h>
#include <accelerator/vulkan/util/texture.h>

#include <vulkan/vulkan.hpp>
#endif

namespace caspar { namespace gpu_test {

// A named color as straight RGBA (0..1). Kept tiny — this producer exists to exercise the GPU
// producer path end to end, not to be a feature.
bool try_get_color(const std::wstring& name, std::array<float, 4>& out)
{
    if (boost::iequals(name, L"RED"))
        out = {1.0f, 0.0f, 0.0f, 1.0f};
    else if (boost::iequals(name, L"GREEN"))
        out = {0.0f, 1.0f, 0.0f, 1.0f};
    else if (boost::iequals(name, L"BLUE"))
        out = {0.0f, 0.0f, 1.0f, 1.0f};
    else if (boost::iequals(name, L"WHITE"))
        out = {1.0f, 1.0f, 1.0f, 1.0f};
    else if (boost::iequals(name, L"BLACK"))
        out = {0.0f, 0.0f, 0.0f, 1.0f};
    else
        return false;
    return true;
}

class gpu_test_producer : public core::frame_producer
{
    const spl::shared_ptr<core::frame_factory> frame_factory_;
    const int                                  width_;
    const int                                  height_;
    const std::array<float, 4>                 color_;
    const std::wstring                         color_str_;

#ifdef ENABLE_VULKAN
    // The producer owns its GPU work via gpu_producer: it holds a command_context on the compute queue
    // (created once) and folds the per-frame hand-off plumbing into produce(). operator bool is false
    // when the channel is not on the Vulkan accelerator — then we use the CPU fallback.
    accelerator::vulkan::gpu_producer gpu_;
#endif

  public:
    gpu_test_producer(const spl::shared_ptr<core::frame_factory>& frame_factory,
                      int                                         width,
                      int                                         height,
                      std::array<float, 4>                        color,
                      std::wstring                                color_str)
        : frame_factory_(frame_factory)
        , width_(width)
        , height_(height)
        , color_(color)
        , color_str_(std::move(color_str))
#ifdef ENABLE_VULKAN
        , gpu_(frame_factory, accelerator::vulkan::queue_type::compute)
#endif
    {
#ifdef ENABLE_VULKAN
        if (!gpu_)
            CASPAR_THROW_EXCEPTION(std::runtime_error("GPU test producer requires Vulkan accelerator"));
#else
        CASPAR_THROW_EXCEPTION(std::runtime_error("GPU test producer requires Vulkan accelerator"));
#endif

        CASPAR_LOG(info) << print() << L" Initialized";
    }

    core::draw_frame receive_impl(const core::video_field field, int nb_samples) override
    {
        // The producer-side counterpart to the screen consumer's downcast: when on the Vulkan
        // accelerator, generate the frame straight on the GPU (a clear, here) and hand over the
        // texture. gpu_producer owns the hand-off plumbing — we declare only the texture and the
        // scope we leave it in (the producer_plane defaults: eGeneral, transfer write), and record
        // just the clear. The acquire transition, the producer->render release, stamping the
        // completion and the import are all handled by produce().
        using namespace accelerator::vulkan;

        auto tex = gpu_.factory().create_producer_texture(width_, height_, 4, common::bit_depth::bit8);

        return core::draw_frame(gpu_.produce(this,
                                             producer_plane{std::move(tex)},
                                             core::pixel_format::rgba,
                                             [&](vk::CommandBuffer cmd, const std::shared_ptr<texture>& tex) {
                                                 vk::ClearColorValue       ccv(color_);
                                                 vk::ImageSubresourceRange range(
                                                     vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
                                                 cmd.clearColorImage(tex->id(), vk::ImageLayout::eGeneral, ccv, range);
                                             }));
    }

    std::wstring         print() const override { return L"gpu_test[" + color_str_ + L"]"; }
    std::wstring         name() const override { return L"gpu_test"; }
    core::monitor::state state() const override { return {}; }
    bool                 is_ready() override { return true; }
};

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(
        L"GPU Test Producer",
        [](const core::frame_producer_dependencies& deps,
           const std::vector<std::wstring>&         params) -> spl::shared_ptr<core::frame_producer> {
            if (params.empty() || !boost::iequals(params.at(0), L"GPUTEST"))
                return core::frame_producer::empty();

            std::array<float, 4> color{1.0f, 0.0f, 0.0f, 1.0f}; // default red
            std::wstring         color_str = L"RED";
            if (params.size() > 1 && try_get_color(params.at(1), color))
                color_str = boost::to_upper_copy(params.at(1));

            try {
                return spl::make_shared<gpu_test_producer>(
                    deps.frame_factory, deps.format_desc.width, deps.format_desc.height, color, color_str);
            } catch (const std::exception& e) {
                CASPAR_LOG(warning) << L"GPU test producer failed: " << e.what();
                return core::frame_producer::empty();
            }
        });
}

}} // namespace caspar::gpu_test
