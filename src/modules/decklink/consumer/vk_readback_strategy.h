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
 *
 * Pure-Vulkan GPU readback strategy for DeckLink output.
 * Replaces cuda_vk_strategy when configured — eliminates CUDA runtime
 * from the DeckLink path, avoiding GPU compute contention between
 * CUDA and Vulkan on the same device.
 */
#pragma once

#include "format_strategy.h"

#include <memory>

namespace caspar { namespace decklink {

/**
 * GPU-accelerated format strategy using native Vulkan compute.
 *
 * Creates its own VkDevice (matched to the mixer's GPU via LUID),
 * imports the mixer's render attachment via VK_KHR_external_memory_win32,
 * runs a v210 packing compute shader, and copies the result to host-visible
 * staging buffers — all without touching the CUDA runtime.
 *
 * Falls back to the wrapped CPU strategy if the GPU path fails.
 */
class vk_readback_strategy final : public format_strategy
{
  public:
    vk_readback_strategy(bool is_hdr, bool use_bt2020,
                         spl::shared_ptr<format_strategy> fallback,
                         bool dma_only = false);
    ~vk_readback_strategy() override;

    BMDPixelFormat        get_pixel_format() override;
    int                   get_row_bytes(int width) override;
    std::shared_ptr<void> allocate_frame_data(const core::video_format_desc& format_desc) override;
    std::shared_ptr<void> convert_frame_for_port(const core::video_format_desc& channel_format_desc,
                                                 const core::video_format_desc& decklink_format_desc,
                                                 const port_configuration&      config,
                                                 const core::const_frame&       frame1,
                                                 const core::const_frame&       frame2,
                                                 BMDFieldDominance              field_dominance) override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// Create a VK readback strategy that wraps the given CPU fallback.
/// Returns the fallback if Vulkan is not available.
/// If dma_only is true, uses vkCmdCopyImageToBuffer (DMA engine) instead of
/// a compute shader, avoiding SM contention with CUDA workloads.
spl::shared_ptr<format_strategy> try_create_vk_readback_strategy(
    bool is_hdr, bool use_bt2020,
    spl::shared_ptr<format_strategy> fallback,
    bool dma_only = false);

}} // namespace caspar::decklink
