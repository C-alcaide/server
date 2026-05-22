/*
 * Copyright (c) 2025 CasparCG Contributors
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
 * CUDA-VK direct GPU decklink strategy header.
 * When the Vulkan accelerator is active, this strategy reads the VK mixer's
 * render attachment directly via CUDA external memory interop and packs v210
 * on GPU, avoiding the VK→CPU readback path entirely.
 */
#pragma once

#include "format_strategy.h"

#include <memory>

namespace caspar { namespace decklink {

/**
 * GPU-accelerated format strategy that reads VK textures via CUDA interop.
 *
 * Replaces the CPU-based hdr_v210_strategy (or sdr_bgra_strategy) when:
 *   - ENABLE_VULKAN is defined at build time
 *   - The Vulkan accelerator is active at runtime
 *   - CUDA is available on the same GPU as the VK mixer
 *
 * Falls back to the wrapped CPU strategy if the GPU path is unavailable
 * for any frame (e.g. empty frame, interlaced, or CUDA error).
 */
class cuda_vk_strategy final : public format_strategy
{
  public:
    /// @param is_hdr       True for HDR v210 output, false for SDR BGRA
    /// @param use_bt2020   True for BT.2020 color matrix, false for BT.709
    /// @param fallback     CPU strategy to use as fallback
    /// @param needs_v210   True when pixel-format=yuv (always use V210 path)
    cuda_vk_strategy(bool is_hdr, bool use_bt2020,
                     spl::shared_ptr<format_strategy> fallback,
                     bool needs_v210 = false);
    ~cuda_vk_strategy() override;

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

/// Create a CUDA-VK GPU-direct strategy that wraps the given CPU fallback.
/// Returns nullptr if CUDA is not available.
spl::shared_ptr<format_strategy> try_create_cuda_vk_strategy(
    bool is_hdr, bool use_bt2020,
    spl::shared_ptr<format_strategy> fallback,
    bool needs_v210 = false);

}} // namespace caspar::decklink
