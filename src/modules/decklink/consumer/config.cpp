/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 * Author: Julian Waller, julian@superfly.tv
 */

#include "config.h"

#include <common/except.h>
#include <common/param.h>
#include <common/ptree.h>

#ifdef WIN32
#include <isa_availability.h>

#define CHECK_INSTRUCTION_SUPPORT(a, v) (__check_arch_support((a), (v)) || __check_isa_support((a), (v)))
#endif

namespace caspar { namespace decklink {

port_configuration parse_output_config(const boost::property_tree::wptree&  ptree,
                                       const core::video_format_repository& format_repository)
{
    port_configuration port_config;
    port_config.device_index = ptree.get(L"device", static_cast<int64_t>(-1));
    port_config.key_only     = ptree.get(L"key-only", port_config.key_only);

    auto format_desc_str = ptree.get(L"video-mode", L"");
    if (!format_desc_str.empty()) {
        auto format_desc = format_repository.find(format_desc_str);
        if (format_desc.format == core::video_format::invalid || format_desc.format == core::video_format::custom)
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid video-mode: " + format_desc_str));
        port_config.format = format_desc;
    }

    auto subregion_tree = ptree.get_child_optional(L"subregion");
    if (subregion_tree) {
        port_config.src_x    = subregion_tree->get(L"src-x", port_config.src_x);
        port_config.src_y    = subregion_tree->get(L"src-y", port_config.src_y);
        port_config.dest_x   = subregion_tree->get(L"dest-x", port_config.dest_x);
        port_config.dest_y   = subregion_tree->get(L"dest-y", port_config.dest_y);
        port_config.region_w = subregion_tree->get(L"width", port_config.region_w);
        port_config.region_h = subregion_tree->get(L"height", port_config.region_h);
    }

    return port_config;
}

vanc_configuration parse_vanc_config(const boost::property_tree::wptree& vanc_tree)
{
    vanc_configuration vanc_config;

    vanc_config.enable            = true;
    vanc_config.op47_line         = vanc_tree.get(L"op47-line", vanc_config.op47_line);
    vanc_config.op47_line_field2  = vanc_tree.get(L"op47-line-field2", vanc_config.op47_line_field2);
    vanc_config.enable_op47       = vanc_config.op47_line > 0;
    vanc_config.op42_sd_line      = vanc_tree.get(L"op42-sd-line", vanc_config.op42_sd_line);
    vanc_config.scte104_line      = vanc_tree.get(L"scte104-line", vanc_config.scte104_line);
    vanc_config.enable_scte104    = vanc_config.scte104_line > 0;
    vanc_config.op47_dummy_header = vanc_tree.get(L"op47-dummy-header", L"");
    vanc_config.hdr_line          = vanc_tree.get(L"hdr-line", vanc_config.hdr_line);
    vanc_config.enable_hdr        = vanc_config.hdr_line > 0;

    return vanc_config;
};

core::color_space get_color_space(const std::wstring& str)
{
    auto color_space_str = boost::to_lower_copy(str);
    if (color_space_str == L"bt709")
        return core::color_space::bt709;
    else if (color_space_str == L"bt2020")
        return core::color_space::bt2020;
    else if (color_space_str == L"bt601")
        return core::color_space::bt601;

    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid decklink color-space: must be bt601, bt709, or bt2020 (SDI cannot signal P3 or Adobe RGB)"));
}

core::color_transfer get_color_transfer(const std::wstring& str)
{
    auto s = boost::to_lower_copy(str);
    if (s == L"pq")
        return core::color_transfer::pq;
    else if (s == L"hlg")
        return core::color_transfer::hlg;
    else if (s == L"sdr")
        return core::color_transfer::sdr;

    CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid decklink color-transfer: must be sdr, pq, or hlg (SDI cannot signal linear/gamma24/gamma26)"));
}

configuration parse_xml_config(const boost::property_tree::wptree&  ptree,
                               const core::video_format_repository& format_repository,
                               const core::channel_info&            channel_info)
{
    configuration config;

    // Both or neither: a display without a view is not a transform, and accepting one
    // silently would render the channel's view while looking configured.
    auto ocio_display = ptree.get(L"ocio-display", L"");
    auto ocio_view    = ptree.get(L"ocio-view", L"");
    if (ocio_display.empty() != ocio_view.empty())
        CASPAR_THROW_EXCEPTION(user_error() << msg_info(
            L"decklink consumer needs <ocio-display> AND <ocio-view>, or neither."));
    config.ocio_display = u8(ocio_display);
    config.ocio_view    = u8(ocio_view);

    auto duplex = ptree.get(L"duplex", L"default");
    if (duplex == L"full") {
        config.duplex = configuration::duplex_t::full_duplex;
    } else if (duplex == L"half") {
        config.duplex = configuration::duplex_t::half_duplex;
    }

    auto latency = ptree.get(L"latency", L"default");
    if (latency == L"low") {
        config.latency = configuration::latency_t::low_latency;
    } else if (latency == L"normal") {
        config.latency = configuration::latency_t::normal_latency;
    } else if (latency == L"sync") {
        config.latency = configuration::latency_t::sync_display;
    }

    auto wait_for_reference = ptree.get(L"wait-for-reference", L"auto");
    if (wait_for_reference == L"disable" || wait_for_reference == L"disabled") {
        config.wait_for_reference = configuration::wait_for_reference_t::disabled;
    } else if (wait_for_reference == L"enable" || wait_for_reference == L"enabled") {
        config.wait_for_reference = configuration::wait_for_reference_t::enabled;
    } else {
        config.wait_for_reference = configuration::wait_for_reference_t::automatic;
    }
    config.wait_for_reference_duration = ptree.get(L"wait-for-reference-duration", config.wait_for_reference_duration);

    {
        auto is_8bit              = channel_info.depth == common::bit_depth::bit8;
        auto default_pixel_format = is_8bit ? L"rgba" : L"yuv";
        auto pixel_format         = ptree.get(L"pixel-format", default_pixel_format);
        if (pixel_format == L"yuv") {
            config.pixel_format = configuration::pixel_format_t::yuv;
        } else if (pixel_format == L"rgba") {
            config.pixel_format = configuration::pixel_format_t::rgba;
        } else {
            CASPAR_THROW_EXCEPTION(user_error() << msg_info(L"Invalid pixel format, must be rgba or yuv"));
        }

        if (channel_info.depth != common::bit_depth::bit8 &&
            config.pixel_format == configuration::pixel_format_t::rgba) {
            CASPAR_THROW_EXCEPTION(user_error()
                                   << msg_info(L"The decklink consumer only supports rgba output on 8-bit channels"));
        }

        if (config.pixel_format != configuration::pixel_format_t::rgba) {
#ifdef WIN32
            if (!CHECK_INSTRUCTION_SUPPORT(__IA_SUPPORT_VECTOR256, 0)) {
#elif defined(__x86_64__) || defined(__i386__)
            if (!__builtin_cpu_supports("avx2")) {
#else
            if (false) {
#endif
                CASPAR_THROW_EXCEPTION(user_error()
                                       << msg_info(L"Your cpu does not support the features needed for yuv output"));
            }
        }
    }

    {
        auto gpu_readback_mode = ptree.get(L"gpu-readback-mode", ptree.get(L"gpu-strategy", L"auto"));
        if (gpu_readback_mode == L"cuda")
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::cuda;
        else if (gpu_readback_mode == L"vulkan")
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::vulkan;
        else if (gpu_readback_mode == L"vulkan-dma")
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::vulkan_dma;
        else if (gpu_readback_mode == L"cpu")
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::cpu;
        else
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::auto_select;
    }

    {
        // Final GPU->card transfer of the packed frame (orthogonal to the pack strategy).
        auto gpu_transfer = ptree.get(L"gpu-transfer", L"auto");
        if (gpu_transfer == L"dvp")
            config.gpu_transfer = configuration::gpu_transfer_t::dvp;
        else if (gpu_transfer == L"copy")
            config.gpu_transfer = configuration::gpu_transfer_t::copy;
        else
            config.gpu_transfer = configuration::gpu_transfer_t::auto_select;
    }

    {
        // OpenGL-mixer v210/BGRA pack location (GL compute vs CPU AVX2).
        auto gpu_pack = ptree.get(L"gpu-pack", L"auto");
        if (gpu_pack == L"gpu")
            config.gpu_pack = configuration::gpu_pack_t::gpu;
        else if (gpu_pack == L"cpu")
            config.gpu_pack = configuration::gpu_pack_t::cpu;
        else
            config.gpu_pack = configuration::gpu_pack_t::auto_select;
    }

    config.primary = parse_output_config(ptree, format_repository);
    if (config.primary.device_index == -1)
        config.primary.device_index = 1;

    // HISTORY, NOT CURRENT BEHAVIOUR -- read the narrowed rule below for what happens today.
    //
    // The GPU readback strategies once implemented only the subregion's SOURCE ORIGIN: `src-x`
    // and `src-y` reached the shaders as push constants and the DMA path as an imageOffset,
    // while `dest-x`, `dest-y`, `width` and `height` reached nothing and were dropped
    // silently, so the wire carried the whole source from the origin, placed at 0,0.
    //
    // (That paragraph was read as present tense on 2026-08-27 and written into an operator
    // guide as a live defect. The measurement below is what motivated the coercion, and the
    // coercion is what fixed it -- a comment explaining a problem in this file is usually
    // followed by its fix.)
    //
    // Measured over an SDI loopback with `640x360 from (100,200) placed at (114,70)`: the
    // CPU strategy puts 360x640 at (70,114) and scores 62.92 dB against the frame that
    // geometry implies; `gpu-readback-mode=vulkan` put 1820x880 at (0,0) and scored
    // 7.91 dB. Same configuration, different picture, no warning.
    //
    // Coerced HERE rather than in `create_format_strategy`, which is where the obvious fix
    // goes and where it does not work: by then the consumer has already told the mixer that
    // no CPU frame data is needed, so a late substitution gets a frame carrying no host
    // pixels and puts NOTHING on the wire — measured, the capture goes flat. The comment on
    // `dma.pixels` in vk_readback_strategy.cpp says exactly this. Deciding at parse time
    // keeps `needs_cpu_frame_data()` consistent with the strategy that will run.
    //
    // `ogl_gpu_pack_eligible` already refuses the OpenGL packer on the same geometry; this
    // is the same rule for the DeckLink readback. Note neither tests `src_x`/`src_y`: an
    // origin-only subregion IS handled on the GPU, and measures 53.55 dB against its model.
    {
        // NARROWED 2026-08-27: the Vulkan COMPUTE readback now implements destination
        // placement, so it is no longer coerced. `vk_readback_v210.comp` and
        // `vk_readback_bgra.comp` walk OUTPUT pixels and decide per pixel whether one falls
        // inside the destination rectangle, which is what makes it cheap -- and is why
        // `dest_x` needs no 6-pixel alignment despite V210 packing six pixels per four words.
        // Walking the source would need read-modify-write on a straddling group; walking the
        // destination computes every group from scratch. Writing every group also blacks the
        // surround, so no clear pass is needed.
        //
        // STILL COERCED, for reasons that are mechanism-specific rather than effort:
        //   * `vulkan-dma` copies image->buffer with a single `region.imageOffset`. There is no
        //     shader in that path to place anything, and a VkBufferImageCopy cannot express a
        //     destination rectangle inside a larger frame.
        //   * `cuda` packs in a CUDA kernel that has not been given the same treatment.
        auto needs_cpu_geometry = [](const port_configuration& p) {
            return p.dest_x != 0 || p.dest_y != 0 || p.region_w != 0 || p.region_h != 0;
        };
        const bool mode_can_place =
            config.gpu_readback_mode == configuration::gpu_readback_mode_t::cpu ||
            config.gpu_readback_mode == configuration::gpu_readback_mode_t::vulkan;
        if (needs_cpu_geometry(config.primary) && !mode_can_place) {
            CASPAR_LOG(warning)
                << L"[decklink] <subregion> sets dest-x/dest-y/width/height, which this "
                   L"gpu-readback-mode cannot place (vulkan-dma has no shader; cuda is not "
                   L"implemented); falling back to gpu-readback-mode=cpu so the geometry is "
                   L"honoured. gpu-readback-mode=vulkan does support it.";
            config.gpu_readback_mode = configuration::gpu_readback_mode_t::cpu;
        }
    }

    auto keyer = ptree.get(L"keyer", L"default");
    if (keyer == L"external") {
        config.keyer = configuration::keyer_t::external_keyer;
    } else if (keyer == L"internal") {
        config.keyer = configuration::keyer_t::internal_keyer;
    } else if (keyer == L"disabled") {
        config.keyer = configuration::keyer_t::disabled_keyer;
    } else if (keyer == L"external_separate_device") {
        config.keyer = configuration::keyer_t::external_keyer;

        auto key_config         = config.primary; // Copy the primary config
        key_config.device_index = ptree.get(L"key-device", static_cast<int64_t>(0));
        if (key_config.device_index == 0) {
            key_config.device_index = config.primary.device_index + 1;
        }
        key_config.key_only = true;
        config.secondaries.push_back(key_config);
    }

    config.embedded_audio    = ptree.get(L"embedded-audio", config.embedded_audio);
    config.base_buffer_depth = ptree.get(L"buffer-depth", config.base_buffer_depth);

    if (ptree.get_child_optional(L"ports")) {
        for (auto& xml_port : ptree | witerate_children(L"ports") | welement_context_iteration) {
            ptree_verify_element_name(xml_port, L"port");

            port_configuration port_config = parse_output_config(xml_port.second, format_repository);

            config.secondaries.push_back(port_config);
        }
    }

    config.color_space   = channel_info.default_color_space;
    auto color_space_str = ptree.get(L"color-space", L"");
    if (!color_space_str.empty())
        config.color_space = get_color_space(color_space_str);

    config.color_transfer   = channel_info.default_color_transfer;
    auto color_transfer_str = ptree.get(L"color-transfer", L"");
    if (!color_transfer_str.empty())
        config.color_transfer = get_color_transfer(color_transfer_str);

    // Note: config.hdr is set by the caller (create_consumer / create_preconfigured_consumer)
    // based on both channel bit-depth and color settings.

    auto hdr_metadata = ptree.get_child_optional(L"hdr-metadata");
    if (hdr_metadata) {
        config.hdr_meta.min_dml  = hdr_metadata->get(L"min-dml", config.hdr_meta.min_dml);
        config.hdr_meta.max_dml  = hdr_metadata->get(L"max-dml", config.hdr_meta.max_dml);
        config.hdr_meta.max_fall = hdr_metadata->get(L"max-fall", config.hdr_meta.max_fall);
        config.hdr_meta.max_cll  = hdr_metadata->get(L"max-cll", config.hdr_meta.max_cll);
    }

    auto vanc = ptree.get_child_optional(L"vanc");
    if (vanc) {
        config.vanc = parse_vanc_config(vanc.get());
    }

    return config;
}

configuration parse_amcp_config(const std::vector<std::wstring>&     params,
                                const core::video_format_repository& format_repository,
                                const core::channel_info&            channel_info)
{
    configuration config;

    if (params.size() > 1)
        config.primary.device_index = std::stoll(params.at(1));

    if (contains_param(L"INTERNAL_KEY", params)) {
        config.keyer = configuration::keyer_t::internal_keyer;
    } else if (contains_param(L"EXTERNAL_KEY", params)) {
        config.keyer = configuration::keyer_t::external_keyer;
    } else if (contains_param(L"DISABLED_KEY", params)) {
        config.keyer = configuration::keyer_t::disabled_keyer;
    } else {
        config.keyer = configuration::keyer_t::default_keyer;
    }

    if (contains_param(L"FULL_DUPLEX", params)) {
        config.duplex = configuration::duplex_t::full_duplex;
    } else if (contains_param(L"HALF_DUPLEX", params)) {
        config.duplex = configuration::duplex_t::half_duplex;
    }

    if (contains_param(L"LOW_LATENCY", params)) {
        config.latency = configuration::latency_t::low_latency;
    }

    config.embedded_audio   = contains_param(L"EMBEDDED_AUDIO", params);
    config.primary.key_only = contains_param(L"KEY_ONLY", params);

    config.color_space    = channel_info.default_color_space;
    config.color_transfer = channel_info.default_color_transfer;

    auto color_space_str = get_param(L"COLOR_SPACE", params);
    if (!color_space_str.empty()) {
        auto cs = boost::to_lower_copy(color_space_str);
        if (cs == L"bt2020")
            config.color_space = core::color_space::bt2020;
        else if (cs == L"bt601")
            config.color_space = core::color_space::bt601;
        else if (cs == L"bt709")
            config.color_space = core::color_space::bt709;
        // P3/Adobe ignored — SDI cannot signal them; channel default is used instead
    }

    auto color_transfer_str = get_param(L"COLOR_TRANSFER", params);
    if (!color_transfer_str.empty()) {
        auto ct = boost::to_lower_copy(color_transfer_str);
        if (ct == L"pq")
            config.color_transfer = core::color_transfer::pq;
        else if (ct == L"hlg")
            config.color_transfer = core::color_transfer::hlg;
        else if (ct == L"sdr")
            config.color_transfer = core::color_transfer::sdr;
        // linear/gamma24/gamma26 ignored — SDI cannot signal them
    }

    // Note: config.hdr is set by the caller based on both channel bit-depth and color settings.

    return config;
}

}} // namespace caspar::decklink
