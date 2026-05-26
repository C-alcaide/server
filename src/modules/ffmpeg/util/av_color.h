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
 */
#pragma once

#include <core/frame/pixel_format.h>

#include <memory>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
extern "C" {
#include <libavutil/frame.h>
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace caspar { namespace ffmpeg {

inline core::color_space get_color_space(const std::shared_ptr<AVFrame>& video,
                                         AVColorSpace fallback = AVCOL_SPC_UNSPECIFIED)
{
    AVColorSpace cs = AVCOL_SPC_UNSPECIFIED;
    if (video) {
        cs = static_cast<AVColorSpace>(video->colorspace);
    }
    if (cs == AVCOL_SPC_UNSPECIFIED) {
        cs = fallback;
    }
    switch (cs) {
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            return core::color_space::bt2020;
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M:
        case AVCOL_SPC_SMPTE240M:
            return core::color_space::bt601;
        default:
            break;
    }

    // Check color_primaries for wide-gamut detection (matrix coefficients don't distinguish P3/Adobe)
    if (video) {
        switch (static_cast<AVColorPrimaries>(video->color_primaries)) {
            case AVCOL_PRI_SMPTE432: return core::color_space::p3_d65;
            case AVCOL_PRI_SMPTE431: return core::color_space::p3_dci;
            default: break;
        }
    }

    return core::color_space::bt709;
}

inline core::color_transfer get_color_transfer(const std::shared_ptr<AVFrame>& video,
                                               AVColorTransferCharacteristic fallback = AVCOL_TRC_UNSPECIFIED)
{
    AVColorTransferCharacteristic trc = AVCOL_TRC_UNSPECIFIED;
    if (video) {
        trc = static_cast<AVColorTransferCharacteristic>(video->color_trc);
    }
    if (trc == AVCOL_TRC_UNSPECIFIED) {
        trc = fallback;
    }
    switch (trc) {
        case AVCOL_TRC_SMPTE2084:
            return core::color_transfer::pq;
        case AVCOL_TRC_ARIB_STD_B67:
            return core::color_transfer::hlg;
        case AVCOL_TRC_LINEAR:
            return core::color_transfer::linear;
        case AVCOL_TRC_GAMMA22:
        case AVCOL_TRC_IEC61966_2_1: // sRGB ~2.2, closest to gamma24
            return core::color_transfer::gamma24;
        case AVCOL_TRC_GAMMA28:
            return core::color_transfer::gamma26;
        default:
            break;
    }
    return core::color_transfer::sdr;
}

}} // namespace caspar::ffmpeg
