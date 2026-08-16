/*
 * The chromaticity coordinates the consumer signals, in one place.
 *
 * These were a struct and three constants at file scope in `decklink_consumer.cpp`, read only
 * by the frame's `GetFloat` implementation. ST 2108-1 emission needs the same numbers to build
 * the mastering display SEI message, and two copies of a colour primary set is the shape of
 * defect this module has been bitten by before -- the per-field transform allowlist that each
 * mixer kept its own copy of, which diverged silently because nothing compared them.
 *
 * Physical constants from published standards do not drift, so the risk is not that one copy
 * becomes wrong. It is that a future colour space gets added to one list and not the other,
 * and the wire then disagrees with the metadata interface about the same frame.
 */
#pragma once

#include <core/frame/pixel_format.h>

namespace caspar { namespace decklink {

/// Field order is red, green, blue, white. NOTE this is NOT the order H.265 uses for the
/// mastering display SEI message, which indexes primaries green-blue-red;
/// `vanc_hdr_strategy.cpp` does the reordering and says so where it happens.
struct ChromaticityCoordinates
{
    double RedX;
    double RedY;
    double GreenX;
    double GreenY;
    double BlueX;
    double BlueY;
    double WhiteX;
    double WhiteY;
};

inline const ChromaticityCoordinates REC_709  = {0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290};
inline const ChromaticityCoordinates REC_2020 = {0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290};
inline const ChromaticityCoordinates REC_601  = {0.630, 0.340, 0.310, 0.595, 0.155, 0.070, 0.3127, 0.3290};

inline const ChromaticityCoordinates& primaries_for(core::color_space cs)
{
    return cs == core::color_space::bt2020 ? REC_2020 : cs == core::color_space::bt601 ? REC_601 : REC_709;
}

}} // namespace caspar::decklink
