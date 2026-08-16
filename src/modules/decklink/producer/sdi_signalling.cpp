// No "../StdAfx.h" here, unlike most of this module. CMake force-includes the precompiled
// header for the `decklink` target anyway, so the explicit include buys nothing there -- and
// it would drag boost and TBB into `decklink_sdi_signalling_test`, which needs neither. The
// parsers deliberately depend on nothing but the colour enums, which is what makes them
// testable without a DeckLink card in the machine. `consumer/vanc.cpp` omits it the same way.
#include "sdi_signalling.h"

namespace caspar { namespace decklink {

namespace {

/// ST 292-1:2018 and ST 425-1:2017 both put these two fields in the same bit positions, which
/// is worth stating because the two standards describe them in differently-shaped tables and
/// it would be easy to conclude they disagree. EBU Tech 3375 reproduces both tables *and* the
/// resulting byte values, and those cross-check:
///
///   HD  1080p/25  SDR 709  85 C5 80 01     3G  1080p/50  SDR 709  89 C9 80 01
///   HD  1080p/25  PQ 2100  85 E5 A0 01     3G  1080p/50  PQ 2100  89 E9 A0 01
///
/// C5h and C9h differ from E5h and E9h only in byte 2 bits 5-4 (00 -> 10, SDR -> PQ); 80h
/// differs from A0h only in byte 3 bits 5-4 (00 -> 10, Rec.709 -> UHDTV). Same positions on
/// both interfaces, established from published values rather than from reading one table.
constexpr uint8_t TRANSFER_SHIFT    = 4;
constexpr uint8_t COLORIMETRY_SHIFT = 4;
constexpr uint8_t FIELD_MASK        = 0x03;

/// Below this, byte 1 names an SD interface predating the reassignment of those reserved
/// bits. See the header for why that distinction cannot be collapsed.
constexpr uint8_t FIRST_HD_INTERFACE_BYTE1 = 0x84;

uint16_t be16(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }

uint32_t be32(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

} // namespace

void parse_vpid(const uint8_t payload[4], sdi_signalling& out)
{
    out.vpid_present = true;
    for (int i = 0; i < 4; ++i) {
        out.vpid[i] = payload[i];
    }

    if (payload[0] < FIRST_HD_INTERFACE_BYTE1) {
        return;
    }

    switch ((payload[1] >> TRANSFER_SHIFT) & FIELD_MASK) {
        case 0:
            out.color_transfer    = core::color_transfer::sdr;
            out.transfer_specified = true;
            break;
        case 1:
            out.color_transfer    = core::color_transfer::hlg;
            out.transfer_specified = true;
            break;
        case 2:
            out.color_transfer    = core::color_transfer::pq;
            out.transfer_specified = true;
            break;
        default:
            // 3h is "Unspecified", which is a statement that the sender does not know --
            // materially different from the sender saying SDR, and the only honest mapping
            // is to leave the caller's fallback in place.
            break;
    }

    switch ((payload[2] >> COLORIMETRY_SHIFT) & FIELD_MASK) {
        case 0:
            out.color_space           = core::color_space::bt709;
            out.colorimetry_specified = true;
            break;
        case 2:
            // "UHDTV" in the interface tables, which is BT.2020's primaries. BT.2100 uses
            // the same primaries and is signalled by this value together with a PQ or HLG
            // transfer, so there is no separate code point to look for.
            out.color_space           = core::color_space::bt2020;
            out.colorimetry_specified = true;
            break;
        default:
            // 1h is Reserved and 3h is Unknown; neither is a colour space.
            break;
    }
}

void parse_st2108(const uint8_t* udw, size_t size, sdi_signalling& out)
{
    out.st2108_present = true;

    // Each metadata frame is [type][length][data...]. Walking by the declared length rather
    // than by a table of known sizes is what makes an unrecognised frame skippable instead of
    // fatal -- a stream carrying ST 2094-40 dynamic metadata we do not parse still yields its
    // mastering display.
    size_t i = 0;
    while (i + 2 <= size) {
        const uint8_t type   = udw[i];
        const uint8_t length = udw[i + 1];
        const uint8_t* data  = udw + i + 2;

        if (i + 2 + length > size) {
            break; // truncated frame; everything after it is unreliable too
        }

        // Data byte 1 begins the encoded SEI payloadType, so the frame's own two-byte SEI
        // header sits at data[0..1] and the message body starts at data[2].
        if (type == 0 && length == 0x1A && data[0] == 0x89 && data[1] == 0x18) {
            // mastering_display_colour_volume(), Rec. ITU-T H.265.
            const uint8_t* m = data + 2;
            // display_primaries x/y for G, B, R then the white point: 16 bytes we do not
            // keep, because there is nowhere in `core` to put primaries.
            out.max_display_mastering_luminance = be32(m + 16) * 0.0001; // units of 0.0001 cd/m2
            out.min_display_mastering_luminance = be32(m + 20) * 0.0001;
            out.mastering_display_present       = true;
        } else if (type == 1 && length == 0x06 && data[0] == 0x90 && data[1] == 0x04) {
            // content_light_level_info(), Rec. ITU-T H.265 -- MaxCLL and MaxFALL per CTA-861.3.
            out.max_content_light_level        = be16(data + 2);
            out.max_frame_average_light_level  = be16(data + 4);
            out.content_light_level_present    = true;
        }

        i += 2 + length;
    }
}

}} // namespace caspar::decklink
