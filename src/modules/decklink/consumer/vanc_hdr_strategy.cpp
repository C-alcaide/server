/*
 * SMPTE ST 2108-1 -- HDR/WCG metadata in the vertical ancillary data space.
 *
 * WHY THIS EXISTS. The consumer already tells the DRIVER what the colour is, through
 * `IDeckLinkVideoFrameMetadataExtensions`. That interface is HDMI-shaped -- the SDK's own
 * helpstring for its 11.5 revision says "HDMI HDR information" -- and on SDI the mastering
 * display volume and content light level have their own transport, which no part of this tree
 * was using. Measured over a 1->4 loopback on 2026-08-16: the ancillary census on the
 * receiving end was EMPTY, at 8-bit and 10-bit ingest, on HD-SDI and 3G, with `<vanc>` off and
 * on. Nothing was being sent, so nothing could be read.
 *
 * WHAT THIS DELIBERATELY DOES NOT SEND: ST 352, the payload identifier that carries
 * colorimetry and transfer characteristics. That packet is the card's to insert, and the
 * driver builds it from the frame metadata the consumer already supplies. Emitting our own
 * would risk two ST 352 packets in one field, which is worse than none -- a conforming
 * receiver is entitled to either. If it turns out the hardware does not insert it, that is a
 * separate change made on evidence, and this packet arriving is how that evidence is
 * obtained: ST 2108-1 is unambiguously the application's to send, so a census that shows
 * 41h/0Ch and no 41h/01h says the round trip works and the VPID is genuinely absent, while a
 * census that stays empty says the ancillary path never carried anything.
 *
 * Interoperability rests on this being the standard rather than a convention: the payloads are
 * H.265 SEI messages, byte for byte the same ones that reach an HDMI display or a distribution
 * encoder, and EBU Tech 3375 names ST 2108 as what production equipment should support for
 * PQ-based systems.
 */
#include "../StdAfx.h"

#include "color_primaries.h"
#include "vanc.h"

#include <common/log.h>

#include <mutex>

namespace caspar { namespace decklink {

namespace {

/// SMPTE ST 2108-1 section 5.1.
constexpr uint8_t ST2108_DID  = 0x41;
constexpr uint8_t ST2108_SDID = 0x0C;

/// Rec. ITU-T H.265: mastering display primaries and the white point are in units of 0.00002,
/// and both mastering luminances in units of 0.0001 cd/m2.
constexpr double CHROMATICITY_UNIT = 0.00002;
constexpr double LUMINANCE_UNIT    = 0.0001;

void push_be16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void push_be32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

uint16_t to_chromaticity(double v)
{
    const double scaled = v / CHROMATICITY_UNIT;
    return static_cast<uint16_t>(scaled < 0.0 ? 0.0 : scaled > 65535.0 ? 65535.0 : scaled + 0.5);
}

uint32_t to_luminance(double cd_m2)
{
    const double scaled = cd_m2 / LUMINANCE_UNIT;
    return static_cast<uint32_t>(scaled < 0.0 ? 0.0 : scaled > 4294967295.0 ? 4294967295.0 : scaled + 0.5);
}

uint16_t to_light_level(double cd_m2)
{
    return static_cast<uint16_t>(cd_m2 < 0.0 ? 0.0 : cd_m2 > 65535.0 ? 65535.0 : cd_m2 + 0.5);
}

class vanc_hdr_strategy : public decklink_vanc_strategy
{
    const uint32_t             line_number_;
    const hdr_meta_configuration hdr_;
    const core::color_space    color_space_;
    mutable std::mutex         mutex_;

  public:
    static const std::wstring Name;

    vanc_hdr_strategy(uint32_t line_number, const hdr_meta_configuration& hdr, core::color_space color_space)
        : line_number_(line_number)
        , hdr_(hdr)
        , color_space_(color_space)
    {
    }

    /// Always. ST 2108-1 section 5.2.1: transmission "shall take place once per video frame
    /// with which the metadata within the packet are associated", and static metadata is
    /// associated with every frame. Unlike OP47 and SCTE-104 this is not a queue of things
    /// somebody pushed -- there is nothing to run out of.
    bool has_data() const override { return true; }

    vanc_packet pop_packet(bool field2) override
    {
        if (field2) {
            // Once per FRAME, not per field. Sending it twice on an interlaced raster would
            // put two mastering display frames on one image, which section 5.3.2 forbids
            // outright: "No more than one HDR/WCG Metadata Frame Type value equal to 0 shall
            // be associated with any video frame."
            return {0, 0, 0, {}};
        }

        const auto& p = primaries_for(color_space_);

        std::vector<uint8_t> udw;
        udw.reserve(40);

        // ---- HDR/WCG Metadata Frame, type 0: Static Metadata Type 1 (mastering display) ----
        udw.push_back(0x00); // frame type
        udw.push_back(0x1A); // frame length: 2 SEI header bytes + 24 payload bytes
        udw.push_back(0x89); // SEI payloadType 137, mastering_display_colour_volume
        udw.push_back(0x18); // SEI payloadSize 24

        // H.265 indexes display_primaries[c] as c=0 GREEN, c=1 BLUE, c=2 RED, which is not the
        // order `ChromaticityCoordinates` declares them in and not the order anyone writes them
        // in prose. Getting this wrong yields a packet that parses cleanly and describes a
        // display that does not exist, so it is spelled out rather than looped.
        push_be16(udw, to_chromaticity(p.GreenX));
        push_be16(udw, to_chromaticity(p.GreenY));
        push_be16(udw, to_chromaticity(p.BlueX));
        push_be16(udw, to_chromaticity(p.BlueY));
        push_be16(udw, to_chromaticity(p.RedX));
        push_be16(udw, to_chromaticity(p.RedY));
        push_be16(udw, to_chromaticity(p.WhiteX));
        push_be16(udw, to_chromaticity(p.WhiteY));
        push_be32(udw, to_luminance(hdr_.max_dml));
        push_be32(udw, to_luminance(hdr_.min_dml));

        // ---- HDR/WCG Metadata Frame, type 1: Static Metadata Type 2 (content light level) ----
        udw.push_back(0x01); // frame type
        udw.push_back(0x06); // frame length: 2 SEI header bytes + 4 payload bytes
        udw.push_back(0x90); // SEI payloadType 144, content_light_level_info
        udw.push_back(0x04); // SEI payloadSize 4
        push_be16(udw, to_light_level(hdr_.max_cll));
        push_be16(udw, to_light_level(hdr_.max_fall));

        return {ST2108_DID, ST2108_SDID, line_number_, udw};
    }

    /// Nothing to push. The metadata is configuration, not traffic -- it describes the
    /// mastering display the channel is graded for, which does not change frame to frame.
    /// Returning false makes an `OP47`-style call naming this strategy report as unknown
    /// rather than silently succeed.
    bool try_push_data(const std::vector<std::wstring>&) override { return false; }

    const std::wstring& get_name() const override { return vanc_hdr_strategy::Name; }
};

} // namespace

const std::wstring vanc_hdr_strategy::Name = L"HDR";

std::shared_ptr<decklink_vanc_strategy>
create_hdr_strategy(uint32_t line_number, const hdr_meta_configuration& hdr, core::color_space color_space)
{
    return std::make_shared<vanc_hdr_strategy>(line_number, hdr, color_space);
}

}} // namespace caspar::decklink
