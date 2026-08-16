/*
 * Reading the colour signalling that an SDI input actually carries.
 *
 * WHY THIS EXISTS RATHER THAN A `GetInt` CALL. `IDeckLinkVideoFrameMetadataExtensions`
 * looks like the answer -- the producer asks it for `bmdDeckLinkFrameMetadataColorspace`
 * and `...HDRElectroOpticalTransferFunc` -- and on an SDI input it returns nothing at all.
 * Measured over a 1->4 loopback on 2026-08-16 with the consumer configured 16-bit/PQ/BT.2020
 * at both ends: `colorspace=unknown transfer=sdr`, both arms. The SDK's own helpstring for
 * the 11.5 revision of that interface says what it is for -- "HDMI HDR information".
 *
 * On SDI the signalling is in the ancillary data, and it is split across TWO standards that
 * carry disjoint halves:
 *
 *   SMPTE ST 352   DID 41h SDID 01h   colorimetry and transfer characteristics
 *   SMPTE ST 2108-1  DID 41h SDID 0Ch   mastering display volume, content light level
 *
 * ST 2108-1 says so itself, in its own introduction: it "does not replace SMPTE ST 352 ...
 * as the primary method for SDI format signaling", and is "an extension to such methods when
 * they lack the capability to transmit HDR/WCG metadata parameters". So a receiver that wants
 * to know whether a feed is PQ BT.2100 reads the VPID, and one that wants the mastering
 * display's peak luminance reads ST 2108-1. Reading only one of the two gets you half an
 * answer, and reading neither -- which is where this module started -- gets you `unknown`.
 *
 * INTEROPERABILITY. Both are open SMPTE standards carried in ST 291-1 ancillary packets, so
 * what is decoded here is what any conforming vendor emits; nothing in this file knows it is
 * talking to a Blackmagic card. That is deliberate, and it is the reason the parsing lives
 * here in plain bytes rather than behind the SDK's metadata abstraction, which is HDMI-shaped
 * and vendor-specific. EBU Tech 3375 publishes the exact four VPID bytes for each HDR case,
 * which gives the tests a ground truth that does not come from us or from Blackmagic.
 */
#pragma once

#include <core/frame/pixel_format.h>

#include <cstdint>
#include <string>

namespace caspar { namespace decklink {

/// What the ancillary data said. Every field carries its own "was this actually present"
/// flag rather than leaning on a sentinel, because the whole difficulty in this area is
/// telling "the source declared Rec.709" apart from "the source declared nothing and 709 is
/// the conventional default" -- a distinction the mixer needs to keep, since an untagged
/// sub-720 signal is conventionally BT.601 and a *tagged* one must be honoured whatever its
/// raster size.
struct sdi_signalling
{
    /// Whether the input frame offered `IDeckLinkVideoFrameAncillaryPackets` at all. False
    /// means the question "did a packet arrive" was never actually asked.
    bool anc_interface_available = false;

    bool    vpid_present = false;
    uint8_t vpid[4]      = {0, 0, 0, 0};

    core::color_space    color_space    = core::color_space::unknown;
    core::color_transfer color_transfer = core::color_transfer::sdr;

    bool colorimetry_specified = false;
    bool transfer_specified    = false;

    /// ST 2108-1 static metadata. Parsed and reported but not plumbed: `core` has nowhere to
    /// put a mastering display volume today, and inventing a field the mixer ignores would
    /// be a claim with nothing behind it. Logging it at least makes an incoming HDR feed's
    /// grading intent visible, and proves the packet arrived.
    bool     st2108_present   = false;
    bool     mastering_display_present = false;
    bool     content_light_level_present = false;
    double   max_display_mastering_luminance = 0.0; // cd/m2
    double   min_display_mastering_luminance = 0.0; // cd/m2
    unsigned max_content_light_level         = 0;   // cd/m2
    unsigned max_frame_average_light_level   = 0;   // cd/m2
};

/// Decode the 4 payload bytes of an ST 352 packet into `out`.
///
/// `byte1` identifies the interface, and it gates whether the colorimetry and transfer bits
/// mean anything: ST 352:2013 lists byte 2 bits 5-4 and byte 3 bits 5-4 as *Reserved*, and it
/// is the later interface standards (ST 292-1:2018 for HD, ST 425-1:2017 for 3G) that give
/// them their HDR meanings. Values below 84h are SD interfaces where those bits were never
/// reassigned, so a zero there means "reserved, unset" and not "Rec.709 SDR" -- reading it as
/// the latter would tag every SD feed 709 and quietly defeat the sub-720 BT.601 convention.
void parse_vpid(const uint8_t payload[4], sdi_signalling& out);

/// Decode the UDW of an ST 2108-1 packet -- a sequence of metadata frames, each
/// `[type][length][payload...]` -- into `out`. Unknown frame types are skipped by their
/// length, which is what lets a stream carrying dynamic metadata we do not understand still
/// yield its static metadata.
void parse_st2108(const uint8_t* udw, size_t size, sdi_signalling& out);

}} // namespace caspar::decklink
