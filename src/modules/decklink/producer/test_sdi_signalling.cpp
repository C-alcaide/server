// Exercising the SDI signalling parsers against published values.
//
// The fork has no C++ test target, and this parser must not ship having never parsed
// anything -- a code path that has never executed is not a correct one, it is an untested
// one, and this tree has already paid for that lesson once when a dead blend-mask path came
// alive multiplying by the wrong channel order.
//
// The vectors are EBU Tech 3375's, not mine: it tabulates the four ST 352 bytes for each HDR
// case on both HD-SDI and 3G-SDI. Checking against them tests the decode against the same
// numbers a third-party vendor implements from, which is the only version of this check that
// says anything about interoperability.
#include "sdi_signalling.h"

#include <cstdio>
#include <cstring>

using namespace caspar;
using namespace caspar::decklink;

static int failures = 0;

static void check(bool ok, const char* what)
{
    if (!ok) {
        std::printf("  FAIL  %s\n", what);
        ++failures;
    } else {
        std::printf("  ok    %s\n", what);
    }
}

static void vpid_case(const char*          label,
                      uint8_t              b1,
                      uint8_t              b2,
                      uint8_t              b3,
                      uint8_t              b4,
                      core::color_space    expect_cs,
                      core::color_transfer expect_ct,
                      bool                 expect_specified)
{
    const uint8_t  word[4] = {b1, b2, b3, b4};
    sdi_signalling out;
    parse_vpid(word, out);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s (%02X %02X %02X %02X)", label, b1, b2, b3, b4);
    const bool ok = out.vpid_present && out.colorimetry_specified == expect_specified &&
                    out.transfer_specified == expect_specified &&
                    (!expect_specified || (out.color_space == expect_cs && out.color_transfer == expect_ct));
    check(ok, buf);
}

int main()
{
    std::printf("ST 352 VPID -- EBU Tech 3375 Table 4, single-link HD-SDI 1080p/25\n");
    vpid_case("SDR BT.709", 0x85, 0xC5, 0x80, 0x01, core::color_space::bt709, core::color_transfer::sdr, true);
    vpid_case("HDR BT.2100 HLG", 0x85, 0xD5, 0xA0, 0x01, core::color_space::bt2020, core::color_transfer::hlg, true);
    vpid_case("HDR BT.2100 PQ", 0x85, 0xE5, 0xA0, 0x01, core::color_space::bt2020, core::color_transfer::pq, true);

    std::printf("ST 352 VPID -- EBU Tech 3375 Table 2, single-link 3G-SDI 1080p/50\n");
    vpid_case("SDR BT.709", 0x89, 0xC9, 0x80, 0x01, core::color_space::bt709, core::color_transfer::sdr, true);
    vpid_case("HDR BT.2100 HLG", 0x89, 0xD9, 0xA0, 0x01, core::color_space::bt2020, core::color_transfer::hlg, true);
    vpid_case("HDR BT.2100 PQ", 0x89, 0xE9, 0xA0, 0x01, core::color_space::bt2020, core::color_transfer::pq, true);

    std::printf("Interfaces predating the reassignment of those reserved bits\n");
    // An SD payload identifier. Byte 2 and 3 zeros here are RESERVED, not "SDR Rec.709", and
    // reading them as colour would tag every SD feed 709 -- silently defeating the sub-720
    // BT.601 convention the mixer depends on.
    vpid_case("SD ST 259 declares nothing", 0x81, 0x00, 0x00, 0x00, core::color_space::unknown,
              core::color_transfer::sdr, false);

    std::printf("Codes that are an absence rather than a value\n");
    {
        // Transfer 3h is "Unspecified" and colorimetry 3h is "Unknown". Neither may be
        // promoted to a real value; the caller's fallback has to survive.
        const uint8_t  word[4] = {0x85, 0xF5, 0xB0, 0x01};
        sdi_signalling out;
        parse_vpid(word, out);
        check(!out.transfer_specified && !out.colorimetry_specified,
              "unspecified/unknown leave the fallback alone (85 F5 B0 01)");
    }
    {
        // Colorimetry 1h is Reserved -- not a colour space either.
        const uint8_t  word[4] = {0x85, 0xC5, 0x90, 0x01};
        sdi_signalling out;
        parse_vpid(word, out);
        check(!out.colorimetry_specified && out.transfer_specified,
              "reserved colorimetry 1h is not a colour space (85 C5 90 01)");
    }

    std::printf("ST 2108-1 static metadata frames\n");
    {
        uint8_t udw[64];
        size_t  n = 0;
        // Static Metadata Type 1: mastering display colour volume, payloadType 137, size 24.
        udw[n++] = 0x00; // frame type
        udw[n++] = 0x1A; // frame length
        udw[n++] = 0x89; // SEI payloadType 137
        udw[n++] = 0x18; // SEI payloadSize 24
        std::memset(udw + n, 0, 16); // primaries and white point: not kept
        n += 16;
        // max 1000 cd/m2 and min 0.005 cd/m2, both in units of 0.0001 cd/m2.
        const uint32_t max_lum = 10000000u, min_lum = 50u;
        udw[n++] = (max_lum >> 24) & 0xFF; udw[n++] = (max_lum >> 16) & 0xFF;
        udw[n++] = (max_lum >> 8) & 0xFF;  udw[n++] = max_lum & 0xFF;
        udw[n++] = (min_lum >> 24) & 0xFF; udw[n++] = (min_lum >> 16) & 0xFF;
        udw[n++] = (min_lum >> 8) & 0xFF;  udw[n++] = min_lum & 0xFF;
        // Static Metadata Type 2: content light level, payloadType 144, size 4.
        udw[n++] = 0x01; udw[n++] = 0x06; udw[n++] = 0x90; udw[n++] = 0x04;
        udw[n++] = 0x03; udw[n++] = 0xE8; // MaxCLL 1000
        udw[n++] = 0x01; udw[n++] = 0x90; // MaxFALL 400

        sdi_signalling out;
        parse_st2108(udw, n, out);
        check(out.mastering_display_present, "mastering display frame found");
        check(out.max_display_mastering_luminance > 999.9 && out.max_display_mastering_luminance < 1000.1,
              "max display mastering luminance is 1000 cd/m2");
        check(out.min_display_mastering_luminance > 0.0049 && out.min_display_mastering_luminance < 0.0051,
              "min display mastering luminance is 0.005 cd/m2");
        check(out.content_light_level_present, "content light level frame found");
        check(out.max_content_light_level == 1000, "MaxCLL is 1000");
        check(out.max_frame_average_light_level == 400, "MaxFALL is 400");
    }
    {
        // A frame type we do not implement, sitting BEFORE one we do. Walking by the declared
        // length is what lets the second one still be found; a parser that gave up on an
        // unknown type would drop the metadata it does understand.
        uint8_t udw[32];
        size_t  n = 0;
        udw[n++] = 0x06; // Dynamic Metadata Type 5 (SL-HDR1) -- not parsed here
        udw[n++] = 0x04;
        udw[n++] = 0xAA; udw[n++] = 0xBB; udw[n++] = 0xCC; udw[n++] = 0xDD;
        udw[n++] = 0x01; udw[n++] = 0x06; udw[n++] = 0x90; udw[n++] = 0x04;
        udw[n++] = 0x00; udw[n++] = 0x64; // MaxCLL 100
        udw[n++] = 0x00; udw[n++] = 0x32; // MaxFALL 50
        sdi_signalling out;
        parse_st2108(udw, n, out);
        check(out.content_light_level_present && out.max_content_light_level == 100 &&
                  out.max_frame_average_light_level == 50,
              "an unknown frame type is skipped by its length, not fatal");
    }
    {
        // A truncated packet must not read past its buffer.
        uint8_t        udw[4] = {0x00, 0x1A, 0x89, 0x18}; // claims 26 bytes, supplies 2
        sdi_signalling out;
        parse_st2108(udw, sizeof(udw), out);
        check(!out.mastering_display_present, "a truncated frame is rejected, not parsed");
    }

    std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
