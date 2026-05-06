/*
 * Copyright (c) 2024 CasparCG contributors
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <cstdint>
#include <array>

namespace caspar { namespace cluster { namespace ptp {

// IEEE 1588-2008 PTP message types
enum class message_type : uint8_t
{
    sync           = 0x0,
    delay_req      = 0x1,
    follow_up      = 0x8,
    delay_resp     = 0x9,
    announce       = 0xB,
};

// PTP port identity (clock identity + port number)
struct port_identity
{
    std::array<uint8_t, 8> clock_identity = {};
    uint16_t               port_number    = 1;
};

// IEEE 1588 timestamp: 48-bit seconds + 32-bit nanoseconds
struct ptp_timestamp
{
    uint16_t seconds_msb = 0;   // upper 16 bits of seconds
    uint32_t seconds_lsb = 0;   // lower 32 bits of seconds
    uint32_t nanoseconds = 0;

    int64_t to_nanoseconds() const
    {
        int64_t sec = (static_cast<int64_t>(seconds_msb) << 32) | seconds_lsb;
        return sec * 1'000'000'000LL + nanoseconds;
    }

    static ptp_timestamp from_nanoseconds(int64_t ns)
    {
        ptp_timestamp ts;
        int64_t       sec  = ns / 1'000'000'000LL;
        int64_t       nsec = ns % 1'000'000'000LL;
        if (nsec < 0) {
            sec -= 1;
            nsec += 1'000'000'000LL;
        }
        ts.seconds_msb = static_cast<uint16_t>((sec >> 32) & 0xFFFF);
        ts.seconds_lsb = static_cast<uint32_t>(sec & 0xFFFFFFFF);
        ts.nanoseconds = static_cast<uint32_t>(nsec);
        return ts;
    }
};

// Common PTP header (34 bytes)
#pragma pack(push, 1)
struct ptp_header
{
    uint8_t       transport_specific_and_type; // upper nibble: transport, lower: message_type
    uint8_t       version_ptp;                 // 0x02 for PTPv2
    uint16_t      message_length;
    uint8_t       domain_number;
    uint8_t       reserved1;
    uint16_t      flags;
    int64_t       correction_field;
    uint32_t      reserved2;
    port_identity source_port_identity;
    uint16_t      sequence_id;
    uint8_t       control_field;
    int8_t        log_message_interval;

    message_type get_type() const { return static_cast<message_type>(transport_specific_and_type & 0x0F); }
    void set_type(message_type t) { transport_specific_and_type = static_cast<uint8_t>(t); }
};

// Sync message body (follows header): origin timestamp
struct sync_message
{
    ptp_header  header;
    ptp_timestamp origin_timestamp;
};

// Follow_Up message body
struct follow_up_message
{
    ptp_header  header;
    ptp_timestamp precise_origin_timestamp;
};

// Delay_Req message body
struct delay_req_message
{
    ptp_header  header;
    ptp_timestamp origin_timestamp;
};

// Delay_Resp message body
struct delay_resp_message
{
    ptp_header    header;
    ptp_timestamp receive_timestamp;
    port_identity requesting_port_identity;
};

// Announce message body (simplified - we only need clock quality for BMCA)
struct announce_message
{
    ptp_header  header;
    ptp_timestamp origin_timestamp;
    uint16_t    current_utc_offset;
    uint8_t     reserved;
    uint8_t     grandmaster_priority1;
    uint32_t    grandmaster_clock_quality; // packed: class(8) + accuracy(8) + variance(16)
    uint8_t     grandmaster_priority2;
    std::array<uint8_t, 8> grandmaster_identity;
    uint16_t    steps_removed;
    uint8_t     time_source;
};
#pragma pack(pop)

}}} // namespace caspar::cluster::ptp
