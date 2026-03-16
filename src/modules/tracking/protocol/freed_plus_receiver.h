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
 * FreeD+ (Stype extended FreeD) protocol implemented from the
 * publicly available Stype specification document.
 */

#pragma once

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// FreeD+ / Stype extended-precision UDP receiver.
///
/// The Stype FreeD+ protocol augments the standard FreeD D1 packet with a
/// 12-byte extension appended after byte 26 (before the checksum), giving
/// higher-precision angle and position values.
///
/// Extension layout (bytes 27-38, big-endian):
///   Bytes 27-30 : High-res pan   (32-bit signed, 1/32768 deg/unit × 2^8 sub-unit)
///   Bytes 31-34 : High-res tilt
///   Bytes 35-38 : High-res roll
/// Total packet: 41 bytes. Byte 40 carries the XOR checksum.
///
/// If the senders only transmit standard 29-byte FreeD packets, use
/// freed_receiver instead.
class freed_plus_receiver : public receiver_base
{
  public:
    explicit freed_plus_receiver(uint16_t port);
    ~freed_plus_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
