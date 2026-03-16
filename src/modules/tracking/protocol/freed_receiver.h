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
 * FreeD protocol specification by Mark Roberts Motion Control Ltd.
 * Implemented from the publicly available specification document.
 */

#pragma once

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// FreeD D-type (D1) UDP receiver.
///
/// Listens on a UDP port and decodes packets adhering to the FreeD D1 format
/// (29-byte, big-endian, XOR checksum). Decoded data is forwarded to
/// tracker_registry::on_data().
///
/// Multiple senders with different camera IDs can share one port. The registry
/// then dispatches data to each binding that matches the incoming camera_id.
class freed_receiver : public receiver_base
{
  public:
    explicit freed_receiver(uint16_t port);
    ~freed_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
