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
 * Open Sound Control (OSC) 1.0 specification by CNMAT/UC Berkeley.
 * Implemented from the publicly available specification.
 */

#pragma once

#include "receiver_base.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace tracking {

/// OSC 1.0 UDP input receiver.
///
/// Listens for OSC bundles/messages using the following address schema:
///
///   /camera/{id}/pan    f  (degrees)
///   /camera/{id}/tilt   f  (degrees)
///   /camera/{id}/roll   f  (degrees)
///   /camera/{id}/zoom   f  (raw 0-65535 or fractional 0.0-1.0 * 65535)
///   /camera/{id}/focus  f  (same scale)
///   /camera/{id}/x      f  (millimetres)
///   /camera/{id}/y      f  (millimetres)
///   /camera/{id}/z      f  (millimetres)
///
/// All angle inputs are expected in degrees and converted to radians internally.
/// Zoom/focus values are treated as uint16 raw (0-65535); float inputs are
/// multiplied by 65535 and clamped if the value is in [0,1].
///
/// The OSC bundle timetag is ignored — timestamp is set on packet receipt.
class osc_receiver : public receiver_base
{
  public:
    explicit osc_receiver(uint16_t port);
    ~osc_receiver() override;

    void        start() override;
    void        stop() override;
    std::string info() const override;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::tracking
