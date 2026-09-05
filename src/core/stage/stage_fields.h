/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// The stage field registry: every property of a screen and of a previz camera declared ONCE,
// in exactly the form `transform_fields.h` declares an `image_transform` field.
//
// It exists for the reason that registry exists. A screen's fifteen properties were settable
// through twelve `PREVIZ SCREEN` subcommands and readable through none -- `SCREEN LIST` reports
// four of them, and there is no `SCREEN <name> INFO` -- so position, rotation, vertical arc,
// resolution, eye mode, design eye and ICVFX were WRITE-ONLY over AMCP. Declaring them here
// makes them describable, publishable and writable through the same machinery the mixer fields
// already use, and the four-of-fifteen problem goes away by construction rather than by adding
// eleven more query verbs.
//
// Two things it deliberately does NOT carry, both from `field_meta`:
//
//   compose  -- screens do not compose. There is one screen per name and two of them never
//               combine into one, so every row is `compose_t::none` and `guard_t::none`. The
//               columns are in the base type only because leaving them there is what let the
//               transform table's 177 rows stay untouched when the base was split out.
//   kf_names -- KEYFRAMES is bound to `image_transform` end to end
//               (`kf_field::get(const core::image_transform&)`), so a screen cannot currently
//               be animated. L57 of the design study says it should be. The rows say `nullptr`
//               rather than inventing names for an animation path that does not exist, which is
//               a gap named honestly rather than one papered over.

#include <core/frame/transform_fields.h>

#include "stage_model.h"

#include <string_view>
#include <vector>

namespace caspar { namespace core { namespace fields {

/// One property of a screen. Same row type as a mixer field, over a different struct.
using screen_field = typed_field<screen_meta>;

/// One property of a previz camera.
using camera_field = typed_field<previz_camera>;

/// Every screen property, in table order.
const std::vector<screen_field>& screen_fields();

/// Every camera property, in table order. Used for BOTH the production camera and the
/// viewport camera -- they are the same struct and differ only in what reads them.
const std::vector<camera_field>& camera_fields();

/// Validate both tables and log their sizes at startup, in the shape `run_compose_self_test`
/// established. Throws `programming_error` on a malformed row -- a bounding rule with nothing to
/// bound, or a missing description -- because both are mistakes in a table that is compiled in,
/// so failing at boot is failing at the only moment anyone can act on it.
void log_stage_fields();

const screen_field* find_screen_field(std::string_view path);
const camera_field* find_camera_field(std::string_view path);

}}} // namespace caspar::core::fields
