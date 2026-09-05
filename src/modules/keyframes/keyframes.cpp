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

#include "keyframes.h"

#include "keyframe_commands.h"
#include "keyframe_fields.h"

#include <common/log.h>
#include <common/utf.h>

namespace caspar { namespace keyframes {

void init(const core::module_dependencies& dependencies)
{
    CASPAR_LOG(info) << L"[keyframes] Initialising keyframe module";

    // The animatable field table is now DERIVED from `core::fields` rather than written
    // here, so the one failure that would be silent is a name changing: a saved timeline
    // animates by name, and a renamed or dropped field animates nothing while still
    // loading and playing. Checked at startup rather than trusted.
    {
        std::vector<std::string> missing;
        std::vector<std::string> added;
        const bool               ok = kf_verify_frozen_names(missing, added);

        CASPAR_LOG(info) << L"[keyframes] " << kf_all_fields().size()
                         << L" animatable components from the transform registry";

        for (const auto& n : added)
            CASPAR_LOG(info) << L"[keyframes] new animatable field: " << u16(n);

        if (!ok) {
            for (const auto& n : missing)
                CASPAR_LOG(error) << L"[keyframes] field no longer animatable: " << u16(n)
                                  << L" -- saved timelines referring to it will animate nothing";
            CASPAR_LOG(error) << L"[keyframes] " << missing.size()
                              << L" of the frozen names are missing from the registry";
        }
    }

    if (dependencies.command_repository)
        register_amcp_commands(dependencies.command_repository);
}

void uninit()
{
    CASPAR_LOG(info) << L"[keyframes] Shutting down keyframe module";
}

}} // namespace caspar::keyframes
