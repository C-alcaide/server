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

#include <common/log.h>

namespace caspar { namespace keyframes {

void init(const core::module_dependencies& dependencies)
{
    CASPAR_LOG(info) << L"[keyframes] Initialising keyframe module";

    if (dependencies.command_repository)
        register_amcp_commands(dependencies.command_repository);
}

void uninit()
{
    CASPAR_LOG(info) << L"[keyframes] Shutting down keyframe module";
}

}} // namespace caspar::keyframes
