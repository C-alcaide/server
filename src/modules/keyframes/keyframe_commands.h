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

#include <protocol/amcp/amcp_command_repository_wrapper.h>
#include <memory>

namespace caspar { namespace keyframes {

void register_amcp_commands(
    const std::shared_ptr<protocol::amcp::amcp_command_repository_wrapper>& repo);

}} // namespace caspar::keyframes
