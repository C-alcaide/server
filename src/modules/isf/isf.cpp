/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf.h"
#include "isf_producer.h"

#include <common/log.h>

#include <core/producer/frame_producer_registry.h>

namespace caspar { namespace isf {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"ISF Producer", create_producer);
    CASPAR_LOG(info) << L"[isf] ISF shader producer registered ( [ISF] <shader-file> ).";
}

}} // namespace caspar::isf
