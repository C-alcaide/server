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

// Two declarations and nothing else, and that is the whole reason this file exists.
//
// The natural home is each backend's `util/transforms.h`, and the shell cannot include
// both: `ogl/util/matrix.h` and `vulkan/util/matrix.h` each define the same operators in
// `boost::numeric::ublas`, so one translation unit including both fails with C2995 on every
// one of them. Neither copy is wrong on its own -- they are per-backend by history -- and
// merging them is a change to the geometry path that has nothing to do with this check.

namespace caspar { namespace accelerator { namespace ogl {
/// Compare the OpenGL backend's hand-written layer composition against
/// `core::fields::compose_colour`, and log the result. See `ogl/util/transforms.h`.
void run_compose_self_test();
}}} // namespace caspar::accelerator::ogl

namespace caspar { namespace accelerator { namespace vulkan {
/// The same for the Vulkan backend.
void run_compose_self_test();
}}} // namespace caspar::accelerator::vulkan
