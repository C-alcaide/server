/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#pragma once

#include <string>
#include <vector>

namespace caspar { namespace isf {

/// Load an image file into a tightly-packed, top-down RGBA8 buffer. Returns false on failure.
bool load_rgba_image(const std::wstring& path, std::vector<unsigned char>& rgba, int& width, int& height);

}} // namespace caspar::isf
