/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 */

#pragma once

#ifdef _DEBUG
#ifdef WIN32
#include <crtdbg.h>
#endif
#endif

#define NOMINMAX

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <common/except.h>
#include <common/log.h>
#include <common/memory.h>

#include <assert.h>
