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

// Central, order-sensitive include of the OpenFX C API + HostSupport C++ headers.
// The HostSupport headers are third-party BSD code that is not warning-clean under the
// project's /W4 /WX, so the whole region is compiled with warnings disabled. The include
// order mirrors HostSupport's own translation units (the API header depends on the
// plugin-cache / API-cache / host declarations being visible first).

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <ofxCore.h>
#include <ofxImageEffect.h>
#include <ofxPixels.h>
#include <ofxGPURender.h>

#include <ofxhBinary.h>
#include <ofxhPropertySuite.h>
#include <ofxhClip.h>
#include <ofxhParam.h>
#include <ofxhMemory.h>
#include <ofxhImageEffect.h>
#include <ofxhPluginAPICache.h>
#include <ofxhPluginCache.h>
#include <ofxhHost.h>
#include <ofxhImageEffectAPI.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
