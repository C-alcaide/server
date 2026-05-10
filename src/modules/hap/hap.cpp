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
 * HAP is an open video codec designed for fast, GPU-accelerated playback
 * using S3TC/DXT compressed textures. https://hap.video/
 */

// hap.cpp
// CasparCG module entry point — registered by casparcg_add_module_project(INIT_FUNCTION "hap::init").
#include "hap.h"
#include "producer/hap_producer.h"

#include <common/log.h>

#include <GL/glew.h>

namespace caspar { namespace hap {

void init(const core::module_dependencies& dependencies)
{
    // Verify GL_EXT_texture_compression_s3tc is available at runtime.
    // This extension is universally supported on desktop GPUs.
    bool s3tc_ok = (GLEW_EXT_texture_compression_s3tc != 0);
    bool bptc_ok = (GLEW_ARB_texture_compression_bptc != 0);

    if (!s3tc_ok) {
        CASPAR_LOG(warning) << L"[hap] GL_EXT_texture_compression_s3tc NOT available"
                            << L" -- HAP/HAP Alpha/HAP Q playback will fail";
    }

    CASPAR_LOG(info) << L"[hap] GL extensions: S3TC=" << (s3tc_ok ? L"yes" : L"NO")
                     << L"  BPTC=" << (bptc_ok ? L"yes" : L"no (HAP R not available)");

    register_hap_producer(dependencies);

    CASPAR_LOG(info) << L"[hap] Module initialised";
}

}} // namespace caspar::hap
