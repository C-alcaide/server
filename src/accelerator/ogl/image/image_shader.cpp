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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */
#include "image_shader.h"

#include "../util/device.h"
#include "../util/shader.h"

#include <common/except.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

#pragma warning(push)
#pragma warning(disable: 4838 4309)
#include "ogl_image_fragment.h"
#include "ogl_image_vertex.h"
#pragma warning(pop)

namespace caspar { namespace accelerator { namespace ogl {

namespace {

/// How many compiled programs the cache keeps alive beyond their users.
///
/// The point is that flipping a channel between two transforms must not recompile on every
/// switch. Four covers a channel alternating between an input transform, a display
/// transform, both, and the base program; past that the least-recently-requested entry is
/// released and its program destroyed once no kernel still holds it.
constexpr size_t MAX_RETAINED_VARIANTS = 4;

struct cache_entry
{
    std::string           id;
    std::weak_ptr<shader> program;
    /// Keeps a program alive while it is among the most recently requested, so a variant
    /// whose last user went away is still there for the next request. Cleared on eviction;
    /// the weak_ptr above is what makes a still-in-use program findable either way.
    std::shared_ptr<shader> retained;
};

std::mutex               g_cache_mutex;
std::vector<cache_entry> g_cache; // most-recently-requested first; bounded by construction

/// Splice a variant's generated code into the base fragment source.
///
/// Returns the base source untouched for the base variant, so that path is byte-identical
/// to what the build embedded -- worth preserving exactly, because the 1 LSB conformance
/// gate is defined against it.
std::string build_fragment_source(const shader_variant& variant, const char* base)
{
    if (variant.is_base())
        return std::string(base);

    // The markers are comments, so a base program containing them is still valid GLSL and
    // the un-spliced shader compiles and runs. That is what makes the substitution safe to
    // get wrong loudly rather than quietly: a missing marker throws here, at configure time,
    // instead of producing a program that silently omits the transform.
    static constexpr const char* DECL_MARKER = "//__CASPAR_OCIO_DECLARATIONS__";
    static constexpr const char* CALL_MARKER = "//__CASPAR_OCIO_TRANSFORM__";

    std::string source(base);

    const auto decl_at = source.find(DECL_MARKER);
    const auto call_at = source.find(CALL_MARKER);
    if (decl_at == std::string::npos || call_at == std::string::npos) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                                   "shader.frag is missing an OCIO splice marker; cannot build variant '" +
                                   variant.id + "'"));
    }

    // Replace the call site first: inserting the (much longer) declarations first would
    // invalidate the offset found for the call.
    source.replace(call_at, std::strlen(CALL_MARKER), variant.transform_call);
    source.replace(decl_at, std::strlen(DECL_MARKER), variant.prologue);

    return source;
}

} // namespace

std::shared_ptr<shader> get_image_shader(const spl::shared_ptr<device>& ogl, const shader_variant& variant)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    // Promote on hit, so the retention window tracks what is actually being used.
    for (size_t i = 0; i < g_cache.size(); ++i) {
        if (g_cache[i].id != variant.id)
            continue;
        if (auto existing = g_cache[i].program.lock()) {
            g_cache[i].retained = existing;
            if (i != 0)
                std::rotate(g_cache.begin(), g_cache.begin() + i, g_cache.begin() + i + 1);
            return existing;
        }
        // Entry is stale -- the program died. Drop it and fall through to a rebuild.
        g_cache.erase(g_cache.begin() + i);
        break;
    }

    // The deleter is alive until the weak pointer is destroyed, so we have
    // to weakly reference ogl, to not keep it alive until atexit
    std::weak_ptr<device> weak_ogl = ogl;

    auto deleter = [weak_ogl](shader* p) {
        auto ogl = weak_ogl.lock();

        if (ogl) {
            ogl->dispatch_async([=] { delete p; });
        }
    };

    std::shared_ptr<shader> program(
        new shader(std::string(reinterpret_cast<const char*>(vertex_shader)),
                   build_fragment_source(variant, reinterpret_cast<const char*>(fragment_shader))),
        deleter);

    g_cache.insert(g_cache.begin(), cache_entry{variant.id, program, program});

    // Release anything past the retention window. Its weak_ptr goes too: a program still
    // held by a kernel stays valid, and the next request for it simply recompiles rather
    // than tracking an entry we have stopped retaining.
    if (g_cache.size() > MAX_RETAINED_VARIANTS)
        g_cache.resize(MAX_RETAINED_VARIANTS);

    return program;
}

size_t image_shader_cache_size()
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    return g_cache.size();
}

}}} // namespace caspar::accelerator::ogl
