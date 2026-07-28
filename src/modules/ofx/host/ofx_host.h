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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace core { class texture; } }
namespace caspar { namespace accelerator { namespace ogl { class device; } } }

namespace caspar { namespace ofx {

/// The OFX effect contexts CasparCG can instantiate.
enum class effect_context
{
    filter,     ///< requires a Source clip (filters an input frame)
    generator,  ///< no input; produces an output frame
    transition  ///< two input clips (SourceFrom/SourceTo) blended by a Transition parameter
};

/// Which field the host is asking the plug-in to render (maps to kOfxImageEffectPropFieldToRender).
enum class field_kind
{
    none,  ///< progressive / unfielded
    both,  ///< both fields present in the frame
    lower,
    upper
};

/// Lightweight, OFX-header-free description of a discovered plug-in.
struct plugin_info
{
    std::string              identifier;   ///< unique OFX plug-in id (e.g. "com.vendor.effect")
    std::string              label;        ///< human-readable label
    std::string              grouping;     ///< menu grouping/path
    std::vector<std::string> contexts;     ///< supported contexts (Filter, General, Generator, ...)
    int                      version_major = 0;
    int                      version_minor = 0;
    std::string              bundle_path;  ///< path of the .ofx bundle the plug-in came from
};

/// A live, created OFX effect instance in the Filter context, sized to a fixed frame size.
/// Renders one frame at a time from a source RGBA buffer into an output RGBA buffer (both
/// 8-bit, bottom-up, as produced by the image bridge).
class effect
{
  public:
    ~effect();

    effect(const effect&)            = delete;
    effect& operator=(const effect&) = delete;

    bool valid() const;

    /// Render a single frame. src_rgba may be null (source disconnected). dst_rgba must be a
    /// width*height*4 byte buffer. Returns true on success.
    bool render(const std::uint8_t* src_rgba,
                std::uint8_t*        dst_rgba,
                int                  width,
                int                  height,
                double               time,
                field_kind           field = field_kind::both);

    /// Zero-copy OpenGL render. Runs the plug-in's OpenGL render on the mixer's GL device thread
    /// against device-owned textures and returns the result as a mixer texture — NO CPU readback
    /// and NO re-upload of the output. src_rgba is bottom-up 8-bit RGBA (width*height*4), the same
    /// orientation as the CPU/self-contained GL path; the output is Y-flipped on the GPU back to the
    /// mixer's top-down convention. Returns nullptr on failure (caller should fall back to render()).
    std::shared_ptr<core::texture> render_gl_zerocopy(accelerator::ogl::device& device,
                                                      const std::uint8_t*       src_rgba,
                                                      int                       width,
                                                      int                       height,
                                                      double                    time,
                                                      field_kind                field = field_kind::both);

    /// True if this effect negotiated an OpenGL-capable render (plug-in advertises GL support and
    /// the host enabled it). Lets the producer decide whether the zero-copy GL path is viable.
    bool opengl_capable() const;

    /// True until a zero-copy render detects the plug-in relies on a non-core (compatibility) GL
    /// profile that cannot run on the mixer's core-profile device. Once false, the producer routes
    /// the plug-in through the self-contained compatibility render path (readback) instead.
    bool zerocopy_gl_supported() const;

    /// True if this effect negotiated a CUDA-capable render (plug-in advertises CUDA support and the
    /// host enabled it). Lets the producer decide whether the CUDA zero-copy path is viable.
    bool cuda_capable() const;

    /// CUDA render into the plug-in's device output buffer, returning that CUDA device pointer (or
    /// nullptr on failure). src_rgba is bottom-up 8-bit RGBA (width*height*4); it is uploaded to a
    /// device buffer, the plug-in renders in CUDA mode, and NO host readback is performed — the
    /// caller copies the returned device buffer straight into a GPU texture (device-to-device).
    void* render_cuda(const std::uint8_t* src_rgba, int width, int height, double time, field_kind field = field_kind::both);

    /// Ask the plug-in whether the given frame is an identity (no-op) — lets the host skip the
    /// render and pass the source through unchanged. Returns false if not identity / on error.
    bool is_identity(int width, int height, double time);

    /// The output premultiplication the plug-in declared via getClipPreferences:
    /// true  = premultiplied (default / mixer-native),
    /// false = unpremultiplied (host must mark the output frame straight-alpha).
    bool output_premultiplied() const;

    /// The negotiated OFX working pixel depth in bytes-per-channel: 1 (8-bit), 2 (16-bit) or
    /// 4 (float32). Determined from the plug-in's supported depths via getClipPreferences.
    int working_bytes() const;

    /// A controllable parameter of the effect.
    struct param
    {
        std::string name;
        std::string type;  ///< OFX param type string (kOfxParamType*)
        std::string label; ///< human-readable label
        int         dimension = 1;     ///< number of scalar components (1/2/3/4)
        bool        has_range = false; ///< true if min/max are meaningful (numeric params)
        double      min       = 0.0;
        double      max       = 0.0;
        double      def       = 0.0;   ///< default value (component 0; for choice = default index)
        std::vector<std::string> choices; ///< option labels for choice params
    };

    /// Enumerate the effect's parameters (in declaration order).
    std::vector<param> params() const;

    /// Set a parameter by name from up to four scalar values (extra values ignored, missing
    /// values treated as 0). Returns true if the parameter exists and was set. Triggers the
    /// plug-in's instanceChanged action.
    bool set_param(const std::string& name, const std::vector<double>& values, double time);

    /// Set a string parameter by name. Returns true if the parameter exists, is a string type,
    /// and was set. Triggers the plug-in's instanceChanged action.
    bool set_param_string(const std::string& name, const std::string& value, double time);

    /// Render a transition frame. src_from/src_to are bottom-up 8-bit RGBA (width*height*4); the
    /// plug-in blends them by the given transition value (0..1, set on the mandatory "Transition"
    /// parameter). dst_rgba receives the width*height*4 result. Returns true on success.
    bool render_transition(const std::uint8_t* src_from,
                           const std::uint8_t* src_to,
                           std::uint8_t*       dst_rgba,
                           int                 width,
                           int                 height,
                           double              time,
                           double              transition,
                           field_kind          field = field_kind::both);

  private:
    friend class host;
    effect();

    struct impl;
    std::unique_ptr<impl> impl_;
};

/// Thin C++ wrapper around the OpenFX HostSupport image-effect host + plug-in cache.
///
/// Owns the OFX host object and the image-effect plug-in cache. Scans the OFX plug-in
/// search path and exposes the discovered plug-ins without leaking any OFX headers.
class host
{
  public:
    host();
    ~host();

    host(const host&)            = delete;
    host& operator=(const host&) = delete;

    /// Add a directory to the front of the OFX plug-in search path. Must be called before scan().
    void add_search_path(const std::string& dir);

    /// Scan the search path (plus the standard OFX locations and the OFX_PLUGIN_PATH env var)
    /// and load/describe the discovered image-effect plug-ins.
    void scan();

    /// The plug-ins discovered by the most recent scan().
    const std::vector<plugin_info>& plugins() const;

    /// The directories that will be / were searched.
    std::vector<std::string> search_paths() const;

    /// Create an effect instance for the plug-in with the given id in the given context, sized
    /// to width x height at the given frame rate. bytes_per_channel is 1 (8-bit) or 2 (16-bit).
    /// Returns nullptr if the plug-in is not found or cannot be instantiated in that context.
    std::unique_ptr<effect> create_effect(const std::string& identifier,
                                          effect_context     context,
                                          int                width,
                                          int                height,
                                          double             frame_rate,
                                          int                bytes_per_channel);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// Process-wide shared OFX host. Created and scanned once; reused by the module init logging
/// and by all ofx producers (the underlying OFX plug-in cache is a singleton).
host& global_host();

/// Configuration hooks (called from module init after reading casparcg.config), with env-var
/// override still honored: CASPARCG_OFX_ENABLE_GL / CASPARCG_OFX_ENABLE_CUDA.
void configure_opengl(bool enabled);
void configure_cuda(bool enabled);
/// Add a plug-in id to the blocklist so it is never instantiated (config-driven or automatic).
void blocklist_plugin(const std::string& identifier);

}} // namespace caspar::ofx
