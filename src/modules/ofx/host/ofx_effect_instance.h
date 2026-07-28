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

#include "ofx_cuda_render.h"
#include "ofx_gl_render.h"
#include "ofx_includes.h"

#include <cstdint>
#include <memory>
#include <string>

namespace caspar { namespace ofx {

/// Per-frame render state shared with the clip instances. The producer fills this in
/// before each render() call; the Source/Output clip instances read from it when the
/// plug-in fetches images.
struct render_context
{
    const std::uint8_t* source_rgba = nullptr; ///< bottom-up RGBA source pixels (may be null)
    const std::uint8_t* source_to_rgba = nullptr; ///< second input for the transition context (SourceTo)
    std::uint8_t*       output_rgba = nullptr;  ///< bottom-up RGBA output pixels
    int                 width       = 0;
    int                 height      = 0;
    int                 row_bytes   = 0;
    int                 bytes_per_channel = 1; ///< source frame depth: 1 = 8-bit, 2 = 16-bit
    int                 working_bytes     = 1; ///< OFX working depth: 1=8-bit, 2=16-bit, 4=float32
    double              time        = 0.0;
    bool                gl          = false;   ///< true while an OpenGL-mode render is in progress
    unsigned int        source_tex  = 0;       ///< GL texture id for the source clip (OpenGL mode)
    unsigned int        output_tex  = 0;       ///< GL texture id for the output clip (OpenGL mode)
    bool                cuda        = false;   ///< true while a CUDA-mode render is in progress
    void*               source_dev  = nullptr; ///< CUDA device ptr for the source clip (CUDA mode)
    void*               source_to_dev = nullptr; ///< CUDA device ptr for the SourceTo clip (transition, CUDA mode)
    void*               output_dev  = nullptr; ///< CUDA device ptr for the output clip (CUDA mode)
    // On-device convert path (CUDA zero-copy): when raw_source is set the host uploads it once and
    // does the swizzle/flip/premultiply on the device via NPP (no CPU passes), then mirrors the
    // plug-in output back to top-down. output_dev_topdown is what the producer copies to the VK array.
    const std::uint8_t* raw_source     = nullptr; ///< raw top-down source (BGRA or RGBA); null = legacy host-RGBA path
    int                 raw_src_stride = 0;       ///< row bytes of raw_source
    bool                raw_is_bgra    = false;   ///< raw_source channel order is BGRA (else RGBA)
    bool                raw_straight   = false;   ///< raw_source has straight (un-premultiplied) alpha
    void*               output_dev_topdown = nullptr; ///< top-down RGBA output, ready for cudaMemcpy2DToArray
    bool                external_gl = false;   ///< GL textures are owned externally (zero-copy on the mixer device); skip the offscreen context + readback
};

/// CasparCG's concrete OFX image-effect instance. Sized to the current video frame; all
/// clips are 8-bit RGBA. Instance creation of parameters uses default values (Phase 2).
class ofx_effect_instance : public OFX::Host::ImageEffect::Instance
{
  public:
    ofx_effect_instance(OFX::Host::ImageEffect::ImageEffectPlugin* plugin,
                        OFX::Host::ImageEffect::Descriptor&        desc,
                        const std::string&                         context,
                        int                                        width,
                        int                                        height,
                        double                                     frame_rate,
                        int                                        bytes_per_channel);

    render_context&       ctx() { return ctx_; }
    const render_context& ctx() const { return ctx_; }

    /// Enable the OpenGL render backend for this effect (host must also advertise support).
    void enable_opengl(bool on) { gl_enabled_ = on; }
    bool opengl_enabled() const { return gl_enabled_ && gl_ != nullptr; }

    /// True if the OpenGL backend was requested for this effect (plug-in advertised GL support
    /// and the host enabled it), regardless of whether the offscreen backend has been created.
    /// Used to decide whether the zero-copy GL path is viable.
    bool opengl_requested() const { return gl_enabled_; }

    /// Enable the CUDA render backend for this effect (host must also advertise support).
    void enable_cuda(bool on) { cuda_enabled_ = on; }

    /// True if the CUDA backend was requested for this effect (plug-in advertised CUDA support and
    /// the host enabled it). Used to decide whether the CUDA zero-copy path is viable.
    bool cuda_requested() const { return cuda_enabled_; }

    // --- OFX::Host::ImageEffect::Instance ---
    OFX::Host::ImageEffect::ClipInstance* newClipInstance(OFX::Host::ImageEffect::Instance*       plugin,
                                                          OFX::Host::ImageEffect::ClipDescriptor* descriptor,
                                                          int index) override;

    /// Overridden to inject kOfxImageEffectPropOpenGLEnabled and drive the GL upload/readback
    /// when the OpenGL backend is active; otherwise delegates to the base (CPU) implementation.
    OfxStatus renderAction(OfxTime            time,
                           const std::string& field,
                           const OfxRectI&    renderRoI,
                           OfxPointD          renderScale,
                           bool               sequentialRender,
                           bool               interactiveRender,
                           bool               draftRender) override;

    const std::string& getDefaultOutputFielding() const override;

    OfxStatus vmessage(const char* type, const char* id, const char* format, va_list args) override;
    OfxStatus setPersistentMessage(const char* type, const char* id, const char* format, va_list args) override;
    OfxStatus clearPersistentMessage() override;

    void   getProjectSize(double& x, double& y) const override;
    void   getProjectOffset(double& x, double& y) const override;
    void   getProjectExtent(double& x, double& y) const override;
    double getProjectPixelAspectRatio() const override;
    double getEffectDuration() const override;
    double getFrameRate() const override;
    double getFrameRecursive() const override;
    void   getRenderScaleRecursive(double& x, double& y) const override;

    // --- Param::SetInstance ---
    OFX::Host::Param::Instance* newParam(const std::string& name, OFX::Host::Param::Descriptor& descriptor) override;
    OfxStatus                   editBegin(const std::string& name) override;
    OfxStatus                   editEnd() override;

    // --- Progress::ProgressI ---
    void progressStart(const std::string& message, const std::string& messageid) override;
    void progressEnd() override;
    bool progressUpdate(double t) override;

    // --- TimeLine::TimeLineI ---
    double timeLineGetTime() override;
    void   timeLineGotoTime(double t) override;
    void   timeLineGetBounds(double& t1, double& t2) override;

  private:
    render_context ctx_;
    int            width_      = 0;
    int            height_     = 0;
    double         frame_rate_ = 25.0;

    // OpenGL backend (created lazily on first GL render).
    bool                        gl_enabled_  = false;
    bool                        gl_attached_ = false;
    std::unique_ptr<gl_backend> gl_;

    // CUDA backend (created lazily on first CUDA render).
    bool                          cuda_enabled_ = false;
    std::unique_ptr<cuda_backend> cuda_;
};

}} // namespace caspar::ofx
