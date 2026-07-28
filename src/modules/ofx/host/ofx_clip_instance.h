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

#include "ofx_includes.h"

#include <cstdint>
#include <string>

namespace caspar { namespace ofx {

class ofx_effect_instance;

/// An OFX image that wraps an externally-owned 8-bit RGBA buffer (no ownership). Used both
/// for the Source clip (points at the bridged input) and the Output clip (points at the
/// buffer the plug-in renders into).
class ofx_image : public OFX::Host::ImageEffect::Image
{
  public:
    ofx_image(OFX::Host::ImageEffect::ClipInstance& clip,
              std::uint8_t*                         data,
              int                                   width,
              int                                   height,
              int                                   row_bytes);
    ~ofx_image() override;
};

/// A Source/Output clip whose images are the current frame's bridged RGBA buffers.
class ofx_clip_instance : public OFX::Host::ImageEffect::ClipInstance
{
  public:
    ofx_clip_instance(ofx_effect_instance* effect, OFX::Host::ImageEffect::ClipDescriptor* desc);
    ~ofx_clip_instance() override;

    const std::string& getUnmappedBitDepth() const override;
    const std::string& getUnmappedComponents() const override;
    const std::string& getPremult() const override;
    double             getAspectRatio() const override;
    double             getFrameRate() const override;
    void               getFrameRange(double& startFrame, double& endFrame) const override;
    const std::string& getFieldOrder() const override;
    bool               getConnected() const override;
    double             getUnmappedFrameRate() const override;
    void               getUnmappedFrameRange(double& start, double& end) const override;
    bool               getContinuousSamples() const override;

    OfxRectD                       getRegionOfDefinition(OfxTime time) const override;
    OFX::Host::ImageEffect::Image* getImage(OfxTime time, const OfxRectD* optionalBounds) override;

#ifdef OFX_SUPPORTS_OPENGLRENDER
    OFX::Host::ImageEffect::Texture*
    loadTexture(OfxTime time, const char* format, const OfxRectD* optionalBounds) override;
#endif

  private:
    ofx_effect_instance* effect_;
    std::string          name_;
};

}} // namespace caspar::ofx
