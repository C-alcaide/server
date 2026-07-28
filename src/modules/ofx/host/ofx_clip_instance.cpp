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

#include "ofx_clip_instance.h"
#include "ofx_effect_instance.h"

namespace caspar { namespace ofx {

ofx_image::ofx_image(OFX::Host::ImageEffect::ClipInstance& clip,
                     std::uint8_t*                         data,
                     int                                   width,
                     int                                   height,
                     int                                   row_bytes)
    : OFX::Host::ImageEffect::Image(clip) // sets depth/components/premult/PAR from the clip
{
    // Full-frame, unit render scale.
    setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 0);
    setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 1);

    setPointerProperty(kOfxImagePropData, data);

    setIntProperty(kOfxImagePropBounds, 0, 0);
    setIntProperty(kOfxImagePropBounds, 0, 1);
    setIntProperty(kOfxImagePropBounds, width, 2);
    setIntProperty(kOfxImagePropBounds, height, 3);

    setIntProperty(kOfxImagePropRegionOfDefinition, 0, 0);
    setIntProperty(kOfxImagePropRegionOfDefinition, 0, 1);
    setIntProperty(kOfxImagePropRegionOfDefinition, width, 2);
    setIntProperty(kOfxImagePropRegionOfDefinition, height, 3);

    setIntProperty(kOfxImagePropRowBytes, row_bytes);
}

ofx_image::~ofx_image() = default; // does not own the pixel buffer

ofx_clip_instance::ofx_clip_instance(ofx_effect_instance* effect, OFX::Host::ImageEffect::ClipDescriptor* desc)
    : OFX::Host::ImageEffect::ClipInstance(effect, *desc)
    , effect_(effect)
    , name_(desc->getName())
{
}

ofx_clip_instance::~ofx_clip_instance() = default;

const std::string& ofx_clip_instance::getUnmappedBitDepth() const
{
    static const std::string byte(kOfxBitDepthByte);
    static const std::string sht(kOfxBitDepthShort);
    return effect_->ctx().bytes_per_channel == 2 ? sht : byte;
}

const std::string& ofx_clip_instance::getUnmappedComponents() const
{
    static const std::string v(kOfxImageComponentRGBA);
    return v;
}

const std::string& ofx_clip_instance::getPremult() const
{
    // The host normalises images to the premultiplied convention (mixer frames are premultiplied;
    // straight-alpha sources are premultiplied on input by the producer).
    static const std::string v(kOfxImagePreMultiplied);
    return v;
}

double ofx_clip_instance::getAspectRatio() const { return 1.0; }

double ofx_clip_instance::getFrameRate() const { return effect_->getFrameRate(); }

void ofx_clip_instance::getFrameRange(double& startFrame, double& endFrame) const
{
    startFrame = 0;
    endFrame   = 0;
}

const std::string& ofx_clip_instance::getFieldOrder() const
{
    static const std::string v(kOfxImageFieldNone);
    return v;
}

bool ofx_clip_instance::getConnected() const { return true; }

double ofx_clip_instance::getUnmappedFrameRate() const { return effect_->getFrameRate(); }

void ofx_clip_instance::getUnmappedFrameRange(double& start, double& end) const
{
    start = 0;
    end   = 0;
}

bool ofx_clip_instance::getContinuousSamples() const { return false; }

OfxRectD ofx_clip_instance::getRegionOfDefinition(OfxTime /*time*/) const
{
    const auto& c = effect_->ctx();
    OfxRectD    v;
    v.x1 = 0;
    v.y1 = 0;
    v.x2 = c.width;
    v.y2 = c.height;
    return v;
}

OFX::Host::ImageEffect::Image* ofx_clip_instance::getImage(OfxTime /*time*/, const OfxRectD* /*optionalBounds*/)
{
    auto& c = effect_->ctx();

    // In CUDA mode the image data pointer is a CUDA device pointer (kOfxImagePropData).
    if (c.cuda) {
        void* dev = (name_ == "Output")   ? c.output_dev
                    : (name_ == "SourceTo") ? c.source_to_dev
                                            : c.source_dev;
        if (dev == nullptr)
            return nullptr;
        return new ofx_image(*this, static_cast<std::uint8_t*>(dev), c.width, c.height, c.row_bytes);
    }

    std::uint8_t* data = (name_ == "Output")   ? c.output_rgba
                         : (name_ == "SourceTo") ? const_cast<std::uint8_t*>(c.source_to_rgba)
                                                 : const_cast<std::uint8_t*>(c.source_rgba);
    if (data == nullptr)
        return nullptr;

    return new ofx_image(*this, data, c.width, c.height, c.row_bytes);
}

#ifdef OFX_SUPPORTS_OPENGLRENDER
namespace {
// GL_TEXTURE_2D without pulling a GL header into this translation unit.
constexpr int kGLTexture2D = 0x0DE1;

/// An OFX texture wrapping a GL texture id owned by the effect's gl_backend.
class ofx_texture : public OFX::Host::ImageEffect::Texture
{
  public:
    ofx_texture(OFX::Host::ImageEffect::ClipInstance& clip, unsigned int tex, int width, int height)
        : OFX::Host::ImageEffect::Texture(clip)
    {
        setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 0);
        setDoubleProperty(kOfxImageEffectPropRenderScale, 1.0, 1);

        setIntProperty(kOfxImageEffectPropOpenGLTextureIndex, static_cast<int>(tex));
        setIntProperty(kOfxImageEffectPropOpenGLTextureTarget, kGLTexture2D);

        setIntProperty(kOfxImagePropBounds, 0, 0);
        setIntProperty(kOfxImagePropBounds, 0, 1);
        setIntProperty(kOfxImagePropBounds, width, 2);
        setIntProperty(kOfxImagePropBounds, height, 3);

        setIntProperty(kOfxImagePropRegionOfDefinition, 0, 0);
        setIntProperty(kOfxImagePropRegionOfDefinition, 0, 1);
        setIntProperty(kOfxImagePropRegionOfDefinition, width, 2);
        setIntProperty(kOfxImagePropRegionOfDefinition, height, 3);

        setIntProperty(kOfxImagePropRowBytes, width * 4);
    }
};
} // namespace

OFX::Host::ImageEffect::Texture*
ofx_clip_instance::loadTexture(OfxTime /*time*/, const char* /*format*/, const OfxRectD* /*optionalBounds*/)
{
    const auto&        c   = effect_->ctx();
    const unsigned int tex = (name_ == "Output") ? c.output_tex : c.source_tex;
    if (!c.gl || tex == 0)
        return nullptr;

    return new ofx_texture(*this, tex, c.width, c.height);
}
#endif

}} // namespace caspar::ofx
