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

#include "ofx_effect_instance.h"
#include "ofx_clip_instance.h"
#include "ofx_param_instance.h"

#include <common/log.h>
#include <common/utf.h>

#include <GL/glew.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace caspar { namespace ofx {

ofx_effect_instance::ofx_effect_instance(OFX::Host::ImageEffect::ImageEffectPlugin* plugin,
                                         OFX::Host::ImageEffect::Descriptor&        desc,
                                         const std::string&                         context,
                                         int                                        width,
                                         int                                        height,
                                         double                                     frame_rate,
                                         int                                        bytes_per_channel)
    : OFX::Host::ImageEffect::Instance(plugin, desc, context, false)
    , width_(width)
    , height_(height)
    , frame_rate_(frame_rate)
{
    ctx_.width             = width;
    ctx_.height            = height;
    ctx_.bytes_per_channel = bytes_per_channel;
    ctx_.row_bytes         = width * 4 * bytes_per_channel;
}

OFX::Host::ImageEffect::ClipInstance*
ofx_effect_instance::newClipInstance(OFX::Host::ImageEffect::Instance* /*plugin*/,
                                     OFX::Host::ImageEffect::ClipDescriptor* descriptor,
                                     int /*index*/)
{
    return new ofx_clip_instance(this, descriptor);
}

OfxStatus ofx_effect_instance::renderAction(OfxTime            time,
                                            const std::string& field,
                                            const OfxRectI&    renderRoI,
                                            OfxPointD          renderScale,
                                            bool               sequentialRender,
                                            bool               interactiveRender,
                                            bool               draftRender)
{
    // Render-mode negotiation: CUDA > OpenGL > CPU.
    if (cuda_enabled_) {
        try {
            if (!cuda_)
                cuda_ = std::make_unique<cuda_backend>();

            ctx_.source_dev = ctx_.source_rgba ? cuda_->upload_source(ctx_.source_rgba, ctx_.width, ctx_.height) : nullptr;
            ctx_.output_dev = cuda_->ensure_output(ctx_.width, ctx_.height);
            ctx_.cuda       = true;

            static const OFX::Host::Property::PropSpec in_spec[] = {
                {kOfxPropTime, OFX::Host::Property::eDouble, 1, true, "0"},
                {kOfxImageEffectPropFieldToRender, OFX::Host::Property::eString, 1, true, ""},
                {kOfxImageEffectPropRenderWindow, OFX::Host::Property::eInt, 4, true, "0"},
                {kOfxImageEffectPropRenderScale, OFX::Host::Property::eDouble, 2, true, "0"},
                {kOfxImageEffectPropSequentialRenderStatus, OFX::Host::Property::eInt, 1, true, "0"},
                {kOfxImageEffectPropInteractiveRenderStatus, OFX::Host::Property::eInt, 1, true, "0"},
                {kOfxImageEffectPropRenderQualityDraft, OFX::Host::Property::eInt, 1, true, "0"},
                {kOfxImageEffectPropCudaEnabled, OFX::Host::Property::eInt, 1, true, "1"},
                OFX::Host::Property::propSpecEnd};

            OFX::Host::Property::Set inArgs(in_spec);
            inArgs.setStringProperty(kOfxImageEffectPropFieldToRender, field);
            inArgs.setDoubleProperty(kOfxPropTime, time);
            inArgs.setIntPropertyN(kOfxImageEffectPropRenderWindow, &renderRoI.x1, 4);
            inArgs.setDoublePropertyN(kOfxImageEffectPropRenderScale, &renderScale.x, 2);
            inArgs.setIntProperty(kOfxImageEffectPropSequentialRenderStatus, sequentialRender);
            inArgs.setIntProperty(kOfxImageEffectPropInteractiveRenderStatus, interactiveRender);
            inArgs.setIntProperty(kOfxImageEffectPropRenderQualityDraft, draftRender);
            inArgs.setIntProperty(kOfxImageEffectPropCudaEnabled, 1);

            const OfxStatus st = mainEntry(kOfxImageEffectActionRender, getHandle(), &inArgs, nullptr);

            if (st == kOfxStatOK || st == kOfxStatReplyDefault) {
                if (ctx_.output_rgba)
                    cuda_->readback_output(ctx_.output_rgba, ctx_.width, ctx_.height);
                static bool logged_ok = false;
                if (!logged_ok) {
                    logged_ok = true;
                    CASPAR_LOG(info) << L"[ofx] CUDA render path active.";
                }
            }
            ctx_.cuda = false;
            return st;
        } catch (...) {
            ctx_.cuda = false;
            static bool logged_ex = false;
            if (!logged_ex) {
                logged_ex = true;
                CASPAR_LOG(warning) << L"[ofx] CUDA render unavailable/failed; falling back.";
            }
            // fall through to CPU below
        }
    }

    // CPU path: delegate to the base implementation.
    if (!gl_enabled_) {
        return OFX::Host::ImageEffect::Instance::renderAction(
            time, field, renderRoI, renderScale, sequentialRender, interactiveRender, draftRender);
    }

    try {
        if (ctx_.external_gl) {
            // Zero-copy path: the caller (producer) has set up ctx_.source_tex / ctx_.output_tex as
            // textures owned by the mixer's GL device, is running us on that device's GL thread
            // (context already current), and will keep the output texture as the frame's texture.
            // No offscreen context, no upload, no readback.
            ctx_.gl = true;
            if (!gl_attached_) {
                contextAttachedAction();
                gl_attached_ = true;
            }
        } else {
            if (!gl_)
                gl_ = std::make_unique<gl_backend>();

            gl_->make_current();

            if (!gl_attached_) {
                contextAttachedAction();
                gl_attached_ = true;
            }

            // Upload source + allocate output textures the clips will hand to the plug-in.
            ctx_.source_tex = ctx_.source_rgba ? gl_->upload_source(ctx_.source_rgba, ctx_.width, ctx_.height) : 0;
            ctx_.output_tex = gl_->ensure_output(ctx_.width, ctx_.height);
            ctx_.gl         = true;
        }

        // Build the render inArgs, adding kOfxImageEffectPropOpenGLEnabled which HostSupport's
        // own renderAction does not set.
        static const OFX::Host::Property::PropSpec in_spec[] = {
            {kOfxPropTime, OFX::Host::Property::eDouble, 1, true, "0"},
            {kOfxImageEffectPropFieldToRender, OFX::Host::Property::eString, 1, true, ""},
            {kOfxImageEffectPropRenderWindow, OFX::Host::Property::eInt, 4, true, "0"},
            {kOfxImageEffectPropRenderScale, OFX::Host::Property::eDouble, 2, true, "0"},
            {kOfxImageEffectPropSequentialRenderStatus, OFX::Host::Property::eInt, 1, true, "0"},
            {kOfxImageEffectPropInteractiveRenderStatus, OFX::Host::Property::eInt, 1, true, "0"},
            {kOfxImageEffectPropRenderQualityDraft, OFX::Host::Property::eInt, 1, true, "0"},
            {kOfxImageEffectPropOpenGLEnabled, OFX::Host::Property::eInt, 1, true, "1"},
            OFX::Host::Property::propSpecEnd};

        OFX::Host::Property::Set inArgs(in_spec);
        inArgs.setStringProperty(kOfxImageEffectPropFieldToRender, field);
        inArgs.setDoubleProperty(kOfxPropTime, time);
        inArgs.setIntPropertyN(kOfxImageEffectPropRenderWindow, &renderRoI.x1, 4);
        inArgs.setDoublePropertyN(kOfxImageEffectPropRenderScale, &renderScale.x, 2);
        inArgs.setIntProperty(kOfxImageEffectPropSequentialRenderStatus, sequentialRender);
        inArgs.setIntProperty(kOfxImageEffectPropInteractiveRenderStatus, interactiveRender);
        inArgs.setIntProperty(kOfxImageEffectPropRenderQualityDraft, draftRender);
        inArgs.setIntProperty(kOfxImageEffectPropOpenGLEnabled, 1);

        // The OFX OpenGL render model requires the HOST to bind the output texture as the render
        // target and set up the viewport (and, for fixed-function plug-ins, the projection); the
        // plug-in then draws into the bound framebuffer. Without this the plug-in draws nowhere and
        // the output stays cleared.
        GLuint render_fbo = 0;
        GLint  prev_fbo   = 0;
        GLint  prev_vp[4] = {0, 0, 0, 0};
        if (ctx_.output_tex) {
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
            glGetIntegerv(GL_VIEWPORT, prev_vp);
            glGenFramebuffers(1, &render_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, render_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx_.output_tex, 0);
            glViewport(0, 0, ctx_.width, ctx_.height);
            if (!ctx_.external_gl) {
                // Self-contained compatibility context: set up a fixed-function projection so
                // legacy plug-ins' pixel-space glVertex2f(0..w,0..h) maps 1:1. These calls are
                // INVALID in a core profile, so they are only used on the (compatibility) offscreen
                // context — never on the zero-copy core-profile mixer device.
                glMatrixMode(GL_PROJECTION);
                glPushMatrix();
                glLoadIdentity();
                glOrtho(0, ctx_.width, 0, ctx_.height, -1, 1);
                glMatrixMode(GL_MODELVIEW);
                glPushMatrix();
                glLoadIdentity();
            }
            glClearColor(0.f, 0.f, 0.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        const OfxStatus st = mainEntry(kOfxImageEffectActionRender, getHandle(), &inArgs, nullptr);

        if (render_fbo) {
            if (!ctx_.external_gl) {
                glMatrixMode(GL_PROJECTION);
                glPopMatrix();
                glMatrixMode(GL_MODELVIEW);
                glPopMatrix();
            }
            glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
            glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
            glDeleteFramebuffers(1, &render_fbo);
        }

        if (st == kOfxStatOK || st == kOfxStatReplyDefault) {
            if (!ctx_.external_gl && ctx_.output_rgba)
                gl_->readback_output(ctx_.output_rgba, ctx_.width, ctx_.height);
            static bool logged_ok = false;
            if (!logged_ok) {
                logged_ok = true;
                CASPAR_LOG(info) << (ctx_.external_gl ? L"[ofx] OpenGL zero-copy render path active."
                                                      : L"[ofx] OpenGL render path active.");
            }
        } else {
            static bool logged_bad = false;
            if (!logged_bad) {
                logged_bad = true;
                CASPAR_LOG(warning) << L"[ofx] OpenGL render returned status " << st << L".";
            }
        }

        ctx_.gl = false;
        return st;
    } catch (...) {
        ctx_.gl = false;
        static bool logged_ex = false;
        if (!logged_ex) {
            logged_ex = true;
            CASPAR_LOG(warning) << L"[ofx] OpenGL render threw; falling back (source passes through).";
        }
        return kOfxStatFailed;
    }
}

const std::string& ofx_effect_instance::getDefaultOutputFielding() const
{
    static const std::string v(kOfxImageFieldNone);
    return v;
}

OfxStatus ofx_effect_instance::vmessage(const char* type, const char* /*id*/, const char* format, va_list args)
{
    char buf[1024];
    if (format)
        std::vsnprintf(buf, sizeof(buf), format, args);
    else
        buf[0] = '\0';
    const bool err = type && (std::strstr(type, "Error") || std::strstr(type, "Fatal"));
    if (err)
        CASPAR_LOG(warning) << L"[ofx] plug-in message: " << u16(buf);
    else
        CASPAR_LOG(info) << L"[ofx] plug-in message: " << u16(buf);
    return kOfxStatOK;
}
OfxStatus ofx_effect_instance::setPersistentMessage(const char* /*type*/, const char* /*id*/, const char* format, va_list args)
{
    char buf[1024];
    if (format)
        std::vsnprintf(buf, sizeof(buf), format, args);
    else
        buf[0] = '\0';
    CASPAR_LOG(warning) << L"[ofx] plug-in persistent message: " << u16(buf);
    return kOfxStatOK;
}
OfxStatus ofx_effect_instance::clearPersistentMessage() { return kOfxStatOK; }

void ofx_effect_instance::getProjectSize(double& x, double& y) const
{
    x = width_;
    y = height_;
}
void ofx_effect_instance::getProjectOffset(double& x, double& y) const
{
    x = 0;
    y = 0;
}
void ofx_effect_instance::getProjectExtent(double& x, double& y) const
{
    x = width_;
    y = height_;
}
double ofx_effect_instance::getProjectPixelAspectRatio() const { return 1.0; }
double ofx_effect_instance::getEffectDuration() const { return 1.0; }
double ofx_effect_instance::getFrameRate() const { return frame_rate_; }
double ofx_effect_instance::getFrameRecursive() const { return 0.0; }
void   ofx_effect_instance::getRenderScaleRecursive(double& x, double& y) const { x = y = 1.0; }

OFX::Host::Param::Instance* ofx_effect_instance::newParam(const std::string&            name,
                                                          OFX::Host::Param::Descriptor& descriptor)
{
    return create_param_instance(this, name, descriptor);
}

OfxStatus ofx_effect_instance::editBegin(const std::string&) { return kOfxStatErrMissingHostFeature; }
OfxStatus ofx_effect_instance::editEnd() { return kOfxStatErrMissingHostFeature; }

void ofx_effect_instance::progressStart(const std::string&, const std::string&) {}
void ofx_effect_instance::progressEnd() {}
bool ofx_effect_instance::progressUpdate(double) { return true; }

double ofx_effect_instance::timeLineGetTime() { return ctx_.time; }
void   ofx_effect_instance::timeLineGotoTime(double) {}
void   ofx_effect_instance::timeLineGetBounds(double& t1, double& t2)
{
    t1 = 0;
    t2 = 0;
}

}} // namespace caspar::ofx
