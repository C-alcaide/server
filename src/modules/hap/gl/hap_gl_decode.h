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
 */

// hap_gl_decode.h
// GL compressed texture upload + FBO render pass for HAP decoding.
//
// Uploads raw DXT/BC block data as GL compressed textures, then renders a
// fullscreen triangle to decompress them into standard RGBA8 textures that
// the CasparCG mixer can consume. The GPU's texture unit does the actual
// DXT/BC decompression in hardware at zero compute cost.
// ---------------------------------------------------------------------------
#pragma once

#include "../util/hap_frame_parser.h"

#include <GL/glew.h>
#include <accelerator/ogl/util/texture.h>

#include <cstdint>
#include <memory>

namespace caspar { namespace hap {

// One decode slot: compressed source texture(s) + output RGBA texture.
struct HapDecodeSlot
{
    // Compressed source textures (created with glTexStorage2D + GL_COMPRESSED_*)
    GLuint   compressed_tex     = 0;  // primary texture (all variants)
    GLuint   compressed_alpha   = 0;  // alpha texture (HapM only)
    GLenum   compressed_format  = 0;  // GL internal format of primary
    GLenum   alpha_format       = 0;  // GL internal format of alpha (HapM)

    // Output: standard uncompressed RGBA8 texture for the mixer
    std::shared_ptr<accelerator::ogl::texture> output_tex;

    int      width   = 0;
    int      height  = 0;
    bool     initialized = false;
};

// Manages the GL shader program, FBO, and VAO for the decompression render pass.
class HapGLDecoder
{
  public:
    HapGLDecoder();
    ~HapGLDecoder();

    HapGLDecoder(const HapGLDecoder&)            = delete;
    HapGLDecoder& operator=(const HapGLDecoder&) = delete;

    // Initialize a decode slot's compressed textures for the given dimensions and format.
    void init_slot(HapDecodeSlot& slot, int width, int height,
                   HapTextureFormat primary_fmt, HapTextureFormat alpha_fmt = HapTextureFormat::RGB_DXT1);

    // Upload DXT data to the compressed texture, run the render pass, and
    // produce an uncompressed RGBA8 result in slot.output_tex.
    // For HapM: both texture_data and alpha_data must be non-empty.
    void decode(HapDecodeSlot& slot,
                HapVariant variant,
                const uint8_t* texture_data, size_t texture_size,
                const uint8_t* alpha_data = nullptr, size_t alpha_size = 0);

  private:
    void compile_shaders();
    void create_fbo();

    GLuint program_passthrough_ = 0;  // HAP, HAP Alpha
    GLuint program_ycocg_       = 0;  // HAP Q
    GLuint program_ycocg_alpha_ = 0;  // HAP Q Alpha (two textures)
    GLuint fbo_                 = 0;
    GLuint vao_                 = 0;  // empty VAO for attributeless rendering

    bool initialized_ = false;
};

// Get the GL compressed internal format for a HAP texture format.
inline GLenum hap_texture_to_gl(HapTextureFormat fmt)
{
    switch (fmt) {
    case HapTextureFormat::RGB_DXT1:   return GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
    case HapTextureFormat::RGBA_DXT5:  return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    case HapTextureFormat::YCoCg_DXT5: return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; // same block format, different interpretation
    case HapTextureFormat::A_RGTC1:    return GL_COMPRESSED_RED_RGTC1;          // BC4 single-channel alpha
    case HapTextureFormat::RGBA_BC7:   return GL_COMPRESSED_RGBA_BPTC_UNORM;
    default:                           return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    }
}

// Calculate the size in bytes of compressed DXT/BC data for given dimensions + format.
inline size_t dxt_data_size(int width, int height, GLenum gl_format)
{
    int bw = (width  + 3) / 4;
    int bh = (height + 3) / 4;
    // DXT1 and RGTC1/BC4 use 8-byte blocks; all others (DXT5, BC7, etc.) use 16-byte blocks.
    int block_size = (gl_format == GL_COMPRESSED_RGB_S3TC_DXT1_EXT ||
                      gl_format == GL_COMPRESSED_RED_RGTC1) ? 8 : 16;
    return (size_t)bw * bh * block_size;
}

}} // namespace caspar::hap
