/*
 * Copyright (c) 2026 CasparCG Contributors
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

#include "gl_export_bridge.h"

#include "platform_config.h"
#include "texture.h"

#include <common/except.h>
#include <common/log.h>

#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <EGL/egl.h>
#include <unistd.h> // dup()
#endif

#include <mutex>

namespace caspar { namespace accelerator { namespace vulkan {

namespace {

PFNGLCREATEMEMORYOBJECTSEXTPROC glCreateMemoryObjectsEXT_ = nullptr;
PFNGLDELETEMEMORYOBJECTSEXTPROC glDeleteMemoryObjectsEXT_ = nullptr;
PFNGLTEXTURESTORAGEMEM2DEXTPROC glTextureStorageMem2DEXT_ = nullptr;

#ifdef _WIN32
PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC glImportMemoryWin32HandleEXT_ = nullptr;
#else
using PFNGLIMPORTMEMORYFDEXTPROC = void (*)(GLuint, GLuint64, GLenum, GLint);
PFNGLIMPORTMEMORYFDEXTPROC glImportMemoryFdEXT_ = nullptr;
#endif

std::once_flag ext_flag;
bool           ext_loaded = false;

void load_extensions()
{
    std::call_once(ext_flag, [] {
#ifdef _WIN32
        auto get = [](const char* n) { return wglGetProcAddress(n); };
#else
        auto get = [](const char* n) { return eglGetProcAddress(n); };
#endif
        glCreateMemoryObjectsEXT_ = (PFNGLCREATEMEMORYOBJECTSEXTPROC)get("glCreateMemoryObjectsEXT");
        glDeleteMemoryObjectsEXT_ = (PFNGLDELETEMEMORYOBJECTSEXTPROC)get("glDeleteMemoryObjectsEXT");
        glTextureStorageMem2DEXT_ = (PFNGLTEXTURESTORAGEMEM2DEXTPROC)get("glTextureStorageMem2DEXT");
#ifdef _WIN32
        glImportMemoryWin32HandleEXT_ = (PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC)get("glImportMemoryWin32HandleEXT");
        ext_loaded = glCreateMemoryObjectsEXT_ && glImportMemoryWin32HandleEXT_ && glTextureStorageMem2DEXT_;
#else
        glImportMemoryFdEXT_ = (PFNGLIMPORTMEMORYFDEXTPROC)get("glImportMemoryFdEXT");
        ext_loaded = glCreateMemoryObjectsEXT_ && glImportMemoryFdEXT_ && glTextureStorageMem2DEXT_;
#endif
        if (!ext_loaded)
            CASPAR_LOG(warning) << L"[gl_export_bridge] GL_EXT_memory_object is not available; "
                                   L"Vulkan textures cannot be shared with OpenGL on this driver.";
    });
}

/// The GL sized format matching a Vulkan texture created by
/// create_exportable_texture, whose formats come from an equivalent table.
GLenum internal_format_for(int stride, bool is_16bit)
{
    static const GLenum FORMAT[2][5] = {{0, GL_R8, GL_RG8, GL_RGB8, GL_RGBA8},
                                        {0, GL_R16, GL_RG16, GL_RGB16, GL_RGBA16}};
    if (stride < 1 || stride > 4)
        return 0;
    return FORMAT[is_16bit ? 1 : 0][stride];
}

} // anonymous namespace

bool gl_import_supported()
{
    load_extensions();
    return ext_loaded;
}

std::string gl_import_memory_as_texture(void*              handle,
                                        unsigned long long size,
                                        unsigned int       internal_format,
                                        int                width,
                                        int                height,
                                        unsigned int&      out_memory_object,
                                        unsigned int&      out_texture)
{
    load_extensions();
    if (!ext_loaded)
        return "GL_EXT_memory_object is not available";
    if (!handle || size == 0 || internal_format == 0 || width <= 0 || height <= 0)
        return "invalid arguments (no handle, zero size, or unsupported format)";

    while (glGetError() != GL_NO_ERROR) {}

    GLuint mem_obj = 0;
    glCreateMemoryObjectsEXT_(1, &mem_obj);

#ifdef _WIN32
    // With an OPAQUE_WIN32 NT handle GL duplicates it, so the texture keeps
    // owning the original and we must not close it here.
    glImportMemoryWin32HandleEXT_(mem_obj, size, platform::kGlHandleType, handle);
#else
    // glImportMemoryFdEXT consumes the fd, and the Vulkan texture owns the one
    // it handed us, so import a duplicate.
    const int fd = dup(static_cast<int>(reinterpret_cast<intptr_t>(handle)));
    glImportMemoryFdEXT_(mem_obj, size, platform::kGlHandleType, fd);
#endif

    if (GLenum err = glGetError(); err != GL_NO_ERROR) {
        glDeleteMemoryObjectsEXT_(1, &mem_obj);
        return "glImportMemory failed, GL error 0x" + std::to_string(err);
    }

    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Optimal is the default, but say it: linear is what the probe found GL
    // rejects on Vulkan-allocated memory, so being explicit documents the
    // requirement at the point it matters.
    glTextureParameteri(tex, GL_TEXTURE_TILING_EXT, GL_OPTIMAL_TILING_EXT);
    glTextureStorageMem2DEXT_(tex, 1, internal_format, width, height, mem_obj, 0);

    if (GLenum err = glGetError(); err != GL_NO_ERROR) {
        glDeleteTextures(1, &tex);
        glDeleteMemoryObjectsEXT_(1, &mem_obj);
        return "glTextureStorageMem2DEXT failed, GL error 0x" + std::to_string(err);
    }

    out_memory_object = mem_obj;
    out_texture       = tex;
    return {};
}

void gl_release_imported_texture(unsigned int& memory_object, unsigned int& tex)
{
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
    if (memory_object) {
        load_extensions();
        if (glDeleteMemoryObjectsEXT_)
            glDeleteMemoryObjectsEXT_(1, &memory_object);
        memory_object = 0;
    }
}

gl_shared_texture::gl_shared_texture(std::shared_ptr<texture> vk_tex)
    : vk_tex_(std::move(vk_tex))
{
    if (!vk_tex_)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("gl_shared_texture: null Vulkan texture"));

    auto handle = vk_tex_->export_native_handle();
    if (handle == platform::kInvalidHandle)
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(
                                   "gl_shared_texture: the Vulkan texture has no exportable memory "
                                   "(created with create_texture rather than create_exportable_texture?)"));

    const auto format = internal_format_for(vk_tex_->stride(), vk_tex_->depth() != common::bit_depth::bit8);

#ifdef _WIN32
    void* raw = handle;
#else
    void* raw = reinterpret_cast<void*>(static_cast<intptr_t>(handle));
#endif

    auto err = gl_import_memory_as_texture(raw,
                                           static_cast<unsigned long long>(vk_tex_->alloc_size()),
                                           format,
                                           vk_tex_->width(),
                                           vk_tex_->height(),
                                           gl_memory_object_,
                                           gl_texture_);
    if (!err.empty())
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("gl_shared_texture: " + err));
}

gl_shared_texture::~gl_shared_texture() { gl_release_imported_texture(gl_memory_object_, gl_texture_); }

int gl_shared_texture::width() const { return vk_tex_->width(); }
int gl_shared_texture::height() const { return vk_tex_->height(); }

}}} // namespace caspar::accelerator::vulkan
