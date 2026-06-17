///////////////////////////
//
// SFML - Simple and Fast Multimedia Library
// Copyright (C) 2007-2009 Laurent Gomila (laurent.gom@gmail.com)
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
///////////////////////////
#include "gl_check.h"

#include "../except.h"
#include "../log.h"

#include <GL/glew.h>

#include <atomic>
#include <chrono>

namespace caspar { namespace gl {

// Rate-limit GL error logging to prevent log-file floods that can exhaust
// disk space and destabilise the server.  Errors are still drained from the
// GL error queue (mandatory) and the last error still throws, but only the
// first few messages per second are actually written to the log.
static constexpr int          GL_ERROR_LOG_LIMIT  = 10;   // max messages per window
static constexpr std::int64_t GL_ERROR_WINDOW_NS  =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)).count();

void SMFL_GLCheckError(const std::string& /*unused*/, const char* func, const char* file, unsigned int line)
{
    // Get the last error
    GLenum LastErrorCode = GL_NO_ERROR;

    // Thread-local rate limiter — each GL thread gets its own counter.
    thread_local int64_t tl_window_start = 0;
    thread_local int     tl_error_count  = 0;
    thread_local bool    tl_suppressed   = false;

    for (GLenum ErrorCode = glGetError(); ErrorCode != GL_NO_ERROR; ErrorCode = glGetError()) {
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        if (now - tl_window_start > GL_ERROR_WINDOW_NS) {
            tl_window_start = now;
            tl_error_count  = 0;
            tl_suppressed   = false;
        }
        ++tl_error_count;
        if (tl_error_count <= GL_ERROR_LOG_LIMIT) {
            std::string str(reinterpret_cast<const char*>(glewGetErrorString(ErrorCode)));
            CASPAR_LOG(error) << "OpenGL Error: " << ErrorCode << L" " << str;
        } else if (!tl_suppressed) {
            tl_suppressed = true;
            CASPAR_LOG(error) << "GL error flood detected - suppressing further GL error "
                                 "messages on this thread for up to 1 second.";
        }
        LastErrorCode = ErrorCode;
    }

    if (LastErrorCode != GL_NO_ERROR) {
        // Decode the error code
        switch (LastErrorCode) {
            case GL_INVALID_ENUM:
                CASPAR_THROW_EXCEPTION(ogl_invalid_enum()
                                       << msg_info(
                                              "an unacceptable value has been specified for an enumerated argument")
                                       << error_info("GL_INVALID_ENUM"));

            case GL_INVALID_VALUE:
                CASPAR_THROW_EXCEPTION(ogl_invalid_value() << msg_info("a numeric argument is out of range")
                                                           << error_info("GL_INVALID_VALUE"));

            case GL_INVALID_OPERATION:
                CASPAR_THROW_EXCEPTION(ogl_invalid_operation()
                                       << msg_info("the specified operation is not allowed in the current state")
                                       << error_info("GL_INVALID_OPERATION"));

            case GL_STACK_OVERFLOW:
                CASPAR_THROW_EXCEPTION(ogl_stack_overflow() << msg_info("this command would cause a stack overflow")
                                                            << error_info("GL_STACK_OVERFLOW"));

            case GL_STACK_UNDERFLOW:
                CASPAR_THROW_EXCEPTION(ogl_stack_underflow() << msg_info("this command would cause a stack underflow")
                                                             << error_info("GL_STACK_UNDERFLOW"));

            case GL_OUT_OF_MEMORY:
                CASPAR_THROW_EXCEPTION(ogl_out_of_memory()
                                       << msg_info("there is not enough memory left to execute the command")
                                       << error_info("GL_OUT_OF_MEMORY"));

            case GL_INVALID_FRAMEBUFFER_OPERATION_EXT:
                CASPAR_THROW_EXCEPTION(
                    ogl_invalid_framebuffer_operation_ext()
                    << msg_info("the object bound to FRAMEBUFFER_BINDING_EXT is not \"framebuffer complete\"")
                    << error_info("GL_INVALID_FRAMEBUFFER_OPERATION_EXT"));
        }
    }
}

}} // namespace caspar::gl
