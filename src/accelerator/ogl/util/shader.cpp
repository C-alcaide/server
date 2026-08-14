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
#include "shader.h"

#include <common/gl/gl_check.h>
#include <common/log.h>
#include <common/utf.h>

#include <GL/glew.h>

#include <sstream>
#include <unordered_map>

namespace caspar { namespace accelerator { namespace ogl {

struct shader::impl
{
    GLuint                                 program_;
    std::unordered_map<std::string, GLint> uniform_locations_;
    std::unordered_map<std::string, GLint> attrib_locations_;

    impl(const impl&)            = delete;
    impl& operator=(const impl&) = delete;

  public:
    impl(const std::string& vertex_source_str, const std::string& fragment_source_str)
        : program_(0)
    {
        GLint success;

        const char* vertex_source = vertex_source_str.c_str();

        auto vertex_shader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);

        GL(glShaderSourceARB(vertex_shader, 1, &vertex_source, NULL));
        GL(glCompileShaderARB(vertex_shader));

        GL(glGetObjectParameterivARB(vertex_shader, GL_OBJECT_COMPILE_STATUS_ARB, &success));
        if (success == GL_FALSE) {
            char info[2048];
            GL(glGetInfoLogARB(vertex_shader, sizeof(info), 0, info));
            GL(glDeleteObjectARB(vertex_shader));
            std::stringstream str;
            str << "Failed to compile vertex shader:" << std::endl << info << std::endl;
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(str.str()));
        }

        const char* fragment_source = fragment_source_str.c_str();

        auto fragmemt_shader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

        GL(glShaderSourceARB(fragmemt_shader, 1, &fragment_source, NULL));
        GL(glCompileShaderARB(fragmemt_shader));

        GL(glGetObjectParameterivARB(fragmemt_shader, GL_OBJECT_COMPILE_STATUS_ARB, &success));
        if (success == GL_FALSE) {
            char info[2048];
            GL(glGetInfoLogARB(fragmemt_shader, sizeof(info), 0, info));
            GL(glDeleteObjectARB(fragmemt_shader));
            std::stringstream str;
            str << "Failed to compile fragment shader:" << std::endl << info << std::endl;
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(str.str()));
        }

        program_ = glCreateProgramObjectARB();

        GL(glAttachObjectARB(program_, vertex_shader));
        GL(glAttachObjectARB(program_, fragmemt_shader));

        GL(glLinkProgramARB(program_));

        GL(glDeleteObjectARB(vertex_shader));
        GL(glDeleteObjectARB(fragmemt_shader));

        GL(glGetObjectParameterivARB(program_, GL_OBJECT_LINK_STATUS_ARB, &success));
        if (success == GL_FALSE) {
            char info[2048];
            GL(glGetInfoLogARB(program_, sizeof(info), 0, info));
            GL(glDeleteObjectARB(program_));
            std::stringstream str;
            str << "Failed to link shader program:" << std::endl << info << std::endl;
            CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info(str.str()));
        }
        GL(glUseProgramObjectARB(program_));
    }

    ~impl() { glDeleteProgram(program_); }

    GLint get_uniform_location(const char* name)
    {
        auto it = uniform_locations_.find(name);
        if (it == uniform_locations_.end())
            it = uniform_locations_.insert(std::make_pair(name, glGetUniformLocation(program_, name))).first;
        return it->second;
    }

    GLint get_attrib_location(const char* name)
    {
        auto it = attrib_locations_.find(name);
        if (it == attrib_locations_.end())
            it = attrib_locations_.insert(std::make_pair(name, glGetAttribLocation(program_, name))).first;
        return it->second;
    }

    /// Which uniform did GL reject, and what is it in the program that is actually bound?
    ///
    /// `GL(glUniform1f(...))` reports the call and nothing else, so a rejected uniform in a
    /// 2000-line kernel with a hundred of them says only "1282 somewhere in set()". A
    /// mismatched uniform is not a rare accident either: every OCIO variant is its own
    /// program, and a name that is a float in one and something else in another is exactly
    /// what a spliced shader can produce.
    ///
    /// Deliberately asks GL for the program's own idea of the type rather than reporting
    /// what we passed — the whole question is where the two disagree.
    ///
    /// It LOGS AND THEN THROWS. An earlier revision only logged, which turned a draw-
    /// aborting error into a silent one and made the failing case appear to pass — the
    /// picture was right because the rest of the draw completed, and the defect underneath
    /// was still there. Reporting better must not mean reporting less.
    void report_uniform_error(const std::string& name, GLint location)
    {
        const GLenum code = glGetError();
        if (code == GL_NO_ERROR)
            return;

        GLint  current = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &current);

        std::string actual = "(no such active uniform)";
        if (location >= 0 && current != 0) {
            GLint  size = 0;
            GLenum type = 0;
            GLchar buf[256] = {0};
            GLsizei len = 0;
            // The index and the location are different numbers; walk the active uniforms to
            // find the one at this location rather than assuming they coincide.
            GLint count = 0;
            glGetProgramiv(static_cast<GLuint>(current), GL_ACTIVE_UNIFORMS, &count);
            for (GLint i = 0; i < count; ++i) {
                glGetActiveUniform(static_cast<GLuint>(current), static_cast<GLuint>(i),
                                   sizeof(buf), &len, &size, &type, buf);
                if (glGetUniformLocation(static_cast<GLuint>(current), buf) == location) {
                    std::stringstream s;
                    s << "GL type 0x" << std::hex << type << std::dec << " named '" << buf << "'";
                    actual = s.str();
                    break;
                }
            }
        }

        CASPAR_LOG(error) << L"[shader] GL rejected uniform '" << u16(name) << L"' at location "
                          << location << L" in program " << current << L"; the bound program has "
                          << u16(actual) << L" there. A uniform set as one type and declared as "
                          << L"another is the usual cause, and so is setting a uniform before "
                          << L"`use()` binds the program it belongs to.";

        // Same exception the GL() macro would have raised, so callers see no change in
        // behaviour — only a log line that says which uniform.
        if (code == GL_INVALID_OPERATION)
            CASPAR_THROW_EXCEPTION(caspar::gl::ogl_invalid_operation()
                                   << msg_info("the specified operation is not allowed in the "
                                               "current state, setting uniform '" + name + "'")
                                   << error_info("GL_INVALID_OPERATION"));
        CASPAR_THROW_EXCEPTION(caspar::gl::ogl_invalid_value()
                               << msg_info("GL rejected uniform '" + name + "'")
                               << error_info("GL_INVALID_VALUE"));
    }

    void set(const std::string& name, bool value) { set(name, value ? 1 : 0); }

    void set(const std::string& name, int value)
    {
        const auto loc = get_uniform_location(name.c_str());
        glUniform1i(loc, value);
        report_uniform_error(name, loc);
    }

    void set(const std::string& name, float value)
    {
        const auto loc = get_uniform_location(name.c_str());
        glUniform1f(loc, value);
        report_uniform_error(name, loc);
    }

    void set(const std::string& name, double value0, double value1)
    {
        GL(glUniform2f(get_uniform_location(name.c_str()), static_cast<float>(value0), static_cast<float>(value1)));
    }
    void set(const std::string& name, double value0, double value1, double value2)
    {
        GL(glUniform3f(get_uniform_location(name.c_str()),
                       static_cast<float>(value0),
                       static_cast<float>(value1),
                       static_cast<float>(value2)));  // fixed: was value1
    }

    void set(const std::string& name, double value0, double value1, double value2, double value3)
    {
        GL(glUniform4f(get_uniform_location(name.c_str()),
                       static_cast<float>(value0),
                       static_cast<float>(value1),
                       static_cast<float>(value2),
                       static_cast<float>(value3)));
    }

    void set(const std::string& name, double value)
    {
        GL(glUniform1f(get_uniform_location(name.c_str()), static_cast<float>(value)));
    }
    void set_matrix3(const std::string& name, const float* value)
    {
        GL(glUniformMatrix3fv(get_uniform_location(name.c_str()), 1, GL_TRUE, value));
    }

    void set_float_array(const std::string& name, const float* values, int count)
    {
        GL(glUniform1fv(get_uniform_location(name.c_str()), count, values));
    }

    void use() { GL(glUseProgramObjectARB(program_)); }
};

shader::shader(const std::string& vertex_source_str, const std::string& fragment_source_str)
    : impl_(new impl(vertex_source_str, fragment_source_str))
{
}
shader::~shader() {}
void shader::set(const std::string& name, bool value) { impl_->set(name, value); }
void shader::set(const std::string& name, int value) { impl_->set(name, value); }
void shader::set(const std::string& name, float value) { impl_->set(name, value); }
void shader::set(const std::string& name, double value0, double value1) { impl_->set(name, value0, value1); }
void shader::set(const std::string& name, double value0, double value1, double value2)
{
    impl_->set(name, value0, value1, value2);
}
void shader::set(const std::string& name, double value0, double value1, double value2, double value3)
{
    impl_->set(name, value0, value1, value2, value3);
}
void  shader::set(const std::string& name, double value) { impl_->set(name, value); }
void  shader::set_matrix3(const std::string& name, const float* value) { impl_->set_matrix3(name, value); }
void  shader::set_float_array(const std::string& name, const float* values, int count) { impl_->set_float_array(name, values, count); }
GLint shader::get_attrib_location(const char* name) { return impl_->get_attrib_location(name); }
int   shader::id() const { return impl_->program_; }
void  shader::use() const { impl_->use(); }

}}} // namespace caspar::accelerator::ogl
