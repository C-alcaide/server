/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf_gl_context.h"

#include <GL/glew.h>

#include <SFML/Window/Context.hpp>

#include <mutex>
#include <stdexcept>

namespace caspar { namespace isf {

namespace {
std::once_flag g_glew_init_flag;
bool           g_glew_ok = false;
} // namespace

struct gl_context::impl
{
    sf::Context context; ///< offscreen GL context, active on the creating thread

    impl()
    {
        context.setActive(true);
        std::call_once(g_glew_init_flag, [] {
            glewExperimental = GL_TRUE;
            g_glew_ok        = (glewInit() == GLEW_OK);
        });
        if (!g_glew_ok)
            throw std::runtime_error("[isf] GLEW init failed for the self-contained OpenGL context");
    }
};

gl_context::gl_context()
    : impl_(std::make_unique<impl>())
{
}
gl_context::~gl_context() = default;

bool gl_context::make_current() { return impl_->context.setActive(true); }

}} // namespace caspar::isf
