/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#pragma once

#include <memory>

namespace caspar { namespace isf {

/// A self-contained offscreen OpenGL context (SFML), used to run ISF shaders when the active mixer
/// is not the OpenGL mixer (e.g. the Vulkan mixer). Rendered output is read back to the CPU.
class gl_context
{
  public:
    gl_context(); ///< may throw if a context / GLEW cannot be created
    ~gl_context();

    gl_context(const gl_context&)            = delete;
    gl_context& operator=(const gl_context&) = delete;

    bool make_current();

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::isf
