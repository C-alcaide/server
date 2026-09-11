/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// WHAT ONE NODE PASS NEEDS, as the two kernels see it.
//
// The prototype put a whole `grade_node` -- a struct with an ellipse, an exposure and a CDL --
// into `draw_params`. That works for one fixed record and stops working the moment a class can
// be one of several, because the kernel would have to switch on a variant.
//
// So a pass carries an OP and a slice of the values array instead. The kernel switches on the
// op to decide which uniforms to upload, and the shader switches on the same number to decide
// which operation to run. One enum, two readers, and adding a class is a case in each rather
// than a new field in this struct.
//
// POINTERS INTO THE PLAN'S ARRAY, not copies. The plan and the values both outlive the draw --
// they are held by `shared_ptr` in the transform the draw was built from -- and copying six
// doubles per pass per frame to avoid a raw pointer would be caution about the wrong thing. The
// lifetime is the caller's to guarantee and the evaluator is the only caller.

#include <cstdint>

namespace caspar { namespace core { namespace graph {

/// One node's full-screen pass.
struct node_draw
{
    /// Index into `node_classes()`. -1 means "this is not a node pass", which is what makes a
    /// single flag unnecessary: the op IS the flag.
    std::int32_t op = -1;

    /// This step's parameters, in the class's port order. Null only when `op < 0`.
    const double* values = nullptr;
    std::uint32_t values_count = 0;

    /// THE FUSED MASK's parameters, or null. A mask with one consumer costs no pass: its
    /// parameters ride along here and the shader evaluates it inline, which is what makes a
    /// windowed grade one draw rather than two.
    const double* mask_values = nullptr;

    /// Is a SECOND image bound? `mix` and `over` take one; everything else does not, and an
    /// unbound `b` must make them return `a` rather than mix toward black -- a muted edge
    /// blacking a layer during a show is the one failure nobody would forgive.
    bool has_in1 = false;

    /// Is a MATERIALISED mask texture bound? (v1: never -- every mask is fused. The flag is
    /// here because the evaluator already distinguishes the two cases and the shader will need
    /// to when `mask_combine` arrives.)
    bool has_mask_texture = false;

    /// WHICH MASK CLASS the fused mask is, as an index into `node_classes()`, or -1 when there
    /// is no fused mask.
    ///
    /// A fused mask is evaluated by its CONSUMER from the consumer's own uniforms, and `op`
    /// above is the CONSUMER's class -- so without this the consumer has no way to know whether
    /// it was handed an ellipse, a rectangle or a gradient. It rendered every one of them as an
    /// ellipse: correct as a materialised pass, wrong the moment the same mask had one
    /// consumer, which is the common case. Measured by `grade-graph`'s rect arm, where a
    /// rectangle and an ellipse of identical `center`/`radius` came back byte-identical.
    std::int32_t mask_op = -1;

    explicit operator bool() const { return op >= 0; }
};

}}} // namespace caspar::core::graph
