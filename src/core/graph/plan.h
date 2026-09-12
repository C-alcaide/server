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

// THE COMPILED FORM: what the frame path sees, and the split that makes it affordable.
//
// TWO OBJECTS REACH `image_transform`, because they have two different FLOW types, and getting
// this wrong is the most expensive mistake available in the whole design:
//
//   `node_plan`   the ATTRIBUTE half -- topology, classes, order, `last_use`, the pass count.
//                 Held by `shared_ptr<const>`, compared by POINTER IDENTITY, and reallocated
//                 ONLY when the document's structure changes. That is exactly what
//                 `grade_nodes` already did and the reason it worked: `image_transform`'s
//                 `operator==` is what the still-frame cache compares, and a pointer is one
//                 word rather than a deep compare of a graph.
//
//   `node_values` the SIGNAL half -- every node parameter, every `bypass`, every edge's `mute`,
//                 as a FLAT `vector<double>` with offsets the plan carries. Compared BY VALUE.
//                 A timeline ramping `exposure` writes one double per tick into an array that
//                 already exists.
//
// WHY NOT ONE OBJECT. The obvious design is a compiled pointer holding the values, reallocated
// whenever a value changes. A timeline ramping one parameter at 50 Hz would then allocate a
// graph fifty times a second -- and worse, the still-frame fingerprint would move on every tick
// by ALLOCATION rather than by value, so a paused, unchanging graph would look different every
// frame and defeat the cache it exists to feed. The split is not an optimisation; the single
// object is incorrect.
//
// AND THE SPLIT IS ONLY SAFE BECAUSE THE REGISTRY DECLARES WHICH IS WHICH. A port's
// `port_flow` says whether its value is a `signal` (in the array) or an `attribute` (in the
// plan). Declaring a signal as an attribute costs a reallocation per tick; declaring an
// attribute as a signal makes a change silently not take effect. `node_registry_self_test`
// cannot catch either, because both are legal C++ -- the declaration is the contract.

#include "model.h"
#include "registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace caspar { namespace core { namespace graph {

/// Every node parameter, flat. Indexed by the offsets in `node_plan::value_index`.
///
/// `double` rather than `float` because that is what the overlays, the registry and the
/// control API all carry, and converting at the boundary is where a rounding difference
/// between the published value and the rendered one would hide.
using node_values = std::vector<double>;

/// ONE STEP of the compiled graph: consume up to three textures, produce one.
///
/// Indices are into the evaluator's own `outputs` array, which is parallel to `steps`. -1 means
/// "not connected", and what that CONTRIBUTES is the port's `disconnected_default` rather than
/// transparent black -- see `registry.h`, and note that a muted edge takes the same path.
struct node_step
{
    /// Which class, as an index into `node_classes()`. An index rather than a string because
    /// this is read per draw: the evaluator switches on it and the shader takes it as `gn_op`.
    std::int32_t cls = -1;

    /// The node's id, for `value_index` lookups and for a preview request naming it. Not read
    /// on the frame path.
    std::string id;

    /// Inputs, as step indices. `in0` is the PRIMARY input -- what a bypassed node aliases.
    std::int32_t in0 = -1;
    std::int32_t in1 = -1;
    std::int32_t mask = -1;

    /// Does this step draw into an attachment of its own? False for `input` (it IS the head
    /// pass's output), for `output` (it is the tail), and for a mask that is FUSED into its
    /// consumer's uniforms rather than materialised.
    bool produces_image = false;

    /// An analytic mask with exactly ONE consumer needs no attachment: its parameters go into
    /// the consumer's own uniforms and the shader evaluates it inline. That is what the
    /// prototype's `grade_node_mask` already does, and it is why a windowed grade costs one
    /// pass rather than two.
    bool fused_mask = false;

    /// This step is a mask generator that must be MATERIALISED -- drawn into its own
    /// attachment for its consumers to sample, because more than one of them reads it.
    ///
    /// Distinct from `produces_image`, which means "produces a full colour image and costs an
    /// image pass". A materialised mask costs a pass too, but it produces a MASK: its
    /// consumers bind it as a texture and multiply by it rather than treating it as a picture.
    /// Kept as its own flag rather than derived in the evaluator from `fused_mask` plus the
    /// class group, because the evaluator would then need the registry to decide what to draw.
    ///
    /// EXACTLY `group == "mask" && !fused_mask`. The invariant is asserted in
    /// `graph_plan_self_test`: a mask is fused or materialised and never neither, which is what
    /// it WAS -- `has_mask_texture` was declared in `node_draw.h` and read by nothing, so a
    /// mask with two consumers was silently dropped and both of them graded the whole image.
    bool produces_mask = false;

    /// THE LAST STEP THAT READS THIS ONE's OUTPUT. After it, the attachment goes back to the
    /// pool. Computed at compile time because the evaluator must not search forwards per step
    /// per frame -- and because an attachment released too early is a garbage read that looks
    /// like a maths bug.
    std::int32_t last_use = -1;

    /// Where this step's parameters begin in `node_values`.
    std::uint32_t values_offset = 0;

    /// How many doubles it owns, so a writer can range-check without the registry.
    std::uint32_t values_count = 0;

    /// This step's STRING parameter, as an index into `node_plan::strings`, or -1 for none.
    ///
    /// WHY A TABLE AND NOT A `std::string` HERE. `node_step` is read per draw, per step, per
    /// frame, and the whole point of the values array is that a step is a few integers next to
    /// each other -- a `std::string` member would put an allocation and a pointer chase on the
    /// frame path for something only one class has.
    ///
    /// ONE, NOT A RANGE, because the only string a class carries today selects it: an ISF
    /// shader's `path`. When a second one appears this becomes an offset/count pair like the
    /// values, and the shape is deliberately the same so that change is mechanical.
    ///
    /// NOT read by any evaluator yet. It exists so a renderer CAN reach a shader path, which
    /// is the piece both the OpenGL and Vulkan ISF paths block on.
    std::int32_t string_index = -1;
};

/// The compiled graph. Immutable, shared, and compared by pointer.
struct node_plan
{
    /// Every string any step refers to, in compile order.
    ///
    /// HERE RATHER THAN ON THE STEP so a step stays trivially copyable and cheap to walk, and
    /// so two steps naming the same shader share one entry. A plan is immutable and compared by
    /// POINTER, so this table is fixed for the plan's whole life -- which is what makes an
    /// index into it safe to hold on the frame path.
    std::vector<std::string> strings;

    /// The NAME of every value slot, parallel to the values array: `value_names[k]` names
    /// `values[k]` for every k the plan owns.
    ///
    /// FOR A FOREIGN RENDERER, which is the only reader. An ISF shader's parameters are named
    /// by the shader FILE, so the evaluator cannot hand them over as bare numbers -- the module
    /// on the other side of the link boundary has to know which input each slot is. Matching by
    /// POSITION would work today and break silently the first time either side inserted a port:
    /// every parameter would land on the wrong input, the shader would still compile and still
    /// render, and nothing would report it.
    ///
    /// Built at COMPILE time and never on the frame path -- a plan is immutable and compared by
    /// pointer, so this is fixed for its whole life and costs one pointer to index.
    ///
    /// Empty for a plan compiled before any class needed it; a reader must range-check rather
    /// than assume it is as long as the values array.
    std::vector<std::string> value_names;

    graph_stage stage = graph_stage::working;

    /// Topologically ordered. `steps.front()` is the `input` and `steps.back()` the `output`,
    /// which the compiler guarantees so the evaluator needs no search for either.
    std::vector<node_step> steps;

    /// How many steps actually draw. **Zero takes the existing single-draw fast path**, which
    /// is what keeps a graph with everything bypassed byte-identical to no graph at all.
    std::int32_t image_passes = 0;

    /// Reported to the client, not acted on: a legal-but-lossy join the author should see.
    std::vector<coercion> coercions;

    /// Which document this was compiled from, and which revision of it.
    ///
    /// THE NAME IS HERE SO THE EVALUATOR KNOWS WHAT IT IS DRAWING. The mixer sees a tree of
    /// layers and items and has no stage layer index at all -- so a request addressed to "node
    /// `e` of graph `look`" can only be matched inside the frame if the plan carries the name.
    /// Addressing a preview by DOCUMENT rather than by layer is also the better interface, for
    /// the reason `detach` takes no layer: a client knows the look's name, and making it also
    /// remember where the look is attached only gives it something to get wrong.
    std::string  document_name;
    std::int64_t document_revision = 0;

    /// `node/<id>/<param>` -> index into `node_values`. The one map between an ADDRESS and the
    /// flat array, so `resolve_drivers` writes a node parameter with one lookup and no parsing.
    std::unordered_map<std::string, std::uint32_t> value_index;

    /// How many doubles `node_values` must hold for this plan.
    std::uint32_t values_size = 0;
};

/// Compile a validated document. Null if it has `error` faults.
///
/// ALLOCATES, and the pointer it returns IS the fingerprint -- so this runs at PUT and at
/// ATTACH and never in the tick.
std::shared_ptr<const node_plan> compile(const graph_document& doc,
                                         const std::vector<std::string>& order,
                                         const std::vector<graph_fault>& faults);

/// The document's parameter values, flattened in the plan's own layout.
///
/// Separate from `compile` because the two have different lifetimes: the plan is rebuilt when
/// the structure changes and this is rebuilt whenever a value does, which for a slider drag is
/// fifty times a second.
node_values values_of(const graph_document& doc, const node_plan& plan);

/// Aborts the boot on a disagreement between the compiler and its own rules.
void graph_plan_self_test();

}}} // namespace caspar::core::graph
