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

#include "plan.h"

#include "validate.h"

#include <core/producer/producer_params.h>

#include <common/except.h>
#include <common/log.h>

#include <algorithm>
#include <set>
#include <map>

namespace caspar { namespace core { namespace graph {

namespace {

std::int32_t class_index(const std::string& id)
{
    const auto& t = node_classes();
    for (std::size_t i = 0; i < t.size(); ++i)
        if (t[i].id == id)
            return static_cast<std::int32_t>(i);
    return -1;
}

} // namespace

std::shared_ptr<const node_plan> compile(const graph_document&           doc,
                                         const std::vector<std::string>& order,
                                         const std::vector<graph_fault>& faults)
{
    if (has_error(faults) || order.empty())
        return nullptr;

    auto plan               = std::make_shared<node_plan>();
    plan->stage             = doc.stage;
    plan->document_name     = doc.name;
    plan->document_revision = doc.revision;

    // ---- index the document ------------------------------------------------------------
    std::map<std::string, const graph_node*> node_of;
    for (const auto& n : doc.nodes)
        node_of[n.id] = &n;

    /// step index by node id, filled as the order is walked
    std::map<std::string, std::int32_t> step_of;

    // Which input port of which node each edge feeds, and from where. Built once rather than
    // searched per step: a step's three inputs would otherwise be three scans of the edge list.
    struct incoming
    {
        std::string from_node;
        std::string from_port;
        bool        muted = false;
    };
    std::map<std::string, incoming> feeds; // "node.port" -> where it comes from
    std::map<std::string, int>      fan_out_count;

    // WHICH NODES FEED A MASK NODE, which is the third condition on fusability and the one the
    // self-test caught me missing. A fused mask is evaluated by its CONSUMER from the
    // consumer's own uniforms -- so the consumer has to be a draw that does that. A
    // `mask_combine` is not: it is a pass that SAMPLES its inputs. So an ellipse feeding a
    // combine has one consumer and still cannot be fused, and `fan_out_count <= 1` alone says
    // the opposite.
    std::set<std::string> feeds_a_mask;
    for (const auto& e : doc.edges) {
        feeds[e.to_node + "." + e.to_port] = incoming{e.from_node, e.from_port, e.muted};
        ++fan_out_count[e.from_node];
        const auto to = node_of.find(e.to_node);
        if (to != node_of.end()) {
            const auto* tc = find_node_class(to->second->cls);
            if (tc && tc->group == "mask")
                feeds_a_mask.insert(e.from_node);
        }
    }

    // ---- one step per node, in evaluation order ----------------------------------------
    //
    // `input` FIRST AND `output` LAST, which the validator has already guaranteed (exactly one
    // of each, and the output reachable) -- so this places them by construction and the
    // evaluator needs no search for either. Kahn's gives the rest in dependency order.
    plan->steps.reserve(order.size());
    for (const auto& id : order) {
        const auto* n = node_of[id];
        const auto* c = find_node_class(n->cls);
        if (!c)
            return nullptr; // validated, so unreachable -- and a null plan is the safe answer

        node_step st;
        st.cls            = class_index(n->cls);
        st.id             = id;
        st.values_offset  = plan->values_size;
        st.produces_image = c->produces_image && c->id != "input";

        // A MASK GENERATOR IS FUSED when exactly one thing reads it. Analytic by construction in
        // v1 (there is only `mask_ellipse`), so the consumer can evaluate it inline from its own
        // uniforms and the mask costs no pass at all -- which is what makes a windowed grade one
        // draw rather than two. With more than one consumer it would have to be materialised,
        // and that is the commit that adds `mask_combine`.
        if (c->group == "mask") {
            // FUSABLE ONLY IF ANALYTIC, and fan-out is the second condition rather than the
            // only one. A fused mask is evaluated by its CONSUMER from the consumer's own
            // uniforms, so a mask that has to READ something -- `mask_qualifier` keys the
            // pixel, `mask_combine` reads two masks -- cannot be fused at any fan-out. Fusing
            // one would silently ignore its inputs, which is the same shape as the defect
            // commit 8a fixed: a plan saying something the evaluator cannot honour.
            const bool analytic =
                std::none_of(c->ports.begin(), c->ports.end(), [](const port_desc& p) {
                    return p.direction == port_direction::input &&
                           (p.domain == port_domain::image || p.domain == port_domain::mask);
                });
            st.fused_mask =
                analytic && fan_out_count[id] <= 1 && feeds_a_mask.find(id) == feeds_a_mask.end();
            // ...and if it is NOT fused it must be MATERIALISED. The two are exhaustive for a
            // mask, which `graph_plan_self_test` asserts -- because "neither" is exactly the
            // state that shipped: nothing implemented the texture path, so a fanned-out mask
            // became no mask and its consumers graded everywhere.
            st.produces_mask = !st.fused_mask;
        }

        // THE INSTANCE'S PORTS, not the class's. For every class but a dynamic one these are the
        // same list; for `isf` the shader file its `path` names declares the rest, and compiling
        // from the class alone would give those parameters NO slots -- so they would validate,
        // publish and be addressable, and then drive nothing. That is the "accepted everywhere
        // and driven by nothing" shape `graph-stack` exists to catch.
        std::string        ports_reason; // a bad selector is the validator's to report, not ours
        const auto         inst_ports = instance_ports(*c, n->string_param(c->ports_selector), ports_reason);

        // Every value input gets a slot, in PORT ORDER, so the shader's uniform upload can walk
        // the ports and the offsets agree by construction rather than by a second table.
        for (const auto& port : inst_ports) {
            if (port.domain != port_domain::value || port.direction != port_direction::input)
                continue;

            // A STRING PARAMETER HAS NO NUMERIC SLOT. It goes in the plan's string table
            // instead, because `node_values` is an array of doubles and putting a path in one
            // is not a narrowing question -- there is no number to store. Without this the
            // selector would silently consume a slot and every later offset would be one out.
            if (port.param.type == fields::value_type::string) {
                if (port.param.name == c->ports_selector) {
                    const auto value = n->string_param(port.param.name);
                    if (!value.empty()) {
                        st.string_index = static_cast<std::int32_t>(plan->strings.size());
                        plan->strings.push_back(value);
                    }
                }
                continue;
            }

            plan->value_index["node/" + id + "/" + port.param.name] = plan->values_size;
            plan->values_size += port.param.arity;
            st.values_count += port.param.arity;

            // THE SAME NAME, FOR EVERY COMPONENT OF AN ARITY > 1 PORT. A foreign renderer wants
            // the PORT, not the component -- an ISF `point2D` is one input taking two numbers --
            // so the two slots of `center` are both named `center` and the reader takes
            // `arity` of them from the first match. Numbering them `center.0`/`center.1` would
            // make the module parse a name apart that core had just put together.
            //
            // HERE rather than in a second loop, so the name table and the offsets cannot drift:
            // one append per slot, in the same statement that allocated it.
            plan->value_names.insert(plan->value_names.end(), port.param.arity, port.param.name);
        }

        step_of[id] = static_cast<std::int32_t>(plan->steps.size());
        plan->steps.push_back(std::move(st));
    }

    // ---- resolve the inputs, now that every node has a step index -----------------------
    for (std::size_t i = 0; i < plan->steps.size(); ++i) {
        auto&       st = plan->steps[i];
        const auto* c  = &node_classes()[st.cls];

        // A MUTED EDGE IS TREATED AS ABSENT, which is the whole point of mute being a VALUE
        // rather than topology: the plan is identical either way, so muting costs no recompile
        // and a timeline can step it. What an absent input contributes is the port's
        // `disconnected_default` -- never transparent black. A muted edge blacking a layer
        // during a show is the one failure nobody would forgive.
        const auto resolve = [&](const char* port) -> std::int32_t {
            const auto it = feeds.find(st.id + "." + port);
            if (it == feeds.end() || it->second.muted)
                return -1;
            const auto s = step_of.find(it->second.from_node);
            return s == step_of.end() ? -1 : s->second;
        };

        // WHICH PORT IS THE PRIMARY INPUT depends on the class, and it is read off the ports
        // rather than hard-coded: the first required image input. That is what a bypassed node
        // aliases, so getting it wrong makes bypass show the wrong picture.
        // WHAT A MASK INPUT MEANS DEPENDS ON WHETHER THIS NODE IS ITSELF A MASK.
        //
        // For a grade or a combine, the `mask` port is a MODIFIER -- what to multiply by -- and
        // it belongs in `st.mask`, the slot the evaluator reads for a fused or materialised
        // mask. For `mask_combine` the mask ports ARE THE OPERANDS: it has two of them, and
        // `st.mask` is one slot, so routing them there would silently drop `b`.
        //
        // So a mask node's mask inputs take the operand slots `in0`/`in1`, which is also how
        // the evaluator binds them -- `in0` -> plane 0, `in1` -> plane 1 -- and means a combine
        // needs no new plumbing at all.
        const bool is_mask_node = c->group == "mask";
        // INSTANCE PORTS: a shader declaring a SECOND image input must have it routed, or the
        // edge validates (the validator resolves per instance) and the compiler wires nothing --
        // "accepted and does nothing", the same family as a parameter with no value slot.
        // The selector comes from the PLAN'S OWN string table rather than from the document,
        // because the document node is not in scope here -- and that is the table's purpose:
        // once compiled, a step carries everything the evaluator and the later passes need.
        const auto  routing_selector =
            st.string_index >= 0 && static_cast<std::size_t>(st.string_index) < plan->strings.size()
                ? plan->strings[st.string_index]
                : std::string{};
        std::string routing_reason;
        const auto  routing_ports = instance_ports(*c, routing_selector, routing_reason);
        for (const auto& p : routing_ports) {
            if (p.direction != port_direction::input)
                continue;
            const bool operand = p.domain == port_domain::image ||
                                 (is_mask_node && p.domain == port_domain::mask);
            if (operand && st.in0 == -1)
                st.in0 = resolve(p.param.name.c_str());
            else if (operand && st.in1 == -1)
                st.in1 = resolve(p.param.name.c_str());
            else if (p.domain == port_domain::mask && st.mask == -1)
                st.mask = resolve(p.param.name.c_str());
        }
    }

    // ---- `last_use`, walked BACKWARDS ---------------------------------------------------
    //
    // The last step that reads each output. Computed here because the evaluator must not search
    // forwards per step per frame -- and because an attachment released one step too early is a
    // garbage read that presents as a maths bug in a node nobody edited.
    for (std::int32_t i = static_cast<std::int32_t>(plan->steps.size()) - 1; i >= 0; --i) {
        const auto& st = plan->steps[i];
        for (const auto src : {st.in0, st.in1, st.mask}) {
            if (src < 0)
                continue;
            auto& producer = plan->steps[src];
            if (producer.last_use < i)
                producer.last_use = i;
        }
    }

    // A MATERIALISED MASK COSTS A PASS TOO, and the cap has to see it -- otherwise a document
    // of sixteen image nodes plus fanned-out masks passes validation and then asks the frame
    // path for more draws than the cap promised.
    for (const auto& st : plan->steps)
        if (st.produces_image || st.produces_mask)
            ++plan->image_passes;

    for (const auto& f : faults)
        if (f.sev == graph_fault::severity::coercion)
            plan->coercions.push_back(coercion{true, false, "", "", f.reason});

    return plan;
}

node_values values_of(const graph_document& doc, const node_plan& plan)
{
    node_values out(plan.values_size, 0.0);

    for (const auto& n : doc.nodes) {
        const auto* c = find_node_class(n.cls);
        if (!c)
            continue;
        // INSTANCE PORTS AGAIN, and this one fills the array the evaluator reads. The slots are
        // allocated per instance above; if the DEFAULTS were written from the class, a shader
        // parameter would keep whatever the array was initialised to rather than its declared
        // default -- a node that renders at zero until something writes it.
        std::string init_reason;
        const auto  init_ports = instance_ports(*c, n.string_param(c->ports_selector), init_reason);
        for (const auto& port : init_ports) {
            if (port.domain != port_domain::value || port.direction != port_direction::input)
                continue;
            const auto it = plan.value_index.find("node/" + n.id + "/" + port.param.name);
            if (it == plan.value_index.end())
                continue;

            // ABSENT MEANS AT ITS DEFAULT -- the document's `params` is sparse, and the
            // descriptor is what fills the gap. That is the same rule the publication and
            // `describe_graph` follow, and it has to be the same in all three or a parameter
            // nobody has touched would render as zero.
            const auto  pv = n.params.find(port.param.name);
            const auto& v  = pv == n.params.end() ? port.param.default_value : pv->second;
            for (std::uint8_t k = 0; k < port.param.arity && k < v.size(); ++k) {
                double d = 0;
                if (as_number(v[k], d))
                    out[it->second + k] = d;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------
// THE BOOT SELF-TEST
//
// What a compiler gets wrong is not usually the topology -- the validator has already sorted
// it -- but the bookkeeping: an offset that overlaps, a `last_use` one step early, a bypassed
// node aliasing the wrong input, a muted edge that still counts as connected. Each of those is
// silent and each renders a plausible wrong picture.
// ---------------------------------------------------------------------------------------

void graph_plan_self_test()
{
    const auto fail = [](const std::string& what) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("graph plan self-test: " + what));
    };

    const auto node = [](std::string id, std::string cls) {
        graph_node n;
        n.id  = std::move(id);
        n.cls = std::move(cls);
        return n;
    };
    const auto edge = [](std::string id, std::string fn, std::string fp, std::string tn,
                         std::string tp, bool muted = false) {
        graph_edge e;
        e.id        = std::move(id);
        e.from_node = std::move(fn);
        e.from_port = std::move(fp);
        e.to_node   = std::move(tn);
        e.to_port   = std::move(tp);
        e.muted     = muted;
        return e;
    };
    const auto build = [&](graph_document d) {
        std::vector<std::string> order;
        const auto               fs = validate(d, order);
        if (has_error(fs))
            fail("a fixture this test depends on does not validate: " + fs.front().reason);
        return compile(d, order, fs);
    };

    // ---- a chain: input -> exposure -> cdl -> output ----------------------------------
    {
        graph_document d;
        d.name  = "chain";
        d.nodes = {node("i", "input"), node("e", "exposure"), node("c", "cdl"),
                   node("o", "output")};
        d.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "e", "out", "c", "in"),
                   edge("e3", "c", "out", "o", "in")};
        const auto p = build(d);
        if (!p)
            fail("a valid chain did not compile");
        if (p->steps.size() != 4)
            fail("a four-node chain produced " + std::to_string(p->steps.size()) + " steps");
        if (p->steps.front().id != "i" || p->steps.back().id != "o")
            fail("the compiler must place `input` first and `output` last, so the evaluator "
                 "needs no search for either -- got " +
                 p->steps.front().id + ".." + p->steps.back().id);
        // TWO passes, not four: `input` is the head pass's own output and `output` is the tail.
        if (p->image_passes != 2)
            fail("a chain of two grading nodes is " + std::to_string(p->image_passes) +
                 " image passes; it must be 2 -- `input` and `output` draw nothing of their own");

        // EVERY OFFSET DISTINCT AND IN RANGE. An overlap is the defect that makes two
        // parameters share a slider, and it is invisible until somebody drags one.
        std::vector<bool> seen(p->values_size, false);
        for (const auto& kv : p->value_index) {
            if (kv.second >= p->values_size)
                fail("value offset for " + kv.first + " is past the end of the array");
            if (seen[kv.second])
                fail("two parameters share the offset " + std::to_string(kv.second) +
                     " -- one of them is " + kv.first);
            seen[kv.second] = true;
        }
        // THE NAME TABLE IS EXACTLY AS LONG AS THE VALUES ARRAY, asserted at BOOT because a
        // reader indexes one with the other's offset. A short table is an out-of-range read on
        // the frame path; a long one means a slot was named twice and every name after it is
        // shifted -- which for a foreign renderer is every parameter on the wrong input, still
        // compiling and still rendering. Neither is visible from a rendered picture.
        if (p->value_names.size() != p->values_size)
            fail("the plan names " + std::to_string(p->value_names.size()) + " value slots but "
                 "owns " + std::to_string(p->values_size) +
                 " -- a foreign renderer indexes one by the other, so they cannot differ");
        for (const auto& kv : p->value_index) {
            const auto slash = kv.first.rfind('/');
            if (slash == std::string::npos || kv.second >= p->value_names.size())
                continue;
            if (p->value_names[kv.second] != kv.first.substr(slash + 1))
                fail("slot " + std::to_string(kv.second) + " is addressed as " + kv.first +
                     " and named `" + p->value_names[kv.second] +
                     "` -- the address and the name must be the same port");
        }

        if (p->value_index.find("node/e/gain") == p->value_index.end())
            fail("`node/e/gain` has no slot, so nothing could drive it");
        if (p->value_index.find("node/c/slope") == p->value_index.end())
            fail("`node/c/slope` has no slot");

        // `last_use`: the exposure's output is read by the cdl (step 2) and nothing later.
        const auto ei = std::find_if(p->steps.begin(), p->steps.end(),
                                     [](const node_step& s) { return s.id == "e"; });
        const auto ci = std::find_if(p->steps.begin(), p->steps.end(),
                                     [](const node_step& s) { return s.id == "c"; });
        if (ei->last_use != static_cast<std::int32_t>(ci - p->steps.begin()))
            fail("the exposure's last_use is " + std::to_string(ei->last_use) +
                 ", and the only thing reading it is the cdl at step " +
                 std::to_string(ci - p->steps.begin()) +
                 " -- an attachment released one step early is a garbage read");
    }

    // ---- values_of: sparse params, defaults filled from the DESCRIPTOR ------------------
    {
        graph_document d;
        d.name  = "vals";
        d.nodes = {node("i", "input"), node("e", "exposure"), node("o", "output")};
        d.nodes[1].params["gain"] = monitor::vector_t{3.0};
        d.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "e", "out", "o", "in")};
        const auto p = build(d);
        const auto v = values_of(d, *p);
        if (v.size() != p->values_size)
            fail("values_of produced " + std::to_string(v.size()) + " for a plan wanting " +
                 std::to_string(p->values_size));
        if (v[p->value_index.at("node/e/gain")] != 3.0)
            fail("the document's own value did not reach the array");
        // `mix` DEFAULTS TO 1.0 and is absent from the document. Zero here would make every
        // grading node a no-op the moment somebody did not set its mix -- the same class as a
        // forgotten mask reading 0 instead of 1.
        if (v[p->value_index.at("node/e/mix")] != 1.0)
            fail("an ABSENT parameter must come from its DESCRIPTOR's default, not from zero: "
                 "`mix` read " +
                 std::to_string(v[p->value_index.at("node/e/mix")]) + " and its default is 1.0");
    }

    // ---- A MUTED EDGE IS ABSENT, and the plan is otherwise IDENTICAL -------------------
    //
    // The identity is the point: mute is a VALUE, so it must not change the plan, or a timeline
    // stepping it would reallocate a graph mid-show.
    {
        graph_document d;
        d.name  = "mute";
        d.nodes = {node("i", "input"), node("a", "exposure"), node("b", "exposure"),
                   node("m", "mix"), node("o", "output")};
        d.edges = {edge("e1", "i", "out", "a", "in"), edge("e2", "i", "out", "b", "in"),
                   edge("e3", "a", "out", "m", "a"), edge("e4", "b", "out", "m", "b"),
                   edge("e5", "m", "out", "o", "in")};
        const auto live = build(d);

        auto muted     = d;
        muted.edges[3] = edge("e4", "b", "out", "m", "b", /*muted*/ true);
        const auto mp  = build(muted);

        if (!live || !mp)
            fail("the mute fixtures did not compile");
        if (live->steps.size() != mp->steps.size() || live->values_size != mp->values_size ||
            live->image_passes != mp->image_passes)
            fail("muting an edge changed the PLAN: " + std::to_string(live->steps.size()) + "/" +
                 std::to_string(live->image_passes) + " became " +
                 std::to_string(mp->steps.size()) + "/" + std::to_string(mp->image_passes) +
                 ". Mute is a VALUE -- if it changes the plan, a timeline stepping it would "
                 "reallocate a graph mid-show");

        const auto mix_of = [](const std::shared_ptr<const node_plan>& p) {
            return *std::find_if(p->steps.begin(), p->steps.end(),
                                 [](const node_step& s) { return s.id == "m"; });
        };
        if (mix_of(live).in1 < 0)
            fail("the live fixture's `mix.b` has no input, so the muted case proves nothing");
        if (mix_of(mp).in1 != -1)
            fail("a MUTED edge still resolved as connected. An unconnected `mix.b` returns `a`; "
                 "a connected one mixes toward it, so this is the difference between a muted "
                 "branch leaving the picture alone and it changing the picture");
    }

    // ---- A FUSED MASK COSTS NO PASS, and a fanned-out one is not fused ------------------
    {
        graph_document d;
        d.name  = "mask";
        d.nodes = {node("i", "input"), node("k", "mask_ellipse"), node("e", "exposure"),
                   node("o", "output")};
        d.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "k", "out", "e", "mask"),
                   edge("e3", "e", "out", "o", "in")};
        const auto p = build(d);
        const auto k = std::find_if(p->steps.begin(), p->steps.end(),
                                    [](const node_step& s) { return s.id == "k"; });
        if (!k->fused_mask)
            fail("a mask with ONE consumer must be fused -- inline in the consumer's uniforms, "
                 "which is what makes a windowed grade one draw rather than two");
        if (p->image_passes != 1)
            fail("a masked exposure is " + std::to_string(p->image_passes) +
                 " image passes; a fused mask costs none, so it must be 1");

        auto two = d;
        two.nodes.push_back(node("e2n", "exposure"));
        two.edges.push_back(edge("e4", "i", "out", "e2n", "in"));
        two.edges.push_back(edge("e5", "k", "out", "e2n", "mask"));
        two.edges.push_back(edge("e6", "e2n", "out", "o", "in"));
        two.edges.erase(two.edges.begin() + 2); // the old e->o edge; `o` takes one input
        const auto tp = build(two);
        const auto tk = std::find_if(tp->steps.begin(), tp->steps.end(),
                                     [](const node_step& s) { return s.id == "k"; });
        if (tk->fused_mask)
            fail("a mask read by TWO consumers cannot be fused: it would be evaluated twice "
                 "from two different uniform sets, and the two would drift the moment either "
                 "consumer's own parameters differed");

        // AND IT MUST THEREFORE BE MATERIALISED. This half is the one that was missing, and it
        // was missing in the worst possible way: the plan said "not fused" correctly, and
        // NOTHING said what to do instead, so the evaluator dropped the mask and both
        // consumers graded the whole image. `grade-graph` measures the picture; this asserts
        // the plan can never again say "not fused" without also saying "materialised".
        if (!tk->produces_mask)
            fail("a mask that is not FUSED must be MATERIALISED -- the two are exhaustive. "
                 "Neither is what shipped, and it renders as no mask at all");
        if (tp->image_passes != 3)
            fail("two masked exposures sharing one mask is " + std::to_string(tp->image_passes) +
                 " passes; it must be 3 -- two exposures plus the materialised mask, which the "
                 "16-pass cap has to count");

        for (const auto& s2 : p->steps)
            if (s2.fused_mask && s2.produces_mask)
                fail("step '" + s2.id + "' claims to be both fused and materialised");
    }

    // ---- a mask that has to READ something can never be fused ---------------------------
    //
    // `mask_qualifier` keys the pixel and `mask_combine` reads two masks, so neither can be
    // evaluated from a consumer's uniforms at ANY fan-out. Asserted at ONE consumer, which is
    // the case fan-out alone would have fused -- the condition that actually discriminates.
    {
        graph_document d;
        d.name  = "combine";
        d.nodes = {node("i", "input"),  node("k1", "mask_ellipse"), node("k2", "mask_ellipse"),
                   node("cm", "mask_combine"), node("e", "exposure"), node("o", "output")};
        d.edges = {edge("e1", "i", "out", "e", "in"),
                   edge("e2", "k1", "out", "cm", "a"),
                   edge("e3", "k2", "out", "cm", "b"),
                   edge("e4", "cm", "out", "e", "mask"),
                   edge("e5", "e", "out", "o", "in")};
        const auto p2 = build(d);
        if (!p2)
            fail("a document combining two masks did not compile");

        const auto cm = std::find_if(p2->steps.begin(), p2->steps.end(),
                                     [](const node_step& s) { return s.id == "cm"; });
        if (cm->fused_mask)
            fail("`mask_combine` was FUSED with one consumer -- it reads two mask textures, so "
                 "fusing it means evaluating it from uniforms alone and silently ignoring both "
                 "of its inputs");
        if (!cm->produces_mask)
            fail("`mask_combine` must be MATERIALISED: it is a mask and it is not fusable");

        // ITS OPERANDS MUST BE IN in0/in1, not in `mask`. There is one `mask` slot and a
        // combine has two inputs, so routing them as modifiers drops `b` with no error.
        if (cm->in0 < 0 || cm->in1 < 0)
            fail("`mask_combine`'s two mask inputs must land in in0/in1 -- one of them is "
                 "unresolved, which renders as a combine of one mask");
        if (cm->in0 == cm->in1)
            fail("`mask_combine`'s two inputs resolved to the SAME step");

        // The two ellipses feed only the combine, so they are fused into IT -- no. They feed a
        // node that cannot evaluate them inline, so they must be materialised as well: three
        // mask passes plus the exposure.
        const auto k1 = std::find_if(p2->steps.begin(), p2->steps.end(),
                                     [](const node_step& s) { return s.id == "k1"; });
        if (k1->fused_mask)
            fail("an ellipse feeding a `mask_combine` cannot be fused: its consumer is not a "
                 "draw that evaluates mask uniforms, it is a pass that SAMPLES its inputs");
    }

    CASPAR_LOG(info) << L"[graph-plan] self-test: all checks passed";
}

}}} // namespace caspar::core::graph
