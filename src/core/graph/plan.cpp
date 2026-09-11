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
    for (const auto& e : doc.edges) {
        feeds[e.to_node + "." + e.to_port] = incoming{e.from_node, e.from_port, e.muted};
        ++fan_out_count[e.from_node];
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
        if (c->group == "mask")
            st.fused_mask = fan_out_count[id] <= 1;

        // Every value input gets a slot, in PORT ORDER, so the shader's uniform upload can walk
        // the class's ports and the offsets agree by construction rather than by a second table.
        for (const auto& port : c->ports) {
            if (port.domain != port_domain::value || port.direction != port_direction::input)
                continue;
            plan->value_index["node/" + id + "/" + port.param.name] = plan->values_size;
            plan->values_size += port.param.arity;
            st.values_count += port.param.arity;
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
        for (const auto& p : c->ports) {
            if (p.direction != port_direction::input)
                continue;
            if (p.domain == port_domain::image && st.in0 == -1)
                st.in0 = resolve(p.param.name.c_str());
            else if (p.domain == port_domain::image && st.in1 == -1)
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

    for (const auto& st : plan->steps)
        if (st.produces_image)
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
        for (const auto& port : c->ports) {
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
    }

    CASPAR_LOG(info) << L"[graph-plan] self-test: all checks passed";
}

}}} // namespace caspar::core::graph
