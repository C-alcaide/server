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

#include "validate.h"

#include "registry.h"

#include <core/producer/producer_params.h>

#include <common/except.h>
#include <common/log.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>

namespace caspar { namespace core { namespace graph {

namespace {

/// At most this many IMAGE-producing steps. Refused rather than clamped.
///
/// INHERITED FROM THE PROTOTYPE'S HELP TEXT and not measured -- the prototype advertised 16 nodes
/// and this keeps that promise. `grade-graph-cost` is what will give a number to set it from; the
/// cap exists now because an unbounded graph can exhaust the attachment pool on the frame path,
/// which is a worse failure than a refusal at PUT.
constexpr int max_image_steps = 16;

graph_fault err(std::string node, std::string edge, std::string port, std::string reason)
{
    graph_fault f;
    f.sev    = graph_fault::severity::error;
    f.node   = std::move(node);
    f.edge   = std::move(edge);
    f.port   = std::move(port);
    f.reason = std::move(reason);
    return f;
}

graph_fault warn(std::string node, std::string edge, std::string port, std::string reason)
{
    auto f = err(std::move(node), std::move(edge), std::move(port), std::move(reason));
    f.sev  = graph_fault::severity::coercion;
    return f;
}

/// Does an id contain a character the ADDRESS GRAMMAR splits on?
///
/// `node/<id>/<param>` is split on '/', and `<param>.N` on '.', so an id carrying either would
/// make its own address ambiguous -- and an ambiguous address is a write that silently lands
/// somewhere else. Refused at PUT rather than escaped, because escaping would put a quoting rule
/// into a grammar that has none.
bool id_is_addressable(const std::string& id)
{
    return !id.empty() && id.find('/') == std::string::npos && id.find('.') == std::string::npos;
}

/// Check one parameter value against its port. The core-side half of the contract
/// `check_and_bound` keeps on the HTTP side -- see `validate.h` for why there are two.
void check_param(const std::string& node_id, const port_desc& p, const monitor::vector_t& v,
                 std::vector<graph_fault>& out)
{
    if (p.direction == port_direction::output) {
        out.push_back(err(node_id, "", p.param.name,
                          "'" + p.param.name + "' is an OUTPUT and carries no value to set"));
        return;
    }
    if (p.domain != port_domain::value) {
        out.push_back(err(node_id, "", p.param.name,
                          "'" + p.param.name + "' is " + domain_name(p.domain) +
                              " and takes an EDGE, not a value"));
        return;
    }
    if (v.size() != p.param.arity) {
        out.push_back(err(node_id, "", p.param.name,
                          "'" + p.param.name + "' takes " + std::to_string(p.param.arity) +
                              " component(s) and was given " + std::to_string(v.size())));
        return;
    }
    if (!p.param.min || !p.param.max)
        return; // no declared range: nothing to enforce, exactly as `bounding_t::free` means

    for (std::size_t i = 0; i < v.size(); ++i) {
        double d = 0.0;
        if (!as_number(v[i], d))
            continue; // an enumeration given by NAME: the class's own table validates it
        // `wrap` normalises rather than refusing, which is what it means: 400 degrees is a legal
        // way to say 40. Nothing in v1's registry declares it; the arm is here so the first port
        // that does gets the same answer as the mixer registry gives.
        if (p.param.bounding == fields::bounding_t::wrap) {
            const auto span = *p.param.max - *p.param.min;
            if (span > 0.0)
                d = *p.param.min + std::fmod(std::fmod(d - *p.param.min, span) + span, span);
        }
        if (d < *p.param.min || d > *p.param.max) {
            out.push_back(err(node_id, "", p.param.name,
                              "'" + p.param.name + "' component " + std::to_string(i) +
                                  " is " + std::to_string(d) + ", outside " +
                                  std::to_string(*p.param.min) + ".." +
                                  std::to_string(*p.param.max)));
            return; // one fault per parameter: a vec3 out of range on all three is one mistake
        }
    }
}

} // namespace

bool has_error(const std::vector<graph_fault>& fs)
{
    return std::any_of(fs.begin(), fs.end(), [](const graph_fault& f) { return f.is_error(); });
}

std::vector<graph_fault> validate(const graph_document& doc, std::vector<std::string>& topo_order)
{
    std::vector<graph_fault> out;
    topo_order.clear();

    // ---- nodes: ids, classes, ports, parameter values ---------------------------------
    std::map<std::string, const graph_node*>  by_id;
    std::map<std::string, const node_class*>  cls_of;
    //: The RESOLVED port list per node, so the edge pass below sees the same ports the
    //: parameter pass did. A dynamic class resolved twice could otherwise disagree with itself
    //: -- a shader file replaced between the two passes would make an edge dangle against a
    //: parameter that had just validated.
    std::map<std::string, std::vector<port_desc>> ports_of;
    /// Nodes that exist and whose class does not. Their edges are left alone: the class fault is
    /// the one cause, and repeating it per edge would make one typo look like several problems.
    std::set<std::string>                     unclassified;
    for (const auto& n : doc.nodes) {
        if (!id_is_addressable(n.id)) {
            out.push_back(err(n.id, "", "",
                              n.id.empty()
                                  ? "a node has no id. Ids are the client's and are required: an "
                                    "id is what `node/<id>/<param>` addresses, so it is the one "
                                    "thing that has to survive an edit"
                                  : "node id '" + n.id +
                                        "' contains '/' or '.', which the address grammar splits "
                                        "on -- its own parameters would not be addressable"));
            continue;
        }
        if (!by_id.emplace(n.id, &n).second) {
            out.push_back(err(n.id, "", "",
                              "two nodes share the id '" + n.id +
                                  "', which makes every edge and every address naming it "
                                  "ambiguous"));
            continue;
        }
        const auto* c = find_node_class(n.cls);
        if (!c) {
            out.push_back(err(n.id, "", "",
                              "no node class '" + n.cls +
                                  "'. The installed classes are listed at /v1/catalog/node"));
            // KNOWN BUT UNCLASSIFIED, and recording it matters for the MESSAGES rather than the
            // verdict. Without this, every edge touching the node also reported "no node 'b' to
            // take an edge to" -- which is false and misleading: the node exists, its CLASS does
            // not, and an editor told that would highlight three things for one typo. One cause,
            // one fault to highlight.
            unclassified.insert(n.id);
            continue;
        }
        cls_of[n.id] = c;

        // THE PORTS OF THIS INSTANCE. For every class but a dynamic one this is the class's own
        // list; for a dynamic one it comes from the file its selector names.
        //
        // ONE FAULT FOR A BAD SELECTOR, NOT ONE PER PORT. A shader path that cannot be read
        // makes every parameter on that node unknown and every edge to it dangle, and reporting
        // all of them would have an editor highlight a dozen things for one typo -- the same
        // mistake the unclassified-node case above exists to avoid. So the selector's own
        // failure is reported once and the node is then checked against the static ports it
        // still has.
        std::string ports_reason;
        auto        inst = instance_ports(*c, n.string_param(c->ports_selector), ports_reason);
        if (!ports_reason.empty())
            out.push_back(err(n.id, "", c->ports_selector, ports_reason));
        ports_of[n.id] = inst;

        for (const auto& kv : n.params) {
            const auto* p = find_port_in(inst, kv.first);
            if (!p) {
                out.push_back(err(n.id, "", kv.first,
                                  "class '" + n.cls + "' has no port '" + kv.first + "'"));
                continue;
            }
            check_param(n.id, *p, kv.second, out);
        }
    }

    // ---- the roots --------------------------------------------------------------------
    //
    // EXACTLY ONE EACH. Two outputs would mean two answers to "what does the layer draw"; none
    // means the graph produces nothing and every pass is wasted.
    int inputs = 0, outputs = 0;
    for (const auto& kv : cls_of) {
        if (kv.second->id == "input")
            ++inputs;
        else if (kv.second->id == "output")
            ++outputs;
    }
    if (inputs != 1)
        out.push_back(err("", "", "",
                          "a graph needs exactly one `input` node and has " +
                              std::to_string(inputs) +
                              ". It is a node rather than an implicit edge so a client can draw "
                              "where the picture enters"));
    if (outputs != 1)
        out.push_back(err("", "", "",
                          "a graph needs exactly one `output` node and has " +
                              std::to_string(outputs) +
                              ". Two would be two answers to what the layer draws"));

    // ---- edges: ids, endpoints, direction, domain, fan-in ------------------------------
    std::set<std::string>                              edge_ids;
    std::map<std::string, std::string>                 input_taken; // "node.port" -> edge id
    std::map<std::string, std::vector<std::string>>    succ;        // node -> nodes it feeds
    std::map<std::string, int>                         indeg;
    for (const auto& kv : cls_of)
        indeg[kv.first] = 0;

    for (const auto& e : doc.edges) {
        // An edge with no id cannot be reached here: the codec assigns `e<N>` before validating,
        // precisely so a fault has something to point at. Checked anyway, because `validate` is
        // also called on documents the codec did not build (the self-test, and `preview`).
        if (e.id.empty()) {
            out.push_back(err("", "", "", "an edge has no id, so no fault could name it"));
            continue;
        }
        if (!edge_ids.insert(e.id).second) {
            out.push_back(err("", e.id, "", "two edges share the id '" + e.id + "'"));
            continue;
        }

        // An edge into or out of a node whose class is unknown is SILENT here. The class fault
        // already names the cause; a second fault saying the node does not exist would be both
        // wrong and duplicated per edge.
        if (unclassified.count(e.from_node) || unclassified.count(e.to_node))
            continue;

        const auto fn = cls_of.find(e.from_node);
        const auto tn = cls_of.find(e.to_node);
        if (fn == cls_of.end()) {
            out.push_back(err("", e.id, "", "no node '" + e.from_node + "' to take an edge from"));
            continue;
        }
        if (tn == cls_of.end()) {
            out.push_back(err("", e.id, "", "no node '" + e.to_node + "' to take an edge to"));
            continue;
        }
        // RESOLVED PORTS ON BOTH ENDS, so an edge to a shader's own input is checked against
        // what that shader declares rather than against the class's static list.
        const auto* fp = find_port_in(ports_of[e.from_node], e.from_port);
        const auto* tp = find_port_in(ports_of[e.to_node], e.to_port);
        if (!fp) {
            out.push_back(err("", e.id, e.from_port,
                              "class '" + fn->second->id + "' has no port '" + e.from_port + "'"));
            continue;
        }
        if (!tp) {
            out.push_back(err("", e.id, e.to_port,
                              "class '" + tn->second->id + "' has no port '" + e.to_port + "'"));
            continue;
        }

        // DIRECTION, checked rather than inferred, so a client that drew the edge backwards is
        // told which end is which instead of getting a domain error about the same join.
        if (fp->direction != port_direction::output) {
            out.push_back(err("", e.id, e.from_port,
                              "'" + e.from_node + "." + e.from_port +
                                  "' is an INPUT; an edge comes FROM an output"));
            continue;
        }
        if (tp->direction != port_direction::input) {
            out.push_back(err("", e.id, e.to_port,
                              "'" + e.to_node + "." + e.to_port +
                                  "' is an OUTPUT; an edge goes TO an input"));
            continue;
        }

        // FAN-IN IS ONE. Fan-out is unlimited -- that is what makes this a graph rather than a
        // chain -- but two edges into one input have no defined order, and picking one silently
        // would make the picture depend on the order the client happened to send them in.
        const auto key = e.to_node + "." + e.to_port;
        const auto it  = input_taken.find(key);
        if (it != input_taken.end()) {
            out.push_back(err("", e.id, e.to_port,
                              "'" + key + "' already takes edge '" + it->second +
                                  "'. An input takes ONE edge: two have no defined order. Fan-OUT "
                                  "from an output is unlimited"));
            continue;
        }
        input_taken[key] = e.id;

        // DOMAIN, through the ONE coercion table.
        const auto co = coerce(*fp, *tp);
        if (!co.legal) {
            out.push_back(err("", e.id, e.to_port,
                              "cannot join " + std::string(domain_name(fp->domain)) + " to " +
                                  domain_name(tp->domain) + ": " + co.note));
            continue;
        }
        if (!co.exact)
            out.push_back(warn("", e.id, e.to_port,
                               std::string(domain_name(fp->domain)) + " into " +
                                   domain_name(tp->domain) + ": " + co.note));

        succ[e.from_node].push_back(e.to_node);
        ++indeg[e.to_node];
    }

    // ---- required inputs --------------------------------------------------------------
    for (const auto& kv : cls_of) {
        for (const auto& p : kv.second->ports) {
            if (p.direction != port_direction::input || !p.required)
                continue;
            if (!input_taken.count(kv.first + "." + p.param.name))
                out.push_back(err(kv.first, "", p.param.name,
                                  "'" + p.param.name + "' is required and nothing is connected to "
                                  "it. An unconnected required input is an error rather than a "
                                  "default: there is no picture to default TO"));
        }
    }

    // ---- cycles, by Kahn's, and the fault NAMES the edge that closes the loop -----------
    //
    // Sorted at PUT and never in the tick. A cycle found on air is a hang or a garbage frame; a
    // cycle found at PUT is a message with an edge id in it.
    if (!has_error(out)) {
        auto                     deg = indeg;
        std::vector<std::string> ready;
        for (const auto& kv : deg)
            if (kv.second == 0)
                ready.push_back(kv.first);
        // Sorted so the order is deterministic for the same document rather than depending on
        // map iteration -- a client comparing two `order` arrays should see a difference only
        // when the graph differs.
        std::sort(ready.begin(), ready.end());

        while (!ready.empty()) {
            const auto id = ready.front();
            ready.erase(ready.begin());
            topo_order.push_back(id);
            for (const auto& s : succ[id]) {
                if (--deg[s] == 0) {
                    ready.push_back(s);
                    std::sort(ready.begin(), ready.end());
                }
            }
        }

        if (topo_order.size() != cls_of.size()) {
            // Whatever is left has a non-zero in-degree, so every one of them is in or below a
            // cycle. The edge to name is one whose BOTH ends are still unplaced -- that is an
            // edge of the cycle itself rather than one merely downstream of it, which is the
            // difference between "delete this" and "look somewhere upstream".
            std::set<std::string> placed(topo_order.begin(), topo_order.end());
            std::string           named;
            for (const auto& e : doc.edges) {
                if (!placed.count(e.from_node) && !placed.count(e.to_node) &&
                    cls_of.count(e.from_node) && cls_of.count(e.to_node)) {
                    named = e.id;
                    break;
                }
            }
            out.push_back(err("", named, "",
                              "the graph has a cycle: " +
                                  std::to_string(cls_of.size() - topo_order.size()) +
                                  " node(s) cannot be ordered" +
                                  (named.empty() ? std::string()
                                                 : ", and edge '" + named +
                                                       "' is one of the edges closing it")));
            topo_order.clear();
        }
    }

    // ---- the output must be REACHABLE --------------------------------------------------
    //
    // Separate from "it exists". An output with an edge into it can still be fed by a subgraph
    // the input never reaches, which renders black -- and black is exactly what a muted edge is
    // designed never to produce, so it cannot be the answer here either.
    if (!has_error(out) && !topo_order.empty()) {
        std::string input_id, output_id;
        for (const auto& kv : cls_of) {
            if (kv.second->id == "input")
                input_id = kv.first;
            else if (kv.second->id == "output")
                output_id = kv.first;
        }
        std::set<std::string>    seen{input_id};
        std::vector<std::string> stack{input_id};
        while (!stack.empty()) {
            const auto id = stack.back();
            stack.pop_back();
            for (const auto& s : succ[id])
                if (seen.insert(s).second)
                    stack.push_back(s);
        }
        if (!seen.count(output_id))
            out.push_back(err(output_id, "", "",
                              "the `output` node is not reachable from `input`, so the layer would "
                              "render black. A path with no live route renders the input "
                              "unchanged, which is only possible if there IS a path"));
    }

    // ---- the pass cap ------------------------------------------------------------------
    if (!has_error(out)) {
        int passes = 0;
        for (const auto& id : topo_order) {
            const auto* c = cls_of[id];
            if (c->produces_image && c->id != "input")
                ++passes;
        }
        if (passes > max_image_steps)
            out.push_back(err("", "", "",
                              std::to_string(passes) + " image-producing nodes; the cap is " +
                                  std::to_string(max_image_steps) +
                                  ". REFUSED rather than clamped: a clamped graph renders "
                                  "something nobody authored"));
    }

    if (has_error(out))
        topo_order.clear();
    return out;
}

// ---------------------------------------------------------------------------------------
// THE BOOT SELF-TEST
//
// Each case is one thing the validator must catch, written as the smallest document that shows
// it. The reason this aborts the boot rather than living in a unit test: `validate` is the only
// thing standing between a typo and a graph that stores, answers 200 and renders nothing, and
// that failure is discovered on air.
// ---------------------------------------------------------------------------------------

void graph_validate_self_test()
{
    const auto fail = [](const std::string& what) {
        CASPAR_THROW_EXCEPTION(caspar_exception()
                               << msg_info("graph validate self-test: " + what));
    };

    const auto node = [](std::string id, std::string cls) {
        graph_node n;
        n.id  = std::move(id);
        n.cls = std::move(cls);
        return n;
    };
    const auto edge = [](std::string id, std::string fn, std::string fp, std::string tn,
                         std::string tp) {
        graph_edge e;
        e.id        = std::move(id);
        e.from_node = std::move(fn);
        e.from_port = std::move(fp);
        e.to_node   = std::move(tn);
        e.to_port   = std::move(tp);
        return e;
    };

    // The minimal working document, and everything below is a mutation of it. If this one does
    // not validate, every other case's failure is uninterpretable -- so it is checked first.
    graph_document base;
    base.name  = "t";
    base.nodes = {node("i", "input"), node("e", "exposure"), node("o", "output")};
    base.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "e", "out", "o", "in")};

    std::vector<std::string> order;
    {
        const auto fs = validate(base, order);
        if (has_error(fs))
            fail("the minimal input->exposure->output document does not validate: " +
                 fs.front().reason);
        if (order.size() != 3 || order.front() != "i" || order.back() != "o")
            fail("topological order is wrong: expected i..o, got " +
                 std::to_string(order.size()) + " entries");
    }

    const auto expect_error = [&](const char* what, const graph_document& d,
                                  const char* must_name) {
        std::vector<std::string> o;
        const auto               fs = validate(d, o);
        if (!has_error(fs))
            fail(std::string(what) + ": accepted, and it must not be");
        if (!o.empty())
            fail(std::string(what) + ": an order was produced for a document with an error");
        if (must_name && *must_name) {
            const bool named = std::any_of(fs.begin(), fs.end(), [&](const graph_fault& f) {
                return f.node == must_name || f.edge == must_name || f.port == must_name;
            });
            if (!named)
                fail(std::string(what) + ": the fault does not name '" + must_name +
                     "', so a client cannot highlight it");
        }
    };

    { // an unknown class
        auto d = base;
        d.nodes[1].cls = "expsure";
        expect_error("an unknown class", d, "e");
    }
    { // an unknown port on an edge
        auto d = base;
        d.edges[0].to_port = "inn";
        expect_error("an unknown port", d, "inn");
    }
    { // an unknown parameter
        auto d = base;
        d.nodes[1].params["gian"] = monitor::vector_t{1.0};
        expect_error("an unknown parameter", d, "gian");
    }
    { // a parameter out of its declared range
        auto d = base;
        d.nodes[1].params["gain"] = monitor::vector_t{1e6};
        expect_error("a parameter out of range", d, "gain");
    }
    { // a parameter of the wrong arity
        auto d = base;
        d.nodes[1].params["gain"] = monitor::vector_t{1.0, 2.0};
        expect_error("a parameter of the wrong arity", d, "gain");
    }
    { // duplicate node ids
        auto d = base;
        d.nodes.push_back(node("e", "exposure"));
        expect_error("a duplicate node id", d, "e");
    }
    { // an id the address grammar cannot express
        auto d         = base;
        d.nodes[1].id  = "a/b";
        d.edges[0].to_node = "a/b";
        d.edges[1].from_node = "a/b";
        expect_error("a node id containing '/'", d, "a/b");
    }
    { // an edge drawn backwards
        auto d = base;
        d.edges[0] = edge("e1", "e", "in", "i", "out");
        expect_error("an edge from an input", d, "e1");
    }
    { // two edges into one input
        auto d = base;
        d.nodes.push_back(node("e3", "exposure"));
        d.edges.push_back(edge("e3a", "i", "out", "e3", "in"));
        d.edges.push_back(edge("e3b", "e3", "out", "o", "in"));
        expect_error("two edges into one input", d, "e3b");
    }
    { // a required input left unconnected
        auto d = base;
        d.edges.erase(d.edges.begin());
        expect_error("an unconnected required input", d, "in");
    }
    { // no output
        auto d = base;
        d.nodes.pop_back();
        d.edges.pop_back();
        expect_error("no output node", d, "");
    }
    { // two outputs
        auto d = base;
        d.nodes.push_back(node("o2", "output"));
        d.edges.push_back(edge("e3", "e", "out", "o2", "in"));
        expect_error("two output nodes", d, "");
    }
    { // A CYCLE, and the fault must name an edge OF the cycle
        auto d = base;
        d.nodes.push_back(node("f", "exposure"));
        // e -> f -> e, with i -> e still there, so `e`'s input is taken; move it.
        d.edges = {edge("e1", "i", "out", "f", "in"), edge("e2", "f", "out", "e", "in"),
                   edge("e3", "e", "out", "f", "mask"), edge("e4", "e", "out", "o", "in")};
        // `e3` closes it: f -> e -> f. `mask` takes an image by coercion, which is legal, so the
        // cycle is the only thing wrong -- deliberately, so this case cannot pass for the wrong
        // reason.
        std::vector<std::string> o;
        const auto               fs = validate(d, o);
        if (!has_error(fs))
            fail("a cycle was accepted");
        const bool names_a_cycle_edge = std::any_of(fs.begin(), fs.end(), [](const graph_fault& f) {
            return f.reason.find("cycle") != std::string::npos && !f.edge.empty();
        });
        if (!names_a_cycle_edge)
            fail("the cycle fault names no edge, so a client cannot tell which one to delete");
    }
    { // A COERCION IS REPORTED AND DOES NOT REFUSE. The whole `severity` split rests on this.
        auto d = base;
        d.nodes.push_back(node("m", "exposure"));
        d.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "e", "out", "m", "in"),
                   edge("e3", "e", "out", "m", "mask"), edge("e4", "m", "out", "o", "in")};
        std::vector<std::string> o;
        const auto               fs = validate(d, o);
        if (has_error(fs))
            fail("an image into a mask port was REFUSED; it is legal and lossy, so it must be "
                 "reported and the document must still compile");
        if (o.empty())
            fail("a document with only coercion faults produced no order");
        const bool reported = std::any_of(fs.begin(), fs.end(), [](const graph_fault& f) {
            return f.sev == graph_fault::severity::coercion && f.edge == "e3";
        });
        if (!reported)
            fail("the image->mask coercion was applied SILENTLY; the client has to be able to "
                 "draw a warning on that edge");
    }
    { // an unreachable output: an exposure chain that never touches `input`
        auto d = base;
        d.nodes.push_back(node("g", "exposure"));
        // i -> e -> (nothing); g -> o, and g's own input unconnected would be a required-input
        // error, so feed g from e and take o from g... that IS reachable. Instead: make a second
        // input-less source by feeding `o` from a mask node, whose output needs no input.
        d.nodes.pop_back();
        d.nodes.push_back(node("k", "mask_ellipse"));
        d.edges = {edge("e1", "i", "out", "e", "in"), edge("e2", "k", "out", "o", "in")};
        // `k` needs nothing, `o` takes its mask by coercion, and `input` reaches only `e`.
        std::vector<std::string> o;
        const auto               fs = validate(d, o);
        if (!has_error(fs))
            fail("an output not reachable from input was accepted; the layer would render black");
    }
    { // the pass cap: 17 exposures in a row
        graph_document d;
        d.name = "cap";
        d.nodes.push_back(node("i", "input"));
        std::string prev = "i", prev_port = "out";
        for (int k = 0; k < 17; ++k) {
            const auto id = "n" + std::to_string(k);
            d.nodes.push_back(node(id, "exposure"));
            d.edges.push_back(edge("c" + std::to_string(k), prev, prev_port, id, "in"));
            prev      = id;
            prev_port = "out";
        }
        d.nodes.push_back(node("o", "output"));
        d.edges.push_back(edge("cz", prev, prev_port, "o", "in"));
        expect_error("17 image passes against a cap of 16", d, "");
        // ...and 16 must be ACCEPTED, or the cap is off by one and every check above passes.
        auto ok16 = d;
        ok16.nodes.erase(ok16.nodes.begin() + 17); // drop n16
        ok16.edges.erase(ok16.edges.begin() + 16); // and the edge into it
        ok16.edges.back() = edge("cz", "n15", "out", "o", "in");
        std::vector<std::string> o;
        const auto               fs = validate(ok16, o);
        if (has_error(fs))
            fail("16 image passes was refused against a cap of 16 -- the cap is off by one: " +
                 fs.front().reason);
    }

    CASPAR_LOG(info) << L"[graph-validate] self-test: all checks passed";
}

}}} // namespace caspar::core::graph
