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

#include "api_graph.h"

#include "json_state.h"

#include <core/graph/registry.h>

#include <common/log.h>

#include <algorithm>

namespace caspar { namespace protocol { namespace http {

namespace gr = core::graph;

namespace {

const json::value* member(const json::object& o, const char* key)
{
    const auto it = o.find(key);
    return it == o.end() ? nullptr : &it->value();
}

bool as_double(const json::value& v, double& out)
{
    if (v.is_double()) {
        out = v.as_double();
        return true;
    }
    if (v.is_int64()) {
        out = static_cast<double>(v.as_int64());
        return true;
    }
    if (v.is_uint64()) {
        out = static_cast<double>(v.as_uint64());
        return true;
    }
    return false;
}

/// One JSON value -> one `monitor::vector_t`, the same three shapes a parameter takes anywhere
/// else in this API: a number, a boolean, a name, or an array of those.
bool decode_value(const json::value& v, core::monitor::vector_t& out, std::string& reason)
{
    const auto push = [&](const json::value& e) {
        double d = 0;
        if (e.is_bool())
            out.push_back(e.as_bool());
        else if (e.is_string())
            out.push_back(std::string(e.as_string().c_str()));
        else if (as_double(e, d))
            out.push_back(d);
        else
            return false;
        return true;
    };
    if (v.is_array()) {
        for (const auto& e : v.as_array())
            if (!push(e)) {
                reason = "array elements must be numbers, booleans or names";
                return false;
            }
        return true;
    }
    if (!push(v)) {
        reason = "a value is a number, a boolean, a name, or an array of those";
        return false;
    }
    return true;
}

/// `ui` is stored as raw JSON TEXT and never interpreted -- but it IS checked for
/// well-formedness, because a malformed blob would break the next GET rather than the PUT that
/// stored it, and a client would have no way to tell which of its documents was the bad one.
///
/// Serialised back from the parsed value rather than kept as the client's bytes, so what a GET
/// returns is guaranteed to parse. The server still has no opinion about the CONTENT: where a box
/// is on a client's canvas is not a thing this server should ever have a schema for.
bool decode_ui(const json::value& v, std::string& out, std::string& reason)
{
    if (v.is_null()) {
        out.clear();
        return true;
    }
    if (!v.is_object() && !v.is_array()) {
        reason = "`ui` takes an object or an array. It is stored uninterpreted -- the server has "
                 "no opinion about where a box is -- but it has to be JSON, or the next GET "
                 "would break rather than this PUT";
        return false;
    }
    out = json::serialize(v);
    return true;
}

json::value ui_to_json(const std::string& ui)
{
    if (ui.empty())
        return nullptr;
    // Parsed on the way OUT, so the client gets its structure back rather than a string
    // containing JSON. It was serialised from a parsed value on the way in, so this cannot throw
    // -- but the guard stays, because "cannot throw" about a stored blob is the kind of claim
    // that stops being true when someone adds another writer.
    json::error_code ec;
    auto             v = json::parse(ui, ec);
    return ec ? json::value(nullptr) : v;
}

json::object fault_to_json(const gr::graph_fault& f)
{
    json::object o;
    o["severity"] = gr::severity_name(f.sev);
    if (!f.node.empty())
        o["node"] = f.node;
    if (!f.edge.empty())
        o["edge"] = f.edge;
    if (!f.port.empty())
        o["port"] = f.port;
    o["reason"] = f.reason;
    return o;
}

json::array faults_to_json(const std::vector<gr::graph_fault>& fs)
{
    json::array a;
    for (const auto& f : fs)
        a.push_back(fault_to_json(f));
    return a;
}

json::object edge_to_json(const gr::graph_edge& e)
{
    json::object o;
    o["id"]   = e.id;
    o["from"] = e.from_node + "." + e.from_port;
    o["to"]   = e.to_node + "." + e.to_port;
    if (e.muted)
        o["muted"] = true;
    return o;
}

json::object node_to_json(const gr::graph_node& n)
{
    json::object o;
    o["id"]    = n.id;
    o["class"] = n.cls;
    if (!n.label.empty())
        o["label"] = n.label;
    json::object ps;
    for (const auto& kv : n.params)
        ps[kv.first] = vector_to_json(kv.second);
    o["params"] = std::move(ps);
    if (!n.ui.empty())
        o["ui"] = ui_to_json(n.ui);
    return o;
}

json::object document_to_json(const gr::stored_graph& e)
{
    json::object o;
    o["name"]     = e.document.name;
    o["revision"] = e.document.revision;
    o["stage"]    = gr::stage_name(e.document.stage);
    json::array ns;
    for (const auto& n : e.document.nodes)
        ns.push_back(node_to_json(n));
    o["nodes"] = std::move(ns);
    json::array es;
    for (const auto& x : e.document.edges)
        es.push_back(edge_to_json(x));
    o["edges"] = std::move(es);
    if (!e.document.ui.empty())
        o["ui"] = ui_to_json(e.document.ui);

    o["faults"] = faults_to_json(e.faults);
    o["ok"]     = e.ok();
    json::array order;
    for (const auto& id : e.order)
        order.push_back(json::string(id));
    o["order"] = std::move(order);
    return o;
}

/// A decode failure, which is a `bad_request` and stores NOTHING.
struct decode_error
{
    std::string node, edge, reason;
    bool        bad() const { return !reason.empty(); }
};

decode_error decode_document(const std::string& body, const std::string& path_name,
                             gr::graph_document& out)
{
    json::error_code ec;
    const auto       parsed = json::parse(body, ec);
    if (ec)
        return {"", "", "the body is not JSON: " + ec.message()};
    if (!parsed.is_object())
        return {"", "", "a graph document is a JSON object"};
    const auto& o = parsed.as_object();

    if (const auto* n = member(o, "name")) {
        if (!n->is_string())
            return {"", "", "\"name\" takes a string"};
        out.name = n->as_string().c_str();
        // The same refusal `put_timeline` makes, for the same reason: guessing which of two names
        // is right would store the document under one the client does not expect, and every
        // later reference to it would miss.
        if (!path_name.empty() && out.name != path_name)
            return {"", "",
                    "the document's \"name\" is \"" + out.name + "\" and the path says \"" +
                        path_name + "\" -- one of them is wrong, and guessing which would store "
                                    "the graph under a name the client does not expect"};
    } else {
        out.name = path_name;
    }
    if (out.name.empty())
        return {"", "", "a graph needs a name"};

    if (const auto* st = member(o, "stage")) {
        if (!st->is_string() || !gr::stage_from_name(st->as_string().c_str(), out.stage))
            return {"", "",
                    "\"stage\" is \"working\" (the default -- scene-linear, before tone-map and "
                    "the OETF, where `MIXER CDL` runs) or \"display\" (on the picture as "
                    "encoded). Measured 2026-09-11: the same CDL lands 42 LSB apart between the "
                    "two, so this is a real choice"};
    }

    if (const auto* ui = member(o, "ui")) {
        std::string reason;
        if (!decode_ui(*ui, out.ui, reason))
            return {"", "", "ui: " + reason};
    }

    if (const auto* ns = member(o, "nodes")) {
        if (!ns->is_array())
            return {"", "", "\"nodes\" takes an array"};
        for (const auto& nv : ns->as_array()) {
            if (!nv.is_object())
                return {"", "", "each node is an object with an id and a class"};
            const auto&   no = nv.as_object();
            gr::graph_node n;
            if (const auto* id = member(no, "id")) {
                if (!id->is_string())
                    return {"", "", "a node's \"id\" takes a string"};
                n.id = id->as_string().c_str();
            }
            if (n.id.empty())
                return {"", "",
                        "every node needs an \"id\", and it is the CLIENT'S: an id is what "
                        "`node/<id>/<param>` addresses, so it is the one thing that has to "
                        "survive an edit"};
            if (const auto* c = member(no, "class")) {
                if (!c->is_string())
                    return {n.id, "", "a node's \"class\" takes a string"};
                n.cls = c->as_string().c_str();
            }
            if (const auto* l = member(no, "label")) {
                if (!l->is_string())
                    return {n.id, "", "a node's \"label\" takes a string"};
                n.label = l->as_string().c_str();
            }
            if (const auto* ui = member(no, "ui")) {
                std::string reason;
                if (!decode_ui(*ui, n.ui, reason))
                    return {n.id, "", "ui: " + reason};
            }
            if (const auto* ps = member(no, "params")) {
                if (!ps->is_object())
                    return {n.id, "", "\"params\" takes an object of port name to value"};
                for (const auto& kv : ps->as_object()) {
                    core::monitor::vector_t vec;
                    std::string             reason;
                    if (!decode_value(kv.value(), vec, reason))
                        return {n.id, "", std::string(kv.key()) + ": " + reason};
                    n.params[std::string(kv.key())] = std::move(vec);
                }
            }
            // `bypass` may also be sent as a top-level convenience, because it is the one
            // parameter a client toggles constantly and `{"bypass": true}` beside the class reads
            // better than burying it in `params`. It IS a parameter: the same port, the same
            // address, the same overlay. One spelling in the store.
            if (const auto* b = member(no, "bypass")) {
                if (!b->is_bool())
                    return {n.id, "", "\"bypass\" takes true or false"};
                n.params["bypass"] = core::monitor::vector_t{b->as_bool()};
            }
            out.nodes.push_back(std::move(n));
        }
    }

    if (const auto* es = member(o, "edges")) {
        if (!es->is_array())
            return {"", "", "\"edges\" takes an array"};
        int next_id = 1;
        for (const auto& ev : es->as_array()) {
            if (!ev.is_object())
                return {"", "", "each edge is an object with \"from\" and \"to\""};
            const auto&   eo = ev.as_object();
            gr::graph_edge e;
            if (const auto* id = member(eo, "id")) {
                if (!id->is_string())
                    return {"", "", "an edge's \"id\" takes a string"};
                e.id = id->as_string().c_str();
            }
            const auto split = [](const json::value* v, const char* which, std::string& node,
                                  std::string& port, std::string& reason) {
                if (!v || !v->is_string()) {
                    reason = std::string("an edge needs \"") + which + "\": \"<node>.<port>\"";
                    return false;
                }
                const std::string s = v->as_string().c_str();
                // The LAST '.', and it is unambiguous because the validator refuses a node id
                // containing '.' or '/'. That refusal exists for the address grammar and this
                // spelling gets to lean on it.
                const auto        dot = s.rfind('.');
                if (dot == std::string::npos || dot == 0 || dot + 1 == s.size()) {
                    reason = std::string("\"") + s + "\" is not \"<node>.<port>\"";
                    return false;
                }
                node = s.substr(0, dot);
                port = s.substr(dot + 1);
                return true;
            };
            std::string reason;
            if (!split(member(eo, "from"), "from", e.from_node, e.from_port, reason))
                return {"", e.id, reason};
            if (!split(member(eo, "to"), "to", e.to_node, e.to_port, reason))
                return {"", e.id, reason};
            if (const auto* m = member(eo, "muted")) {
                if (!m->is_bool())
                    return {"", e.id, "\"muted\" takes true or false"};
                e.muted = m->as_bool();
            }
            if (e.id.empty()) {
                // SERVER-ASSIGNED, and assigned HERE rather than in the validator so that every
                // fault the validator raises has an id to point at. Drawing an edge is a gesture
                // with no natural name, so requiring one would only make clients invent `e17`.
                for (;;) {
                    auto       candidate = "e" + std::to_string(next_id++);
                    const bool taken     = std::any_of(
                        out.edges.begin(), out.edges.end(),
                        [&](const gr::graph_edge& x) { return x.id == candidate; });
                    if (!taken) {
                        e.id = std::move(candidate);
                        break;
                    }
                }
            }
            out.edges.push_back(std::move(e));
        }
    }
    return {};
}

api_reply decode_failed(const decode_error& e)
{
    json::object d;
    d["node"]   = e.node;
    d["edge"]   = e.edge;
    d["reason"] = e.reason;
    // `bad_request`, and NOT `graph_invalid`. The line is the same one commit 0 drew for the
    // timeline: this is not a document at all, or its grammar is wrong, so there is nothing to
    // store and nothing for an editor to highlight inside a document it does not have.
    // `graph_invalid` means "the server understood you and the graph will not run", which is a
    // different thing a client acts on differently.
    return api_reply::fail(api_code::bad_request,
                           (e.node.empty() ? (e.edge.empty() ? std::string("graph")
                                                             : "edge '" + e.edge + "'")
                                           : "node '" + e.node + "'") +
                               ": " + e.reason,
                           json::array{d});
}

api_reply no_store()
{
    return api_reply::fail(api_code::internal, "no graph store is wired into this build");
}

json::object attachment_to_json(const core::graph::graph_store& st, const std::string& name)
{
    json::object o;
    const auto   a = st.attached(name);
    if (!a)
        return o;
    o["channel"] = a->channel;
    o["layer"]   = a->layer;
    return o;
}

} // namespace

api_reply list_graphs(const api_context& ctx)
{
    if (!ctx.graphs)
        return no_store();
    json::array a;
    for (const auto& name : ctx.graphs->names()) {
        const auto e = ctx.graphs->get(name);
        if (!e)
            continue;
        json::object o;
        o["name"]     = name;
        o["revision"] = e->document.revision;
        o["stage"]    = gr::stage_name(e->document.stage);
        o["nodes"]    = static_cast<std::int64_t>(e->document.nodes.size());
        o["edges"]    = static_cast<std::int64_t>(e->document.edges.size());
        o["ok"]       = e->ok();
        o["faults"]   = static_cast<std::int64_t>(e->faults.size());
        const auto at = attachment_to_json(*ctx.graphs, name);
        if (!at.empty())
            o["attached"] = at;
        a.push_back(std::move(o));
    }
    json::object out;
    out["graphs"] = std::move(a);
    // BOTH counters, because a client polling this is asking one of two different questions and
    // the answer differs: `revision` moves when the SHAPE of something changes and is what the
    // stage fingerprint carries, `values_revision` moves on a parameter write and deliberately
    // is not in the fingerprint.
    out["revision"]        = ctx.graphs->revision();
    out["values_revision"] = ctx.graphs->values_revision();
    return api_reply::ok_with(std::move(out));
}

api_reply put_graph(const api_context& ctx, const std::string& name, const std::string& body)
{
    if (!ctx.graphs)
        return no_store();
    if (name.empty())
        return api_reply::fail(api_code::bad_request, "PUT /v1/graph/{name} takes a name");

    gr::graph_document doc;
    if (const auto e = decode_document(body, name, doc); e.bad())
        return decode_failed(e);

    // `label` names the GESTURE for the undo history. Optional, and when a client sends none the
    // entry reads "an edit" -- which is honest rather than invented.
    std::string label;
    {
        json::error_code ec;
        const auto       parsed = json::parse(body, ec);
        if (!ec && parsed.is_object()) {
            if (const auto* l = member(parsed.as_object(), "label"); l && l->is_string())
                label = l->as_string().c_str();
        }
    }

    const auto entry = ctx.graphs->put(std::move(doc), label);
    auto       result = document_to_json(*entry);
    result["attached"] = attachment_to_json(*ctx.graphs, name);
    result["can_undo"] = ctx.graphs->can_undo(name);

    if (!entry->ok()) {
        // STORED, and the reply says so by returning the document. An operator editing a look on
        // air must not lose the grade that is rendering because they mistyped a port name.
        return api_reply{api_code::graph_invalid,
                         "the graph was stored and will not run",
                         faults_to_json(entry->faults),
                         std::move(result)};
    }
    return api_reply::ok_with(std::move(result));
}

api_reply get_graph(const api_context& ctx, const std::string& name)
{
    if (!ctx.graphs)
        return no_store();
    if (name.empty())
        return list_graphs(ctx);
    const auto e = ctx.graphs->get(name);
    if (!e)
        return api_reply::fail(api_code::graph_not_found, "no graph named '" + name + "'");
    auto result        = document_to_json(*e);
    result["attached"] = attachment_to_json(*ctx.graphs, name);
    result["can_undo"] = ctx.graphs->can_undo(name);
    result["can_redo"] = ctx.graphs->can_redo(name);
    return api_reply::ok_with(std::move(result));
}

api_reply delete_graph(const api_context& ctx, const std::string& name)
{
    if (!ctx.graphs)
        return no_store();
    if (name.empty())
        return api_reply::fail(api_code::bad_request, "DELETE /v1/graph/{name} takes a name");
    if (!ctx.graphs->erase(name))
        return api_reply::fail(api_code::graph_not_found, "no graph named '" + name + "'");
    json::object o;
    o["name"] = name;
    // DETACHED, not refused. A client deleting a look means "take it off air"; refusing would
    // leave them holding a document they cannot get rid of without first remembering where it
    // was attached. Flagged as a policy call rather than an obvious one.
    o["detached"] = true;
    return api_reply::ok_with(std::move(o));
}

api_reply get_graph_history(const api_context& ctx, const std::string& name)
{
    if (!ctx.graphs)
        return no_store();
    if (!ctx.graphs->get(name))
        return api_reply::fail(api_code::graph_not_found, "no graph named '" + name + "'");

    json::array a;
    for (const auto& h : ctx.graphs->history(name)) {
        json::object o;
        o["label"]    = h.label;
        o["kind"]     = h.kind;
        o["revision"] = h.revision;
        o["at"] = static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(h.at.time_since_epoch()).count());
        a.push_back(std::move(o));
    }
    json::object out;
    out["name"]     = name;
    out["entries"]  = std::move(a);
    out["can_undo"] = ctx.graphs->can_undo(name);
    out["can_redo"] = ctx.graphs->can_redo(name);
    return api_reply::ok_with(std::move(out));
}

api_reply graph_history_verb(const api_context& ctx, const std::string& name,
                             const std::string& verb)
{
    if (!ctx.graphs)
        return no_store();
    if (!ctx.graphs->get(name))
        return api_reply::fail(api_code::graph_not_found, "no graph named '" + name + "'");

    std::shared_ptr<const gr::stored_graph> e;
    if (verb == "undo")
        e = ctx.graphs->undo(name);
    else if (verb == "redo")
        e = ctx.graphs->redo(name);
    else
        return api_reply::fail(api_code::bad_request,
                               "'" + verb + "' is not a history verb. Take `undo` or `redo`");

    if (!e)
        return api_reply::fail(api_code::bad_request,
                               verb == "undo" ? "nothing to undo" : "nothing to redo");
    auto result        = document_to_json(*e);
    result["attached"] = attachment_to_json(*ctx.graphs, name);
    result["can_undo"] = ctx.graphs->can_undo(name);
    result["can_redo"] = ctx.graphs->can_redo(name);
    CASPAR_LOG(info) << L"[api] graph " << u16(name) << L" " << u16(verb) << L" -> revision "
                     << e->document.revision;
    return api_reply::ok_with(std::move(result));
}

}}} // namespace caspar::protocol::http
