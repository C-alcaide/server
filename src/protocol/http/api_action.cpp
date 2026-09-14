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

#include "api_action.h"

#include "api_timeline.h"

#include <tuple>
#include "api_value.h"

#include <common/log.h>
#include <common/utf.h>

#include <boost/algorithm/string/case_conv.hpp>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>

namespace caspar { namespace protocol { namespace http {

namespace {

std::vector<std::string> split_path(std::string_view p)
{
    std::vector<std::string> out;
    size_t                   i = 0;
    while (i <= p.size()) {
        const auto j = p.find('/', i);
        const auto n = (j == std::string_view::npos ? p.size() : j) - i;
        if (n > 0)
            out.emplace_back(p.substr(i, n));
        if (j == std::string_view::npos)
            break;
        i = j + 1;
    }
    return out;
}

bool is_number(const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; }

api_reply parse_action_path(const std::string& path, action_target& out)
{
    const auto seg = split_path(path);

    // /channel/{n}/{verb}  or  /channel/{n}/stage/layer/{m}/{verb}
    if (seg.size() == 3 && seg[0] == "channel") {
        if (!is_number(seg[1]))
            return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + seg[1]);
        out.channel = std::atoi(seg[1].c_str());
        out.verb    = seg[2];
        return api_reply{};
    }
    if (seg.size() == 6 && seg[0] == "channel" && seg[2] == "stage" && seg[3] == "layer") {
        if (!is_number(seg[1]))
            return api_reply::fail(api_code::channel_not_found, "channel index is not a number: " + seg[1]);
        if (!is_number(seg[4]))
            return api_reply::fail(api_code::layer_not_found, "layer index is not a number: " + seg[4]);
        out.channel = std::atoi(seg[1].c_str());
        out.layer   = std::atoi(seg[4].c_str());
        out.verb    = seg[5];
        return api_reply{};
    }
    return api_reply::fail(api_code::unknown_path,
                           "an action path is /channel/{n}/{verb} or /channel/{n}/stage/layer/{m}/{verb}, not " + path);
}

/// The AMCP reply code mapped onto the status vocabulary.
///
/// Lossy, and knowingly so: AMCP answers a missing file and a malformed argument with the
/// same 404 and 400 it gives everything else, so a client gets less detail here than from a
/// native endpoint. This is why delegating is temporary -- but it is much better than the
/// API growing its own clip parser beside the producer registry's.
api_code code_from_amcp(int amcp_code)
{
    if (amcp_code >= 200 && amcp_code < 300)
        return api_code::ok;
    if (amcp_code == 404)
        return api_code::unknown_path;
    if (amcp_code >= 400 && amcp_code < 500)
        return api_code::bad_request;
    if (amcp_code >= 500)
        return api_code::internal;
    return api_code::internal;
}

std::wstring quote_if_needed(const std::wstring& s)
{
    return s.find(L' ') == std::wstring::npos ? s : (L"\"" + s + L"\"");
}

/// Build the AMCP line for a `load`/`play` that carries a clip.
///
/// Assembling text and handing it to the parser is not elegant. The alternative is
/// reimplementing `LOAD`'s argument handling -- the producer registry lookup, the
/// per-producer parameter conventions, `LOOP`, `SEEK`, `LENGTH`, transitions -- next to the
/// original, which is the exact duplication that let `MIXER EXPOSURE` diverge.
std::wstring build_amcp(const action_target& t, const json::object& body, bool play)
{
    std::wostringstream cmd;
    cmd << (play ? L"PLAY " : L"LOAD ") << t.channel;
    if (t.layer >= 0)
        cmd << L"-" << t.layer;

    if (auto* c = body.if_contains("clip"))
        if (c->is_string())
            cmd << L" " << quote_if_needed(u16(std::string(c->as_string().c_str())));

    if (auto* s = body.if_contains("seek"))
        if (s->is_int64() || s->is_uint64() || s->is_double())
            cmd << L" SEEK " << static_cast<int64_t>(s->to_number<double>());

    if (auto* l = body.if_contains("length"))
        if (l->is_int64() || l->is_uint64() || l->is_double())
            cmd << L" LENGTH " << static_cast<int64_t>(l->to_number<double>());

    if (auto* lp = body.if_contains("loop"))
        if (lp->is_bool() && lp->as_bool())
            cmd << L" LOOP";

    return cmd.str();
}

/// `INPUT ...` from a JSON body.
///
/// Goes through AMCP rather than reaching the stage directly, and that is deliberate: the
/// channel-level form has to dispatch to the image mixer FIRST (previz consumes the event when it
/// is active) and only then to the stage, and `video_channel::input` is the single place that
/// decides. `api_context` hands out a `stage_base`, not a channel, so a direct route here would
/// have to duplicate that decision -- which is how two dispatch orders end up disagreeing.
///
/// Returns an empty string when the body is not a valid input description; the caller reports why.
std::wstring build_input_amcp(const action_target& t, const json::object& body, std::string& why)
{
    const auto str = [&](const char* k) -> std::string {
        auto* v = body.if_contains(k);
        return v && v->is_string() ? std::string(v->as_string().c_str()) : std::string();
    };
    const auto num = [&](const char* k, double dflt) -> double {
        auto* v = body.if_contains(k);
        if (!v)
            return dflt;
        if (v->is_double())
            return v->as_double();
        if (v->is_int64())
            return static_cast<double>(v->as_int64());
        return dflt;
    };
    const auto has = [&](const char* k) { return body.if_contains(k) != nullptr; };

    std::wostringstream cmd;
    cmd << L"INPUT " << t.channel;
    if (t.layer >= 0)
        cmd << L"-" << t.layer;

    auto type   = str("type");
    auto action = str("action");
    boost::to_lower(type);
    boost::to_lower(action);

    if (type == "mouse") {
        if (action == "move" || action == "wheel") {
            if (!has("x") || !has("y")) {
                why = "a mouse " + action + " needs x and y in 0..1";
                return {};
            }
        }
        if (action == "move") {
            cmd << L" MOUSE MOVE " << num("x", 0) << L" " << num("y", 0);
            if (has("modifiers"))
                cmd << L" " << static_cast<uint32_t>(num("modifiers", 0));
        } else if (action == "down" || action == "up") {
            auto button = str("button");
            boost::to_upper(button);
            if (button != "LEFT" && button != "MIDDLE" && button != "RIGHT") {
                why = "button must be left, middle or right";
                return {};
            }
            if (!has("x") || !has("y")) {
                why = "a mouse " + action + " needs x and y in 0..1";
                return {};
            }
            cmd << L" MOUSE " << (action == "down" ? L"DOWN " : L"UP ") << u16(button) << L" " << num("x", 0) << L" "
                << num("y", 0);
            if (has("modifiers"))
                cmd << L" " << static_cast<uint32_t>(num("modifiers", 0));
        } else if (action == "wheel") {
            cmd << L" MOUSE WHEEL " << num("x", 0) << L" " << num("y", 0) << L" " << num("dx", 0) << L" "
                << num("dy", 0);
        } else if (action == "leave") {
            cmd << L" MOUSE LEAVE";
        } else {
            why = "a mouse action is move, down, up, wheel or leave, not \"" + action + "\"";
            return {};
        }

    } else if (type == "key") {
        if (action != "down" && action != "up") {
            why = "a key action is down or up, not \"" + action + "\"";
            return {};
        }
        if (!has("key")) {
            why = "a key event needs a numeric virtual-key code in \"key\"";
            return {};
        }
        cmd << L" KEY " << (action == "down" ? L"DOWN " : L"UP ") << static_cast<int>(num("key", 0));
        if (has("modifiers"))
            cmd << L" " << static_cast<uint32_t>(num("modifiers", 0));

    } else if (type == "text") {
        const auto text = str("text");
        if (text.empty()) {
            why = "a text event needs a non-empty \"text\"";
            return {};
        }
        // Quoted, because AMCP splits on whitespace and typed text routinely contains it.
        cmd << L" TEXT \"" << u16(text) << L"\"";

    } else {
        why = "type must be mouse, key or text, not \"" + type + "\"";
        return {};
    }

    return cmd.str();
}

/// Queue one verb against a stage and return its future WITHOUT waiting.
///
/// The separation matters inside a batch and nowhere else. A batch queues its ops against
/// `stage_delayed` executors that are deliberately BLOCKED until every channel is locked;
/// waiting on one of those futures before the release is a guaranteed deadlock, and it is
/// exactly what the first version did -- `run_action` called `.get()`, the batch hung, and
/// the socket timed out with nothing in the log.
/// `{"op": "timeline", "name": "show", "verb": "play"}` -- one transport verb in a batch.
///
/// WHY A BATCH NEEDS THIS AT ALL. Starting two channels' documents on one frame is the whole
/// point of a batch, and before this a client could pin two field writes to a frame and not the
/// two `PLAY`s that made them mean anything. The verb table itself is `parse_transport_verb`,
/// shared with the route, so the two cannot drift.
///
/// The CHANNEL is taken from the document rather than the op, unlike every other op in a batch.
/// A document declares which channel drives it, so asking the client to restate it would create
/// a second source of truth whose only possible contribution is to disagree. An explicit
/// `channel` IS accepted and is then checked against the document -- a client that thinks it is
/// driving channel 2 and is not has made an error worth a message.
///
/// `at_frame` is NOT read from the op. The batch pins one frame for everything it carries, and
/// an op naming a different one would break the single guarantee a batch makes.
api_reply parse_timeline_batch_op(const api_context& ctx, const json::object& op, batch_op& out)
{
    if (!ctx.timelines)
        return api_reply::fail(api_code::internal, "no timeline store is wired into this build");

    const auto* n = op.if_contains("name");
    if (!n || !n->is_string())
        return api_reply::fail(api_code::field_missing, "a timeline op needs a document \"name\"");
    out.timeline.name = n->as_string().c_str();

    const auto entry = ctx.timelines->get(out.timeline.name);
    if (!entry)
        return api_reply::fail(api_code::timeline_not_found,
                               "no document '" + out.timeline.name + "'");
    out.channel = entry->document.channel;

    if (const auto* c = op.if_contains("channel")) {
        if (!c->is_number())
            return api_reply::fail(api_code::field_wrong_type, "channel takes an index");
        const auto claimed = static_cast<int>(c->to_number<double>());
        if (claimed != out.channel)
            return api_reply::fail(api_code::bad_request,
                                   "document '" + out.timeline.name + "' declares channel " +
                                       std::to_string(out.channel) + " and the op says " +
                                       std::to_string(claimed));
    }

    const auto* v = op.if_contains("verb");
    if (!v || !v->is_string())
        return api_reply::fail(api_code::field_missing, "a timeline op needs a \"verb\"");
    const std::string verb = v->as_string().c_str();

    if (op.if_contains("at_frame") != nullptr)
        return api_reply::fail(api_code::bad_request,
                               "at_frame belongs on the batch, not on an op: a batch lands on one "
                               "frame and an op that named its own would break that");

    return parse_transport_verb(verb, op, out.timeline.cmd);
}



/// Nine segments, `.../mixer/node/<id>/<param>` -- the same test the write route applies, and
/// applied here for the same reason: `resolve_write_target` reads seven segments and would take
/// `node` for a field name.
bool is_node_batch_path(const std::string& path)
{
    const auto seg = split_path(path);
    return seg.size() == 9 && seg[0] == "channel" && seg[2] == "stage" && seg[3] == "layer" &&
           seg[5] == "mixer" && seg[6] == "node";
}

/// One JSON value -> one `monitor::vector_t`, the three shapes a parameter takes anywhere in
/// this API: a scalar, a boolean, or an array of those. A node parameter is not validated
/// against its descriptor here -- see `parse_node_batch_op` for why that would be wrong rather
/// than merely absent.
bool decode_batch_value(const json::value& v, core::monitor::vector_t& out)
{
    const auto push = [&](const json::value& e) {
        double d = 0;
        if (e.is_bool())
            out.push_back(e.as_bool());
        else if (e.is_double()) {
            out.push_back(e.as_double());
        } else if (e.is_int64()) {
            out.push_back(static_cast<double>(e.as_int64()));
        } else if (e.is_uint64()) {
            out.push_back(static_cast<double>(e.as_uint64()));
        } else if (e.is_string()) {
            out.push_back(std::string(e.as_string().c_str()));
        } else
            return false;
        (void)d;
        return true;
    };
    if (v.is_array()) {
        for (const auto& e : v.as_array())
            if (!push(e))
                return false;
        return true;
    }
    return push(v);
}

/// `{"op": "graph", "name": "look", "verb": "attach", "layer": 1}` -- one graph verb in a batch.
///
/// WHY A BATCH NEEDS THIS. Putting a look on air is a CUT, and before this a client could pin
/// the field writes around it to a frame and not the attach that made them mean anything. The
/// same argument the timeline op already makes, for the same reason.
///
/// ADDRESSED BY DOCUMENT NAME and not by a path, like the timeline op: a graph document is
/// server-wide rather than a property of a channel, so it is the second thing in this API with
/// no path. `layer` is read only by `attach`; `detach` takes the layer from the STORE, because
/// a client taking a look off air knows the look's name and making it also remember where the
/// look is gives it something to get wrong.
api_reply parse_graph_batch_op(const api_context& ctx, const json::object& op, batch_op& out)
{
    if (!ctx.graphs)
        return api_reply::fail(api_code::internal, "no graph store is wired into this build");

    const auto* n = op.if_contains("name");
    if (!n || !n->is_string())
        return api_reply::fail(api_code::field_missing, "a graph op needs a document \"name\"");
    out.graph.name = n->as_string().c_str();

    if (!ctx.graphs->get(out.graph.name))
        return api_reply::fail(api_code::graph_not_found,
                               "no graph named '" + out.graph.name + "'");

    const auto* v = op.if_contains("verb");
    if (!v || !v->is_string())
        return api_reply::fail(api_code::field_missing, "a graph op needs a \"verb\"");
    out.graph.verb = v->as_string().c_str();
    if (out.graph.verb != "attach" && out.graph.verb != "detach" && out.graph.verb != "undo" &&
        out.graph.verb != "redo")
        return api_reply::fail(api_code::bad_request,
                               "no such graph verb: " + out.graph.verb +
                                   ". Known: attach, detach, undo, redo");

    // THE CHANNEL, and where it comes from differs by verb -- which is worth being explicit
    // about rather than defaulting. `attach` is told; everything else reads the store, because
    // the document already knows where it is.
    if (out.graph.verb == "attach") {
        const auto* c = op.if_contains("channel");
        const auto* l = op.if_contains("layer");
        if (!c || !c->is_number() || !l || !l->is_number())
            return api_reply::fail(api_code::field_missing,
                                   "a graph attach needs \"channel\" and \"layer\"");
        out.channel     = static_cast<int>(c->to_number<double>());
        out.graph.layer = static_cast<int>(l->to_number<double>());
    } else {
        const auto where = ctx.graphs->attached(out.graph.name);
        if (!where)
            return api_reply::fail(api_code::bad_request,
                                   "'" + out.graph.name + "' is not attached to anything, so a " +
                                       out.graph.verb + " has no channel to land on");
        out.channel     = where->channel;
        out.graph.layer = where->layer;
    }

    if (op.if_contains("at_frame") != nullptr)
        return api_reply::fail(api_code::bad_request,
                               "at_frame belongs on the batch, not on an op: a batch lands on one "
                               "frame and an op that named its own would break that");
    return api_reply{};
}

/// A NODE-PARAMETER write inside a batch.
///
/// VALIDATED ONLY AS FAR AS ITS ADDRESS, which is the same limit the `action` op already
/// declares and for the same reason: whether the value is in range depends on the document
/// ATTACHED AT THE TIME, and a batch that validated against the document as it is now would
/// still be wrong by the time its frame arrives. `prepare_*` touches no stage by design.
///
/// So the batch's all-or-nothing promise covers the ADDRESS being resolvable and stops there --
/// stated, because the alternative is to pretend otherwise.
api_reply parse_node_batch_op(const std::string& path, const json::object& op, batch_op& out)
{
    const auto seg = split_path(path);
    if (seg.size() != 9 || !is_number(seg[1]) || !is_number(seg[4]))
        return api_reply::fail(api_code::bad_request,
                               "a node path is channel/N/stage/layer/M/mixer/node/<id>/<param>: " +
                                   path);
    out.channel    = std::atoi(seg[1].c_str());
    out.node.layer = std::atoi(seg[4].c_str());
    out.node.path  = "node/" + seg[7] + "/" + seg[8];

    const auto* v = op.if_contains("value");
    if (!v)
        return api_reply::fail(api_code::field_missing, "no value for " + path);
    if (!decode_batch_value(*v, out.node.value))
        return api_reply::fail(api_code::field_wrong_type,
                               "a node parameter takes a number, a boolean, a name, or an array "
                               "of those");

    // THE GESTURE LABEL, which is the whole reason a batch of node writes is different from
    // three separate ones: consecutive writes carrying the same label fold into ONE undo entry,
    // so a slider drag is one step back rather than fifty.
    if (const auto* l = op.if_contains("label"); l && l->is_string())
        out.node.label = l->as_string().c_str();

    if (op.if_contains("at_frame") != nullptr)
        return api_reply::fail(api_code::bad_request,
                               "at_frame belongs on the batch, not on an op");
    return api_reply{};
}

api_reply queue_verb(const std::shared_ptr<core::stage_base>& stage,
                     const action_target&                     target,
                     std::future<void>&                       out)
{
    const auto& verb = target.verb;

    if (target.layer < 0) {
        if (verb == "clear")
            out = stage->clear();
        else if (verb == "clear_transforms")
            out = stage->clear_transforms();
        else
            return api_reply::fail(api_code::unknown_path, "a channel takes clear or clear_transforms, not " + verb);
        return api_reply{};
    }

    if (verb == "play")
        out = stage->play(target.layer);
    else if (verb == "stop")
        out = stage->stop(target.layer);
    else if (verb == "pause")
        out = stage->pause(target.layer);
    else if (verb == "resume")
        out = stage->resume(target.layer);
    else if (verb == "preview")
        out = stage->preview(target.layer);
    else if (verb == "clear")
        out = stage->clear(target.layer);
    else if (verb == "clear_transforms")
        out = stage->clear_transforms(target.layer);
    else
        return api_reply::fail(api_code::unknown_path, "no such action: " + verb);

    return api_reply{};
}

} // namespace

api_reply run_action(const api_context& ctx, const std::string& path, const std::string& body, const std::string& peer)
{
    action_target target;
    if (auto r = parse_action_path(path, target); r.code != api_code::ok)
        return r;

    json::object obj;
    if (!body.empty()) {
        try {
            auto doc = json::parse(body);
            if (!doc.is_object())
                return api_reply::fail(api_code::bad_request, "body must be a JSON object");
            obj = doc.as_object();
        } catch (const std::exception&) {
            return api_reply::fail(api_code::bad_request, "body is not valid JSON");
        }
    }

    if (!ctx.stage)
        return api_reply::fail(api_code::internal, "the API was built without access to the channels");
    const auto stage = ctx.stage(target.channel);
    if (!stage)
        return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(target.channel));

    const auto& verb     = target.verb;
    const bool  has_clip = obj.if_contains("clip") != nullptr;

    const auto log_it = [&](const std::wstring& what) {
        const std::string label =
            obj.if_contains("label") && obj.at("label").is_string() ? std::string(obj.at("label").as_string().c_str())
                                                                    : "";
        CASPAR_LOG(info) << L"[api] " << u16(peer) << L" POST " << u16(path) << L" " << what
                         << (label.empty() ? L"" : (L" label=\"" + u16(label) + L"\""));
    };

    try {
        if (verb == "input") {
            if (!ctx.amcp)
                return api_reply::fail(api_code::internal, "this build cannot deliver input through the API");

            std::string why;
            const auto  cmd = build_input_amcp(target, obj, why);
            if (cmd.empty())
                return api_reply::fail(api_code::bad_request, why.empty() ? "not a valid input event" : why);

            const auto rep = ctx.amcp(cmd);
            log_it(L"via-amcp " + cmd);

            json::object r;
            r["via"]  = "amcp";
            r["sent"] = u8(cmd);
            r["code"] = rep.code;
            if (!rep.text.empty())
                r["reply"] = u8(rep.text);
            const auto code = code_from_amcp(rep.code);
            if (code != api_code::ok)
                return api_reply::fail(code, u8(rep.text), json::array{std::move(r)});
            return api_reply::ok_with(std::move(r));
        }

        if ((verb == "load" || verb == "play") && has_clip) {
            if (!ctx.amcp)
                return api_reply::fail(api_code::internal, "this build cannot load clips through the API");
            const auto cmd = build_amcp(target, obj, verb == "play");
            const auto rep = ctx.amcp(cmd);
            log_it(L"via-amcp " + cmd);

            json::object r;
            r["via"]  = "amcp";
            r["sent"] = u8(cmd);
            r["code"] = rep.code;
            if (!rep.text.empty())
                r["reply"] = u8(rep.text);
            const auto code = code_from_amcp(rep.code);
            if (code != api_code::ok)
                return api_reply::fail(code, u8(rep.text), json::array{std::move(r)});
            return api_reply::ok_with(std::move(r));
        }

        std::future<void> f;
        if (auto r = queue_verb(stage, target, f); r.code != api_code::ok)
            return r;
        f.get();
    } catch (const std::exception& e) {
        // NO `catch (...)` AFTER THIS. There was one, and under `/EHa` its only remaining job was
        // to swallow a STRUCTURED exception -- every C++ exception is already handled above. An
        // access violation here must reach a crash dump rather than become a 500.
        return api_reply::fail(api_code::internal, e.what());
    }

    log_it(u16(verb));

    json::object r;
    r["path"]   = path;
    r["action"] = verb;
    return api_reply::ok_with(std::move(r));
}

// -----------------------------------------------------------------------------------------
// Batch
// -----------------------------------------------------------------------------------------

api_reply validate_batch(const api_context& ctx, const std::string& body, batch_plan& out)
{
    json::value doc;
    try {
        doc = json::parse(body);
    } catch (const std::exception&) {
        return api_reply::fail(api_code::bad_request, "body is not valid JSON");
    }
    if (!doc.is_object())
        return api_reply::fail(api_code::bad_request, "body must be a JSON object");
    const auto& o = doc.as_object();

    const auto* ops = o.if_contains("ops");
    if (!ops || !ops->is_array() || ops->as_array().empty())
        return api_reply::fail(api_code::field_missing, "a batch needs a non-empty ops array");

    // `queue` is accepted and echoed but not acted on. The field is reserved now so a
    // client written today does not change when independent queues arrive; pretending it
    // works would be worse than saying it is one queue.
    out.queue =
        o.if_contains("queue") && o.at("queue").is_string() ? std::string(o.at("queue").as_string().c_str()) : "";
    out.label =
        o.if_contains("label") && o.at("label").is_string() ? std::string(o.at("label").as_string().c_str()) : "";

    const auto& arr = ops->as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
        const auto fail = [&](api_reply r) {
            json::object d;
            d["index"]   = static_cast<std::int64_t>(i);
            d["code"]    = to_string(r.code);
            d["message"] = r.message;
            if (!r.details.empty())
                d["detail"] = r.details;
            return api_reply::fail(api_code::batch_op_failed,
                                   "op " + std::to_string(i) + " is invalid; nothing was applied",
                                   json::array{std::move(d)});
        };

        if (!arr[i].is_object())
            return fail(api_reply::fail(api_code::bad_request, "an op must be an object"));
        const auto& op = arr[i].as_object();

        const std::string kind =
            op.if_contains("op") && op.at("op").is_string() ? std::string(op.at("op").as_string().c_str()) : "set";
        // A TIMELINE OP IS ADDRESSED BY DOCUMENT NAME, not by a state path, so the path
        // requirement below does not apply to it and it is handled before that requirement is
        // imposed. A document is server-wide -- it is not a property of a channel or a layer --
        // which is why it is the one thing in this API with no path.
        if (kind == "timeline") {
            batch_op pl;
            pl.index       = i;
            pl.is_timeline = true;
            if (auto r = parse_timeline_batch_op(ctx, op, pl); r.code != api_code::ok)
                return fail(std::move(r));
            if (!ctx.stage || !ctx.stage(pl.channel))
                return fail(api_reply::fail(api_code::channel_not_found,
                                            "no channel " + std::to_string(pl.channel)));
            out.ops.push_back(std::move(pl));
            continue;
        }

        // A GRAPH OP IS ADDRESSED BY DOCUMENT NAME, like a timeline op, so it is handled before
        // the path requirement below is imposed.
        if (kind == "graph") {
            batch_op pl;
            pl.index    = i;
            pl.is_graph = true;
            if (auto r = parse_graph_batch_op(ctx, op, pl); r.code != api_code::ok)
                return fail(std::move(r));
            if (!ctx.stage || !ctx.stage(pl.channel))
                return fail(api_reply::fail(api_code::channel_not_found,
                                            "no channel " + std::to_string(pl.channel)));
            out.ops.push_back(std::move(pl));
            continue;
        }

        const auto* p = op.if_contains("path");
        if (!p || !p->is_string())
            return fail(api_reply::fail(api_code::field_missing, "an op needs a path"));
        const std::string path = p->as_string().c_str();

        // A NODE PARAMETER IS A `set` WITH A DIFFERENT WRITER, and it has to be split out here
        // rather than inside `prepare_set`: `resolve_write_target` reads a seven-segment path
        // and would try to read `node` as a field name, answering `unknown_path` for a
        // parameter that exists. The write path's route already tests this first for exactly
        // the same reason.
        if (kind == "set" && is_node_batch_path(path)) {
            batch_op pl;
            pl.index   = i;
            pl.is_node = true;
            if (auto r = parse_node_batch_op(path, op, pl); r.code != api_code::ok)
                return fail(std::move(r));
            if (!ctx.stage || !ctx.stage(pl.channel))
                return fail(api_reply::fail(api_code::channel_not_found,
                                            "no channel " + std::to_string(pl.channel)));
            out.ops.push_back(std::move(pl));
            continue;
        }

        batch_op pl;
        pl.index = i;
        if (kind == "set") {
            pl.is_set = true;
            if (auto r = prepare_set(path, op, pl.set); r.code != api_code::ok)
                return fail(std::move(r));
            pl.channel = pl.set.target.channel;
        } else if (kind == "action") {
            // An action is validated only as far as its path -- whether a `play` will
            // succeed is not knowable without running it, and pretending otherwise would
            // make the all-or-nothing promise a lie rather than a limit.
            if (auto r = parse_action_path(path, pl.action); r.code != api_code::ok)
                return fail(std::move(r));

            // A clip load inside a batch is refused rather than half-supported. It has to
            // go through AMCP to build the producer, which runs against the channel's REAL
            // stage -- so it would land whenever AMCP got to it, outside the frame the rest
            // of the batch is pinned to. Silently breaking the batch's one guarantee is
            // worse than saying no.
            if (op.if_contains("clip") != nullptr)
                return fail(api_reply::fail(api_code::bad_request,
                                            "a clip load cannot be part of an atomic batch; "
                                            "POST it to /v1/action first, then batch the rest"));

            pl.channel = pl.action.channel;
        } else {
            return fail(api_reply::fail(api_code::bad_request,
                                        "unknown op in a batch: " + kind +
                                            " (set, action, timeline, graph)"));
        }

        if (!ctx.stage || !ctx.stage(pl.channel))
            return fail(api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(pl.channel)));
        out.ops.push_back(std::move(pl));
    }

    // The reference channel for `at_frame` is the LOWEST channel the batch touches, chosen
    // rather than left to the caller so two clients naming the same frame mean the same
    // instant. Every channel this server drives advances on the same tick, so the choice is
    // only visible if a channel is added or removed mid-show.
    out.reference_channel = out.ops.front().channel;
    for (const auto& op : out.ops)
        out.reference_channel = std::min(out.reference_channel, op.channel);

    if (auto* at = o.if_contains("at_frame")) {
        if (!at->is_int64() && !at->is_uint64() && !at->is_double())
            return api_reply::fail(api_code::field_wrong_type, "at_frame must be a frame number");
        out.at_frame = static_cast<std::int64_t>(at->to_number<double>());
    }
    if (auto* in = o.if_contains("in_frames")) {
        if (out.at_frame != 0)
            return api_reply::fail(api_code::bad_request, "give at_frame or in_frames, not both");
        if (!in->is_int64() && !in->is_uint64() && !in->is_double())
            return api_reply::fail(api_code::field_wrong_type, "in_frames must be a number of frames");
        out.at_frame = -static_cast<std::int64_t>(in->to_number<double>()); //< resolved by the caller
    }

    return api_reply{};
}

api_reply apply_batch(const api_context& ctx, const batch_plan& plan)
{
    if (!ctx.concrete_stage || !ctx.channel_count)
        return api_reply::fail(api_code::internal, "the API was built without batch support");

    // One `stage_delayed` per channel, each holding a blocked executor. Every op is QUEUED
    // against its channel's delayed stage, every touched channel is then locked, and only
    // then are the executors released -- so a channel's tick cannot run between the first
    // op landing and the last. That is what makes a two-channel batch land on one frame.
    const int                                          n = ctx.channel_count();
    std::map<int, std::shared_ptr<core::stage_delayed>> delayed;
    std::vector<std::future<void>>                     results;
    std::vector<std::unique_lock<std::mutex>>          locks;
    /// (op index, document name, "did the channel own it") for each `timeline` op.
    std::vector<std::tuple<std::size_t, std::string, std::future<bool>>> verdicts;

    try {
        for (const auto& pl : plan.ops) {
            if (delayed.find(pl.channel) != delayed.end())
                continue;
            if (pl.channel < 1 || pl.channel > n)
                return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(pl.channel));
            auto st = ctx.concrete_stage(pl.channel);
            if (!st)
                return api_reply::fail(api_code::channel_not_found, "no channel " + std::to_string(pl.channel));
            delayed[pl.channel] = std::make_shared<core::stage_delayed>(st, pl.channel);
        }

        for (const auto& pl : plan.ops) {
            auto& st = delayed[pl.channel];
            if (pl.is_timeline) {
                // THROUGH THE DELAYED STAGE like every other op -- `stage_base` carries
                // `timeline_command` for exactly this. Reaching past it to the real stage would
                // not merely land on the wrong frame: the delayed stage is holding that
                // channel's executor, so the call would block this thread against a lock it is
                // itself responsible for releasing.
                //
                // The command then queues into the document's pending list and the tick applies
                // a frame's worth at once under `stop > pause > run`, so a batch's `PLAY` and a
                // batch's field writes land on the same tick.
                //
                // A SEPARATE VECTOR because this one answers `bool`, and the answer is checked
                // after the release rather than discarded: false means the channel does not own
                // the document, which validation could not rule out -- the store is mutable and
                // a `DELETE` can land between the two.
                verdicts.emplace_back(pl.index, pl.timeline.name,
                                      st->timeline_command(pl.timeline.name, pl.timeline.cmd));
                continue;
            }
            if (pl.is_graph) {
                // The same `bool` shape a timeline op has, and checked after the release for the
                // same reason: `attach` can fail because another layer claimed the document
                // between validation and the frame, and the store is mutable in between.
                verdicts.emplace_back(pl.index, pl.graph.name,
                                      st->graph_command(pl.graph.name, pl.graph.verb,
                                                        pl.graph.layer));
                continue;
            }
            if (pl.is_node) {
                // NOT `apply_transform`. A node parameter's constant lives in the attached
                // DOCUMENT, so `set_node_param` is the only writer that reaches it -- and the
                // transform path this used to fall into had no node arm at all, so the write
                // did nothing while the batch reported success.
                verdicts.emplace_back(pl.index, pl.node.path,
                                      st->set_node_param(pl.node.layer, pl.node.path,
                                                         pl.node.value, pl.node.label));
                continue;
            }
            if (pl.is_set) {
                results.push_back(
                    st->apply_transform(pl.set.target.layer, set_closure(pl.set), pl.set.duration, pl.set.tween));
            } else {
                // The same verb table a lone action uses, queued rather than waited on --
                // see `queue_verb`.
                std::future<void> f;
                if (auto r = queue_verb(st, pl.action, f); r.code != api_code::ok) {
                    json::object dd;
                    dd["index"]   = static_cast<std::int64_t>(pl.index);
                    dd["code"]    = to_string(r.code);
                    dd["message"] = r.message;
                    for (auto& kv : delayed)
                        kv.second->abort();
                    return api_reply::fail(api_code::batch_op_failed,
                                           "op " + std::to_string(pl.index) + " failed while applying",
                                           json::array{std::move(dd)});
                }
                results.push_back(std::move(f));
            }
        }

        for (auto& kv : delayed)
            if (kv.second->count_queued() > 0)
                locks.push_back(kv.second->get_lock());

        for (auto& kv : delayed)
            kv.second->release();
    } catch (const std::exception& e) {
        // NO `catch (...)` AFTER THIS, for the reason given at the other site: it would catch a
        // structured exception and nothing else, since C++ exceptions land here. Losing the
        // `abort()` of the delayed locks to an access violation costs nothing -- the process is
        // already corrupt and the right outcome is a dump, not a tidy 500.
        for (auto& kv : delayed)
            kv.second->abort();
        CASPAR_LOG_CURRENT_EXCEPTION();
        return api_reply::fail(api_code::internal, std::string("the batch threw while being applied: ") + e.what());
    }

    try {
        for (auto& kv : delayed)
            kv.second->wait();
        locks.clear();
        for (auto& f : results)
            f.get();
    } catch (const std::exception& e) {
        CASPAR_LOG_CURRENT_EXCEPTION();
        return api_reply::fail(api_code::internal, std::string("the batch threw while settling: ") + e.what());
    }

    // A TIMELINE OP THAT THE CHANNEL REFUSED. Reported AFTER the batch has landed rather than
    // rolled back, and that is a limit stated rather than hidden: the field writes have already
    // been applied by the time this is knowable, so the honest answer is "the batch landed and
    // this op did not" instead of a rollback the design cannot perform.
    for (auto& v : verdicts) {
        if (std::get<2>(v).get())
            continue;
        json::object dd;
        dd["index"]   = static_cast<std::int64_t>(std::get<0>(v));
        dd["code"]    = to_string(api_code::timeline_not_found);
        dd["message"] = "the channel no longer owns document '" + std::get<1>(v) +
                        "' -- it was deleted or replaced between validation and apply";
        return api_reply::fail(api_code::batch_op_failed,
                               "op " + std::to_string(std::get<0>(v)) +
                                   " was refused after the rest of the batch had landed",
                               json::array{std::move(dd)});
    }

    CASPAR_LOG(info) << L"[api] " << u16(plan.peer) << L" batch ops=" << plan.ops.size() << L" channels="
                     << delayed.size() << (plan.at_frame > 0 ? (L" at_frame=" + std::to_wstring(plan.at_frame)) : L"")
                     << (plan.label.empty() ? L"" : (L" label=\"" + u16(plan.label) + L"\""));

    json::object r;
    r["ops"]      = static_cast<std::int64_t>(plan.ops.size());
    r["channels"] = static_cast<std::int64_t>(delayed.size());
    if (!plan.queue.empty())
        r["queue"] = plan.queue;
    if (plan.at_frame > 0)
        r["at_frame"] = plan.at_frame;
    return api_reply::ok_with(std::move(r));
}

api_reply run_batch(const api_context& ctx, const std::string& body, const std::string& peer, batch_plan& deferred)
{
    batch_plan plan;
    plan.peer = peer;
    if (auto r = validate_batch(ctx, body, plan); r.code != api_code::ok)
        return r;

    if (plan.at_frame == 0)
        return apply_batch(ctx, plan);

    // Deferred. The caller owns the waiting -- it is the only part of this file that knows
    // what frame the server is on, and holding an HTTP socket open for fifty frames to
    // avoid telling it would be a worse trade than any it saves.
    deferred = std::move(plan);
    return api_reply{};
}

}}} // namespace caspar::protocol::http
