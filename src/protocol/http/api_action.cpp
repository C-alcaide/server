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
#include "api_value.h"

#include <common/log.h>
#include <common/utf.h>

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

/// Queue one verb against a stage and return its future WITHOUT waiting.
///
/// The separation matters inside a batch and nowhere else. A batch queues its ops against
/// `stage_delayed` executors that are deliberately BLOCKED until every channel is locked;
/// waiting on one of those futures before the release is a guaranteed deadlock, and it is
/// exactly what the first version did -- `run_action` called `.get()`, the batch hung, and
/// the socket timed out with nothing in the log.
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
        } catch (...) {
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
        return api_reply::fail(api_code::internal, e.what());
    } catch (...) {
        return api_reply::fail(api_code::internal, "the action threw");
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
    } catch (...) {
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
        const auto* p = op.if_contains("path");
        if (!p || !p->is_string())
            return fail(api_reply::fail(api_code::field_missing, "an op needs a path"));
        const std::string path = p->as_string().c_str();

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
            return fail(api_reply::fail(api_code::bad_request, "unknown op in a batch: " + kind));
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
        for (auto& kv : delayed)
            kv.second->abort();
        CASPAR_LOG_CURRENT_EXCEPTION();
        return api_reply::fail(api_code::internal, std::string("the batch threw while being applied: ") + e.what());
    } catch (...) {
        for (auto& kv : delayed)
            kv.second->abort();
        CASPAR_LOG_CURRENT_EXCEPTION();
        return api_reply::fail(api_code::internal, "the batch threw while being applied");
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
