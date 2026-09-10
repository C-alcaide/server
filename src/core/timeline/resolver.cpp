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

#include "resolver.h"

#include <common/log.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <set>

namespace caspar { namespace core { namespace timeline {

const instance* resolved_timeline::active_on(const std::string& layer, flicks t) const
{
    const auto it = by_layer.find(layer);
    if (it == by_layer.end())
        return nullptr;

    const instance* best = nullptr;
    for (const auto i : it->second) {
        const auto& inst = instances[i];
        if (!inst.contains(t))
            continue;
        if (!best) {
            best = &inst;
            continue;
        }
        if (inst.priority != best->priority) {
            if (inst.priority > best->priority)
                best = &inst;
            continue;
        }
        // Equal priority: LAST-STARTED wins. `by_layer` is in start order and ties fall through
        // to declaration order, so `>=` here means "the later of two simultaneous starts", which
        // is the document order a client controls.
        if (inst.start >= best->start)
            best = &inst;
    }
    return best;
}

namespace {

// ---------------------------------------------------------------------------------------
// Pass 1 -- flatten the tree, composing the remaps
// ---------------------------------------------------------------------------------------

struct flat_object
{
    const timeline_object* obj = nullptr;
    std::string            parent_id;
    time_remap             composed;
    std::size_t            order = 0; //< document order, for tie-breaking
    bool                   inherited_disabled = false;
};

void flatten(const std::vector<timeline_object>& objs, const std::string& parent_id,
             const time_remap& outer, bool parent_disabled, std::vector<flat_object>& out)
{
    for (const auto& o : objs) {
        flat_object f;
        f.obj                = &o;
        f.parent_id          = parent_id;
        f.composed           = outer.then(o.remap);
        f.order              = out.size();
        f.inherited_disabled = parent_disabled || o.disabled;
        out.push_back(f);
        if (!o.children.empty())
            flatten(o.children, o.id, f.composed, f.inherited_disabled, out);
    }
}

// ---------------------------------------------------------------------------------------
// Pass 2 -- dependency order
// ---------------------------------------------------------------------------------------

void collect_refs(const enable_spec& e, std::vector<const time_expr*>& out)
{
    for (const auto* p : {&e.start, &e.end, &e.duration, &e.while_})
        if (*p)
            out.push_back(&**p);
}

/// Every object id an object's expressions depend on, class references expanded to members.
std::vector<std::string> dependencies_of(const flat_object&                                  f,
                                         const std::map<std::string, std::vector<std::string>>& by_class)
{
    std::vector<std::string> deps;
    for (const auto& spec : f.obj->enable) {
        std::vector<const time_expr*> exprs;
        collect_refs(spec, exprs);
        for (const auto* e : exprs) {
            if (e->k == time_expr::kind::class_start || e->k == time_expr::kind::class_end) {
                const auto it = by_class.find(e->ref);
                if (it != by_class.end())
                    for (const auto& id : it->second)
                        if (id != f.obj->id)
                            deps.push_back(id);
            } else if (e->is_reference()) {
                deps.push_back(e->ref);
            }
        }
    }
    // A CHILD DEPENDS ON ITS PARENT, always, even with no expression: its absolute time is the
    // parent's start plus its own, and `one_at_a_time` makes it depend on its siblings too.
    if (!f.parent_id.empty())
        deps.push_back(f.parent_id);
    std::sort(deps.begin(), deps.end());
    deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
    return deps;
}

} // namespace

resolved_timeline resolve(const timeline_document& doc, const trigger_log& triggers)
{
    resolved_timeline out;
    out.revision = doc.revision;

    std::vector<flat_object> flat;
    flatten(doc.objects, "", time_remap{}, false, flat);

    // Index by id, and refuse duplicates: two objects with one id makes every reference to it
    // ambiguous, and picking one silently is how a show animates the wrong thing.
    std::map<std::string, std::size_t> by_id;
    for (std::size_t i = 0; i < flat.size(); ++i) {
        const auto& id = flat[i].obj->id;
        if (id.empty()) {
            out.errors.push_back({"", "", "every object needs an id -- references and published "
                                          "ownership both key on it"});
            continue;
        }
        if (!by_id.emplace(id, i).second)
            out.errors.push_back({id, "", "duplicate object id"});
    }

    std::map<std::string, std::vector<std::string>> by_class;
    for (const auto& f : flat)
        for (const auto& c : f.obj->classes)
            by_class[c].push_back(f.obj->id);

    if (!out.errors.empty())
        return out;

    // ---- dependency order, by depth-first search with a cycle report ----------------
    std::map<std::string, int>   state; // 0 unseen, 1 in progress, 2 done
    std::vector<std::size_t>     order;
    bool                         cycle = false;

    const std::function<void(const std::string&)> visit = [&](const std::string& id) {
        if (cycle)
            return;
        const auto it = by_id.find(id);
        if (it == by_id.end())
            return; // an unknown reference; reported when the expression is evaluated
        auto& st = state[id];
        if (st == 2)
            return;
        if (st == 1) {
            out.errors.push_back({id, "", "circular reference: this object's start depends on "
                                          "itself, directly or through another object"});
            cycle = true;
            return;
        }
        st = 1;
        for (const auto& dep : dependencies_of(flat[it->second], by_class))
            visit(dep);
        st = 2;
        order.push_back(it->second);
    };

    for (const auto& f : flat)
        visit(f.obj->id);
    if (cycle)
        return out;

    // ---- evaluate, in that order ----------------------------------------------------
    //
    // Each object's instances are appended as they resolve, and later objects read them back
    // through `out.by_object`. That is the whole reason for the ordering pass: `#a.end + 5` is
    // answerable only once `a` has an end.

    struct span
    {
        flicks                start;
        std::optional<flicks> end;
    };
    std::map<std::string, std::vector<span>> spans; //< by object id, in occurrence order

    const auto object_start = [&](const std::string& id) -> std::optional<flicks> {
        const auto it = spans.find(id);
        if (it == spans.end() || it->second.empty())
            return std::nullopt;
        return it->second.front().start;
    };
    const auto object_end = [&](const std::string& id) -> std::optional<std::optional<flicks>> {
        const auto it = spans.find(id);
        if (it == spans.end() || it->second.empty())
            return std::nullopt;
        return it->second.back().end; //< the LAST occurrence's end, which may itself be open
    };

    // `resolve_expr` answers a time, or nothing plus a reason. `open` distinguishes "this
    // reference is to something that has not ended" from "this reference is broken", which are
    // different answers and used to be the same one.
    const auto resolve_expr = [&](const time_expr& e, const std::string& owner, bool& open,
                                  std::string& reason) -> std::optional<flicks> {
        open = false;
        switch (e.k) {
            case time_expr::kind::literal:
                return e.literal;

            case time_expr::kind::always:
                return flicks{0};

            case time_expr::kind::trigger: {
                const auto t = triggers.next_after(e.trigger, 0);
                if (!t) {
                    open = true;
                    out.pending_triggers.push_back(e.trigger);
                    return std::nullopt;
                }
                return *t + e.literal;
            }

            case time_expr::kind::ref_start: {
                const auto s = object_start(e.ref);
                if (!s) {
                    reason = "no object '" + e.ref + "'";
                    return std::nullopt;
                }
                return *s + e.literal;
            }

            case time_expr::kind::ref_end: {
                const auto en = object_end(e.ref);
                if (!en) {
                    reason = "no object '" + e.ref + "'";
                    return std::nullopt;
                }
                if (!*en) {
                    open = true;
                    return std::nullopt;
                }
                return **en + e.literal;
            }

            case time_expr::kind::ref_duration: {
                const auto s  = object_start(e.ref);
                const auto en = object_end(e.ref);
                if (!s || !en) {
                    reason = "no object '" + e.ref + "'";
                    return std::nullopt;
                }
                if (!*en) {
                    open = true;
                    return std::nullopt;
                }
                return (**en - *s) + e.literal;
            }

            case time_expr::kind::class_start: {
                const auto it = by_class.find(e.ref);
                if (it == by_class.end()) {
                    reason = "no object carries the class '" + e.ref + "'";
                    return std::nullopt;
                }
                std::optional<flicks> earliest;
                for (const auto& id : it->second) {
                    if (id == owner)
                        continue; //< a member referencing its own class does not reference itself
                    if (const auto s = object_start(id))
                        earliest = earliest ? std::min(*earliest, *s) : *s;
                }
                if (!earliest) {
                    reason = "class '" + e.ref + "' has no resolvable member";
                    return std::nullopt;
                }
                return *earliest + e.literal;
            }

            case time_expr::kind::class_end: {
                const auto it = by_class.find(e.ref);
                if (it == by_class.end()) {
                    reason = "no object carries the class '" + e.ref + "'";
                    return std::nullopt;
                }
                std::optional<flicks> latest;
                bool                  any_open = false;
                for (const auto& id : it->second) {
                    if (id == owner)
                        continue;
                    if (const auto en = object_end(id)) {
                        if (!*en)
                            any_open = true;
                        else
                            latest = latest ? std::max(*latest, **en) : **en;
                    }
                }
                if (any_open) {
                    // The LATEST end of a set containing an open one is open. Taking the latest
                    // definite end instead would place a dependent object before something in the
                    // set had finished, which is the one thing the expression promises not to do.
                    open = true;
                    return std::nullopt;
                }
                if (!latest) {
                    reason = "class '" + e.ref + "' has no resolvable member";
                    return std::nullopt;
                }
                return *latest + e.literal;
            }
        }
        reason = "unhandled expression kind";
        return std::nullopt;
    };

    for (const auto idx : order) {
        const auto& f  = flat[idx];
        const auto& o  = *f.obj;

        // `one_at_a_time`: a child starts where the previous SIBLING ended. Computed here rather
        // than expressed as `#prev.end` in the document, because the client should not have to
        // rewrite every child's expression when one is inserted in the middle.
        std::optional<flicks> sequence_cursor;
        std::optional<flicks> group_start;
        bool                  waits_for_go = false;
        int                   cue_index    = 0;
        if (!f.parent_id.empty()) {
            const auto pit = by_id.find(f.parent_id);
            if (pit != by_id.end() && flat[pit->second].obj->play.one_at_a_time) {
                const auto& parent = *flat[pit->second].obj;
                for (const auto& sib : parent.children) {
                    if (sib.id == o.id)
                        break;
                    ++cue_index;
                    const auto en = object_end(sib.id);
                    if (en && *en)
                        sequence_cursor = **en;
                }
                group_start = object_start(f.parent_id);
                if (!sequence_cursor)
                    sequence_cursor = group_start.value_or(0);

                // A CUE STACK WITHOUT `auto_play` WAITS FOR A GO between cues, which is what
                // distinguishes a cue stack from a sequence. `auto_play` is Hippotizer's and
                // WATCHOUT's "follow-on"; without it every cue after the first starts on the
                // Nth firing of the group's trigger, so an operator's GO advances the stack.
                //
                // The FIRST cue does not wait: the group's own start is when the stack begins,
                // and making cue 1 need a GO as well would mean two actions to start a show.
                waits_for_go = !parent.play.auto_play && cue_index > 0;
            }
        }

        auto specs = o.enable;
        if (specs.empty()) {
            // No enable at all: active for as long as its parent is, or from 0 for a root. That
            // is what makes a group a container rather than something that has to declare a span
            // covering its children.
            enable_spec s;
            time_expr   e;
            e.k = time_expr::kind::always;
            s.while_ = e;
            specs.push_back(s);
        }

        for (const auto& spec : specs) {
            std::string           reason;
            bool                  open_start = false, open_end = false;
            std::optional<flicks> start, end;

            if (spec.while_) {
                if (spec.while_->k == time_expr::kind::always) {
                    if (!f.parent_id.empty()) {
                        start = object_start(f.parent_id).value_or(0);
                        if (const auto pe = object_end(f.parent_id))
                            end = *pe;
                    } else {
                        start = 0;
                    }
                } else {
                    start = resolve_expr(time_expr{time_expr::kind::ref_start, spec.while_->literal,
                                                   spec.while_->ref, {}},
                                         o.id, open_start, reason);
                    end = resolve_expr(time_expr{time_expr::kind::ref_end, spec.while_->literal,
                                                 spec.while_->ref, {}},
                                       o.id, open_end, reason);
                }
            } else {
                if (spec.start) {
                    start = resolve_expr(*spec.start, o.id, open_start, reason);
                } else if (waits_for_go) {
                    // THE Nth FIRING AT-OR-AFTER THE GROUP'S START. Counted from the group's
                    // start rather than from the previous cue's END, and that is the semantic
                    // choice rather than an implementation detail: a GO takes the next cue NOW,
                    // whenever it is pressed, which is what a lighting desk does and what an
                    // operator expects. Counting from the previous cue's end would mean a GO
                    // pressed while a cue was still running did nothing -- and would leave cue 3
                    // waiting for a THIRD firing, which is what the first version did and what
                    // `resolver_self_test` said at boot.
                    //
                    // A GO pressed early therefore starts the next cue while the current one's
                    // span is still open; last-started-wins gives the layer to the new cue,
                    // which is again what a desk does.
                    const auto from = group_start.value_or(0);
                    const auto it   = triggers.fired.find("go");
                    int        seen = 0;
                    std::optional<flicks> at;
                    if (it != triggers.fired.end()) {
                        for (const auto t : it->second) {
                            if (t < from)
                                continue;
                            if (++seen == cue_index) {
                                at = t;
                                break;
                            }
                        }
                    }
                    if (at) {
                        start = *at;
                    } else {
                        // Waiting. NOT an error and not a start of zero: an unfired cue has no
                        // position, and giving it one would put it on air.
                        open_start = true;
                        out.pending_triggers.push_back("go");
                    }
                } else if (sequence_cursor) {
                    start = *sequence_cursor;
                } else {
                    start = 0;
                }

                if (spec.end)
                    end = resolve_expr(*spec.end, o.id, open_end, reason);

                if (!end && spec.duration && start) {
                    bool        dopen = false;
                    std::string dreason;
                    if (const auto d = resolve_expr(*spec.duration, o.id, dopen, dreason))
                        end = *start + *d;
                    else if (!dopen)
                        reason = dreason;
                }

                // Over-determined and disagreeing is an ERROR, not a precedence rule. All three
                // of start, end and duration given, with end != start + duration, is a document
                // whose author believes two contradictory things; answering with one of them is
                // how a show goes wrong in a way nobody can debug.
                if (spec.start && spec.end && spec.duration && start && end) {
                    bool        dopen = false;
                    std::string dreason;
                    if (const auto d = resolve_expr(*spec.duration, o.id, dopen, dreason)) {
                        if (*end != *start + *d)
                            out.errors.push_back(
                                {o.id, "", "start, end and duration are all given and disagree: "
                                           "end is not start + duration"});
                    }
                }
            }

            if (!start) {
                if (!open_start)
                    out.errors.push_back({o.id, "", reason.empty() ? "unresolvable start" : reason});
                continue;
            }
            if (!end && !open_end && !reason.empty())
                out.errors.push_back({o.id, "", reason});

            // A REPEATING SPEC expands here rather than at evaluation time, so `#a.end` on the
            // third repeat means the third repeat's end and the client's `resolved` view shows
            // three objects rather than one it has to unroll itself.
            const int count = spec.repeating.on ? (spec.repeating.count > 0 ? spec.repeating.count : 1)
                                                : 1;
            for (int r = 0; r < count; ++r) {
                instance in;
                in.object_id    = o.id;
                in.layer        = o.layer;
                in.priority     = o.priority;
                in.composed     = f.composed;
                in.repeat_index = r;
                in.start        = *start + f.composed.offset + r * spec.repeating.period;
                if (end)
                    in.end = *end + f.composed.offset + r * spec.repeating.period;

                if (in.end && *in.end < in.start) {
                    out.errors.push_back({o.id, "", "end is before start"});
                    continue;
                }

                if (f.inherited_disabled)
                    continue; //< resolved, so `#its.end` still works, but never active

                spans[o.id].push_back({in.start, in.end});
                out.instances.push_back(std::move(in));
            }
            // An object whose instances were all skipped as disabled still needs a span, so a
            // reference to it resolves rather than reporting a missing object.
            if (f.inherited_disabled) {
                const auto s = *start + f.composed.offset;
                spans[o.id].push_back({s, end ? std::optional<flicks>(*end + f.composed.offset)
                                              : std::nullopt});
            }
        }
    }

    // ---- sort and index -------------------------------------------------------------
    //
    // Stable, so equal starts keep DOCUMENT ORDER -- which is what `active_on`'s final tie-break
    // relies on. An unstable sort would make the winner of two simultaneous cues depend on the
    // standard library's implementation.
    std::stable_sort(out.instances.begin(), out.instances.end(),
                     [](const instance& a, const instance& b) { return a.start < b.start; });

    for (std::size_t i = 0; i < out.instances.size(); ++i) {
        out.by_object[out.instances[i].object_id].push_back(i);
        if (!out.instances[i].layer.empty())
            out.by_layer[out.instances[i].layer].push_back(i);
    }

    std::sort(out.pending_triggers.begin(), out.pending_triggers.end());
    out.pending_triggers.erase(std::unique(out.pending_triggers.begin(), out.pending_triggers.end()),
                               out.pending_triggers.end());
    return out;
}

// =======================================================================================
// Self-test
// =======================================================================================

namespace {

timeline_object obj(const char* id, const char* layer, std::optional<double> start,
                    std::optional<double> end)
{
    timeline_object o;
    o.id    = id;
    o.layer = layer;
    enable_spec s;
    if (start) {
        time_expr e;
        e.literal = from_seconds(*start);
        s.start   = e;
    }
    if (end) {
        time_expr e;
        e.literal = from_seconds(*end);
        s.end     = e;
    }
    o.enable.push_back(s);
    return o;
}

enable_spec spec_expr(const char* start_text, const char* end_text = nullptr)
{
    parse_context ctx;
    enable_spec   s;
    time_expr     e;
    std::string   err;
    if (start_text) {
        if (!parse_time_expr(start_text, ctx, e, err))
            std::abort();
        s.start = e;
    }
    if (end_text) {
        if (!parse_time_expr(end_text, ctx, e, err))
            std::abort();
        s.end = e;
    }
    return s;
}

} // namespace

void resolver_self_test()
{
    expression_self_test();

    const auto req = [](bool ok, const char* what) {
        if (!ok) {
            CASPAR_LOG(fatal) << L"timeline::resolver_self_test: " << what;
            std::abort();
        }
    };
    const trigger_log no_triggers;

    // ---- 1. a chain of `#a.end` references ----------------------------------------------
    {
        timeline_document d;
        d.objects.push_back(obj("a", "1-10", 0.0, 4.0));

        timeline_object b;
        b.id    = "b";
        b.layer = "1-11";
        b.enable.push_back(spec_expr("#a.end", "#a.end + 2"));
        d.objects.push_back(b);

        timeline_object c;
        c.id    = "c";
        c.layer = "1-12";
        c.enable.push_back(spec_expr("#b.end + 1", "#b.end + 3"));
        d.objects.push_back(c);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a three-object chain resolves");
        req(r.instances.size() == 3, "three instances");

        const auto find = [&](const char* id) -> const instance& {
            return r.instances[r.by_object.at(id).front()];
        };
        req(find("b").start == from_seconds(4.0), "b starts where a ends");
        req(*find("b").end == from_seconds(6.0), "and ends two seconds later");
        req(find("c").start == from_seconds(7.0), "c starts one second after b ends");
        req(*find("c").end == from_seconds(9.0), "and runs two seconds");

        // THE ORDER OF DECLARATION MUST NOT MATTER. `c` before `a` is the same document.
        timeline_document reversed;
        reversed.objects.assign(d.objects.rbegin(), d.objects.rend());
        const auto r2 = resolve(reversed, no_triggers);
        req(r2.ok(), "the same document declared backwards resolves");
        req(r2.instances[r2.by_object.at("c").front()].start == from_seconds(7.0),
            "and c lands in the same place -- the dependency pass, not the declaration order");
    }

    // ---- 2. a class reference with an offset ---------------------------------------------
    {
        timeline_document d;
        auto              lt1 = obj("lt1", "1-10", 6.0, 8.0);
        lt1.classes           = {"lt"};
        auto lt2              = obj("lt2", "1-11", 2.0, 12.0);
        lt2.classes           = {"lt"};
        d.objects.push_back(lt1);
        d.objects.push_back(lt2);

        timeline_object tail;
        tail.id    = "tail";
        tail.layer = "1-12";
        tail.enable.push_back(spec_expr(".lt.start + 5", ".lt.end"));
        d.objects.push_back(tail);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a class reference resolves");
        const auto& t = r.instances[r.by_object.at("tail").front()];
        req(t.start == from_seconds(7.0), ".lt.start is the EARLIEST member start (2), plus 5");
        req(*t.end == from_seconds(12.0), "and .lt.end is the LATEST member end (12)");
    }

    // ---- 3. priority, then last-started -------------------------------------------------
    {
        timeline_document d;
        auto              bg = obj("bg", "1-10", 0.0, 100.0);
        auto              cue = obj("cue", "1-10", 10.0, 20.0);
        d.objects.push_back(bg);
        d.objects.push_back(cue);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "two objects on one layer resolve");
        req(r.active_on("1-10", from_seconds(5.0))->object_id == "bg", "before the cue, bg owns it");
        req(r.active_on("1-10", from_seconds(15.0))->object_id == "cue",
            "during the cue, LAST-STARTED wins -- not the long background object");
        req(r.active_on("1-10", from_seconds(25.0))->object_id == "bg", "after it, bg again");
        req(r.active_on("1-10", from_seconds(150.0)) == nullptr, "and nothing after everything");

        // PRIORITY OUTRANKS LAST-STARTED. Same document, bg promoted.
        timeline_document p = d;
        p.objects[0].priority = 10;
        const auto rp         = resolve(p, no_triggers);
        req(rp.active_on("1-10", from_seconds(15.0))->object_id == "bg",
            "a higher priority holds the layer through a later cue");
    }

    // ---- 4. a transparent anchor, referenced three times --------------------------------
    {
        timeline_document d;
        timeline_object   anchor = obj("interview", "", 3.0, 30.0);
        d.objects.push_back(anchor);
        for (const char* id : {"x", "y", "z"}) {
            timeline_object o;
            o.id    = id;
            o.layer = std::string("1-1") + id[0];
            o.enable.push_back(spec_expr("#interview.end", "#interview.end + 1"));
            d.objects.push_back(o);
        }

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "an anchor with three dependents resolves");
        req(r.instances.size() == 4, "four instances, the anchor included");
        req(r.by_layer.find("") == r.by_layer.end(), "the anchor occupies NO layer");
        for (const char* id : {"x", "y", "z"})
            req(r.instances[r.by_object.at(id).front()].start == from_seconds(30.0),
                "and every dependent starts where it ends");
    }

    // ---- 5. a nested group at rate 2 ----------------------------------------------------
    {
        timeline_document d;
        timeline_object   g;
        g.id         = "g";
        g.is_group   = true;
        g.remap.rate = 2;
        g.remap.offset = from_seconds(10.0);
        g.enable.push_back(spec_expr("0", "100"));

        timeline_object inner = obj("inner", "1-10", 0.0, 4.0);
        g.children.push_back(inner);
        d.objects.push_back(g);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a nested group resolves");
        const auto& in = r.instances[r.by_object.at("inner").front()];
        req(in.start == from_seconds(10.0), "the group's offset places the child");
        req(boost::rational_cast<double>(in.composed.rate) == 2.0, "and its rate composes onto it");
        // A child four local seconds long, at rate 2, is two seconds of DOCUMENT time -- and the
        // mapping is what `local_at` inverts, which is the property the tick depends on.
        req(in.local_at(from_seconds(11.0)) == from_seconds(2.0),
            "one document second into the child is two LOCAL seconds at rate 2");
        req(in.local_at(from_seconds(10.0)) == 0, "and its own start is local zero");
    }

    // ---- 6. repeating, three times ------------------------------------------------------
    {
        timeline_document d;
        timeline_object   o;
        o.id    = "blink";
        o.layer = "1-10";
        enable_spec s        = spec_expr("0", "1");
        s.repeating.on       = true;
        s.repeating.period   = from_seconds(5.0);
        s.repeating.count    = 3;
        o.enable.clear();
        o.enable.push_back(s);
        d.objects.push_back(o);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a repeating spec resolves");
        req(r.instances.size() == 3, "into three instances, not one the caller has to unroll");
        req(r.instances[0].start == 0, "the first at 0");
        req(r.instances[1].start == from_seconds(5.0), "the second a period later");
        req(r.instances[2].start == from_seconds(10.0), "and the third");
        req(r.instances[2].repeat_index == 2, "each carrying its own index");
        req(*r.instances[1].end == from_seconds(6.0), "with the span carried along");
    }

    // ---- 7. one_at_a_time + auto_play: children in sequence, no GO needed ---------------
    //
    // `auto_play` was added to this fixture when cue stacks learned to WAIT: back-to-back
    // sequencing is the auto_play behaviour, and without the flag every cue after the first now
    // waits for a GO. The case below covers that half. This test threw `invalid unordered_map
    // key` the moment the distinction existed, which is a fixture that had encoded the only
    // behaviour there was.
    {
        timeline_document d;
        timeline_object   stack;
        stack.id                 = "stack";
        stack.is_group           = true;
        stack.play.one_at_a_time = true;
        stack.play.auto_play     = true;
        stack.enable.push_back(spec_expr("2", "100"));

        // NO start on any child. That is the point: a cue stack's children say how LONG they
        // are, and where they fall follows from the ones before.
        for (const char* id : {"c1", "c2", "c3"}) {
            timeline_object c;
            c.id    = id;
            c.layer = "1-10";
            enable_spec s;
            time_expr   dur;
            dur.literal = from_seconds(4.0);
            s.duration  = dur;
            c.enable.push_back(s);
            stack.children.push_back(c);
        }
        d.objects.push_back(stack);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a cue stack resolves");
        req(r.instances[r.by_object.at("c1").front()].start == from_seconds(2.0),
            "the first cue starts where the group does");
        req(r.instances[r.by_object.at("c2").front()].start == from_seconds(6.0),
            "the second where the first ended");
        req(r.instances[r.by_object.at("c3").front()].start == from_seconds(10.0),
            "and the third where the second did");
    }

    // ---- 7b. one_at_a_time WITHOUT auto_play: each cue waits for a GO --------------------
    {
        timeline_document d;
        timeline_object   stack;
        stack.id                 = "stack";
        stack.is_group           = true;
        stack.play.one_at_a_time = true;
        stack.play.auto_play     = false; //< the difference from the case above
        stack.enable.push_back(spec_expr("0", "1000"));

        for (const char* id : {"c1", "c2", "c3"}) {
            timeline_object c;
            c.id    = id;
            c.layer = "1-10";
            enable_spec s;
            time_expr   dur;
            dur.literal = from_seconds(4.0);
            s.duration  = dur;
            c.enable.push_back(s);
            stack.children.push_back(c);
        }
        d.objects.push_back(stack);

        // Nothing fired: the FIRST cue is live and the other two are waiting.
        const auto r0 = resolve(d, no_triggers);
        req(r0.ok(), "an unfired cue stack resolves");
        req(r0.by_object.count("c1") == 1, "the first cue does not wait -- the group's start is "
                                           "when the stack begins, and needing a GO to start as "
                                           "well would mean two actions to start a show");
        req(r0.by_object.count("c2") == 0, "the second waits");
        req(r0.by_object.count("c3") == 0, "and so does the third");
        req(!r0.pending_triggers.empty(), "and the trigger they wait on is published");

        // One GO, at 10 s: cue 2 starts there. Cue 3 still waits.
        trigger_log one;
        one.fired["go"] = {from_seconds(10.0)};
        const auto r1   = resolve(d, one);
        req(r1.ok() && r1.by_object.count("c2") == 1, "one GO starts the second cue");
        req(r1.instances[r1.by_object.at("c2").front()].start == from_seconds(10.0),
            "at the moment the GO fired, not where the first cue ended");
        req(r1.by_object.count("c3") == 0, "and the third still waits");

        // Two GOs: the SECOND one starts cue 3. Counting matters -- one GO must not advance
        // two cues, which a "has anything fired" test would do.
        trigger_log two;
        two.fired["go"] = {from_seconds(10.0), from_seconds(25.0)};
        const auto r2   = resolve(d, two);
        req(r2.by_object.count("c3") == 1, "two GOs start the third cue");
        req(r2.instances[r2.by_object.at("c3").front()].start == from_seconds(25.0),
            "at the SECOND firing -- one GO must not advance two cues");
        req(r2.instances[r2.by_object.at("c2").front()].start == from_seconds(10.0),
            "and the second cue is still where its own GO put it");

        // A GO BEFORE THE STACK BEGAN does not advance it. The group starts at 0 here, so
        // a firing at a negative position is the only way to express "before" -- which is
        // what a seek backwards produces.
        trigger_log early;
        early.fired["go"] = {from_seconds(-5.0)};
        const auto r3     = resolve(d, early);
        req(r3.by_object.count("c2") == 0,
            "a GO from before the stack began does not advance it");
    }

    // ---- 8. an OPEN end, and what it does to a dependent ---------------------------------
    {
        timeline_document d;
        timeline_object   held;
        held.id    = "held";
        held.layer = "1-10";
        enable_spec s;
        time_expr   st;
        st.literal = 0;
        s.start    = st;
        time_expr trig;
        trig.k       = time_expr::kind::trigger;
        trig.trigger = "go";
        s.end        = trig;
        held.enable.push_back(s);
        d.objects.push_back(held);

        timeline_object after;
        after.id    = "after";
        after.layer = "1-11";
        after.enable.push_back(spec_expr("#held.end"));
        d.objects.push_back(after);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "an open end is not an error");
        const auto& h = r.instances[r.by_object.at("held").front()];
        req(!h.end, "the held object's end is OPEN, not a large number");
        req(h.contains(from_seconds(1000.0)), "so it is still active far out");
        req(r.pending_triggers.size() == 1 && r.pending_triggers[0] == "go",
            "and the trigger it waits on is published, so the transport knows GO means something");
        req(r.by_object.find("after") == r.by_object.end(),
            "the dependent does NOT resolve while the reference is open -- and that is not an "
            "error either, because firing the trigger re-resolves");

        // Fire it, and both resolve.
        trigger_log fired;
        fired.fired["go"] = {from_seconds(12.0)};
        const auto r2     = resolve(d, fired);
        req(r2.ok(), "with the trigger fired it resolves");
        req(*r2.instances[r2.by_object.at("held").front()].end == from_seconds(12.0),
            "the held object ends when the trigger fired");
        req(r2.instances[r2.by_object.at("after").front()].start == from_seconds(12.0),
            "and the dependent starts there");
        req(r2.pending_triggers.empty(), "nothing pending any more");
    }

    // ---- 9. the failures, each named ----------------------------------------------------
    {
        timeline_document d;
        timeline_object   a;
        a.id    = "a";
        a.layer = "1-10";
        a.enable.push_back(spec_expr("#b.end", "#b.end + 1"));
        timeline_object b;
        b.id    = "b";
        b.layer = "1-11";
        b.enable.push_back(spec_expr("#a.end", "#a.end + 1"));
        d.objects.push_back(a);
        d.objects.push_back(b);

        const auto r = resolve(d, no_triggers);
        req(!r.ok(), "a cycle is refused");
        req(!r.errors.empty() && !r.errors.front().object.empty(),
            "and the error NAMES an object rather than saying the document is invalid");
    }
    {
        timeline_document d;
        d.objects.push_back(obj("a", "1-10", 0.0, 4.0));
        d.objects.push_back(obj("a", "1-11", 0.0, 4.0));
        const auto r = resolve(d, no_triggers);
        req(!r.ok(), "a duplicate id is refused -- every reference to it would be ambiguous");
    }
    {
        timeline_document d;
        timeline_object   o;
        o.id    = "x";
        o.layer = "1-10";
        o.enable.push_back(spec_expr("#nothing.end"));
        d.objects.push_back(o);
        const auto r = resolve(d, no_triggers);
        req(!r.ok(), "a reference to an object that does not exist is refused");
        req(r.errors.front().reason.find("nothing") != std::string::npos,
            "and the reason quotes the id the author typed");
    }
    {
        timeline_document d;
        d.objects.push_back(obj("a", "1-10", 8.0, 2.0));
        const auto r = resolve(d, no_triggers);
        req(!r.ok(), "an end before its start is refused");
    }
    {
        // Over-determined and disagreeing: start 0, end 10, duration 4.
        timeline_document d;
        timeline_object   o;
        o.id    = "x";
        o.layer = "1-10";
        enable_spec s = spec_expr("0", "10");
        time_expr   dur;
        dur.literal = from_seconds(4.0);
        s.duration  = dur;
        o.enable.clear();
        o.enable.push_back(s);
        d.objects.push_back(o);
        const auto r = resolve(d, no_triggers);
        req(!r.ok(), "start, end and duration all given and disagreeing is an ERROR");
    }

    // ---- 10. a disabled object still resolves its references -----------------------------
    {
        timeline_document d;
        auto              a = obj("a", "1-10", 0.0, 4.0);
        a.disabled          = true;
        d.objects.push_back(a);
        timeline_object b;
        b.id    = "b";
        b.layer = "1-11";
        b.enable.push_back(spec_expr("#a.end", "#a.end + 1"));
        d.objects.push_back(b);

        const auto r = resolve(d, no_triggers);
        req(r.ok(), "a document referencing a disabled object resolves");
        req(r.by_object.find("a") == r.by_object.end(), "the disabled object is not active");
        req(r.instances[r.by_object.at("b").front()].start == from_seconds(4.0),
            "and yet #a.end still answers -- switching an object off must not break the show "
            "around it, which deleting it would");
    }

    CASPAR_LOG(info) << L"[timeline-resolver] self-test: all checks passed";
}

}}} // namespace caspar::core::timeline
