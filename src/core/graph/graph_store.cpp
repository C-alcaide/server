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

#include "graph_store.h"

#include "registry.h"

#include <core/producer/producer_params.h>

#include <common/except.h>
#include <common/log.h>

#include <algorithm>

namespace caspar { namespace core { namespace graph {

std::shared_ptr<stored_graph> graph_store::build(graph_document doc)
{
    auto entry      = std::make_shared<stored_graph>();
    entry->document = std::move(doc);
    entry->faults   = validate(entry->document, entry->order);
    // COMPILED HERE, on the caller's thread, and only when it can run. A null plan is what
    // `ok() == false` MEANS on the frame path: the stage keeps whatever plan it had and
    // publishes `graph_stale`, which is how an operator editing on air keeps the grade that is
    // rendering.
    entry->plan = compile(entry->document, entry->order, entry->faults);
    if (entry->plan)
        entry->values = values_of(entry->document, *entry->plan);
    return entry;
}

void graph_store::remember(slot& s, const std::string& label, const char* kind, bool coalesce)
{
    // COALESCING IS WHAT MAKES UNDO MEAN "the gesture", not "the write".
    //
    // A slider drag is fifty `patch_params` calls in a second. Fifty history entries would make
    // "undo" mean "move the slider back one fiftieth of the way", which is not what anybody
    // pressing it wants -- and a client cannot fix it from outside, because it does not know how
    // many writes the server received.
    //
    // The rule is the LABEL: consecutive changes of the same kind carrying the same non-empty
    // label are one entry. So a client names its gesture once and every write inside it folds in.
    // An unlabelled write never coalesces, because with no label there is nothing to say two
    // writes belong together -- guessing from timing alone would fold two deliberate nudges into
    // one. (Time-based coalescing is the alternative and is flagged as unmeasured in the plan.)
    const bool fold = coalesce && !label.empty() && !s.entries.empty() &&
                      s.entries.back().label == label && s.entries.back().kind == kind;

    if (!fold) {
        if (s.current)
            s.past.push_back(s.current);
        while (s.past.size() > history_depth)
            s.past.pop_front();
        history_entry h;
        h.label = label.empty() ? std::string("an edit") : label;
        h.kind  = kind;
        h.at    = std::chrono::system_clock::now();
        // The revision this entry would return TO -- the one that was current when the gesture
        // began. A client presenting "undo: back to revision N" needs the destination, not the
        // revision the gesture is about to create, which does not exist yet at this point.
        h.revision = s.current ? s.current->document.revision : 0;
        s.entries.push_back(h);
        while (s.entries.size() > history_depth)
            s.entries.pop_front();
    }
    // REDO IS CLEARED BY ANY FORWARD CHANGE, folded or not: once the document has moved, the old
    // future was computed against something that no longer exists.
    s.future.clear();
}

std::shared_ptr<const stored_graph> graph_store::put(graph_document doc, const std::string& label)
{
    // Validated OUTSIDE the lock. A large document's validation is the one expensive thing here,
    // and holding the lock across it would block every tick that wants to read any graph --
    // including layers on other channels with nothing to do with this edit.
    const auto name = doc.name;
    {
        std::lock_guard<std::mutex> l(lock_);
        doc.revision = ++revision_;
    }
    auto entry = build(std::move(doc));

    std::lock_guard<std::mutex> l(lock_);
    auto&                       s = slots_[name];
    // A PUT is never coalesced: it replaces the whole document, so two of them are two edits
    // even under one label.
    remember(s, label, "put", /*coalesce*/ false);
    s.current    = entry;
    s.last_label = label;
    return entry;
}

std::shared_ptr<const stored_graph> graph_store::get(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    return it == slots_.end() ? nullptr : it->second.current;
}

bool graph_store::erase(const std::string& name)
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || !it->second.current)
        return false;
    // The attachment goes with it. The stage notices on its next tick, because it re-reads the
    // store per tick and finds nothing -- which is the same path a DETACH takes.
    slots_.erase(it);
    ++revision_;
    return true;
}

std::vector<std::string> graph_store::names() const
{
    std::vector<std::string>    out;
    std::lock_guard<std::mutex> l(lock_);
    out.reserve(slots_.size());
    for (const auto& kv : slots_)
        if (kv.second.current)
            out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

std::shared_ptr<const stored_graph>
graph_store::patch_params(const std::string&                             name,
                          const std::map<std::string, monitor::vector_t>& values,
                          const std::string&                             label)
{
    std::shared_ptr<const stored_graph> old;
    {
        std::lock_guard<std::mutex> l(lock_);
        const auto                  it = slots_.find(name);
        if (it == slots_.end() || !it->second.current)
            return nullptr;
        old = it->second.current;
    }

    auto doc = old->document;
    bool changed = false;
    for (const auto& kv : values) {
        // `node/<id>/<param>` -- the address form. A key that does not parse, or names a node or
        // port that is not there, is IGNORED rather than refused: this is called from the stage
        // executor against a document that may have been re-PUT since the driver was resolved,
        // and refusing on the frame path over a stale key would take the channel down for an
        // edit the operator already made.
        const auto& path = kv.first;
        if (path.compare(0, 5, "node/") != 0)
            continue;
        const auto slash = path.find('/', 5);
        if (slash == std::string::npos)
            continue;
        const auto id    = path.substr(5, slash - 5);
        auto       param = path.substr(slash + 1);
        // A `.N` component suffix is not accepted here. `patch_params` writes whole parameters;
        // a per-component write is read-modify-write and the CALLER does it, because only the
        // caller knows what the other components were meant to be -- the same division
        // `resolve_drivers`' write lambda already makes for a mixer field.
        if (param.find('.') != std::string::npos)
            continue;

        const auto n = std::find_if(doc.nodes.begin(), doc.nodes.end(),
                                    [&](const graph_node& x) { return x.id == id; });
        if (n == doc.nodes.end())
            continue;
        const auto* c = find_node_class(n->cls);
        if (!c || !find_port(*c, param))
            continue;
        n->params[param] = kv.second;
        changed          = true;
    }
    if (!changed)
        return old;

    // Re-validated, because a value CAN break a document: a parameter out of its declared range
    // is an error fault. Cheap -- no topological work changes -- and it keeps one answer to "is
    // this document buildable" rather than two depending on how it was last written.
    auto entry = build(std::move(doc));

    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    // Someone may have PUT or erased it while this was validating. Their write is newer, so it
    // wins: storing a patch built on a document that has been replaced would silently undo a PUT
    // that was already acknowledged.
    if (it == slots_.end() || it->second.current != old)
        return it == slots_.end() ? nullptr : it->second.current;

    // VALUES ONLY: `values_revision_` moves and `revision_` does NOT, so this does not enter the
    // stage fingerprint and a slider drag does not make every client re-walk the tree. The
    // document's own `revision` field keeps the value it had, for the same reason -- a client
    // comparing revisions is asking "has the structure changed".
    entry->document.revision = old->document.revision;
    // AND THE PLAN POINTER IS CARRIED OVER, not the newly compiled one -- which is the whole
    // attribute/signal split made concrete. A value change does not alter the topology, so the
    // plan is the SAME OBJECT, and `image_transform::operator==` compares it by pointer: reusing
    // it is what lets a timeline ramp a node parameter without the still-frame fingerprint
    // moving by allocation on every tick. `build` compiled one; it is dropped on the floor,
    // which costs a graph-sized allocation per write and is the price of one code path for both.
    if (old->plan && entry->plan) {
        entry->plan   = old->plan;
        entry->values = values_of(entry->document, *entry->plan);
    }
    remember(it->second, label, "patch", /*coalesce*/ true);
    it->second.current = entry;
    ++values_revision_;
    return entry;
}

bool graph_store::claim(const std::string& name, attachment where)
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || !it->second.current)
        return false;
    if (it->second.where && it->second.where->valid()) {
        // Already attached. Re-attaching to the SAME place is not an error -- it is idempotent,
        // which a client retrying a command after a timeout depends on.
        if (it->second.where->channel == where.channel && it->second.where->layer == where.layer)
            return true;
        return false;
    }
    // ONE DOCUMENT PER LAYER TOO, in the other direction: a layer already holding another graph
    // has to let go of it first, or the layer would have two answers to what it renders.
    for (auto& kv : slots_) {
        if (kv.first == name)
            continue;
        if (kv.second.where && kv.second.where->channel == where.channel &&
            kv.second.where->layer == where.layer)
            return false;
    }
    it->second.where = where;
    ++revision_;
    return true;
}

void graph_store::release(const std::string& name)
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || !it->second.where)
        return;
    it->second.where.reset();
    ++revision_;
}

std::optional<attachment> graph_store::attached(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || !it->second.where)
        return std::nullopt;
    return it->second.where;
}

std::optional<std::string> graph_store::attached_to(int channel, int layer) const
{
    std::lock_guard<std::mutex> l(lock_);
    for (const auto& kv : slots_) {
        if (kv.second.where && kv.second.where->channel == channel &&
            kv.second.where->layer == layer)
            return kv.first;
    }
    return std::nullopt;
}

std::shared_ptr<const stored_graph> graph_store::undo(const std::string& name)
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || it->second.past.empty())
        return nullptr;
    auto& s = it->second;
    s.future.push_back(s.current);
    s.current = s.past.back();
    s.past.pop_back();
    if (!s.entries.empty())
        s.entries.pop_back();
    // UNDO IS A STRUCTURE CHANGE even when what it undoes was a value patch, because the client
    // that is drawing this document has to re-read it: the values it is showing are no longer the
    // ones in the store, and there is no per-value notification. The asymmetry with
    // `patch_params` is deliberate -- a patch is a change the client just made and already knows
    // about; an undo may be another client's.
    ++revision_;
    return s.current;
}

std::shared_ptr<const stored_graph> graph_store::redo(const std::string& name)
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end() || it->second.future.empty())
        return nullptr;
    auto& s = it->second;
    s.past.push_back(s.current);
    s.current = s.future.back();
    s.future.pop_back();
    history_entry h;
    h.label    = "redo";
    h.kind     = "put";
    h.at       = std::chrono::system_clock::now();
    h.revision = s.current ? s.current->document.revision : 0;
    s.entries.push_back(h);
    ++revision_;
    return s.current;
}

std::vector<history_entry> graph_store::history(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    if (it == slots_.end())
        return {};
    return {it->second.entries.begin(), it->second.entries.end()};
}

bool graph_store::can_undo(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    return it != slots_.end() && !it->second.past.empty();
}

bool graph_store::can_redo(const std::string& name) const
{
    std::lock_guard<std::mutex> l(lock_);
    const auto                  it = slots_.find(name);
    return it != slots_.end() && !it->second.future.empty();
}

std::int64_t graph_store::revision() const
{
    std::lock_guard<std::mutex> l(lock_);
    return revision_;
}

std::int64_t graph_store::values_revision() const
{
    std::lock_guard<std::mutex> l(lock_);
    return values_revision_;
}

// ---------------------------------------------------------------------------------------
// THE BOOT SELF-TEST
//
// The store's rules are the kind that look obviously right and are easy to get backwards, and
// three of them are load-bearing in a way no battery can see from outside in one run: which
// counter a change moves, that a patch does NOT move the structure counter, and that undo folds
// a labelled run into one step.
// ---------------------------------------------------------------------------------------

void graph_store_self_test()
{
    const auto fail = [](const std::string& what) {
        CASPAR_THROW_EXCEPTION(caspar_exception() << msg_info("graph store self-test: " + what));
    };

    const auto doc = [](const char* name, double gain) {
        graph_document d;
        d.name = name;
        graph_node i, e, o;
        i.id = "i", i.cls = "input";
        e.id = "e", e.cls = "exposure";
        e.params["gain"] = monitor::vector_t{gain};
        o.id = "o", o.cls = "output";
        d.nodes = {i, e, o};
        graph_edge a, b;
        a.id = "e1", a.from_node = "i", a.from_port = "out", a.to_node = "e", a.to_port = "in";
        b.id = "e2", b.from_node = "e", b.from_port = "out", b.to_node = "o", b.to_port = "in";
        d.edges = {a, b};
        return d;
    };

    graph_store st;

    const auto e1 = st.put(doc("g", 1.0));
    if (!e1 || !e1->ok())
        fail("a valid document did not store, or did not validate");
    if (e1->document.revision != 1)
        fail("the first document's revision is " + std::to_string(e1->document.revision) +
             ", not 1. The revision is the STORE's counter and is server-assigned");
    if (st.revision() != 1)
        fail("a put did not move the structure revision");

    // ---- THE TWO COUNTERS ARE SEPARATE, which is `L376`'s rule ------------------------
    const auto struct_before = st.revision();
    const auto values_before = st.values_revision();
    const auto e2 = st.patch_params("g", {{"node/e/gain", monitor::vector_t{2.0}}}, "drag");
    if (!e2)
        fail("patch_params on an existing document returned null");
    if (st.revision() != struct_before)
        fail("patch_params moved the STRUCTURE revision. It must not: a slider drag would then "
             "make every client re-walk every channel's tree fifty times a second");
    if (st.values_revision() == values_before)
        fail("patch_params did not move the VALUES revision, so nothing can tell a value "
             "changed");
    {
        const auto& n = e2->document.nodes[1];
        double      g = 0;
        if (n.params.count("gain") == 0 || !as_number(n.params.at("gain").front(), g) || g != 2.0)
            fail("patch_params did not write the value");
    }
    if (e2->document.revision != e1->document.revision)
        fail("patch_params moved the document's own revision; a client comparing revisions is "
             "asking whether the STRUCTURE changed");

    // A key naming nothing is ignored rather than refused -- it arrives from the stage executor
    // against a document that may have moved on.
    if (!st.patch_params("g", {{"node/nosuch/gain", monitor::vector_t{1.0}}}))
        fail("a patch naming an absent node returned null; it must be ignored, because this is "
             "called from the frame path against a document that may have been re-PUT");

    // ---- COALESCING: a labelled run is ONE undo ---------------------------------------
    for (int i = 0; i < 10; ++i)
        st.patch_params("g", {{"node/e/gain", monitor::vector_t{2.0 + i * 0.1}}}, "drag");
    const auto h = st.history("g");
    // put + the first labelled patch = 2 entries, and the ten more must have folded into the
    // second. Eleven would mean undo means "one fiftieth of a slider drag".
    if (h.size() != 2)
        fail("a run of 11 patches under one label produced " + std::to_string(h.size()) +
             " history entries; consecutive same-label changes must coalesce to one");
    if (!st.can_undo("g"))
        fail("nothing to undo after a put and a patch run");

    const auto undone = st.undo("g");
    if (!undone)
        fail("undo returned null");
    {
        double g = 0;
        as_number(undone->document.nodes[1].params.at("gain").front(), g);
        if (g != 1.0)
            fail("undo did not go back past the whole coalesced gesture: gain is " +
                 std::to_string(g) + ", not the 1.0 it was before the drag");
    }
    if (!st.can_redo("g"))
        fail("redo is not available after an undo");
    if (!st.redo("g"))
        fail("redo returned null");
    {
        double g = 0;
        as_number(st.get("g")->document.nodes[1].params.at("gain").front(), g);
        if (g == 1.0)
            fail("redo did not restore the drag");
    }
    // A PUT CLEARS REDO. Once the document has branched, the old future was computed against
    // something that no longer exists.
    st.undo("g");
    st.put(doc("g", 5.0));
    if (st.can_redo("g"))
        fail("a put left redo available, so a client could apply an edit made against a document "
             "that no longer exists");

    // ---- ATTACHMENT ------------------------------------------------------------------
    st.put(doc("h", 1.0));
    if (!st.claim("g", attachment{1, 10}))
        fail("claiming an unattached document failed");
    if (!st.claim("g", attachment{1, 10}))
        fail("re-claiming the SAME place failed; it must be idempotent for a client retrying "
             "after a timeout");
    if (st.claim("g", attachment{1, 11}))
        fail("a second claim elsewhere succeeded; one document attaches once");
    if (st.claim("h", attachment{1, 10}))
        fail("two documents claimed the same layer; a layer would have two answers to what it "
             "renders");
    if (!st.attached("g") || st.attached("g")->layer != 10)
        fail("the attachment does not read back");
    if (!st.attached_to(1, 10) || *st.attached_to(1, 10) != "g")
        fail("attached_to does not find the document by layer");
    st.release("g");
    if (st.attached("g"))
        fail("release left the attachment");
    if (!st.claim("h", attachment{1, 10}))
        fail("the layer was not free after a release");

    // ---- ERASE detaches --------------------------------------------------------------
    if (!st.erase("h"))
        fail("erasing an existing document failed");
    if (st.attached_to(1, 10))
        fail("erase left the layer attached to a document that no longer exists");
    if (st.erase("h"))
        fail("erasing twice succeeded");
    if (st.get("h"))
        fail("an erased document still reads back");

    // ---- A DOCUMENT WITH ERRORS IS STORED --------------------------------------------
    auto bad = doc("g", 1.0);
    bad.nodes[1].cls = "expsure";
    const auto stored = st.put(std::move(bad), "typo");
    if (!stored)
        fail("a document with errors was not stored. It must be: a graph is edited ON AIR, and "
             "an operator who mistypes a class name must not lose the grade that is rendering");
    if (stored->ok())
        fail("a document naming an unknown class validated");
    if (!st.get("g") || st.get("g")->ok())
        fail("the stored broken document does not read back with its faults");

    CASPAR_LOG(info) << L"[graph-store] self-test: all checks passed";
}

}}} // namespace caspar::core::graph
