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

#include "target.h"

#include <core/stage/stage_fields.h>

#include <common/log.h>

#include <cstdlib>

namespace caspar { namespace core { namespace address {

const char* kind_name(target_kind k)
{
    switch (k) {
        case target_kind::image:
            return "image";
        case target_kind::audio:
            return "audio";
        case target_kind::producer_param:
            return "producer_param";
        case target_kind::screen:
            return "screen";
        case target_kind::camera:
            return "camera";
        case target_kind::view_camera:
            return "view_camera";
        default:
            return "none";
    }
}

void split(std::string_view path, std::string& field, std::uint8_t& component)
{
    component = 0;
    field     = std::string(path);

    // `producer/<name>` may legitimately contain a dot in the parameter's OWN name, so the
    // suffix is recognised only when what follows it is entirely digits. `producer/foo.bar` is
    // one name; `midtone.1` is a component. Carried over from `binding::split_target`, which
    // this replaces, along with the reason.
    const auto dot = path.rfind('.');
    if (dot == std::string_view::npos || dot + 1 >= path.size())
        return;

    const auto tail = path.substr(dot + 1);
    if (tail.find_first_not_of("0123456789") != std::string_view::npos)
        return;

    const auto n = std::atoi(std::string(tail).c_str());
    if (n < 0 || n > 3)
        return;

    field     = std::string(path.substr(0, dot));
    component = static_cast<std::uint8_t>(n);
}

namespace {

bool eat(std::string_view& s, std::string_view prefix)
{
    if (s.size() <= prefix.size() || s.compare(0, prefix.size(), prefix) != 0)
        return false;
    s.remove_prefix(prefix.size());
    return true;
}

} // namespace

target parse(std::string_view path)
{
    target out;
    if (path.empty())
        return out;

    // A `mixer/` prefix is what the published state tree and the HTTP path both carry, so an
    // address copied out of either resolves unedited. It carries no information here -- there is
    // no second namespace it could distinguish from -- so it is stripped rather than dispatched
    // on.
    eat(path, "mixer/");
    if (path.empty())
        return out;

    out.path = std::string(path);
    split(path, out.field, out.component);

    std::string_view rest = path;

    if (eat(rest, "producer/")) {
        // The name keeps its own dots: only the stage can tell `foo.1` from component 1 of
        // `foo`, because only the stage knows which parameters the producer declares. So the
        // tail is the name and the component is whatever `split` makes of it -- which the caller
        // must re-check against the declared arity.
        out.kind = target_kind::producer_param;
        split(rest, out.field, out.component);
        if (out.field.empty())
            return target{};
        return out;
    }

    if (eat(rest, "previz/")) {
        std::string_view tail = rest;
        if (eat(tail, "screen/")) {
            const auto slash = tail.find('/');
            if (slash == std::string_view::npos || slash == 0 || slash + 1 >= tail.size())
                return target{};
            out.object = std::string(tail.substr(0, slash));
            split(tail.substr(slash + 1), out.field, out.component);
            const auto* f = fields::find_screen_field(out.field);
            if (!f)
                return target{};
            out.kind = target_kind::screen;
            out.meta = f;
            return out;
        }

        const bool view = tail.compare(0, 12, "view_camera/") == 0 && tail.size() > 12;
        if (view || (tail.compare(0, 7, "camera/") == 0 && tail.size() > 7)) {
            tail.remove_prefix(view ? 12 : 7);
            split(tail, out.field, out.component);
            const auto* f = fields::find_camera_field(out.field);
            if (!f)
                return target{};
            out.kind = view ? target_kind::view_camera : target_kind::camera;
            out.meta = f;
            return out;
        }
        return target{};
    }

    // The two static tables, image first so it stays the fast path and so a name can never
    // resolve to both. `find_audio_field` is why `volume` is addressable at all: the audio half
    // of the transform had no registry, so every table-driven surface -- the tree, the write
    // path, the publication's change test -- was blind to it while `MIXER VOLUME` set it.
    if (const auto* f = fields::find(out.field)) {
        out.kind = target_kind::image;
        out.meta = f;
        return out;
    }
    if (const auto* a = fields::find_audio_field(out.field)) {
        out.kind = target_kind::audio;
        out.meta = a;
        return out;
    }
    return target{};
}

void target_self_test()
{
    const auto req = [](bool ok, const char* what) {
        if (!ok) {
            CASPAR_LOG(fatal) << L"address::target_self_test: " << what;
            std::abort();
        }
    };

    // The image half, whole and by component, with and without the tolerated prefix.
    {
        const auto t = parse("opacity");
        req(t.kind == target_kind::image, "`opacity` is an image field");
        req(t.component == 0 && t.field == "opacity", "`opacity` has no component");
        req(t.meta && t.meta->arity == 1, "`opacity` is arity 1");
    }
    {
        const auto t = parse("mixer/fill_translation.1");
        req(t.kind == target_kind::image, "`mixer/fill_translation.1` is an image field");
        req(t.field == "fill_translation" && t.component == 1, "component 1 split off");
        req(t.path == "fill_translation.1", "the stripped path is stored, suffix kept");
        req(t.meta && t.meta->arity == 2, "fill_translation is arity 2");
    }

    // The audio half -- the row this commit adds. `volume` resolving is the whole point.
    {
        const auto t = parse("volume");
        req(t.kind == target_kind::audio, "`volume` resolves, and as audio");
        req(t.meta && t.meta->arity == 1, "`volume` is arity 1");
        req(fields::find("volume") == nullptr, "`volume` is NOT in the image table");
    }
    {
        const auto t = parse("mixer/volume");
        req(t.kind == target_kind::audio, "`mixer/volume` resolves the same way");
    }

    // Producer parameters keep their dots. This is the case that made `split` conditional.
    {
        const auto t = parse("producer/brightness");
        req(t.kind == target_kind::producer_param, "a producer parameter is classified");
        req(t.field == "brightness" && t.meta == nullptr, "and left for the stage to resolve");
    }
    {
        const auto t = parse("producer/foo.bar");
        req(t.field == "foo.bar" && t.component == 0, "a non-numeric dot stays in the name");
    }
    {
        const auto t = parse("producer/foo.2");
        req(t.field == "foo" && t.component == 2, "a numeric dot is a component even here");
    }

    // The stage registries, including the two cameras that share one table.
    {
        const auto t = parse("previz/camera/position.2");
        req(t.kind == target_kind::camera, "the production camera");
        req(t.field == "position" && t.component == 2, "and its third component");
    }
    {
        const auto t = parse("previz/view_camera/position");
        req(t.kind == target_kind::view_camera, "the viewport camera is a different kind");
        req(t.meta == parse("previz/camera/position").meta, "off the same row");
    }
    {
        // `position`, not `pos_x`: the screen table declares one vec3 row over three struct
        // members, so the address for a screen's X is component 0 of `position`. This test
        // asserted `pos_x` on its first run and aborted the boot, which is what it is for --
        // the address space is the TABLE's names, not the struct's.
        const auto t = parse("previz/screen/wall/position.0");
        req(t.kind == target_kind::screen && t.object == "wall", "the screen name is carried");
        req(t.field == "position" && t.component == 0, "and the field is the tail");
        req(t.meta && t.meta->arity == 3, "a screen position is arity 3");
    }
    {
        const auto t = parse("previz/screen/back wall/size");
        req(t.kind == target_kind::screen && t.object == "back wall", "a screen name may have a space");
    }
    req(!parse("previz/screen/wall/pos_x"), "a struct member is not an address");

    // Failures, each of which was accepted somewhere before.
    req(!parse(""), "the empty path does not resolve");
    req(!parse("mixer/"), "a bare prefix does not resolve");
    req(!parse("no_such_field"), "an unknown name does not resolve");
    req(!parse("previz/screen/wall"), "a screen with no field does not resolve");
    req(!parse("previz/screen//pos_x"), "a screen with no name does not resolve");
    req(!parse("previz/nonsense/x"), "an unknown previz noun does not resolve");
    req(!parse("producer/"), "a producer parameter needs a name");

    // A component index outside the slot range is NOT a component -- it stays in the name and
    // then fails to resolve, rather than silently becoming component 0 of a real field. `.4` on
    // a two-component field would otherwise write component 0.
    req(!parse("fill_translation.4"), "`.4` is out of range and does not resolve");

    CASPAR_LOG(info) << L"[core] address self-test: all checks passed, over "
                     << fields::all().size() << L" image, " << fields::audio_fields().size()
                     << L" audio, " << fields::screen_fields().size() << L" screen and "
                     << fields::camera_fields().size() << L" camera rows.";
}

}}} // namespace caspar::core::address
