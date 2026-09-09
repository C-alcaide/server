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

#include "api_openapi.h"

#include <core/stage/stage_fields.h>
#include "api_status.h"
#include "json_state.h"

#include <common/utf.h>
#include <core/frame/transform_fields.h>

#include <sstream>

namespace caspar { namespace protocol { namespace http {

namespace fields = core::fields;

namespace {

json::object envelope_schema()
{
    json::array codes;
    for (int c = 0; c <= static_cast<int>(api_code::internal); ++c)
        codes.push_back(json::value(to_string(static_cast<api_code>(c))));

    json::object status;
    status["type"] = "object";
    json::object sprops;
    json::object code;
    code["type"] = "string";
    code["enum"] = std::move(codes);
    code["description"] =
        "The application status. `ok` means the request succeeded; everything else is a named "
        "failure, and the HTTP status stays 200 because the transport succeeded.";
    sprops["code"]    = std::move(code);
    json::object msg;
    msg["type"]       = "string";
    sprops["message"] = std::move(msg);
    json::object det;
    det["type"]        = "array";
    det["description"] = "Structured context: the failing index of a batch, the range that was "
                         "violated, the component that violated it.";
    det["items"]       = json::object();
    sprops["details"]  = std::move(det);
    status["properties"] = std::move(sprops);
    status["required"]   = json::array{json::value("code"), json::value("message")};

    json::object env;
    env["type"] = "object";
    json::object props;
    props["status"] = std::move(status);
    json::object server;
    server["type"]        = "string";
    server["description"] = "Which server answered. A client attached to two servers tells them "
                            "apart by this rather than by which socket the reply arrived on.";
    props["server"]       = std::move(server);
    props["result"]       = json::object();
    env["properties"]     = std::move(props);
    env["required"]       = json::array{json::value("status"), json::value("server")};
    env["description"]    = "Carried by EVERY reply, success or failure.";
    return env;
}

/// One JSON Schema per registry field, generated. This is the half that must not be
/// hand-written.
json::object field_schemas()
{
    json::object out;
    for (const auto& f : fields::all()) {
        json::object sch;
        const bool   writable =
            (static_cast<uint8_t>(f.access) & static_cast<uint8_t>(fields::access_t::write)) != 0;

        json::object value;
        switch (f.type) {
            case fields::value_type::boolean:
            case fields::value_type::blob: value["type"] = "boolean"; break;
            case fields::value_type::integer: value["type"] = "integer"; break;
            case fields::value_type::string: value["type"] = "string"; break;
            case fields::value_type::enumeration: {
                value["type"] = "string";
                json::array names;
                for (auto n : fields::split_list(f.values))
                    names.push_back(json::value(std::string(n)));
                if (!names.empty())
                    value["enum"] = std::move(names);
                value["description"] = "Accepts the name or its zero-based ordinal.";
                break;
            }
            default: value["type"] = "number"; break;
        }
        if (f.range && f.type != fields::value_type::enumeration) {
            value["minimum"] = f.range->lo;
            value["maximum"] = f.range->hi;
        }

        if (f.arity > 1) {
            json::object arr;
            arr["type"]     = "array";
            arr["items"]    = std::move(value);
            arr["minItems"] = static_cast<std::int64_t>(f.arity);
            arr["maxItems"] = static_cast<std::int64_t>(f.arity);
            sch             = std::move(arr);
        } else {
            sch = std::move(value);
        }

        sch["readOnly"] = !writable;
        if (f.description)
            sch["title"] = f.description;

        json::object x;
        x["compose"]        = compose_name(f.compose);
        x["bounding"]       = bounding_name(f.bounding);
        x["compose_clamps"] = f.compose_clamps;
        x["default"]        = vector_to_json(f.defaults());
        if (f.unit && *f.unit)
            x["unit"] = f.unit;
        if (f.kf_names) {
            json::array kf;
            for (auto n : fields::split_list(f.kf_names))
                kf.push_back(json::value(std::string(n)));
            x["keyframe_names"] = std::move(kf);
        }
        sch["x-casparcg"] = std::move(x);

        out[f.path] = std::move(sch);
    }
    return out;
}

json::object envelope_response(const char* description)
{
    json::object schema;
    schema["$ref"] = "#/components/schemas/Envelope";
    json::object media;
    media["schema"] = std::move(schema);
    json::object content;
    content["application/json"] = std::move(media);
    json::object resp;
    resp["description"] = description;
    resp["content"]     = std::move(content);
    json::object out;
    out["200"] = std::move(resp);
    return out;
}

json::object path_param(const char* name, const char* description)
{
    json::object schema;
    schema["type"] = "string";
    json::object p;
    p["name"]        = name;
    p["in"]          = "path";
    p["required"]    = true;
    p["schema"]      = std::move(schema);
    p["description"] = description;
    return p;
}

json::object op(const char* summary, const char* description, json::array params, const char* body_ref = nullptr)
{
    json::object o;
    o["summary"]     = summary;
    o["description"] = description;
    if (!params.empty())
        o["parameters"] = std::move(params);
    if (body_ref) {
        json::object schema;
        schema["$ref"] = body_ref;
        json::object media;
        media["schema"] = std::move(schema);
        json::object content;
        content["application/json"] = std::move(media);
        json::object rb;
        rb["required"] = true;
        rb["content"]  = std::move(content);
        o["requestBody"] = std::move(rb);
    }
    o["responses"] = envelope_response("The uniform envelope. Check `status.code`, not the HTTP status.");
    return o;
}

} // namespace

json::object openapi(const http_config& cfg)
{
    json::object doc;
    doc["openapi"] = "3.1.0";

    json::object info;
    info["title"] = "CasparVP control API";
    info["version"] = "1";
    info["description"] =
        "The server's own state, addressable and self-describing. This document is GENERATED: the "
        "mixer field list under components/schemas comes from the same table the server validates "
        "and composes with, so it cannot describe a field the server does not have.";
    doc["info"] = std::move(info);

    json::object server;
    server["url"]         = "http://" + u8(cfg.host.empty() ? std::wstring(L"127.0.0.1") : cfg.host) + ":" +
                    std::to_string(cfg.port);
    server["description"] = u8(cfg.name);
    doc["servers"]        = json::array{std::move(server)};

    // --- paths ---------------------------------------------------------------------------
    json::object paths;

    {
        json::object p;
        p["get"] = op("The whole address space",
                      "Every channel, every layer, every published key, and -- with <extent>mixer -- every "
                      "mixer parameter of every existing layer with its type, range and default. OSCQuery "
                      "shaped; fork-specific attributes are under the `casparcg` key of each node.",
                      {});
        paths["/v1/tree"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("What this server can be asked to PLAY",
                      "Installed OFX plug-ins and the ISF shaders under the media folder, each with its "
                      "id -- exactly what PLAY takes -- its label, its section and what the format "
                      "declares about it. The counterpart of .../foreground/params, which describes a "
                      "producer that is ALREADY running: this says what exists to run, so a client can "
                      "offer the choice and then draw the panel with no per-effect knowledge. Built on "
                      "demand, because a .fs can be dropped into the media folder while the server runs.",
                      {});
        paths["/v1/catalog"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("One format's catalogue",
                      "The same, narrowed to `ofx` or `isf`. An unknown kind is unknown_path rather than "
                      "an empty list, because an empty list is what a server with none installed "
                      "correctly returns and the two must not look the same.",
                      json::array{path_param("kind", "ofx or isf")});
        paths["/v1/catalog/{kind}"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("A sub-tree",
                      "The same, rooted at a path such as /channel/1/stage/layer/10/mixer.",
                      json::array{path_param("path", "A tree path, e.g. channel/1/stage/layer/10/mixer")});
        paths["/v1/tree/{path}"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("Read one value",
                      "A mixer parameter at its default is not published, so an unpublished path answers "
                      "with the descriptor's default and sets `result.is_default`. Absent means at its "
                      "default, never unknown. The same holds for the 3D stage, at "
                      "`/channel/{n}/mixer/previz/camera/{field}`, `.../view_camera/{field}` and "
                      "`.../screen/{name}/{field}`.",
                      json::array{path_param("path", "channel/1/stage/layer/10/mixer/opacity")});
        p["put"] = op("Write one value",
                      "Body: {\"value\": ..., \"duration\": frames, \"tween\": name, \"label\": text}, or an "
                      "in-place operation {\"op\": \"toggle\"|\"add\"|\"cas\", ...}. Out of range is refused, "
                      "never clipped. The reply's `value` is what the field now HOLDS. "
                      "STAGE fields take `op: set` only and are not tweenable: the previz renderer "
                      "cannot make a read-modify-write atomic, and KEYFRAMES is bound to "
                      "image_transform so a screen cannot be animated in this build.",
                      json::array{path_param("path", "channel/1/stage/layer/10/mixer/opacity")},
                      "#/components/schemas/ValueWrite");
        paths["/v1/value/{path}"] = std::move(p);
    }
    {
        json::object p;
        p["post"] = op("Transport and clear",
                       "play, stop, pause, resume, preview, clear, clear_transforms. `play` or `load` with a "
                       "`clip` in the body is delegated to AMCP, which builds the producer.",
                       json::array{path_param("path", "channel/1/stage/layer/10/play")});
        paths["/v1/action/{path}"] = std::move(p);
    }
    {
        json::object p;
        p["post"] = op("Several ops, one frame",
                       "Every op is validated before any is applied. Applied atomically across every channel "
                       "touched. `at_frame` or `in_frames` schedules it; a frame already past is refused.",
                       {},
                       "#/components/schemas/Batch");
        paths["/v1/batch"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("A challenge",
                      "With <auth>password</auth>, returns a salt and a single-use challenge. Answer with "
                      "Authorization: Caspar <challenge>:<sha256(sha256(password+salt)+challenge)>. With "
                      "<auth>off</auth>, returns {\"auth\":\"off\"} and nothing is required.",
                      {});
        paths["/v1/auth"] = std::move(p);
    }
    {
        json::object p;
        p["get"] = op("This document", "Generated from the server's own field table.", {});
        paths["/v1/openapi.json"] = std::move(p);
    }
    {
        json::object get;
        get["summary"]     = "Live changes (WebSocket)";
        get["description"] = "Not an HTTP endpoint: send an Upgrade request here, then one "
                             "{\"op\":\"subscribe\",\"prefixes\":[...],\"throttle_ms\":n} message. Prefixes must "
                             "start /channel/{index} and match on segment boundaries. Described here because "
                             "OpenAPI has no way to say it, and leaving it out would say it does not exist.";
        get["responses"]   = envelope_response("101 on success; the envelope only on refusal.");
        json::object p;
        p["get"]            = std::move(get);
        paths["/v1/events"] = std::move(p);
    }

    doc["paths"] = std::move(paths);

    // --- components ------------------------------------------------------------------------
    json::object schemas;
    schemas["Envelope"] = envelope_schema();

    {
        json::object w;
        w["type"] = "object";
        json::object props;
        props["value"] = json::object();
        json::object o;
        o["type"] = "string";
        o["enum"] = json::array{json::value("set"), json::value("toggle"), json::value("add"), json::value("cas")};
        props["op"] = std::move(o);
        props["expect"] = json::object();
        json::object d;
        d["type"]         = "integer";
        d["description"]  = "Frames to tween over.";
        props["duration"] = std::move(d);
        json::object t;
        t["type"]      = "string";
        props["tween"] = std::move(t);
        json::object l;
        l["type"]        = "string";
        l["description"] = "Free text, logged with the write and the client's address.";
        props["label"]   = std::move(l);
        w["properties"]  = std::move(props);
        schemas["ValueWrite"] = std::move(w);
    }
    {
        json::object b;
        b["type"] = "object";
        json::object props;
        json::object ops;
        ops["type"]        = "array";
        ops["items"]       = json::object();
        ops["description"] = "Each is {\"op\":\"set\"|\"action\", \"path\": ..., ...}.";
        props["ops"]       = std::move(ops);
        json::object at;
        at["type"]         = "integer";
        at["description"]  = "A frame from this server's /channel/{n}/frame. Mutually exclusive with in_frames.";
        props["at_frame"]  = std::move(at);
        json::object in;
        in["type"]         = "integer";
        props["in_frames"] = std::move(in);
        json::object lb;
        lb["type"]       = "string";
        props["label"]   = std::move(lb);
        json::object q;
        q["type"]        = "string";
        q["description"] = "Accepted and echoed; there is one queue.";
        props["queue"]   = std::move(q);
        b["properties"]  = std::move(props);
        b["required"]    = json::array{json::value("ops")};
        schemas["Batch"] = std::move(b);
    }

    {
        json::object mixer;
        mixer["type"]        = "object";
        mixer["description"] = "Every mixer parameter, generated from core::fields::all(). "
                               "`readOnly` marks the ones a PUT refuses.";
        mixer["properties"]  = field_schemas();
        schemas["MixerFields"] = std::move(mixer);
    }

    json::object components;
    components["schemas"] = std::move(schemas);
    doc["components"]     = std::move(components);

    return doc;
}

std::string docs_page(const http_config& cfg)
{
    std::ostringstream o;
    o << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
         "<title>CasparVP control API</title><style>"
         "body{font:14px/1.55 system-ui,sans-serif;margin:0;background:#1e1e1e;color:#d4d4d4}"
         "main{max-width:60rem;margin:0 auto;padding:2rem 1.5rem 4rem}"
         "h1{font-size:1.5rem;margin:0 0 .25rem}h2{font-size:1.05rem;margin:2rem 0 .5rem;color:#9cdcfe}"
         "p{max-width:44rem}code{background:#2d2d2d;padding:.1em .35em;border-radius:3px}"
         "table{border-collapse:collapse;width:100%;margin:.5rem 0 1rem;font-size:13px}"
         "th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #333;vertical-align:top}"
         "th{color:#888;font-weight:600}tbody tr:hover{background:#252525}"
         ".m{color:#888}.ro{color:#d0a02a}.w{overflow-x:auto}"
         "</style></head><body><main>";

    o << "<h1>CasparVP control API</h1><p class=\"m\">" << u8(cfg.name) << " &middot; port " << cfg.port
      << " &middot; extent " << u8(cfg.extent) << " &middot; auth " << u8(cfg.auth) << "</p>";

    o << "<p>The machine-readable form of this page is <code>/v1/openapi.json</code>, generated from "
         "the same field table the server validates and composes with. This page is plain HTML with no "
         "script and no external reference, so it works on a server with no route to the internet.</p>";

    o << "<h2>Endpoints</h2><div class=\"w\"><table><thead><tr><th>Method</th><th>Path</th><th>What</th>"
         "</tr></thead><tbody>";
    struct row
    {
        const char* m;
        const char* p;
        const char* d;
    };
    static const row rows[] = {
        {"GET", "/v1/tree", "The whole address space"},
        {"GET", "/v1/catalog", "What can be PLAYed: installed OFX plug-ins and ISF shaders"},
        {"GET", "/v1/catalog/{kind}", "The same, narrowed to ofx or isf"},
        {"GET", "/v1/tree/{path}", "A sub-tree"},
        {"GET", "/v1/value/{path}", "One value; an unpublished mixer or stage field reads as its default"},
        {"PUT", "/v1/value/{path}", "Write one value; op set|toggle|add|cas, with duration and tween"},
        {"POST", "/v1/action/{path}", "play, stop, pause, resume, preview, clear, clear_transforms"},
        {"POST", "/v1/batch", "Several ops, one frame, all or nothing; at_frame or in_frames"},
        {"GET", "/v1/auth", "A salt and a single-use challenge"},
        {"GET", "/v1/openapi.json", "This, machine-readable"},
        {"WS", "/v1/events", "Prefix subscription with a per-connection diff"},
        {"GET", "?HOST_INFO", "On any path: name, pid, subscriptions, extensions"},
    };
    for (const auto& r : rows)
        o << "<tr><td><code>" << r.m << "</code></td><td><code>" << r.p << "</code></td><td>" << r.d
          << "</td></tr>";
    o << "</tbody></table></div>";

    o << "<h2>Mixer fields</h2><p class=\"m\">" << fields::all().size()
      << " parameters, under <code>/channel/{n}/stage/layer/{m}/mixer/{name}</code>. "
         "Generated &mdash; this list is the server's, not a copy of it.</p>"
         "<div class=\"w\"><table><thead><tr><th>Name</th><th>Type</th><th>Range</th><th>Compose</th>"
         "<th>Keyframe names</th></tr></thead><tbody>";
    for (const auto& f : fields::all()) {
        const bool writable =
            (static_cast<uint8_t>(f.access) & static_cast<uint8_t>(fields::access_t::write)) != 0;
        o << "<tr><td><code>" << f.path << "</code>";
        if (!writable)
            o << " <span class=\"ro\">read-only</span>";
        o << "</td><td>" << type_name(f.type);
        if (f.arity > 1)
            o << "[" << static_cast<int>(f.arity) << "]";
        o << "</td><td>";
        if (f.range)
            o << f.range->lo << " .. " << f.range->hi;
        else if (f.type == fields::value_type::enumeration && f.values)
            o << "<span class=\"m\">" << f.values << "</span>";
        else
            o << "<span class=\"m\">&mdash;</span>";
        o << "</td><td>" << compose_name(f.compose) << "</td><td class=\"m\">"
          << (f.kf_names ? f.kf_names : "&mdash;") << "</td></tr>";
    }
    o << "</tbody></table></div>";

    // The stage, from its own two tables. Generated for the same reason the mixer list above is:
    // a hand-written copy of a table is a claim that goes stale, and this one would go stale in
    // two places at once.
    const auto stage_table = [&](const char*        heading,
                                 const char*        addr,
                                 const auto&        table) {
        o << "<h2>" << heading << "</h2><p class=\"m\">" << table.size() << " properties, under <code>"
          << addr << "</code>. Generated.</p>"
             "<div class=\"w\"><table><thead><tr><th>Name</th><th>Type</th><th>Range</th><th>Unit</th>"
             "<th>Description</th></tr></thead><tbody>";
        for (const auto& f : table) {
            const bool writable =
                (static_cast<uint8_t>(f.access) & static_cast<uint8_t>(fields::access_t::write)) != 0;
            o << "<tr><td><code>" << f.path << "</code>";
            if (!writable)
                o << " <span class=\"ro\">read-only</span>";
            o << "</td><td>" << type_name(f.type);
            if (f.arity > 1)
                o << "[" << static_cast<int>(f.arity) << "]";
            o << "</td><td>";
            if (f.range)
                o << f.range->lo << " .. " << f.range->hi;
            else if (f.type == fields::value_type::enumeration && f.values)
                o << "<span class=\"m\">" << f.values << "</span>";
            else
                o << "<span class=\"m\">&mdash;</span>";
            o << "</td><td class=\"m\">" << (f.unit && *f.unit ? f.unit : "&mdash;") << "</td><td class=\"m\">"
              << (f.description ? f.description : "&mdash;") << "</td></tr>";
        }
        o << "</tbody></table></div>";
    };

    stage_table("Stage &mdash; screen", "/channel/{n}/mixer/previz/screen/{name}/{field}",
                fields::screen_fields());
    stage_table("Stage &mdash; camera", "/channel/{n}/mixer/previz/{camera|view_camera}/{field}",
                fields::camera_fields());

    o << "</main></body></html>";
    return o.str();
}

}}} // namespace caspar::protocol::http
