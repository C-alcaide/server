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

#include "api_status.h"

namespace caspar { namespace protocol { namespace http {

const char* to_string(api_code c)
{
    switch (c) {
        case api_code::ok: return "ok";
        case api_code::unknown_path: return "unknown_path";
        case api_code::not_writable: return "not_writable";
        case api_code::field_wrong_type: return "field_wrong_type";
        case api_code::field_out_of_range: return "field_out_of_range";
        case api_code::field_missing: return "field_missing";
        case api_code::field_conflict: return "field_conflict";
        case api_code::channel_not_found: return "channel_not_found";
        case api_code::layer_not_found: return "layer_not_found";
        case api_code::producer_not_ready: return "producer_not_ready";
        case api_code::batch_op_failed: return "batch_op_failed";
        case api_code::not_supported_on_backend: return "not_supported_on_backend";
        case api_code::unauthorized: return "unauthorized";
        case api_code::bad_request: return "bad_request";
        case api_code::internal: return "internal";
    }
    return "internal";
}

json::object envelope(const api_reply& r, const std::string& server_name)
{
    json::object status;
    status["code"]    = to_string(r.code);
    status["message"] = r.message;
    if (!r.details.empty())
        status["details"] = r.details;

    json::object out;
    out["status"] = std::move(status);
    out["server"] = server_name;
    out["result"] = r.result;
    return out;
}

}}} // namespace caspar::protocol::http
