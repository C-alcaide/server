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

#include "model.h"

namespace caspar { namespace core { namespace graph {

const char* stage_name(graph_stage s)
{
    return s == graph_stage::working ? "working" : "display";
}

bool stage_from_name(std::string_view n, graph_stage& out)
{
    if (n == "working") {
        out = graph_stage::working;
        return true;
    }
    if (n == "display") {
        out = graph_stage::display;
        return true;
    }
    return false;
}

const char* severity_name(graph_fault::severity s)
{
    return s == graph_fault::severity::error ? "error" : "coercion";
}

}}} // namespace caspar::core::graph
