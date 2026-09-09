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

#include "../StdAfx.h"

#include "producer_params.h"

namespace caspar { namespace core {

bool as_number(const monitor::data_t& d, double& out)
{
    if (const auto* p = boost::get<double>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<float>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<int32_t>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<int64_t>(&d)) { out = static_cast<double>(*p); return true; }
    if (const auto* p = boost::get<uint32_t>(&d)) { out = *p; return true; }
    if (const auto* p = boost::get<uint64_t>(&d)) { out = static_cast<double>(*p); return true; }
    if (const auto* p = boost::get<bool>(&d)) { out = *p ? 1.0 : 0.0; return true; }
    return false;
}

bool as_numbers(const monitor::vector_t& v, std::vector<double>& out)
{
    std::vector<double> tmp;
    tmp.reserve(v.size());
    for (const auto& e : v) {
        double d = 0.0;
        if (!as_number(e, d))
            return false;
        tmp.push_back(d);
    }
    out = std::move(tmp);
    return true;
}

param_snapshot snapshot_of(const param_desc& p)
{
    param_snapshot s;
    s.name          = p.name;
    s.type          = p.type;
    s.access        = p.access;
    s.min           = p.min;
    s.max           = p.max;
    s.bounding      = p.bounding;
    s.label         = p.label;
    s.unit          = p.unit;
    s.values        = p.values;
    s.description   = p.description;
    s.group         = p.group;
    s.index         = p.index;
    s.step          = p.step;
    s.arity         = p.arity;
    s.default_value = p.default_value;
    if (p.get)
        s.value = p.get();
    return s;
}

}} // namespace caspar::core
