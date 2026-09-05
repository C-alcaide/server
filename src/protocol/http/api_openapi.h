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

#pragma once

#include "http_config.h"
#include "boost_prelude.h"

#include <string>

namespace caspar { namespace protocol { namespace http {

/// `GET /v1/openapi.json` -- OpenAPI 3.1, generated.
///
/// The endpoints are written out by hand here because there are eight of them and they do
/// not change on their own. The FIELD LIST is not: every writable mixer parameter, its type,
/// its arity, its range and its description come from `core::fields::all()`, so the document
/// cannot describe a field the server does not have, or miss one it does.
///
/// That is the whole reason to generate it rather than commit a `.yaml`. A checked-in
/// specification is a second copy of every claim in this file, and this tree has measured
/// what happens to second copies.
json::object openapi(const http_config& cfg);

/// `GET /v1/docs` -- a self-contained page that renders the document above.
///
/// NOT Swagger UI, which the plan for this branch called for. Swagger UI is ~1.5 MB of
/// third-party JavaScript and CSS that would have to be vendored into the repository and
/// embedded in the binary, carrying its own licence, to give a developer a form to click --
/// or loaded from a CDN, which a studio server has no route to. This is a few kilobytes of
/// plain HTML with no script and no external reference, and it lists exactly what the
/// generated document contains.
std::string docs_page(const http_config& cfg);

}}} // namespace caspar::protocol::http
