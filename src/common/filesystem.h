/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Helge Norberg, helge.norberg@svt.se
 */

#pragma once

#include <boost/filesystem/path.hpp>

#include <optional>

namespace caspar {

std::optional<boost::filesystem::path>
find_file_within_dir_or_absolute(const std::wstring&                                        parent_dir,
                                 const std::wstring&                                        filename,
                                 const std::function<bool(const boost::filesystem::path&)>& is_valid_file);

/// Is `resolved` inside `base`?
///
/// For containing a client-supplied path to a folder. Note this is deliberately
/// NOT what find_file_within_dir_or_absolute above does: that one accepts absolute
/// paths by design, which is right for reading media an operator points at, and
/// wrong for anything that writes.
///
/// A plain string-prefix test is not enough -- it has no boundary at the separator,
/// so a base of "D:\casparcg\media" also accepts "D:\casparcg\media_evil\x".
/// lexically_relative gives the boundary: the result is empty when the paths are
/// unrelated and begins with ".." when `resolved` is outside `base`.
///
/// Purely lexical, so callers must canonicalize (or weakly_canonicalize) first --
/// otherwise a symlink inside `base` still escapes it.
bool is_within_base(const boost::filesystem::path& resolved, const boost::filesystem::path& base);

boost::filesystem::path get_relative(const boost::filesystem::path& file, const boost::filesystem::path& relative_to);
boost::filesystem::path get_relative_without_extension(const boost::filesystem::path& file,
                                                       const boost::filesystem::path& relative_to);

} // namespace caspar
