import os

file_path_cpp = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\av_producer.cpp"
with open(file_path_cpp, "r", encoding="utf-8") as f:
    c = f.read()

import re

old_impl = r"Impl\(std::shared_ptr<core::frame_factory>.*?scale_mode\)\n\s*: frame_factory_\(frame_factory\)"

new_impl = """Impl(std::shared_ptr<core::frame_factory> frame_factory,
         core::video_format_desc              format_desc,
         std::string                          name,
         std::string                          path,
         std::string                          vfilter,
         std::string                          afilter,
         std::optional<int64_t>               start,
         std::optional<int64_t>               seek,
         std::optional<int64_t>               duration,
         bool                                 loop,
         int                                  seekable,
         core::frame_geometry::scale_mode     scale_mode,
         bool                                 growing)
        : growing_(growing)
        , frame_factory_(frame_factory)"""

c = re.sub(old_impl, new_impl, c, flags=re.DOTALL)

with open(file_path_cpp, "w", encoding="utf-8") as f:
    f.write(c)

print("success replacing impl")
