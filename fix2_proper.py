import os

file_path_cpp = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\av_producer.cpp"
with open(file_path_cpp, "r", encoding="utf-8") as f:
    c = f.read()

start = c.find("AVProducer::AVProducer(std::shared_ptr")
end = c.find("}", start) + 1

new_cpp = """AVProducer::AVProducer(std::shared_ptr<core::frame_factory> frame_factory,
                       core::video_format_desc              format_desc,
                       std::string                          name,
                       std::string                          path,
                       std::optional<std::string>           vfilter,
                       std::optional<std::string>           afilter,
                       std::optional<int64_t>               start,
                       std::optional<int64_t>               seek,
                       std::optional<int64_t>               duration,
                       std::optional<bool>                  loop,
                       int                                  seekable,
                       core::frame_geometry::scale_mode     scale_mode,
                       bool                                 growing)
    : impl_(new Impl(std::move(frame_factory),
                     std::move(format_desc),
                     std::move(name),
                     std::move(path),
                     std::move(vfilter.value_or("")),
                     std::move(afilter.value_or("")),
                     std::move(start),
                     std::move(seek),
                     std::move(duration),
                     loop.value_or(false),
                     seekable,
                     scale_mode,
                     growing))
{
}"""

c = c[:start] + new_cpp + c[end:]

with open(file_path_cpp, "w", encoding="utf-8") as f:
    f.write(c)

print("success!")
