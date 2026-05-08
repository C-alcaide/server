import os

in_path = r"D:\Github\CasparVP\our_av.cpp"
out_path = r"D:\Github\CasparVP\src\modules\ffmpeg\producer\av_producer.cpp"

with open(in_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. growing_
text = text.replace(
    "std::atomic<bool>    pingpong_{false};  // ping-pong: auto-reverse at each end",
    "std::atomic<bool>    pingpong_{false};  // ping-pong: auto-reverse at each end\n    bool                 growing_{false};"
)

# 2. Impl constructor signature
impl_search = """    Impl(std::shared_ptr<core::frame_factory> frame_factory,
         core::video_format_desc              format_desc,
         std::string                          name,
         std::string                          path,
         std::string                          vfilter,
         std::string                          afilter,
         int64_t                              start,
         int64_t                              seek,
         int64_t                              duration,
         bool                                 loop,
         int                                  seekable,
         core::frame_geometry::scale_mode     scale_mode)
        : graph_(diagnostics::graph_registry::instance().create("av_producer", name))"""
impl_repl = """    Impl(std::shared_ptr<core::frame_factory> frame_factory,
         core::video_format_desc              format_desc,
         std::string                          name,
         std::string                          path,
         std::string                          vfilter,
         std::string                          afilter,
         int64_t                              start,
         int64_t                              seek,
         int64_t                              duration,
         bool                                 loop,
         int                                  seekable,
         core::frame_geometry::scale_mode     scale_mode,
         bool                                 growing)
        : growing_(growing)
        , graph_(diagnostics::graph_registry::instance().create("av_producer", name))"""
text = text.replace(impl_search, impl_repl)

# 3. AVProducer constructor signature
av_search = """AVProducer::AVProducer(std::shared_ptr<core::frame_factory> frame_factory,
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
                       core::frame_geometry::scale_mode     scale_mode)
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
                     scale_mode))"""
av_repl = """AVProducer::AVProducer(std::shared_ptr<core::frame_factory> frame_factory,
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
                     growing))"""
text = text.replace(av_search, av_repl)

# 4. EOF logic
eof_search = """                        buffer_eof_ = (video_filter_.eof && audio_filter_.eof) ||
                                      av_rescale_q(time, TIME_BASE_Q, format_tb_) >= av_rescale_q(end, TIME_BASE_Q, format_tb_);"""
eof_repl = """                        buffer_eof_ = !growing_ && ((video_filter_.eof && audio_filter_.eof) ||
                                      av_rescale_q(time, TIME_BASE_Q, format_tb_) >= av_rescale_q(end, TIME_BASE_Q, format_tb_));"""
text = text.replace(eof_search, eof_repl)

# 5. frame_flush_ logic
flush_search = "if (buffer_eof_ && !frame_flush_ && !in_reverse) {"
flush_repl = "if ((buffer_eof_ || growing_) && !frame_flush_ && !in_reverse) {"
text = text.replace(flush_search, flush_repl)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Restoration complete")
