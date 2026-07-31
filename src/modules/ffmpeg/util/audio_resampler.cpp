#include "audio_resampler.h"
#include "av_assert.h"

#include <common/log.h>

#include <cstddef>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace caspar::ffmpeg {

AudioResampler::AudioResampler(int sample_rate, AVSampleFormat in_sample_fmt)
{
    AVChannelLayout channel_layout     = AV_CHANNEL_LAYOUT_7POINT1;
    AVChannelLayout channel_layout_out = AV_CHANNEL_LAYOUT_HEXADECAGONAL;

    SwrContext* raw_ctx = nullptr;
    FF(swr_alloc_set_opts2(&raw_ctx,
                           &channel_layout_out,
                           AV_SAMPLE_FMT_S32,
                           sample_rate,
                           &channel_layout,
                           in_sample_fmt,
                           sample_rate,
                           0,
                           nullptr));

    ctx = std::shared_ptr<SwrContext>(raw_ctx, [](SwrContext* ptr) { swr_free(&ptr); });

    FF_RET(swr_init(ctx.get()), "swr_init");
}

caspar::array<int32_t> AudioResampler::convert(int frames, const void** src)
{
    // NOTE on sizing: caspar::array's size_ is an ELEMENT count -- size() returns it
    // and end() is ptr_ + size_. This used to be built as
    //
    //     caspar::array<int32_t>(frames * 16 * sizeof(int32_t))
    //
    // i.e. a BYTE count handed to the size_t constructor, which mallocs bytes but
    // stores the argument as size_. The array then claimed 4x as many samples as it
    // held, and audio_mixer::mix() -- which trusts item.samples.size() while its
    // loop runs to the destination size -- read past the end of the allocation and
    // mixed heap garbage into the channel. Go through a vector so the element-count
    // constructor is used, matching portaudio_producer and replay_producer.
    static constexpr int OUT_CHANNELS = 16; // AV_CHANNEL_LAYOUT_HEXADECAGONAL, see the ctor

    // frames comes from CEF's OnAudioStreamPacket; a non-positive count would make
    // the size computation below wrap into an enormous allocation.
    if (frames <= 0)
        return {};

    std::vector<int32_t> buffer(static_cast<std::size_t>(frames) * OUT_CHANNELS);
    auto*                ptr = buffer.data();

    // swr_convert returns the number of samples per channel it actually produced, or
    // a negative error. Discarding it meant a short or failed conversion silently
    // yielded a full-length buffer whose tail was never written.
    const int converted =
        swr_convert(ctx.get(), reinterpret_cast<uint8_t**>(&ptr), frames, reinterpret_cast<const uint8_t**>(src), frames);
    if (converted < 0) {
        // Deliberately not FF_RET/throwing: the only caller is CefAudioHandler::
        // OnAudioStreamPacket, so an exception here would unwind through CEF's C
        // frames. An empty array is safe -- audio_mixer skips items with no samples.
        CASPAR_LOG(warning) << L"[audio_resampler] swr_convert failed (" << converted << L"), dropping packet";
        return {};
    }

    buffer.resize(static_cast<std::size_t>(converted) * OUT_CHANNELS);
    return caspar::array<int32_t>(std::move(buffer));
}

}; // namespace caspar::ffmpeg
