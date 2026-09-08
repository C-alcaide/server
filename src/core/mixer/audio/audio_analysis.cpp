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

#include "../../StdAfx.h"

#include "audio_analysis.h"

#include <common/log.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

namespace caspar { namespace core {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Analysis window. 1024 at 48 kHz is 21.3 ms and 46.9 Hz per bin.
///
/// Chosen against the two things that actually constrain it: a bin narrow enough to separate a
/// bass line from a kick (46.9 Hz does), and a window short enough that a band follows a beat
/// rather than smearing across it (21 ms is half a frame at 25p). 2048 would halve the bin
/// width and double the latency, which is the wrong direction for a reactive parameter.
constexpr std::size_t kWindow = 1024;

/// The lowest and highest band edges, in Hz.
///
/// 40 Hz because below it a normal monitor chain reproduces nothing and a band there reads room
/// rumble; 16 kHz because above it there is no musical content worth binding to and the top bin
/// of a 48 kHz analysis is 24 kHz of mostly nothing.
constexpr double kLowHz  = 40.0;
constexpr double kHighHz = 16000.0;

double to_dbfs(double linear)
{
    if (!(linear > 0.0))
        return silence_dbfs;
    const double db = 20.0 * std::log10(linear);
    return db < silence_dbfs ? silence_dbfs : db;
}

} // namespace

void fft_radix2(std::vector<double>& re, std::vector<double>& im)
{
    const std::size_t n = re.size();
    if (n < 2 || im.size() != n || (n & (n - 1)) != 0)
        return;

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    // Butterflies, doubling the block length each pass.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * kPi / static_cast<double>(len);
        const double wr  = std::cos(ang);
        const double wi  = std::sin(ang);
        for (std::size_t i = 0; i < n; i += len) {
            double cr = 1.0;
            double ci = 0.0;
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::size_t a = i + k;
                const std::size_t b = i + k + len / 2;

                const double xr = re[b] * cr - im[b] * ci;
                const double xi = re[b] * ci + im[b] * cr;

                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;

                // The twiddle advanced by multiplication rather than recomputed with cos/sin per
                // bin. The accumulated error over 512 steps is ~1e-13, which the self-test's
                // Parseval check bounds, and it removes 2 * n * log2(n) transcendental calls from
                // a path that runs on the audio thread.
                const double nr = cr * wr - ci * wi;
                ci              = cr * wi + ci * wr;
                cr              = nr;
            }
        }
    }
}

struct audio_analysis::impl
{
    mutable std::mutex lock;

    int sample_rate;
    int band_count;

    audio_levels        current;
    std::vector<double> edges;

    //: Mono downmix, filled until it holds `kWindow` samples and then transformed.
    std::vector<double> window;

    //: Hann, precomputed. A rectangular window smears a pure tone across every bin, which
    //: makes a band reading meaningless -- and it is the reason the self-test checks the
    //: single-bin case against the UNWINDOWED transform rather than through this class.
    std::vector<double> hann;

    //: Per-channel accumulators, reused. Members rather than locals because `feed` runs on the
    //: audio path of every tick and a 16-element allocation per tick is a pointless one.
    std::vector<double> scratch_sq;
    std::vector<double> scratch_peak;

    //: The last completed window, retained for `spectrum()` and `waveform()`.
    //:
    //: RETAINED rather than recomputed, because both are read from the stage executor and the
    //: transform runs on the audio path -- so recomputing on read would put an FFT on whichever
    //: thread happened to ask, once per reader. Retained rather than published, because a
    //: reader that does not want them must not pay to copy them (see the header).
    //:
    //: `last_bins` is the magnitude per bin, normalised the same way the bands are.
    //: `last_wave` is the window's own samples, so a waveform reader sees exactly what the
    //: spectrum was computed from rather than a differently-aligned slice.
    std::vector<double> last_bins;
    std::vector<double> last_wave;

    impl(int rate, int bands)
        : sample_rate(rate > 0 ? rate : 48000)
        , band_count(bands > 0 ? bands : 3)
    {
        window.reserve(kWindow);
        hann.resize(kWindow);
        for (std::size_t i = 0; i < kWindow; ++i)
            hann[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                                           static_cast<double>(kWindow - 1));
        rebuild_edges();
        current.bands.clear();
    }

    void rebuild_edges()
    {
        // LOG-spaced, because pitch is logarithmic: linear edges put five of six bands above
        // 4 kHz, where almost nothing a VJ wants to react to lives.
        edges.assign(band_count + 1, 0.0);
        const double lo = std::log(kLowHz);
        const double hi = std::log(std::min(kHighHz, sample_rate * 0.5));
        for (int i = 0; i <= band_count; ++i)
            edges[i] = std::exp(lo + (hi - lo) * i / band_count);
    }

    void analyse()
    {
        std::vector<double> re(kWindow, 0.0);
        std::vector<double> im(kWindow, 0.0);
        for (std::size_t i = 0; i < kWindow; ++i)
            re[i] = window[i] * hann[i];

        fft_radix2(re, im);

        // The window's samples, kept before the transform overwrites nothing (it works on `re`,
        // a copy) -- so this is only a copy, and it is what `waveform()` returns.
        last_wave = window;

        std::vector<double> out(band_count, 0.0);
        last_bins.assign(kWindow / 2, 0.0);
        const double        bin_hz = static_cast<double>(sample_rate) / static_cast<double>(kWindow);

        // Only the first half of the spectrum: the input is real, so the second half is its
        // conjugate mirror and counting it would double every band.
        // The per-bin magnitudes are normalised HERE, by the same factor the bands use, so a
        // caller comparing a bin against a band is comparing like with like. A bin normalised
        // differently from the band containing it would be the sort of discrepancy that reads as
        // a defect in whichever one the reader trusts less.
        const double bin_norm = static_cast<double>(kWindow) / 4.0;

        for (std::size_t k = 1; k < kWindow / 2; ++k) {
            const double f   = k * bin_hz;
            const double mag = std::sqrt(re[k] * re[k] + im[k] * im[k]);

            last_bins[k] = mag / bin_norm;

            for (int b = 0; b < band_count; ++b) {
                if (f >= edges[b] && f < edges[b + 1]) {
                    out[b] += mag;
                    break;
                }
            }
        }

        // Normalised so a full-scale sine inside a band reads about 1.
        //
        // The factors: the transform's output scales with the window length, and the Hann window
        // halves the coherent gain. `kWindow / 4` is (N/2) * (1/2) -- N/2 from the one-sided
        // spectrum of a unit sine and 1/2 from the window. It is a CALIBRATION rather than a
        // physical quantity, which is why the battery gates a band reading against a tone at a
        // known level rather than against an absolute number derived here.
        const double norm = static_cast<double>(kWindow) / 4.0;
        for (auto& v : out)
            v = v / norm;

        current.bands = std::move(out);
    }

    void feed(const int32_t* interleaved, std::size_t frames, int channels)
    {
        if (!interleaved || frames == 0 || channels <= 0)
            return;

        constexpr double scale = 1.0 / 2147483648.0; // int32 full scale

        std::lock_guard<std::mutex> g(lock);

        // PER-CHANNEL first, and this is not a refinement -- it is the fix for a defect the
        // first version had.
        //
        // That version averaged every channel into a mono signal and took the RMS of THAT. On
        // this rig the mixer runs a 16-channel layout, so a stereo clip was divided by 16 while
        // only two channels carried anything: a -20 dBFS tone read -41 dBFS, and
        // `binding-audio`'s calibration check caught it on its first run. The peak agreed with
        // the mixer's own long-standing `volume` state exactly, which is what said the error was
        // in the downmix and not in the level maths.
        //
        // So the reported level is the LOUDEST CHANNEL's RMS, which is what a level meter shows
        // and which does not change when a layout gains silent channels. A binding's input range
        // then means the same thing on a stereo channel and on a 16-channel one.
        scratch_sq.assign(static_cast<std::size_t>(channels), 0.0);
        scratch_peak.assign(static_cast<std::size_t>(channels), 0.0);

        for (std::size_t f = 0; f < frames; ++f) {
            for (int c = 0; c < channels; ++c) {
                const double v = interleaved[f * channels + c] * scale;
                scratch_sq[c] += v * v;
                scratch_peak[c] = std::max(scratch_peak[c], std::abs(v));
            }
        }

        double best_rms = 0.0;
        double peak     = 0.0;
        int    active   = 0;
        for (int c = 0; c < channels; ++c) {
            best_rms = std::max(best_rms, std::sqrt(scratch_sq[c] / static_cast<double>(frames)));
            peak     = std::max(peak, scratch_peak[c]);
            if (scratch_peak[c] > 0.0)
                ++active;
        }

        // The spectrum's input is the mean over the channels that CARRY something, for the same
        // reason: dividing by the layout width would scale every band by however many silent
        // channels the configuration happens to declare. `active` can be 0 on digital silence,
        // hence the floor of 1 -- and the window then fills with zeros, which is correct.
        const double divisor = static_cast<double>(active > 0 ? active : 1);

        for (std::size_t f = 0; f < frames && window.size() < kWindow; ++f) {
            double mono = 0.0;
            for (int c = 0; c < channels; ++c)
                mono += interleaved[f * channels + c] * scale;
            window.push_back(mono / divisor);
        }

        current.rms  = best_rms;
        current.dbfs = to_dbfs(current.rms);
        current.peak = peak;

        if (window.size() >= kWindow) {
            analyse();
            // The whole window is dropped rather than slid by a hop, deliberately: a hop needs a
            // copy of the tail on every tick and buys temporal resolution the frame rate cannot
            // use. At 48 kHz a 1024 window completes about every 21 ms, which is faster than a
            // 25p tick, so no frame goes without a fresh spectrum.
            window.clear();
        }
    }
};

audio_analysis::audio_analysis(int sample_rate, int bands)
    : impl_(std::make_unique<impl>(sample_rate, bands))
{
}

audio_analysis::~audio_analysis() = default;

void audio_analysis::feed(const int32_t* interleaved, std::size_t frames, int channels)
{
    impl_->feed(interleaved, frames, channels);
}

audio_levels audio_analysis::levels() const
{
    std::lock_guard<std::mutex> g(impl_->lock);
    return impl_->current;
}

std::vector<double> audio_analysis::band_edges() const
{
    std::lock_guard<std::mutex> g(impl_->lock);
    return impl_->edges;
}

std::vector<double> audio_analysis::spectrum(int bins) const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    const auto& src = impl_->last_bins;
    if (src.empty() || bins <= 0)
        return {};

    // Asking for at least as many as the transform has gives the transform's own, unchanged.
    // There is nothing to interpolate from, and inventing detail would misrepresent the
    // resolution -- a shader drawing 1024 bars from 512 bins should draw 512 and stretch them,
    // which is its decision and not ours to fake.
    if (static_cast<std::size_t>(bins) >= src.size())
        return src;

    // THE PEAK of each group, not the mean and not decimation.
    //
    // Decimation is out for the obvious reason: taking every Nth bin aliases, and a tone landing
    // on a skipped bin vanishes entirely.
    //
    // The MEAN is out for a less obvious one, and it was this function's first implementation. A
    // pure tone occupies one or two of the group's bins, so a mean divides it by the group size:
    // reducing 512 bins to 64 made a full-scale tone read 1/8 of its magnitude, and reducing to
    // 32 would halve that again -- so the same signal read at two resolutions gave two answers,
    // and at 8-bit texture precision the quieter one rounded to zero. Measured: `isf-audio`'s
    // probes read 1 and 0 out of 255 where the direction was right and the magnitude was not.
    //
    // The peak is invariant under the group size for a tone, which is the property a bar wants:
    // a bar answers "is there energy in this range", and a mean answers "how much of this range
    // is energy" -- which is a different and less useful question at display resolutions.
    //
    // NOTE THIS DIFFERS FROM THE BANDS beside it, which SUM. That is deliberate and the two are
    // not inconsistent: a band is a wide frequency range whose TOTAL energy is the quantity an
    // operator means by "the bass", while a reduced-resolution bin is a narrow group standing in
    // for a peak. The first version of this function averaged, which was neither -- and left a
    // bin scaled differently from the band containing it, which is exactly the discrepancy the
    // normalisation comment above claims to avoid.
    std::vector<double> out(static_cast<std::size_t>(bins), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t lo = i * src.size() / out.size();
        const std::size_t hi = std::max(lo + 1, (i + 1) * src.size() / out.size());
        double            peak = 0.0;
        for (std::size_t k = lo; k < hi && k < src.size(); ++k)
            peak = std::max(peak, src[k]);
        out[i] = peak;
    }
    return out;
}

std::vector<double> audio_analysis::waveform(int samples) const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    const auto& src = impl_->last_wave;
    if (src.empty() || samples <= 0)
        return {};

    if (static_cast<std::size_t>(samples) >= src.size())
        return src;

    // The MOST RECENT `samples`, from the end. A waveform display fed the oldest part of the
    // window is a display of the past, and the window is 21 ms -- visible as lag on a transient.
    return std::vector<double>(src.end() - samples, src.end());
}

int audio_analysis::native_bins() { return static_cast<int>(kWindow / 2); }

int audio_analysis::native_window() { return static_cast<int>(kWindow); }

void audio_analysis::set_sample_rate(int rate)
{
    std::lock_guard<std::mutex> g(impl_->lock);
    if (rate <= 0 || rate == impl_->sample_rate)
        return;
    impl_->sample_rate = rate;
    impl_->rebuild_edges();
}

// ---------------------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------------------

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "")
{
    if (ok)
        return;
    ++failures;
    CASPAR_LOG(error) << L"[audio-analysis] SELF-TEST FAILED: " << u16(what)
                      << (detail.empty() ? L"" : (L" -- " + u16(detail)));
}

} // namespace

int audio_analysis_self_test()
{
    failures = 0;

    // ---- DC: all the energy in bin 0 and nothing anywhere else ---------------------------
    {
        const std::size_t   n = 64;
        std::vector<double> re(n, 1.0), im(n, 0.0);
        fft_radix2(re, im);
        check(std::abs(re[0] - static_cast<double>(n)) < 1e-9 && std::abs(im[0]) < 1e-9,
              "fft/dc/bin0", std::to_string(re[0]));
        double rest = 0.0;
        for (std::size_t k = 1; k < n; ++k)
            rest += std::abs(re[k]) + std::abs(im[k]);
        check(rest < 1e-9, "fft/dc/no-leakage", std::to_string(rest));
    }

    // ---- A sinusoid EXACTLY on a bin: magnitude n/2 there, nothing elsewhere -------------
    //
    // On a bin rather than between two, so the expected answer is closed-form. Between bins the
    // energy legitimately splits, and asserting a single peak there would be asserting a defect.
    {
        const std::size_t   n = 64;
        const std::size_t   k = 5;
        std::vector<double> re(n), im(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            re[i] = std::cos(2.0 * kPi * k * i / n);
        fft_radix2(re, im);

        const double mag_k = std::sqrt(re[k] * re[k] + im[k] * im[k]);
        check(std::abs(mag_k - n / 2.0) < 1e-9, "fft/sine/on-bin", std::to_string(mag_k));

        double worst_other = 0.0;
        for (std::size_t j = 1; j < n / 2; ++j) {
            if (j == k)
                continue;
            worst_other = std::max(worst_other, std::sqrt(re[j] * re[j] + im[j] * im[j]));
        }
        check(worst_other < 1e-9, "fft/sine/isolated", std::to_string(worst_other));
    }

    // ---- Parseval: total energy is conserved ---------------------------------------------
    //
    // The check that catches a wrong twiddle, a missed butterfly or an off-by-one in the
    // bit-reversal -- none of which the two cases above necessarily see, because both have
    // special structure. This one uses a deterministic pseudo-random input, which has none.
    {
        const std::size_t   n = 256;
        std::vector<double> re(n), im(n, 0.0);
        uint64_t            x = 0x243F6A8885A308D3ull;
        double              time_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            re[i] = static_cast<double>(static_cast<int64_t>(x >> 11)) / static_cast<double>(1ull << 52) - 1.0;
            time_energy += re[i] * re[i];
        }
        auto re2 = re;
        auto im2 = im;
        fft_radix2(re2, im2);
        double freq_energy = 0.0;
        for (std::size_t k = 0; k < n; ++k)
            freq_energy += re2[k] * re2[k] + im2[k] * im2[k];
        freq_energy /= static_cast<double>(n);
        check(std::abs(freq_energy - time_energy) < 1e-9 * std::max(1.0, time_energy),
              "fft/parseval",
              "time " + std::to_string(time_energy) + " freq " + std::to_string(freq_energy));
    }

    // ---- A non-power-of-two input is LEFT ALONE rather than corrupted ---------------------
    {
        std::vector<double> re{1.0, 2.0, 3.0}, im{0.0, 0.0, 0.0};
        const auto          before = re;
        fft_radix2(re, im);
        check(re == before, "fft/refuses-non-power-of-two");
    }

    // ---- dBFS: the two anchors and the floor ---------------------------------------------
    check(std::abs(to_dbfs(1.0) - 0.0) < 1e-12, "dbfs/full-scale");
    check(std::abs(to_dbfs(0.1) + 20.0) < 1e-9, "dbfs/minus-20", std::to_string(to_dbfs(0.1)));
    check(to_dbfs(0.0) == silence_dbfs, "dbfs/silence-is-floored");
    check(to_dbfs(1e-30) == silence_dbfs, "dbfs/tiny-is-floored");

    // ---- RMS and the downmix, through the class ------------------------------------------
    {
        // A full-scale square wave: every sample is +/- full scale, so the RMS IS full scale.
        // Chosen over a sine because its RMS is closed-form with no 1/sqrt(2) to get wrong.
        audio_analysis          a(48000, 3);
        std::vector<int32_t>    buf(512 * 2);
        for (std::size_t f = 0; f < 512; ++f) {
            const int32_t v = (f % 2) ? 2147483647 : -2147483647;
            buf[f * 2]      = v;
            buf[f * 2 + 1]  = v;
        }
        a.feed(buf.data(), 512, 2);
        const auto l = a.levels();
        check(std::abs(l.rms - 1.0) < 1e-6, "levels/square-rms-is-full-scale", std::to_string(l.rms));
        check(std::abs(l.peak - 1.0) < 1e-6, "levels/peak", std::to_string(l.peak));
        check(std::abs(l.dbfs - 0.0) < 1e-4, "levels/dbfs", std::to_string(l.dbfs));

        // Silence reads zero and the FLOOR, not -inf.
        audio_analysis       b(48000, 3);
        std::vector<int32_t> quiet(512 * 2, 0);
        b.feed(quiet.data(), 512, 2);
        const auto q = b.levels();
        check(q.rms == 0.0 && q.dbfs == silence_dbfs, "levels/silence",
              std::to_string(q.rms) + " " + std::to_string(q.dbfs));
    }

    // ---- The level does not depend on the channel LAYOUT ---------------------------------
    //
    // Two shapes of the same defect, and the second is the one that actually happened.
    //
    // (a) Every channel carrying the signal. A summing downmix would make an 8-channel mix
    //     eight times as loud as the same programme in stereo.
    {
        for (int channels : {1, 2, 8}) {
            audio_analysis       a(48000, 3);
            std::vector<int32_t> buf(256 * channels);
            for (std::size_t f = 0; f < 256; ++f)
                for (int c = 0; c < channels; ++c)
                    buf[f * channels + c] = (f % 2) ? 2147483647 : -2147483647;
            a.feed(buf.data(), 256, channels);
            check(std::abs(a.levels().rms - 1.0) < 1e-6,
                  "levels/level-independent-of-channel-count/" + std::to_string(channels),
                  std::to_string(a.levels().rms));
        }
    }

    // (b) STEREO CONTENT IN A WIDER LAYOUT, which is the normal case on this rig: the mixer
    //     runs a 16-channel layout and a clip carries two. An averaging downmix divides by the
    //     LAYOUT width, so the level falls by 20*log10(16/2) = 18 dB -- and the first version
    //     of this file did exactly that, reading a -20 dBFS tone as -41 dBFS. The reported
    //     level is the loudest channel's RMS, so adding silent channels must change nothing.
    {
        for (int channels : {2, 8, 16}) {
            audio_analysis       a(48000, 3);
            std::vector<int32_t> buf(256 * channels, 0);
            for (std::size_t f = 0; f < 256; ++f) {
                const int32_t v      = (f % 2) ? 2147483647 : -2147483647;
                buf[f * channels]     = v;   // only the first two channels carry anything
                buf[f * channels + 1] = v;
            }
            a.feed(buf.data(), 256, channels);
            check(std::abs(a.levels().rms - 1.0) < 1e-6,
                  "levels/silent-channels-do-not-attenuate/" + std::to_string(channels),
                  std::to_string(a.levels().rms));
        }
    }

    // ---- The bands: a tone lands in the band that CONTAINS it, and not in the others ------
    {
        audio_analysis a(48000, 3);
        const auto     edges = a.band_edges();
        check(edges.size() == 4 && edges.front() > 0.0 && edges.back() > edges.front(),
              "bands/edges-ascending");

        // Two tones, each in a different band, checked separately. ASYMMETRIC by construction:
        // a single tone cannot tell "the bands work" from "every band reports the whole
        // spectrum", which is exactly what a missing edge test would do.
        struct probe
        {
            double hz;
            int    band;
        };
        // 100 Hz is inside band 0 (40 .. ~400 Hz) and 8 kHz inside band 2 (~1.6k .. 16k).
        const probe probes[] = {{100.0, 0}, {8000.0, 2}};

        for (const auto& p : probes) {
            audio_analysis       an(48000, 3);
            std::vector<int32_t> buf(4096 * 2);
            for (std::size_t f = 0; f < 4096; ++f) {
                const double s = std::sin(2.0 * kPi * p.hz * f / 48000.0);
                const auto   v = static_cast<int32_t>(s * 2147483000.0);
                buf[f * 2]     = v;
                buf[f * 2 + 1] = v;
            }
            an.feed(buf.data(), 4096, 2);
            const auto bands = an.levels().bands;
            if (bands.size() != 3) {
                check(false, "bands/populated/" + std::to_string(static_cast<int>(p.hz)),
                      std::to_string(bands.size()) + " bands");
                continue;
            }
            const int   other1 = (p.band + 1) % 3;
            const int   other2 = (p.band + 2) % 3;
            const bool  loudest = bands[p.band] > bands[other1] * 4.0 &&
                                 bands[p.band] > bands[other2] * 4.0;
            check(loudest, "bands/tone-lands-in-its-own-band/" + std::to_string(static_cast<int>(p.hz)),
                  "b0 " + std::to_string(bands[0]) + " b1 " + std::to_string(bands[1]) + " b2 " +
                      std::to_string(bands[2]));
        }
    }

    // ---- spectrum(n) and waveform(n): what the ISF audio textures are built from ----------
    //
    // These two exist for `audioFFT` and `audio`, which are TEXTURES a shader samples -- so a
    // defect here is a wrong picture in someone else's shader rather than a wrong number in
    // ours, and it is worth checking at boot rather than only through a fixture.
    {
        // A tone at a known frequency, then the bin it must land in, computed from the bin
        // width rather than looked up: 8000 Hz at 48 kHz over a 1024 window is bin
        // 8000 / (48000/1024) = 170.67, so bin 170 or 171.
        audio_analysis       a(48000, 3);
        std::vector<int32_t> buf(4096 * 2);
        const double         hz = 8000.0;
        for (std::size_t f = 0; f < 4096; ++f) {
            const double v = std::sin(2.0 * kPi * hz * f / 48000.0);
            const auto   s = static_cast<int32_t>(v * 2147483000.0);
            buf[f * 2]     = s;
            buf[f * 2 + 1] = s;
        }
        a.feed(buf.data(), 4096, 2);

        const auto full = a.spectrum(audio_analysis::native_bins());
        check(full.size() == static_cast<std::size_t>(audio_analysis::native_bins()),
              "spectrum/native-size", std::to_string(full.size()));

        if (!full.empty()) {
            const double bin_hz   = 48000.0 / 1024.0;
            const auto   want_bin = static_cast<std::size_t>(hz / bin_hz);
            std::size_t  loudest  = 0;
            for (std::size_t k = 1; k < full.size(); ++k)
                if (full[k] > full[loudest])
                    loudest = k;
            // Within one bin: the tone does not sit exactly on a bin centre, so the energy
            // legitimately splits between two and either may be the larger.
            check(loudest + 1 >= want_bin && loudest <= want_bin + 1,
                  "spectrum/tone-lands-in-its-own-bin",
                  "loudest " + std::to_string(loudest) + ", want ~" + std::to_string(want_bin));
        }

        // ASKING FOR MORE than the transform has gives the transform's own count, unchanged.
        // Interpolating would misrepresent the resolution to a shader sizing a texture from it.
        const auto over = a.spectrum(audio_analysis::native_bins() * 4);
        check(over.size() == full.size(), "spectrum/no-invented-resolution",
              std::to_string(over.size()) + " vs " + std::to_string(full.size()));

        // REDUCED by taking each group's PEAK, not by truncation. The two are told apart by
        // where the tone goes: a peak keeps it at the same FRACTION of the way up the spectrum,
        // and truncation to a quarter of the bins would drop an 8 kHz tone entirely because it
        // sits above the first quarter.
        const auto quarter = a.spectrum(audio_analysis::native_bins() / 4);
        check(quarter.size() == static_cast<std::size_t>(audio_analysis::native_bins() / 4),
              "spectrum/reduced-size", std::to_string(quarter.size()));
        if (!quarter.empty() && !full.empty()) {
            std::size_t loud_q = 0;
            for (std::size_t k = 1; k < quarter.size(); ++k)
                if (quarter[k] > quarter[loud_q])
                    loud_q = k;
            const double frac_full = 0.0 + static_cast<double>(
                std::distance(full.begin(), std::max_element(full.begin() + 1, full.end())))
                                     / static_cast<double>(full.size());
            const double frac_q = static_cast<double>(loud_q) / static_cast<double>(quarter.size());
            check(std::abs(frac_full - frac_q) < 0.05, "spectrum/reduced-not-truncated",
                  "full at " + std::to_string(frac_full) + " of the way up, reduced at " +
                      std::to_string(frac_q));
        }

        // Every value stays in range after the reduction -- a max of values in 0..1 is in
        // 0..1, so a failure here is an indexing error rather than an arithmetic one.
        bool in_range = true;
        for (double v : quarter)
            if (v < -1e-12 || v > 1.0 + 1e-12)
                in_range = false;
        check(in_range, "spectrum/reduced-in-range");

        // THE MAGNITUDE SURVIVES THE REDUCTION, which is what tells a peak from a mean and is
        // the check that was missing when this function averaged. A tone's own bin is unchanged
        // by grouping under a peak; under a mean it is divided by the group size, so the same
        // signal read at two resolutions gave two answers and the coarser one rounded to zero
        // in an 8-bit texture.
        //
        // Checked at TWO reductions rather than one, because a single one cannot distinguish
        // "the magnitude survives" from "this particular group size happens to agree".
        if (!full.empty()) {
            const double peak_full = *std::max_element(full.begin() + 1, full.end());
            for (int n : {audio_analysis::native_bins() / 4, audio_analysis::native_bins() / 16}) {
                const auto red = a.spectrum(n);
                if (red.empty())
                    continue;
                const double peak_red = *std::max_element(red.begin(), red.end());
                check(std::abs(peak_red - peak_full) < 1e-9,
                      "spectrum/magnitude-survives-reduction/" + std::to_string(n),
                      "full " + std::to_string(peak_full) + " reduced " + std::to_string(peak_red));
            }
        }

        check(a.spectrum(0).empty(), "spectrum/zero-bins-is-empty");
    }

    {
        // The waveform, from a full-scale SQUARE wave: every sample is +/-1, so every returned
        // value must be at one extreme. A sine would leave the check unable to distinguish a
        // correct read from a scaled one.
        audio_analysis       a(48000, 3);
        std::vector<int32_t> buf(2048 * 2);
        for (std::size_t f = 0; f < 2048; ++f) {
            const int32_t v = (f % 2) ? 2147483000 : -2147483000;
            buf[f * 2]      = v;
            buf[f * 2 + 1]  = v;
        }
        a.feed(buf.data(), 2048, 2);

        const auto full = a.waveform(audio_analysis::native_window());
        check(full.size() == static_cast<std::size_t>(audio_analysis::native_window()),
              "waveform/native-size", std::to_string(full.size()));

        bool extremes = !full.empty();
        for (double v : full)
            if (std::abs(std::abs(v) - 1.0) > 1e-3)
                extremes = false;
        check(extremes, "waveform/full-scale-square-is-at-the-extremes");

        // Fewer samples takes the MOST RECENT, which is what a display wants. Checked by
        // identity against the tail of the full window rather than by a property, because "the
        // last N" and "the first N" are both plausible and only one is right.
        const auto tail = a.waveform(64);
        bool       is_tail = tail.size() == 64 && full.size() >= 64;
        if (is_tail)
            for (std::size_t i = 0; i < tail.size(); ++i)
                if (tail[i] != full[full.size() - 64 + i])
                    is_tail = false;
        check(is_tail, "waveform/short-read-takes-the-most-recent");

        check(a.waveform(0).empty(), "waveform/zero-samples-is-empty");

        // Before ANY window has completed, both are empty rather than a zero-filled buffer of
        // the right size. A shader given silence would draw a flat line and look correct, so
        // the producer needs to be able to tell "no data yet" from "digital silence".
        audio_analysis       fresh(48000, 3);
        std::vector<int32_t> few(16 * 2, 0);
        fresh.feed(few.data(), 16, 2);
        check(fresh.spectrum(64).empty() && fresh.waveform(64).empty(),
              "spectrum+waveform/empty-until-a-window-completes");
    }

    if (failures == 0)
        CASPAR_LOG(info) << L"[audio-analysis] self-test: all checks passed";

    return failures;
}

}} // namespace caspar::core
