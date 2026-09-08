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

        std::vector<double> out(band_count, 0.0);
        const double        bin_hz = static_cast<double>(sample_rate) / static_cast<double>(kWindow);

        // Only the first half of the spectrum: the input is real, so the second half is its
        // conjugate mirror and counting it would double every band.
        for (std::size_t k = 1; k < kWindow / 2; ++k) {
            const double f   = k * bin_hz;
            const double mag = std::sqrt(re[k] * re[k] + im[k] * im[k]);
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

    if (failures == 0)
        CASPAR_LOG(info) << L"[audio-analysis] self-test: all checks passed";

    return failures;
}

}} // namespace caspar::core
