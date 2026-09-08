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

// Level and spectrum from the mixed audio, so a parameter can be driven by what is playing.
//
// WHY THIS EXISTS. The audio mixer already walked every sample of every tick to compute a
// per-channel PEAK for the `volume` state and the clipping tag -- and that was the whole of the
// fork's audio measurement. Peak is the wrong quantity for a reactive parameter: it is a single
// sample's absolute value, so it jumps on a click and says nothing about loudness. RMS is what
// a level meter shows and what a VJ tool binds to, and it costs one extra multiply-accumulate
// in a loop that was already running.
//
// The spectrum is the thing VDJ workflows actually want -- Resolume's FFT source with its
// low/mid/high bands and a gain and a fall per band is the single most-used animation source in
// that product, and ISF's `audioFFT` input texture is the most-used feature of that format.
//
// WHY THE FFT IS WRITTEN HERE RATHER THAN TAKEN FROM FFmpeg. `av_tx` would have been the
// obvious choice -- it is a good implementation and it is already built. But **`core` links
// `common`, GLEW and SFML and deliberately not FFmpeg**: the decode path lives in
// `modules/ffmpeg` precisely so that the frame and audio core does not depend on it. Pulling
// avutil into `core` to get 60 lines of radix-2 arithmetic would invert that on the strength of
// one function, and every module that includes a core header would inherit the dependency.
//
// So the transform is here, it is a textbook iterative radix-2 Cooley-Tukey, and it is checked
// at every server start against closed-form cases -- a DC input, a single bin's sinusoid, and
// Parseval's theorem. A hand-written FFT with no test would be the wrong trade; with one, the
// dependency is the worse of the two costs.

#include <core/monitor/monitor.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace caspar { namespace core {

/// What one tick of audio measured. Plain data, copyable, published as-is.
struct audio_levels
{
    /// Linear RMS of the mono downmix over the tick, 0..1. `0` for silence.
    double rms = 0.0;

    /// The same in dBFS, floored at `silence_dbfs` rather than running to -inf: a binding's
    /// input range cannot map an infinity, and a level meter shows a floor for the same reason.
    double dbfs = -120.0;

    /// Peak absolute sample over the tick, 0..1. Kept because the clipping tag needs it and
    /// because a transient-driven parameter wants it rather than RMS.
    double peak = 0.0;

    /// Band energies, one per band, normalised so a full-scale sine in a band reads ~1.
    ///
    /// Empty until a full analysis window has been filled. That is a real state and not an
    /// error: at 48 kHz with a 1024-sample window, the first result arrives after ~21 ms, so a
    /// binding created in the same command as a `PLAY` reads no bands for one or two ticks.
    std::vector<double> bands;
};

/// The floor `dbfs` reports instead of -infinity for digital silence.
constexpr double silence_dbfs = -120.0;

/// Rolling level and spectrum analysis over the mixed output.
///
/// Fed from the audio mixer's own tick, on the mixer thread, and read from anywhere: `levels()`
/// returns a copy under a lock rather than exposing the working buffers.
class audio_analysis
{
  public:
    /// `bands` is how many log-spaced bands the spectrum is reduced to. Three is Resolume's
    /// low/mid/high and the default; more is legal and is what an ISF `audioFFT` texture wants.
    explicit audio_analysis(int sample_rate = 48000, int bands = 3);
    ~audio_analysis();

    audio_analysis(const audio_analysis&)            = delete;
    audio_analysis& operator=(const audio_analysis&) = delete;

    /// One tick of interleaved 32-bit samples, as the mixer produces them.
    void feed(const int32_t* interleaved, std::size_t frames, int channels);

    /// The last measured levels.
    audio_levels levels() const;

    /// The band edges in Hz, `bands + 1` of them. For the published description, so a client
    /// can label a band rather than calling it "band 2".
    std::vector<double> band_edges() const;

    /// The last window's spectrum, reduced to `bins` values in 0..1. Empty until a window has
    /// been filled.
    ///
    /// SEPARATE FROM `levels()` and not a member of `audio_levels`, deliberately. That struct is
    /// COPIED on every `levels()` call -- once per tick for the published state and once more
    /// per binding that reads the source -- so putting 512 bins and 1024 samples in it would put
    /// about 12 KB of copying per reader on the audio path to produce numbers almost every
    /// reader ignores. These two take the lock and return only what was asked for.
    ///
    /// `bins` is REDUCED from the transform's own resolution by averaging adjacent groups, not
    /// by taking the first `bins` of them: the FFT gives `kWindow/2` usable bins covering the
    /// whole spectrum, and truncating would silently discard everything above
    /// `bins * sample_rate / kWindow` Hz. Asking for more bins than the transform has gives the
    /// transform's own count -- there is nothing to interpolate from and inventing detail would
    /// be a lie about the resolution.
    std::vector<double> spectrum(int bins) const;

    /// The last window's mono samples, most recent last, each in -1..1. At most one window's
    /// worth; fewer if `samples` asks for fewer, taking the MOST RECENT ones.
    ///
    /// The most recent rather than the oldest, because a waveform display that lags by a window
    /// is a waveform display of the past. Empty until a window has been filled.
    std::vector<double> waveform(int samples) const;

    /// How many bins the transform itself produces -- `kWindow / 2`. What `spectrum()` will
    /// return if asked for more, and what a caller sizing a texture wants to know.
    static int native_bins();

    /// How many samples one analysis window holds. The ceiling on `waveform()`.
    static int native_window();

    void set_sample_rate(int rate);

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// In-place iterative radix-2 FFT. `re` and `im` must both be a power-of-two length.
///
/// Exposed for the self-test, which is the only reason it is not private: a transform checked
/// only through the class it lives in cannot be checked against closed-form cases.
void fft_radix2(std::vector<double>& re, std::vector<double>& im);

/// Property checks over the transform and the level maths, at every server start. Returns the
/// number of failures and logs each.
int audio_analysis_self_test();

}} // namespace caspar::core
