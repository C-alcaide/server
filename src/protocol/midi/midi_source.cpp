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

#include "midi_source.h"

#include <common/log.h>

#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <windows.h>

#include <mmsystem.h>
#endif

namespace caspar { namespace protocol { namespace midi {

#ifdef _WIN32

namespace {

/// The three status bytes this reads, with the channel nibble already masked off.
constexpr unsigned char kNoteOff        = 0x80;
constexpr unsigned char kNoteOn         = 0x90;
constexpr unsigned char kControlChange  = 0xB0;
constexpr unsigned char kPitchBend      = 0xE0;

} // namespace

struct midi_source::impl
{
    const int device;

    mutable std::mutex            lock;
    std::map<std::string, double> values;
    std::uint64_t                 messages = 0;
    std::string                   device_name;

    HMIDIIN handle = nullptr;
    bool    open   = false;

    static void CALLBACK proc(HMIDIIN, UINT msg, DWORD_PTR user, DWORD_PTR p1, DWORD_PTR)
    {
        // ON A SYSTEM THREAD. The documented consequence of blocking here is dropped input, so
        // this takes a mutex, stores a number and returns. No logging, no allocation beyond the
        // map's own, and nothing that can wait on another thread.
        if (msg != MIM_DATA)
            return;
        auto* self = reinterpret_cast<impl*>(user);
        if (self)
            self->on_data(static_cast<DWORD>(p1));
    }

    void on_data(DWORD packed)
    {
        const auto status = static_cast<unsigned char>(packed & 0xFF);
        const auto d1     = static_cast<unsigned char>((packed >> 8) & 0x7F);
        const auto d2     = static_cast<unsigned char>((packed >> 16) & 0x7F);

        // The channel nibble is DISCARDED, so `cc/7` is controller 7 on any MIDI channel.
        //
        // A deliberate simplification and not an oversight: a controller sends on one channel
        // and an operator does not know or care which, so folding them means `cc/7` works
        // whatever the device is set to. The cost is that two devices on one port with
        // overlapping controller numbers collide -- and the answer to that is two sources, which
        // is what the device index is for.
        const auto kind = static_cast<unsigned char>(status & 0xF0);

        std::lock_guard<std::mutex> g(lock);
        ++messages;

        switch (kind) {
            case kControlChange:
                values["cc/" + std::to_string(d1)] = d2 / 127.0;
                break;

            case kNoteOn:
                // Velocity 0 on a NOTE ON is a note off -- the MIDI specification says so, and
                // a great many devices use it in preference to sending 0x80 at all. Treating it
                // as "the pad is at zero pressure" would leave a bound parameter latched on.
                values["note/" + std::to_string(d1)] = d2 / 127.0;
                break;

            case kNoteOff:
                values["note/" + std::to_string(d1)] = 0.0;
                break;

            case kPitchBend:
                // 14-bit, LSB first, centred at 0x2000. Normalised so centre is 0.5, because a
                // bend wheel's rest position is its middle and a binding wants that to be the
                // middle of its output range.
                values["pitch"] = ((d2 << 7) | d1) / 16383.0;
                break;

            default:
                break;
        }
    }

    explicit impl(int index)
        : device(index)
    {
        const auto count = static_cast<int>(midiInGetNumDevs());
        if (index < 0 || index >= count) {
            CASPAR_LOG(warning) << L"[midi-source] no MIDI input device " << index << L" (the system "
                                << L"reports " << count << L"). The source exists and reports "
                                << L"nothing, so a binding to it will report BROKEN.";
            return;
        }

        MIDIINCAPSW caps{};
        if (midiInGetDevCapsW(index, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
            device_name = u8(std::wstring(caps.szPname));

        const auto res = midiInOpen(&handle,
                                    static_cast<UINT>(index),
                                    reinterpret_cast<DWORD_PTR>(&impl::proc),
                                    reinterpret_cast<DWORD_PTR>(this),
                                    CALLBACK_FUNCTION);
        if (res != MMSYSERR_NOERROR) {
            CASPAR_LOG(warning) << L"[midi-source] could not open MIDI input " << index << L" ("
                                << u16(device_name) << L"), error " << res
                                << L". Another application may hold it exclusively.";
            handle = nullptr;
            return;
        }

        midiInStart(handle);
        open = true;
        CASPAR_LOG(info) << L"[midi-source] listening on MIDI input " << index << L": "
                         << u16(device_name);
    }

    ~impl()
    {
        if (!handle)
            return;
        // Stop, then reset, then close, in that order: `midiInClose` fails with
        // MIDIERR_STILLPLAYING if buffers are outstanding, and `midiInReset` is what returns
        // them. Closing without it leaks the device until the process exits, which on Windows
        // means the next `SOURCE ADD ... MIDI` on the same device fails to open.
        midiInStop(handle);
        midiInReset(handle);
        midiInClose(handle);
        handle = nullptr;
    }
};

std::vector<std::string> input_devices()
{
    std::vector<std::string> out;
    const auto               count = static_cast<int>(midiInGetNumDevs());
    for (int i = 0; i < count; ++i) {
        MIDIINCAPSW caps{};
        if (midiInGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
            out.push_back(u8(std::wstring(caps.szPname)));
        else
            out.push_back("(unreadable)");
    }
    return out;
}

#else // !_WIN32

// The non-Windows stub. It constructs, opens nothing and reports nothing, so a binding to it is
// visibly BROKEN. That is better than refusing `SOURCE ADD ... MIDI` outright: a configuration
// or preset written on Windows then loads on Linux with one dead source rather than an error
// that stops the whole file.
//
// ALSA (`snd_rawmidi_open`) is the Linux equivalent and is not written. It is a real gap and
// the docs say so.

struct midi_source::impl
{
    const int                     device;
    mutable std::mutex            lock;
    std::map<std::string, double> values;
    std::uint64_t                 messages = 0;
    std::string                   device_name;
    bool                          open = false;

    explicit impl(int index)
        : device(index)
    {
        CASPAR_LOG(warning) << L"[midi-source] MIDI input is Windows-only in this build; source "
                            << L"created but nothing will arrive. Bindings to it report BROKEN.";
    }
};

std::vector<std::string> input_devices() { return {}; }

#endif

midi_source::midi_source(int device_index)
    : impl_(std::make_shared<impl>(device_index))
{
}

midi_source::~midi_source() = default;

bool midi_source::value(const std::string& channel, double& out) const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    if (channel == "messages") {
        out = static_cast<double>(impl_->messages);
        return true;
    }

    auto it = impl_->values.find(channel);
    if (it == impl_->values.end())
        return false;
    out = it->second;
    return true;
}

std::vector<std::string> midi_source::channels() const
{
    std::lock_guard<std::mutex> g(impl_->lock);

    // Whatever has ARRIVED, plus the message counter. There is no declared set: a controller has
    // 128 possible controllers and 128 possible notes, and listing 256 channels nobody has
    // touched would bury the four an operator is actually using. So a binding to an untouched
    // controller reports `broken` until the knob is turned once -- which is the right behaviour
    // for hardware, and is why `messages` exists to prove the device is alive.
    std::vector<std::string> out{"messages"};
    for (const auto& kv : impl_->values)
        out.push_back(kv.first);
    return out;
}

std::string midi_source::describe() const
{
    std::lock_guard<std::mutex> g(impl_->lock);
    return "midi " + std::to_string(impl_->device) + " \"" + impl_->device_name + "\" (" +
           (impl_->open ? "listening" : "NOT OPEN") + ", " + std::to_string(impl_->messages) +
           " messages)";
}

bool midi_source::listening() const
{
    std::lock_guard<std::mutex> g(impl_->lock);
    return impl_->open;
}

}}} // namespace caspar::protocol::midi
