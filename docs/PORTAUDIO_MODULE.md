# PortAudio Module

## Overview

The PortAudio module provides professional audio I/O for CasparCG Server via the PortAudio library (v19.7.0, statically linked). It supports ASIO, WASAPI, and DirectSound host APIs on Windows, enabling multi-channel output to professional audio interfaces and low-latency capture from any input device.

The module provides three components:

- **Consumer** — Audio output to any PortAudio-compatible device (USB interfaces, ASIO, Dante, MADI). Acts as a master synchronisation clock for the channel.
- **Producer** — Audio capture from any input device (microphones, line-in, virtual cables, Dante receivers) into a CasparCG layer.
- **Shared utilities** — Device enumeration, host API selection, and a lock-free SPSC ring buffer used by both consumer and producer.

The LTC timecode module also uses PortAudio internally for its audio capture stream.

## Module Structure

```
src/modules/portaudio/
├── portaudio_module.h/cpp          Module lifecycle (init/uninit)
├── consumer/
│   └── portaudio_consumer.h/cpp    Audio output consumer
├── producer/
│   └── portaudio_producer.h/cpp    Audio capture producer
├── util/
│   ├── portaudio_device.h/cpp      Device manager singleton
│   └── spsc_ring_buffer.h          Lock-free ring buffer
└── CMakeLists.txt                  Build config + FetchContent
```

## Architecture

### Consumer (Audio Output)

The consumer bridges CasparCG's push model (one audio buffer per video frame) with PortAudio's pull model (hardware callback requests samples at its own rate).

```
Channel tick                    Hardware callback
    │                                │
    ▼                                ▼
  send()                        stream_callback()
    │                                │
    ▼                                ▼
  Queue ──► Write Thread ──► SPSC Ring Buffer ──► PA Callback ──► DAC
              │         blocks on              reads lock-free
              │         drain_cv_              notifies drain_cv_
              ▼
         future completes
         (master clock)
```

**Key design decisions:**

1. **Master clock via ring buffer backpressure.** The write thread blocks until the hardware drains enough space from the ring buffer. This makes the audio device's crystal oscillator pace the entire channel, identical to how DeckLink consumers work. `has_synchronization_clock()` returns `true`.

2. **Dedicated write thread.** `send()` posts work to a queue and returns a `std::future` immediately. The write thread does the blocking. This allows the PortAudio consumer's future to be awaited in parallel with other consumers (e.g. DeckLink) rather than serialising them.

3. **Hardware-optimal callback size.** The stream is opened with `paFramesPerBufferUnspecified`, letting the driver choose the ideal callback granularity for the device.

4. **Average cadence for buffer sizing.** The ring buffer is sized using the average of the audio cadence array, not the minimum. This prevents cumulative drift on variable-cadence formats like NTSC 29.97fps (cadence pattern 1602, 1601, 1602, 1601, 1602).

5. **Condition variable drain notification.** The PA callback signals `drain_cv_` after every read, replacing a spin-wait loop and reducing CPU usage.

6. **Device hot-unplug detection.** The write thread checks `Pa_IsStreamActive()` before each write and logs a warning if the stream has stopped unexpectedly.

### Producer (Audio Capture)

The producer captures audio from an input device and delivers it as audio-only frames to the CasparCG mixer.

```
Hardware callback                  Channel tick
    │                                  │
    ▼                                  ▼
  stream_callback()              receive_impl()
    │                                  │
    ▼                                  ▼
  SPSC Ring Buffer ◄── write    read ──► get_frame()
  (lock-free)                          │
                                       ▼
                                  1×1 pixel frame
                                  + audio_data()
```

**Key design decisions:**

1. **Lock-free capture callback.** The PA callback writes directly into an SPSC ring buffer with no mutex. This is critical — audio callbacks run on a real-time thread where blocking can cause glitches or driver dropouts.

2. **Minimal video frame allocation.** Audio-only producers create a 1×1 transparent BGRA pixel instead of a full-resolution frame. On a 1080p channel this saves ~8 MB per frame of unnecessary allocation.

3. **5-second ring buffer.** The capture buffer holds up to 5 seconds of audio, preventing unbounded memory growth if the CasparCG mixer stalls momentarily.

### SPSC Ring Buffer

Both consumer and producer use `spsc_ring_buffer`, a single-producer single-consumer lock-free ring buffer for `int32_t` samples.

- Power-of-two capacity (auto-rounded, minimum 64 samples)
- Wrap-around via bitmask (`pos & mask_`)
- `alignas(64)` on atomic counters to prevent false sharing
- `memory_order_acquire`/`release` on counter loads/stores
- Bulk `memcpy` with two-chunk wrap handling

### Device Manager

`portaudio_device_manager` is a singleton that owns the PortAudio lifecycle (`Pa_Initialize` / `Pa_Terminate`) and provides device enumeration and matching.

**Host API priority** (auto mode): ASIO → WASAPI → system default.

**Device name matching** (two-pass):
1. Exact match (case-sensitive)
2. Substring match (case-insensitive)

If no match is found, the default device for the selected host API is used.

### LTC Integration

The LTC timecode module (`src/modules/ltc/`) uses PortAudio independently for its own mono float32 capture stream. It has a dedicated `Pa_Initialize`/`Pa_Terminate` cycle managed by `ltc::uninit()`, which runs before PortAudio's global `Pa_Terminate` in the module shutdown sequence.

## Configuration

### Consumer — XML (casparcg.config)

```xml
<channel>
  <video-mode>1080i5000</video-mode>
  <consumers>
    <portaudio>
      <device>Focusrite</device>
      <host-api>asio</host-api>
      <channels>8</channels>
      <buffer-size>4</buffer-size>
      <delay>0</delay>
    </portaudio>
  </consumers>
</channel>
```

| Element | Default | Description |
|---|---|---|
| `<device>` | *(empty = default device)* | Partial device name, case-insensitive substring match |
| `<host-api>` | `auto` | `auto`, `asio`, `wasapi`, `directsound` (alias: `ds`) |
| `<channels>` | `2` | Number of output channels (clamped to device maximum) |
| `<buffer-size>` | `4` | Ring buffer depth in video frames |
| `<delay>` | `0` | Pipeline delay compensation in video frames (pre-fills silence) |

### Consumer — AMCP

```
ADD [channel] PORTAUDIO [device] [DEVICE=name] [API=api] [CHANNELS=n] [BUFFER=n] [DELAY=n]
```

All parameters are optional. Examples:

```bash
# Default device, stereo, auto API
ADD 1 PORTAUDIO

# Named device with keyword params
ADD 1 PORTAUDIO DEVICE=Focusrite API=asio CHANNELS=8

# Positional device name (first non-keyword arg)
ADD 1 PORTAUDIO "Dante Virtual Soundcard" CHANNELS=16
```

### Producer — AMCP

```
PLAY [channel]-[layer] portaudio [device] [DEVICE=name] [API=api] [CHANNELS=n]
```

Examples:

```bash
# Default capture device, stereo
PLAY 1-10 portaudio

# Specific device
PLAY 1-10 portaudio DEVICE="Line In (Focusrite)" API=asio CHANNELS=2

# Positional device name
PLAY 1-10 portaudio "Microphone"
```

### Producer — XML (casparcg.config)

Producers use the AMCP command syntax inside a `<producer>` element:

```xml
<channel>
  <video-mode>1080i5000</video-mode>
  <producers>
    <producer id="10">portaudio DEVICE=Focusrite API=asio CHANNELS=2</producer>
  </producers>
</channel>
```

### Host API Values

| Value | API | Notes |
|---|---|---|
| `auto` | Auto-select | Tries ASIO first, then WASAPI, then system default |
| `asio` | ASIO | Requires ASIO SDK at build time + ASIO driver at runtime |
| `wasapi` | Windows Audio Session API | Default on most Windows systems, shared or exclusive mode |
| `directsound` / `ds` | DirectSound | Legacy, higher latency |

### ASIO Support

ASIO is optional at build time. Set `ASIOSDK_ROOT_DIR` to the Steinberg ASIO SDK path before configuring CMake:

```
cmake -DASIOSDK_ROOT_DIR=C:/ASIOSDK2.3.3 ../src
```

If not set, the module builds with WASAPI and DirectSound only.

## Operation Manual

### Scenario 1: Professional Audio Interface (ASIO)

**Use case:** Route CasparCG audio to a Focusrite, RME, or Dante interface with 8+ channels via ASIO for lowest latency.

```xml
<portaudio>
  <device>Focusrite</device>
  <host-api>asio</host-api>
  <channels>8</channels>
  <buffer-size>4</buffer-size>
</portaudio>
```

**Requirements:**
- ASIO driver installed for the device
- CasparCG built with `ASIOSDK_ROOT_DIR` set
- Device sample rate set to 48 kHz in the ASIO control panel

**Notes:**
- ASIO provides exclusive device access — no other application can use the device simultaneously.
- ASIO gives the lowest achievable latency and the most stable clock.

### Scenario 2: System Audio Output (WASAPI)

**Use case:** Output to a standard Windows audio device (headphones, USB speakers, HDMI audio) without ASIO drivers.

```xml
<portaudio>
  <host-api>wasapi</host-api>
  <channels>2</channels>
</portaudio>
```

Or via AMCP:
```bash
ADD 1 PORTAUDIO API=wasapi
```

**Notes:**
- WASAPI shared mode allows other applications to use the device simultaneously.
- Set the device to 48 kHz / 24-bit in Windows Sound Settings → Properties → Advanced to avoid resampling.

### Scenario 3: Live Audio Capture (Voice-over, Line In)

**Use case:** Bring live audio from a microphone or line-in into a CasparCG layer.

```bash
PLAY 1-10 portaudio DEVICE="Microphone" CHANNELS=2
```

Or preconfigured in casparcg.config:
```xml
<producers>
  <producer id="10">portaudio DEVICE=Microphone CHANNELS=2</producer>
</producers>
```

The audio appears on layer 10 and is mixed with other layers by the CasparCG mixer. Control its volume with:
```bash
MIXER 1-10 VOLUME 0.8
```

### Scenario 4: Virtual Audio Cable Routing

**Use case:** Capture audio from another application (media player, DAW, browser) via a virtual cable.

1. Install [VB-CABLE](https://vb-audio.com/Cable/), [Voicemeeter](https://vb-audio.com/Voicemeeter/), or similar virtual audio device.
2. Configure the source application to output to the virtual cable.
3. Capture in CasparCG:

```bash
PLAY 1-10 portaudio DEVICE="CABLE Output"
```

### Scenario 5: Dante / AES67 / Ravenna (IP-Based Audio)

**Use case:** Output to or capture from an IP audio network (Dante, AES67, Ravenna).

CasparCG does not implement any IP audio protocol directly. Instead, these protocols are handled by a **virtual soundcard driver** that runs on the same machine and presents the network audio streams as standard ASIO or WASAPI devices. The PortAudio module sees them like any other audio interface.

**Common virtual soundcard drivers:**

| Driver | Protocol | Cost | Notes |
|---|---|---|---|
| Dante Virtual Soundcard (DVS) | Dante + AES67 | ~$30 (one-time) | Most widely used in broadcast. ASIO mode recommended |
| Dante Via | Dante + AES67 | ~$50 (one-time) | Routes any application audio to Dante, more flexible |
| Merging RAVENNA ASIO | AES67 / Ravenna | Free | Works with generic AES67 streams |

**Network requirements:**
- Any GbE NIC works. Intel i210/i225/i226 chipsets are recommended for their hardware PTP (IEEE 1588) timestamping, which provides more stable clock synchronisation.
- A dedicated or VLAN-isolated network is recommended for production use to avoid jitter from other traffic.
- All devices must be on the same PTP clock domain. The virtual soundcard driver handles PTP participation automatically.

**Output (consumer):**
```xml
<portaudio>
  <device>Dante Virtual Soundcard</device>
  <host-api>asio</host-api>
  <channels>16</channels>
</portaudio>
```

**Input (producer):**
```bash
PLAY 1-10 portaudio DEVICE="Dante Virtual Soundcard" API=asio CHANNELS=8
```

**Setup steps:**
1. Install the virtual soundcard driver (e.g. DVS) and configure it for ASIO mode.
2. Open Dante Controller (free) and set up audio subscriptions between devices.
3. Set the sample rate to 48 kHz in both the virtual soundcard and Dante Controller.
4. Configure the PortAudio consumer or producer with the virtual soundcard's device name.

### Scenario 6: DeckLink Video + PortAudio Audio

**Use case:** Output video via DeckLink (with embedded audio disabled) and route audio separately to a dedicated audio interface.

```xml
<channel>
  <video-mode>1080i5000</video-mode>
  <consumers>
    <decklink>
      <device>1</device>
      <embedded-audio>false</embedded-audio>
    </decklink>
    <portaudio>
      <device>RME</device>
      <host-api>asio</host-api>
      <channels>8</channels>
    </portaudio>
  </consumers>
</channel>
```

Both consumers run as master clocks. The channel tick is paced by whichever consumer's `send()` future completes last (both block until their hardware accepts the frame). The DeckLink's genlock and the audio interface's crystal oscillator run independently — the ring buffer absorbs the minor drift between them.

### Scenario 7: LTC Timecode Input

The LTC module uses PortAudio independently for timecode capture. It does not conflict with the PortAudio consumer or producer — each opens its own PA stream.

Configure the LTC capture device in casparcg.config:
```xml
<ltc>
  <device>Line In</device>
</ltc>
```

## Channel Mapping

The consumer maps channels by index: output channel N receives source channel N. If the source has fewer channels than the output, excess output channels are zero-filled (silence).

| Source | Output | Result |
|---|---|---|
| 2ch stereo | 2ch device | L→1, R→2 |
| 2ch stereo | 8ch device | L→1, R→2, ch3-8 silence |
| 8ch source | 2ch device | ch1→1, ch2→2, ch3-8 discarded |
| 8ch source | 8ch device | 1:1 mapping |

For more complex routing (mono expansion, downmix, channel reordering), use FFmpeg audio filters at the source level or route through an external audio matrix before reaching CasparCG.

## Diagnostics

The consumer registers a diagnostics graph with these traces:

| Trace | Color | Description |
|---|---|---|
| `tick-time` | Blue | Time between successive `send()` completions, normalised to frame duration |
| `buffer-fill` | Green | Ring buffer fill ratio (0.0 = empty, 1.0 = full) |
| `underrun` | Red | PA callback requested more samples than available |
| `overflow` | Orange | Write thread could not fit all samples into the ring buffer |

Monitor state (available via OSC):

| Key | Type | Description |
|---|---|---|
| `buffer/fill` | int64 | Samples currently in the ring buffer |
| `buffer/underruns` | int64 | Cumulative underrun count |
| `buffer/overflows` | int64 | Cumulative overflow count |

## Best Practices

1. **Match sample rates.** Set the audio device to 48 kHz (CasparCG's native rate) in its driver or control panel. Mismatched rates cause resampling artefacts or drift.

2. **Use ASIO when available.** ASIO provides exclusive device access, lowest latency, and the most stable clock. WASAPI shared mode adds an extra mixing stage and slightly higher latency.

3. **Keep buffer-size at 4.** The default ring buffer depth of 4 video frames provides a good balance between latency (~80ms at 50fps) and resilience to scheduling jitter. Reduce to 2 for lower latency if the system is dedicated to CasparCG; increase to 6–8 if you see underrun warnings.

4. **Use delay compensation for external pipelines.** If your audio goes through an external processing chain (e.g. Dante network → DSP → amplifiers), set `<delay>` to the number of video frames of additional latency to keep audio and video in sync.

5. **Check logs on startup.** The device manager logs all detected host APIs and devices at initialisation. Use this to verify device names and channel counts before configuring.

6. **One ASIO device per host.** Most ASIO drivers only allow a single application to open the device. If CasparCG holds the ASIO device, DAWs or other applications cannot access it.

7. **Producer captures are audio-only.** The PortAudio producer creates minimal 1×1 transparent video frames. Layer it behind your video content — it will not affect the video output.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No audio output | Wrong device name | Check startup logs for available devices; use a substring that uniquely matches |
| No audio output | Wrong host API | Try `auto` or explicitly set `wasapi` if ASIO is not installed |
| Audio glitches / underruns | Buffer too small | Increase `<buffer-size>` to 6 or 8 |
| Audio glitches / underruns | Sample rate mismatch | Set device to 48 kHz in its control panel |
| Audio drift over time | Clock mismatch with video | The PortAudio consumer is a master clock — drift against DeckLink is absorbed by the ring buffer. If drift is severe (>1 frame), check that both devices are on the same clock domain or use genlock |
| "ASIO host API not found" | ASIO SDK not included in build | Rebuild with `-DASIOSDK_ROOT_DIR=path/to/sdk` |
| "ASIO host API not found" | No ASIO driver installed | Install hardware ASIO driver, ASIO4ALL, or FlexASIO |
| Device not found | Device connected after CasparCG started | Restart CasparCG — device enumeration happens at startup |
| Producer silence | Microphone permissions | Windows Settings → Privacy → Microphone → allow CasparCG |
| Producer silence | Wrong capture device | Try without `DEVICE=` to use the system default |
| "Audio device disconnected" warning | USB device unplugged | Reconnect device and restart the consumer (`REMOVE` then `ADD`) |