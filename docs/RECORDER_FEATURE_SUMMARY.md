# CasparCG Server — Input & Capture Feature Reference

This document covers the input and capture modules added to CasparCG Server: System Audio capture, LTC timecode input, DeckLink input sharing, and DeckLink synchronized capture.

---

## 1. System Audio Producer (`system_audio`)

### Overview

The System Audio producer captures audio from any OS-visible recording device — Line-in jacks, USB audio interfaces, Dante Virtual Soundcard, VB-Cable virtual devices, etc. — and routes it into a CasparCG layer. This enables mixing external audio sources alongside video playback without requiring a DeckLink card.

### Technical Architecture

```
┌──────────────────┐     callback      ┌────────────────────┐    receive_impl    ┌──────────────┐
│  Audio Hardware   │ ──────────────▶  │  miniaudio capture  │ ──────────────▶   │  CasparCG    │
│  (WASAPI/ALSA)   │   S32 samples    │  data_callback()    │   per-frame pull  │  Pipeline    │
└──────────────────┘                   │  ┌──────────────┐   │                   └──────────────┘
                                       │  │  Ring Buffer  │   │
                                       │  │ (mutex-locked)│   │
                                       │  └──────────────┘   │
                                       └────────────────────┘
```

*   **Audio engine:** [miniaudio](https://miniaud.io) (MIT / Unlicense) in capture-only mode. Uses WASAPI on Windows, ALSA/PulseAudio on Linux.
*   **Sample format:** 32-bit signed integer (S32), matching CasparCG's native audio format. No sample-rate conversion is performed — miniaudio is configured to match the channel's `audio_sample_rate` directly.
*   **Channel count:** Matches the CasparCG channel's `audio_channels` setting (typically 2).
*   **Push → Pull bridge:**
    *   **Push side:** miniaudio's high-priority audio thread fires `data_callback()` whenever the hardware delivers a buffer. Samples are appended to a `std::vector<int32_t>` under a mutex lock.
    *   **Pull side:** On each `receive_impl()` call (once per video frame), the producer calculates the required sample count for the frame duration and drains that many samples from the buffer.
*   **Underrun handling:** If the ring buffer contains fewer samples than needed (clock drift, CPU load, late device start), the producer copies what is available and pads the remainder with silence. No pipeline stalls or frame drops occur.
*   **Video frame:** The producer emits a transparent BGRA frame (black/silent fill) — only the audio payload is meaningful. Layer it beneath or above video producers using standard CasparCG mixer commands.
*   **Device selection:** Device names are matched by substring against the OS capture device list. If no match is found, the default OS recording device is used (with a log warning).

### Usage

```
PLAY <channel>-<layer> system_audio [<device_name>]
PLAY <channel>-<layer> system_audio DEVICE="<device_name>"
```

### Examples

```
# Use the default OS recording device
PLAY 1-10 system_audio

# Substring match on device name
PLAY 1-10 system_audio "Focusrite"

# Explicit DEVICE parameter (useful when names contain spaces)
PLAY 1-10 system_audio DEVICE="Focusrite USB Audio"

# Dante Virtual Soundcard
PLAY 1-10 system_audio "Dante"

# VB-Cable virtual input
PLAY 1-10 system_audio "CABLE Output"
```

### Best Practices

*   Run `system_audio` on its own dedicated layer so it can be independently stopped, started, or volume-adjusted via `MIXER VOLUME`.
*   If you experience periodic underruns, ensure the audio device's sample rate matches the channel configuration (default 48 kHz).
*   For multi-channel audio capture (e.g., 8-channel Dante), set the channel's `audio-channels` in `casparcg.config` to match.

---

## 2. LTC Timecode Input (`ltc`)

### Overview

A global module that decodes SMPTE Linear Timecode (LTC) from an audio input in real time. The decoded timecode is available server-wide for timestamping, scheduling, OSC output, or integration with external automation systems.

> **Platform support**: The LTC module runs on both **Windows** and **Linux**. Audio capture uses PortAudio (WASAPI/DirectSound on Windows, ALSA/JACK on Linux). The libltc decoder library is platform-independent.

### Technical Architecture

```
┌──────────────────┐     callback      ┌────────────────────┐     decode      ┌─────────────────┐
│  Audio Device     │ ──────────────▶  │  PortAudio capture  │ ────────────▶  │  libltc decoder  │
│  (Line In, etc.)  │   f32 mono       │  48 kHz, 1 channel  │               │  LTCFrameExt     │
└──────────────────┘                   └────────────────────┘               └────────┬────────┘
                                                                                     │
                                                                              ltc_frame_to_time()
                                                                                     │
                                                                                     ▼
                                                                            ┌─────────────────┐
                                                                            │  SMPTETimecode   │
                                                                            │  (thread-safe    │
                                                                            │   global state)  │
                                                                            └─────────────────┘
```

*   **Audio engine:** [PortAudio](http://www.portaudio.com/) v19.7 captures audio at 48 kHz, mono, float32 format. On Windows it uses WASAPI/DirectSound; on Linux it uses ALSA or JACK.
*   **Decoder:** [libltc](https://github.com/x42/libltc) v1.3.2 (LGPL-2.1+). The `stream_callback()` feeds raw float samples into `ltc_decoder_write_float()`. Decoded frames are read via `ltc_decoder_read()` in the same callback, keeping latency minimal.
*   **Global singleton:** `LTCInput::instance()` provides thread-safe access to the latest timecode from anywhere in the server. Internally uses atomic double-buffering for lock-free delivery from the audio callback thread.
*   **Signal validity:**
    *   A `valid_signal` flag is set whenever a timecode frame is successfully decoded.
    *   If no valid frame arrives within **1 second**, the module considers the signal lost.
    *   On signal loss, `get_current_timecode_string()` falls back to the **system clock** (local time formatted as `HH:MM:SS:00`).
    *   `is_valid()` and `is_using_system_clock()` expose the current state.
*   **Auto-start:** The module initializes on server startup via `init()`. If a device is configured in `casparcg.config` under `<configuration><ltc><device>`, it is used automatically.

### Configuration

**casparcg.config:**
```xml
<configuration>
  <ltc>
    <device>Line In</device>
  </ltc>
</configuration>
```

### AMCP Commands

| Command | Description |
| :--- | :--- |
| `INFO LTC` | Display current timecode, device name, and signal status. |
| `LTC LOAD "<device_name>"` | Select and start capturing from the named audio device. |

### Example Session

```
# Check current status
INFO LTC
# → [ltc] No device selected

# Select a device
LTC LOAD "Line In"

# Verify timecode is being decoded
INFO LTC
# → [ltc] 10:00:00:00 (Device: Line In) [Valid]
```

### Available API (Internal)

| Method | Return | Description |
| :--- | :--- | :--- |
| `get_current_timecode_string()` | `std::string` | `"HH:MM:SS:FF"` from LTC or system clock fallback. |
| `get_current_frame_number(fps)` | `uint32_t` | Absolute frame count since midnight at the given FPS. |
| `is_valid()` | `bool` | `true` if LTC signal received within the last second. |
| `is_using_system_clock()` | `bool` | `true` when fallen back to system time. |
| `get_capture_devices()` | `vector<string>` | Lists all OS capture devices. |
| `set_capture_device(name)` | `bool` | Stops current device, starts the named one. |

---

## 3. DeckLink Input Sharing

### Overview

Multiple CasparCG producers can now use the **same physical DeckLink input** simultaneously. In previous versions, the first producer to open an input would lock the device exclusively. With input sharing, any number of layers or channels can consume the same input feed — useful for PiP layouts, multi-layer effects, or routing one camera to several output channels.

### Technical Architecture

```
                                ┌───────────────────────┐
                                │  SharedDeckLinkInput   │
                                │  (singleton per device)│
┌────────────────┐   hardware   │                       │   fan-out    ┌──────────────────┐
│  DeckLink Card │ ──callback──▶│  VideoInputFrameArrived│─────────────▶│ Producer A (1-10)│
│  Input #1      │              │  VideoInputFormatChanged│────────────▶│ Producer B (1-20)│
└────────────────┘              │                       │─────────────▶│ Producer C (2-10)│
                                └───────────────────────┘              └──────────────────┘
                                     ▲
                                     │ managed by
                              ┌──────┴──────────────┐
                              │ DeckLinkInputManager │
                              │ (global singleton)   │
                              └─────────────────────┘
```

*   **`DeckLinkInputManager`:** Global singleton holding a `map<int, weak_ptr<SharedDeckLinkInput>>` keyed by device index. When a producer requests device `N`, it either gets the existing shared instance or creates a new one.
*   **`SharedDeckLinkInput`:** Wraps the BMD `IDeckLinkInput` interface. Implements `IDeckLinkInputCallback` and fans out every `VideoInputFrameArrived` and `VideoInputFormatChanged` event to all registered listener producers.
*   **Reference counting:**
    *   `enable_video_input()` / `disable_video_input()` track an `enable_ref_count_`. The hardware is initialized on the first call and released only when the last producer disconnects.
    *   `start_streams()` / `stop_streams()` track a `start_ref_count_`. Streams run as long as at least one producer needs them.
    *   The `SharedDeckLinkInput` itself is held by `shared_ptr` — when all producers release their reference, the manager's `weak_ptr` expires and the instance is destroyed.
*   **Format detection:** Only the first producer's `EnableVideoInput` call reaches the hardware. Subsequent producers joining an already-active input receive frames at whatever format the hardware is currently running. They call `get_current_display_mode()` to discover the active format and rebuild their FFmpeg filters accordingly.
*   **Format change handling:** When the BMD SDK fires `VideoInputFormatChanged`, the shared input executes the BMD-recommended sequence: `PauseStreams → EnableVideoInput → FlushStreams → notify listeners → StartStreams`. Listeners are notified **before** `StartStreams` so they can update their internal mode and filters before new frames arrive.
*   **No data copying penalty:** The BMD SDK delivers the same `IDeckLinkVideoInputFrame` pointer to all listeners. Each producer copies the data into its own CasparCG frame independently.

### Usage

No special commands or configuration required — just play the same DeckLink input on multiple layers:

```
# Layer 10: Camera 1 — full frame
PLAY 1-10 decklink 1

# Layer 20: Camera 1 again — with a different transform
PLAY 1-20 decklink 1
MIXER 1-20 FILL 0.6 0.0 0.4 0.4

# Different channel also using Camera 1
PLAY 2-10 decklink 1
```

### Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `DEVICE` or positional `<index>` | — | DeckLink device index (1-based). |
| `FORMAT` | (auto-detect) | Force a specific input format (e.g., `1080i5000`). Disables format detection. |
| `FREEZE_ON_LOST` | off | Freeze last frame instead of showing black on signal loss. |
| `10BIT` | off | Capture in 10-bit YUV (`v210`) instead of 8-bit. |
| `VF` | (none) | FFmpeg video filter string. |
| `AF` | (none) | FFmpeg audio filter string. |
| `FILTER` | (none) | Legacy alias for `VF`. |
| `LENGTH` | ∞ | Limit playback to N frames. |
| `SYNC_GROUP` | 0 | Synchronized capture group ID (see Section 4). |
| `SYNC_PEERS` | 1 | Expected number of producers in the sync group. |

### Constraints

*   The **first producer** to open a device determines the hardware video format. If auto-detection changes the format later, all producers are notified and must rebuild their filters.
*   Producers on channels with **different output formats** (e.g., 1080p50 channel consuming a 1080i50 input) work correctly — FFmpeg deinterlacing and frame-rate conversion filters are applied per-producer automatically.

---

## 4. DeckLink Synchronized Capture (`DeckLinkSyncManager`)

### Overview

The Sync Manager ensures that multiple DeckLink inputs start capturing at precisely the same moment, achieving phase-aligned frame delivery. This is essential for multi-camera setups, Key/Fill pairs, or any workflow where frame-accurate synchronization between inputs is required.

### Technical Architecture

```
┌─────────────────────────────────────────────────┐
│               DeckLinkSyncManager               │
│  (global singleton, polling monitor thread)      │
│                                                  │
│  groups_: map<group_id, Group>                   │
│     Group {                                      │
│       producers: [decklink_producer*, ...]       │
│       expected_peers: N                          │
│       started: bool                              │
│     }                                            │
│                                                  │
│  Monitor loop (50 ms poll):                      │
│    For each group:                               │
│      1. Wait for expected_peers to register      │
│      2. Poll check_signal_locked() on all peers  │
│      3. When ALL signals locked → start leader   │
│         (BMD driver starts entire capture group)  │
└─────────────────────────────────────────────────┘
```

*   **Groups:** Each producer can declare a `SYNC_GROUP` ID and the number of `SYNC_PEERS` expected. Producers with the same group ID are collected into a `Group` struct.
*   **Registration:** When a producer with `SYNC_GROUP > 0` is created, it calls `DeckLinkSyncManager::register_producer()` instead of immediately starting streams. The producer is added to its group.
*   **Monitor thread:** Once a group reaches its expected peer count, a background polling thread starts (50 ms interval). On each tick, it checks `check_signal_locked()` on every producer in every pending group — this queries `bmdDeckLinkStatusVideoInputSignalLocked` via the BMD SDK.
*   **Synchronized start:** When all producers in a group report a locked signal, `start_streams()` is called on the **first producer only** (the leader). The BMD driver, configured with `bmdVideoInputSynchronizeToCaptureGroup` and matching `bmdDeckLinkConfigCaptureGroup` IDs, starts all devices in the group simultaneously at the hardware level.
*   **Lifecycle:** When a producer is destroyed, `unregister_producer()` removes it from the group. The monitor thread continues for other groups.

### Configuration

Sync groups can be set via AMCP parameters or `casparcg.config`.

**AMCP parameters:**
```
PLAY 1-10 decklink 1 SYNC_GROUP 1 SYNC_PEERS 2
PLAY 1-20 decklink 2 SYNC_GROUP 1 SYNC_PEERS 2
```

**casparcg.config:**
```xml
<configuration>
  <decklink-sync>
    <device-1>
      <group>1</group>
      <peers>2</peers>
    </device-1>
    <device-2>
      <group>1</group>
      <peers>2</peers>
    </device-2>
  </decklink-sync>
</configuration>
```

When configured via `casparcg.config`, the sync parameters are automatically applied whenever the device is used — no AMCP parameters needed:

```
PLAY 1-10 decklink 1
PLAY 1-20 decklink 2
# Both automatically join sync group 1 and wait for each other
```

### Example: Key/Fill Synchronized Input

```
# DeckLink 1 = Fill, DeckLink 2 = Key
# Both must start at the exact same frame
PLAY 1-10 decklink 1 SYNC_GROUP 1 SYNC_PEERS 2
PLAY 1-20 decklink 2 SYNC_GROUP 1 SYNC_PEERS 2

# The second command triggers the sync manager.
# Once both signals are locked, capture starts simultaneously.
```

### Best Practices

*   Ensure all DeckLink cards in a sync group share a common genlock/reference signal. Without hardware sync, the BMD driver cannot guarantee phase alignment.
*   Set `SYNC_PEERS` accurately — if the count is wrong, the group will either start prematurely or wait indefinitely.
*   For permanent multi-camera rigs, prefer the `casparcg.config` method to avoid repeating AMCP parameters on every session.
