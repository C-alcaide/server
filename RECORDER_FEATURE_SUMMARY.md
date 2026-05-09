# CasparCG Server Feature Updates

This document outlines recent feature additions to the CasparCG Server, explaining their purpose, inner workings, configuration, and usage.

## 1. System Audio Producer (`system_audio`)

### Overview
A new producer module that captures audio from external system devices (Line-in, USB mics, Dante Virtual Soundcard, VB-Cables) and routes it into CasparCG layers.

### Inner Workings
- **Engine**: Uses `miniaudio` (capture mode) to interface with the OS audio subsystem (WASAPI on Windows).
- **Architecture**:
    - **Push Thread**: High-priority audio thread receives samples from the hardware driver in S32 format.
    - **Ring Buffer**: Samples are stored in a thread-safe lock-protected buffer.
    - **Pull Thread**: When CasparCG requests a frame, the producer consumes the required number of samples from the buffer.
- **Underrun Handling**: If the driver is slower than the frame rate (clock drift or CPU load), the producer inserts silence to match the required sample count, preventing pipeline stalls.

### Configuration & Usage
Use the `PLAY` command with the `system_audio` producer.

**Syntax:**
```bash
PLAY [channel]-[layer] system_audio [device_name]
```

**Examples:**
```bash
# Play default OS recording device
PLAY 1-10 system_audio

# Play specific device (substring match)
PLAY 1-10 system_audio "Focusrite"
# OR
PLAY 1-10 system_audio DEVICE="Focusrite USB Audio"
```

## 2. LTC Timecode Input (`ltc`)

### Overview
A global module that decodes Linear Timecode (LTC) from an audio input and makes it available to the server (e.g., for timestamping, scheduling, or `OSC` output).

### Inner Workings
- **Engine**: Uses `miniaudio` for capture and `libltc` for decoding.
- **Data Flow**:
    - Captures audio from a designated audio input.
    - Decodes the audio stream in real-time to extract SMPTE timecode.
    - Updates a thread-safe global state (`LTCInput::instance`).
- **Validity**: Tracks signal validity. If the signal is lost, it can optionally fall back to internal clock or invalid state.

### Configuration
Configuration is primarily handled via `casparcg.config` (legacy method) or AMCP commands.

**AMCP Commands:**
- **`INFO LTC`**: Displays the current status, captured timecode, and device name.
- **`LTC LOAD [device_name]`**: Selects and initializes the audio device for LTC capture.

**Example Session:**
```bash
# Check status
INFO LTC
# Output: [ltc] No device selected

# Select device
LTC LOAD "Line In"

# Verify
INFO LTC
# Output: [ltc] 10:00:00:00 (Device: Line In)
```

## 3. DeckLink Input Sharing & Sync

### Overview
Enables multiple CasparCG channels or layers to utilize the **same** physical DeckLink input connector simultaneously. Previous versions would lock the device, preventing reuse.

### Inner Workings
- **Shared Input Manager**: A `decklink_producer` now requests access via a `DeckLinkInputManager`.
- **Reference Counting**: The first producer to request `DeckLink 1` initializes the hardware. Subsequent producers requesting `DeckLink 1` receive a reference to the existing stream.
- **Sample Distribution**: The hardware callback distributes the incoming video and audio frames to all registered subscribers (producers) for that device index.
- **Lifecycle**: The hardware device is only stopped and released when the last producer using it is stopped.

### Configuration & Usage
No special configuration is required. Simply route multiple layers to the same input decklink device.

**Constraints:**
- **Video Format**: The first producer to start the input determines the hardware video format.
- **Conversion**: Subsequent producers can consume the shared input even if their channel format differs, as the producer handles frame rate and resolution conversion internally (via FFmpeg filters).
- **Latency**: All layers receive the frame at the same time; processing delays on one channel do not affect the execution of another, as they copy the data.

**Example:**
```bash
# Layer 10 shows Camera 1
PLAY 1-10 decklink 1

# Layer 20 ALSO shows Camera 1 (perhaps with a different transform or effect)
PLAY 1-20 decklink 1
```

## 4. Sync Manager (Beta)

### Overview
A mechanism to synchronize the startup of multiple DeckLink producers to ensure they are phase-aligned, useful for multi-channel Key/Fill inputs or multi-camera setups.

### Inner Workings
- **Groups**: Producers can be assigned to a sync group.
- **Barrier**: When a producer in a group attempts to start, it waits (blocks) until `expected_peers` count is reached.
- **Release**: Once all peers are ready, they start the underlying DeckLink streams simultaneously.

### Usage
(Internal API mostly, but exposed via specific producer parameters if implemented in the future).
