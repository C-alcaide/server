# System Audio Producer Module

## Overview
The **System Audio Producer** is a new module for CasparCG Server that allows capturing audio from external system devices (such as physical line-ins, USB microphones, Dante Virtual Soundcard, or virtual cables) and routing them into a CasparCG layer.

This enables workflows like:
- Integrating live voice-overs.
- Bringing in audio from external applications via virtual cables (VB-Audio, Loopback).
- Using Dante/AES67 audio streams as input sources.

## Changes Made

### 1. New Module Structure
Created `server/src/modules/system_audio/` containing:
- **`system_audio.h/cpp`**: Module entry points and registration logic.
- **`miniaudio.h`**: A single-header audio playback and capture library (no external dependencies required).
- **`producer/system_audio_producer.h/cpp`**: The core producer implementation.

### 2. Build System Integration
- Added `system_audio` to `server/src/modules/CMakeLists.txt`.
- Configured CMake to compile the module and link it with the server core.

## Inner Workings

### Audio Capture Strategy
The module uses `miniaudio` in **Capture Mode**. 
1.  **Push Model (Driver)**: The audio driver pushes samples (in `S32` format) to a high-priority callback function running on a dedicated audio thread.
2.  **Buffering**: Incoming samples are immediately locked and appended to a `std::vector<int32_t>` buffer. This acts as a thread-safe intermediate storage (ring-buffer style) to bridge the gap between the audio driver's push rate and CasparCG's frame pull rate.
3.  **Pull Model (CasparCG)**: When CasparCG's mixer requests a frame via `get_frame()`:
    - The producer calculates exactly how many samples are needed for the current frame duration (e.g., 960 samples for 1080i50 at 48kHz).
    - It retrieves that many samples from the buffer.
    - If insufficient data is available (underrun), it outputs silence or whatever, partial data is available, to prevent glitches.
    - The data is wrapped in a `caspar::array` and sent down the pipeline.

### Device Selection
The producer iterates through available system input devices. It performs a **substring match** on the device name provided in the command. If no device is specified, it defaults to the OS default capture device.

## configuration & Usage

The producer is invoked using the standard **AMCP `PLAY` command**.

### Syntax
```bash
PLAY [channel]-[layer] system_audio [device_name] [buffer_options]
```

### Parameters
- **`device_name`**: (Optional) The name of the audio input device. Partial names work (e.g., "Dante" matches "Dante Virtual Soundcard").
- **`DEVICE=name`**: Alternative syntax for specifying the device.

### Examples

#### 1. Play Default System Input
Uses the default recording device set in Windows Sound Settings.
```bash
PLAY 1-10 system_audio
```

#### 2. Play Specific Device (Simple)
Plays from a device named "Microphone (USB Audio Device)".
```bash
PLAY 1-10 system_audio "Microphone"
```

#### 3. Play Specific Device (Explicit)
Useful if the name contains spaces or complex characters.
```bash
PLAY 1-10 system_audio DEVICE="Dante Virtual Soundcard"
```

#### 4. Route to specific audio mapping
Mix the stereo input to channel 1 & 2.
```bash
PLAY 1-10 system_audio "Line In" MIX 0 1 0 1
```

## Testing & Verification

### Prerequisites
1.  **Virtual Audio Cable (Optional)**: Install [VB-CABLE](https://vb-audio.com/Cable/) or similar to easily route audio from a media player (Spotify/YouTube) into CasparCG for testing.
2.  **Dante (Optional)**: If testing networked audio, ensure Dante Controller sees the transmitter and the Receiver (CasparCG machine) is subscribed.

### Test Procedure
1.  **Start CasparCG Server**.
2.  **Open a Console** (Telnet/Putty to port 5250 or use the official client).
3.  **List Devices** (Check logs):
    - When `system_audio` initializes or fails to find a device, it logs available devices to the console/log file. Use this to find the exact name string.
    *(Note: You can force a log of devices by trying to play a non-existent device)*:
    ```bash
    PLAY 1-1 system_audio "NonExistentDevice"
    ```
4.  **Play Audio**:
    ```bash
    PLAY 1-1 system_audio
    ```
5.  **Verify Output**:
    - Watch the DeckLink/Screen output.
    - Check the `DIAG` window (if enabled) to see audio levels on Layer 1.

### Troubleshooting
- **No Audio**: Check Windows Privacy settings (Microphone access) to ensure CasparCG has permission to access audio devices.
- **Drift/Glitches**: If audio drifts over long periods, the internal sample rate of the device might not match CasparCG's channel sample rate (usually 48kHz). Ensure your system devices are set to **48000 Hz** in Windows Sound Control Panel -> Properties -> Advanced.
