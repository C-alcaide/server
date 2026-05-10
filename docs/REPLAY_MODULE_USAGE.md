# CasparCG Replay Module Usage Guide

This document outlines the usage, commands, and features of the Replay module for CasparCG Server. This module enables high-performance recording and playback of `.mav` files using the Replay codec, with robust support for "growing file" playback (Play-while-Recording) and background exporting.

## 1. Recording (Consumer)

To start recording a channel to a Replay file, use the `ADD` command with the `REPLAY` consumer type.

### Syntax
```bash
ADD <channel> REPLAY <filename>[?<options>]
```
*Note: The filename is relative to the `media/` folder defined in your configuration.*

### Recording Options
You can configure the recording behavior using query parameters appended to the filename:
*   `max_duration=<HH:MM>`: Sets the maximum duration of the circular buffer. Older segments are automatically deleted to maintain this duration. Format must be `Hours:Minutes` (e.g., `01:00` for 1 hour).
*   `segment=<seconds>`: Sets the duration of each individual segment file (default is usually 60 seconds).

### Quality Profiles
You can specify the recording quality by adding a 3rd argument to the `ADD` command or using the default.
Supported profiles: `LQ`, `SQ` (default), `HQ`, `OMT_LQ`, `OMT_SQ`, `OMT_HQ`.

The `OMT` (Open Media Transport) profiles offer updated tuning for modern workflows, often prioritizing full DC precision at optimized bitrates.

**1080p Specifications (at 60fps):**

| Profile | Target Bitrate | DC Precision | Notes |
| :--- | :--- | :--- | :--- |
| **LQ** | 86 Mbps | Low | Legacy Low Quality |
| **OMT_LQ** | 86 Mbps | Low | Same bitrate, adaptable quality |
| **SQ** | **130 Mbps** | **Low** | Default. Good for general replay. |
| **OMT_SQ** | **200 Mbps** | **High** | **Higher Quality than SQ.** Full DC precision. |
| **HQ** | 260 Mbps | High | Mastering quality. |
| **OMT_HQ** | 260 Mbps | High | Same bitrate, adaptable quality. |

**2160p (4K) Specifications (at 60fps):**

| Profile | Target Bitrate | DC Precision | Notes |
| :--- | :--- | :--- | :--- |
| **SQ** | 400 Mbps | Low | Good balance |
| **OMT_SQ** | 300 Mbps | Low | Tuned for bandwidth efficiency |
| **HQ** | 800 Mbps | High | High bandwidth requirement |
| **OMT_HQ** | 600 Mbps | High | Optimized high quality |

*Note: Bitrates scale linearly with frame rate (e.g., 1080p25/30 will be half the 60fps values).*

### Examples

**Standard Recording (SQ Quality):**
```bash
ADD 1 REPLAY "my_recording"
```

**High Quality Recording:**
```bash
ADD 1 REPLAY "my_high_quality_recording" HQ
```

**Circular Buffer Recording (Time-Shift):**
Record the last hour (3600 seconds) in 1-minute segments:
This generates a simple linear recording.

**Circular Buffer Recording (Time-Shift):**
Record the last hour (01:00) in 1-minute segments:
```bash
ADD 1 REPLAY "my_channel_buffer?max_duration=01:00&segment=60"
```
This creates a folder `media/my_channel_buffer/` containing:
*   `my_channel_buffer.mav.000`, `my_channel_buffer.mav.001`, ...
*   `my_channel_buffer.idx.000`, `my_channel_buffer.idx.001`, ...

**Stop Recording:**
```bash
REMOVE 1 REPLAY "my_recording"
```

## 2. Playback (Producer)

To play a Replay file, use the standard `PLAY` command. The producer automatically detects if the file is segmented or growing and updates its duration dynamically.

### Syntax
```bash
PLAY <channel>-<layer> <filename>
```

### Examples
```bash
# Play a file (or folder of segments)
PLAY 1-10 "my_recording"

# Play a live recording buffer
PLAY 1-10 "my_channel_buffer"
```

### Growing File Support
You can `PLAY` a file immediately after (or even before) the recording has finished. If the file is growing, the producer will continue to play new frames as they appear.

### In/Out/Loop Parameters
The `PLAY`, `LOAD`, and `LOADBG` commands support `IN`, `OUT`, `SEEK`, `LENGTH` and `LOOP` parameters directly.

**Syntax:**
```bash
PLAY <channel>-<layer> <filename> [LOOP] [SEEK <val>] [IN <val>] [OUT <val>] [LENGTH <val>]
```
**Examples:**
```bash
# Play file looping from frame 100 to 500
PLAY 1-1 "my_recording" LOOP IN 100 OUT 500

# Play file starting at specific timestamp with duration of 10 seconds (at 25fps = 250 frames)
PLAY 1-1 "my_recording" IN 2026-03-03-12-00-00-00 LENGTH 250

# Play file immediately seeking to the "Live" edge
PLAY 1-1 "growing_file" SEEK LIVE
```

## 3. Timestamp Format

The Replay module supports a high-precision timestamp format for all time-based commands (`SEEK`, `IN`, `OUT`, `EXPORT`).

**Format:** `yyyy-mm-dd-hh-mm-ss-ff`
*   `yyyy`: Year
*   `mm`: Month
*   `dd`: Day
*   `hh`: Hour (24h)
*   `mm`: Minute
*   `ss`: Second
*   `ff`: Frame number (within that second)

*Alternative ISO format:* `2026-03-03T12:00:00` is also supported.

**Frame Integers:**
You can also use simple integer frame numbers (e.g., `0`, `100`, `5000`) instead of timestamps.

## 4. Commands

The Replay producer supports standard AMCP commands via `CALL`.

### SEEK
Jumps to a specific frame or timestamp.

```bash
# Jump to frame 1000
CALL 1-10 SEEK 1000

# Jump to a specific time
CALL 1-10 SEEK 2026-03-03-12-00-00-00

# Jump to the "Live" edge (latest recorded frame)
CALL 1-10 SEEK LIVE
```

### IN / OUT
Sets the playback boundaries (in-point and out-point). Playback will start at `IN` and stop (or loop) at `OUT`.

```bash
# Set IN point to a timestamp
CALL 1-10 IN 2026-03-03-10-00-00-00

# Set OUT point to frame 5000
CALL 1-10 OUT 5000
```

### EXPORT
Exports clips to an external file (using FFmpeg) in the background. Supports concatenation of multiple source clips.

**Syntax:**
```bash
CALL <channel>-<layer> EXPORT <output_filename> [<input_filename> <in> <out>]...
```

**Examples:**

*   **Simple Export:** Export frames 100-200 of the currently playing file to `out.mp4`.
    ```bash
    CALL 1-10 EXPORT "out.mp4" 100 200
    ```

*   **Time-Based Export:** Export a specific time range.
    ```bash
    CALL 1-10 EXPORT "highlight.mp4" 2026-03-03-12-00-00-00 2026-03-03-12-05-00-00
    ```

*   **Explicit Input File:** Export from a specific file (not the one playing).
    ```bash
    CALL 1-10 EXPORT "out.mp4" "other_recording.mav" 0 500
    ```

*   **Concatenation:** Combine two clips into one file.
    ```bash
    # usage: output input1 in out input2 in out
    CALL 1-10 EXPORT "combined.mp4" "part1.mav" 0 100 "part2.mav" 50 150
    ```

*   **Mixed Concatenation:** Export 50 frames from current file, then append a clip from another file.
    ```bash
    CALL 1-10 EXPORT "promo.mp4" 0 50 "intro.mav" 0 100
    ```

### SPEED
Adjusts playback speed.
```bash
CALL 1-10 SPEED 2.0  # 200% speed
CALL 1-10 SPEED 0.5  # 50% speed
```

### LOOP
Toggle looping mode.
```bash
CALL 1-10 LOOP 1     # Enable
CALL 1-10 LOOP 0     # Disable
```

## 5. OSC Data
The Replay producer outputs real-time data via OSC.

**Addresses:**
*   `/channel/1/stage/layer/1/file/time`: Current time in seconds.
*   `/channel/1/stage/layer/1/file/frame`: Current frame number.
*   `/channel/1/stage/layer/1/replay/timestamp`: Current timestamp (microseconds).
*   `/channel/1/stage/layer/1/replay/timestamp_formatted`: Current timestamp as string (`yyyy-mm-dd-hh-mm-ss-ff`).

## 6. Diagnostics

The Replay module registers detailed graphs in the CasparCG Diagnostics window.

*   **Producer Graph**: Shows read bitrate, current FPS, and buffer health.
*   **Consumer Graph**: Shows write buffer usage and encoding performance.
