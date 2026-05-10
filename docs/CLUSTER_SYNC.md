# Cluster Synchronization Module

Multi-machine synchronized playout for CasparCG using IEEE 1588 PTP clock synchronization, global frame numbering, frame-accurate command scheduling, TCP command relay with virtual channel mapping, and a content sync watchdog for automatic drift detection and correction.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [PTP Clock](#ptp-clock)
- [Frame Clock](#frame-clock)
- [Command Scheduler](#command-scheduler)
- [Command Relay](#command-relay)
- [Virtual Channel Map](#virtual-channel-map)
- [Content Sync Watchdog](#content-sync-watchdog)
- [Producer Lifecycle Handling](#producer-lifecycle-handling)
- [AMCP Commands](#amcp-commands)
- [Configuration Reference](#configuration-reference)
- [Configuration Examples](#configuration-examples)
- [Timing Analysis](#timing-analysis)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The cluster module enables multiple CasparCG instances to play content in frame-accurate synchronization. One instance acts as the **master** (timing authority and command router), while the others act as **clients** (timing followers and command receivers).

```
                              ┌─────────────────────────┐
                              │      PTP Multicast       │
                              │   224.0.1.129:319/320    │
                              └───────┬─────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
    ┌──────┴──────┐           ┌───────┴──────┐           ┌──────┴───────┐
    │   MASTER    │           │   CLIENT A   │           │   CLIENT B   │
    │             │           │              │           │              │
    │ PTP Master  │──Sync──▸  │ PTP Client   │           │ PTP Client   │
    │ Frame Clock │           │ Frame Clock  │           │ Frame Clock  │
    │ Scheduler   │           │ Scheduler    │           │ Scheduler    │
    │ Relay  ─────┼───TCP───▸ │ Relay        │           │ Relay        │
    │ Watchdog    │           │ Watchdog     │           │ Watchdog     │
    │ Channel Map │    ┌──────┼──TCP─────────┼───────────┤              │
    └─────────────┘    │      └──────────────┘           └──────────────┘
                       │
                  Virtual Channels
                  1 → Client A, physical 1
                  2 → Client A, physical 2
                  3 → Client B, physical 1
```

### Subsystem Pipeline

```
┌───────────┐     ┌─────────────┐     ┌───────────────────┐
│ PTP Clock │────▸│ Frame Clock │────▸│ Command Scheduler │
│ (ns time) │     │ (frame #)   │     │ (frame-accurate)  │
└───────────┘     └──────┬──────┘     └───────────────────┘
                         │
                         ▼
                  ┌──────────────┐     ┌─────────────────┐
                  │Content Sync  │     │ Command Relay   │
                  │  Watchdog    │     │ (TCP transport)  │
                  └──────────────┘     └─────────────────┘
```

1. **PTP Clock** establishes a common nanosecond-resolution time base across all nodes
2. **Frame Clock** converts PTP nanoseconds to a monotonic global frame number using integer arithmetic
3. **Command Scheduler** executes AMCP commands at their target frame with sub-frame precision
4. **Command Relay** transports timestamped commands from master to clients over TCP
5. **Content Sync Watchdog** continuously monitors producer frame positions and corrects drift

---

## PTP Clock

### Protocol Implementation

The PTP clock implements a subset of IEEE 1588-2008 (PTPv2) using two-step synchronization over UDP multicast:

| Port | Protocol | Messages |
|------|----------|----------|
| 319 (event) | UDP multicast | Sync, Delay_Req |
| 320 (general) | UDP multicast | Follow_Up, Delay_Resp, Announce |

All messages use the standard PTP wire format with `#pragma pack(push, 1)` for correct byte alignment.

### Clock Modes

| Mode | Behavior |
|------|----------|
| `master` | Authoritative time source. Sends Sync + Follow_Up at configured interval. Responds to Delay_Req from clients. Sends Announce every 2 seconds. `now_ns()` returns `system_clock_ns()` directly. |
| `client` | Synchronizes to master. Receives Sync/Follow_Up for offset calculation, sends Delay_Req for path delay measurement. `now_ns()` returns `system_clock_ns() + offset_ns`. |
| `external` | Same as client behavior, but intended for locking to an external PTP grandmaster on the network (not the CasparCG master). |

### Two-Step Synchronization

The master uses **two-step** mode (TWO_STEP flag = `0x0200`), which provides the most precise timestamps:

```
Master                                              Client
  │                                                    │
  ├──── Sync (seq=N) ────────────────────────────────▸ │  T2 = recv timestamp
  │                                                    │
  │  T1 = precise timestamp taken after Sync is sent   │
  │                                                    │
  ├──── Follow_Up (seq=N, T1) ───────────────────────▸ │
  │                                                    │
  │                                   offset = T1 - T2 + delay
  │                                                    │
  │ ◂──────────────────────────── Delay_Req (seq=M) ───┤  T3 = send timestamp
  │                                                    │
  │  T4 = recv timestamp                               │
  │                                                    │
  ├──── Delay_Resp (seq=M, T4) ──────────────────────▸ │
  │                                                    │
  │                                   delay = (T4 - T3) / 2
```

**Timestamps**:
- `T1`: Master's precise send timestamp (from Follow_Up, captured after the Sync packet hits the wire)
- `T2`: Client's local receive timestamp of the Sync packet
- `T3`: Client's local send timestamp of the Delay_Req
- `T4`: Master's receive timestamp of the Delay_Req (from Delay_Resp)

**Calculations**:
- Mean path delay: $\text{delay} = \frac{T4 - T3}{2}$
- Clock offset: $\text{offset} = T1 - T2 + \text{delay}$

### Offset Filter

Raw offset measurements are noisy due to network jitter, OS scheduling, and timestamping imprecision. The module applies an **exponential moving average (EMA)** filter:

$$\text{filtered}_{n} = \alpha \cdot \text{measured}_{n} + (1 - \alpha) \cdot \text{filtered}_{n-1}$$

With $\alpha = 0.125$ (matching the classical NTP/PTP filter constant). This provides:
- **Convergence time**: ~8 measurements to reach 63% of a step change (~1 second at default 125ms interval)
- **Noise rejection**: 7× attenuation of white noise at the Nyquist frequency
- **Step response**: Smooth tracking of slow clock drift without overshooting

### Lock State Machine

```
                  ┌───────────────┐
         start ──▸│ initializing  │
                  └───────┬───────┘
                          │ first measurement
                          ▼
                  ┌───────────────┐
        ┌────────│    locked      │◂───────┐
        │         └───────┬───────┘        │
        │                 │                │
        │  |offset| > 10ms│    |offset| < 1ms
        │                 ▼                │
        │         ┌───────────────┐        │
        └────────▸│ free_running  │────────┘
                  └───────────────┘
```

| Transition | Threshold | Description |
|-----------|-----------|-------------|
| initializing → locked | First measurement with \|offset\| < 1ms | Clock has initial sync |
| locked → free_running | \|offset\| > 10ms | Lost synchronization (network issue, master restart) |
| free_running → locked | \|offset\| < 1ms | Resynchronized to master |

The hysteresis (1ms lock / 10ms unlock) prevents oscillation when the offset is near the threshold.

### Message Intervals

| Message | Default Interval | Purpose |
|---------|-----------------|---------|
| Sync + Follow_Up | 125ms (`sync-interval-ms`) | Offset measurement |
| Announce | 2000ms (fixed) | Master presence / BMCA |
| Delay_Req | 1000ms (fixed) | Path delay measurement |

The socket receive timeout is 50ms, providing a non-blocking feel while keeping the thread responsive to the `running_` flag.

### Clock Identity

Each PTP clock generates a random 8-byte clock identity at construction using `std::random_device`. This identity is embedded in all PTP messages and used for Delay_Req/Delay_Resp correlation. In a production deployment with multiple clusters on the same network, the `ptp-domain` parameter should differ between clusters to provide traffic isolation.

---

## Frame Clock

### Purpose

The frame clock converts PTP nanosecond timestamps into a **monotonically increasing global frame number** that is identical on all cluster nodes (provided their PTP clocks are synchronized). This frame number is the universal reference for scheduling commands and measuring content drift.

### Frame Number Formula

$$\text{frame} = \left\lfloor \frac{(\text{ptp\_ns} - \text{epoch\_origin\_ns}) \times \text{fps\_num}}{\text{fps\_den} \times 10^9} \right\rfloor$$

And the inverse:

$$\text{ptp\_ns} = \text{epoch\_origin\_ns} + \frac{\text{frame} \times \text{fps\_den} \times 10^9}{\text{fps\_num}}$$

### Integer Arithmetic

All frame calculations use **64-bit integer arithmetic** exclusively — no floating-point operations. This eliminates accumulated rounding errors that would cause frame numbers to diverge between nodes over time.

For a 50fps channel (`fps_num=50`, `fps_den=1`):
- Frame period = $\frac{1 \times 10^9}{50} = 20{,}000{,}000$ ns = 20ms
- At $2^{63}$ nanoseconds from epoch, the frame number reaches ~$4.6 \times 10^{12}$ — sufficient for over 2,900 years of continuous operation

For fractional framerates like 29.97fps (`fps_num=30000`, `fps_den=1001`):
- Frame period = $\frac{1001 \times 10^9}{30000} = 33{,}366{,}667$ ns ≈ 33.37ms
- The integer division `(elapsed * 30000) / (1001 * 1'000'000'000)` produces exact frame boundaries with no accumulated error

### Epoch Origin

The `epoch-origin` parameter defines the zero point for frame numbering. All nodes in a cluster **must** use the same epoch origin. It is specified as an ISO-8601 datetime string and parsed to nanoseconds since Unix epoch:

```xml
<epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
```

Choosing a recent date keeps frame numbers small and human-readable in logs. Using the Unix epoch (or omitting the parameter) produces very large frame numbers but is functionally equivalent.

### API

| Method | Description |
|--------|-------------|
| `current_frame()` | Returns the global frame number at the current PTP time |
| `frame_to_ptp_ns(frame)` | Returns the PTP nanosecond timestamp for the start of `frame` |
| `ptp_ns_to_frame(ptp_ns)` | Returns the frame number at `ptp_ns` (floor) |
| `ns_until_frame(target)` | Returns nanoseconds until the start of `target` frame |
| `set_framerate(num, den)` | Updates the framerate (e.g., on channel format change) |

---

## Command Scheduler

### Purpose

The command scheduler executes AMCP commands at precise global frame boundaries. Commands are placed in a priority queue keyed by target frame number and dispatched when the frame clock reaches or passes that frame.

### Scheduling Modes

| Mode | Usage |
|------|-------|
| `schedule(target_frame, command, priority)` | Execute at an explicit global frame |
| `schedule_now(command, sync_margin)` | Execute at `current_frame + sync_margin` |

The `sync_margin` (default: 3 frames) provides a safety buffer for network propagation, deserialization, and queue insertion. At 50fps, 3 frames = 60ms — sufficient for LAN round-trip plus processing.

### Dispatch Loop

The dispatch loop runs on a dedicated thread with adaptive sleep:

```
┌──────────────────────────────────────────────────────────┐
│ while (running):                                         │
│   now = frame_clock.current_frame()                      │
│                                                          │
│   while (queue not empty AND queue.top.frame <= now):    │
│     cmd = queue.pop()                                    │
│     if (now - cmd.frame > 1): log WARNING "late"         │
│     executor(cmd.command_text)                           │
│                                                          │
│   if (queue not empty):                                  │
│     ns_until = frame_clock.ns_until_frame(next_frame)    │
│     if ns_until > 2ms:  sleep(500µs)                     │
│     elif ns_until > 100µs:  sleep(50µs)                  │
│     else: spin (no sleep)                                │
│   else:                                                  │
│     sleep(1ms)                                           │
└──────────────────────────────────────────────────────────┘
```

The three-tier sleep strategy balances CPU efficiency with timing precision:
- **> 2ms away**: Sleep 500µs — minimal CPU, coarse timing
- **100µs – 2ms**: Sleep 50µs — moderate CPU, sub-millisecond precision
- **< 100µs**: Spin — maximum precision for the final approach to the frame boundary

### Priority

Commands scheduled for the same frame are ordered by priority (higher value = executes first). This ensures deterministic ordering when multiple operations must happen simultaneously (e.g., a MIXER transform followed by a PLAY).

### Lateness Detection

If a command executes more than 1 frame late, a warning is logged with the command text (truncated to 60 characters). This indicates either:
- Network propagation exceeded the sync margin
- The scheduler thread was preempted by a higher-priority OS thread
- The command queue is backlogged

---

## Command Relay

### Wire Protocol

Commands are transmitted over TCP using a simple text protocol:

```
FRAME:<target_frame> <AMCP_COMMAND_TEXT>\r\n
```

Example:
```
FRAME:1234567 PLAY 1-10 clip1\r\n
```

### Connection Topology

**Master → Client** connections use persistent TCP sockets with `TCP_NODELAY` enabled for minimal latency:

```
Master                          Client A
  │                                │
  ├──── connect() ────────────────▸│ listen_socket
  │    TCP_NODELAY=1               │
  │                                │
  ├──── "FRAME:123 PLAY 1-1 x" ──▸│ parse → scheduler.schedule(123, "PLAY 1-1 x")
  │                                │
  ├──── "FRAME:126 STOP 1-1" ────▸│ parse → scheduler.schedule(126, "STOP 1-1")
  │                                │
```

### Master Mode

The master maintains TCP connections to all configured members. If a connection drops, it automatically retries every 2 seconds. Commands are routed based on the virtual channel map:

1. Extract the virtual channel from the command
2. Look up the target host in the channel map
3. If the target is `"local"`: execute on this instance's scheduler
4. If the target is a remote host: rewrite the command (virtual → physical channel) and send over TCP

### Client Mode

The client opens a TCP listener on the configured `relay-port` (default: 5250). When the master connects, the client spawns a receive thread that:

1. Reads lines terminated by `\r\n`
2. Parses `FRAME:<N>` prefix to extract the target frame
3. Feeds the command to the local scheduler: `scheduler.schedule(target_frame, command)`

### Channel Rewriting

When a command is forwarded to a remote member, the virtual channel reference is rewritten to the physical channel:

```
Virtual channel 3 → host 192.168.1.20, physical channel 1

Original:  "PLAY 3-10 clip1"
Rewritten: "PLAY 1-10 clip1"
```

This allows the master to use a unified virtual channel namespace while each client operates on its own physical channels.

---

## Virtual Channel Map

The virtual channel map provides a layer of indirection between the master's unified channel numbering and the physical channel numbers on each client.

### Configuration

```xml
<channels>
    <channel virtual="1" host="192.168.1.10" physical="1" />
    <channel virtual="2" host="192.168.1.10" physical="2" />
    <channel virtual="3" host="192.168.1.20" physical="1" />
    <channel virtual="4" host="local"        physical="1" />
</channels>
```

| Virtual | Host | Physical | Meaning |
|---------|------|----------|---------|
| 1 | 192.168.1.10 | 1 | Client A, channel 1 |
| 2 | 192.168.1.10 | 2 | Client A, channel 2 |
| 3 | 192.168.1.20 | 1 | Client B, channel 1 |
| 4 | local | 1 | Master's own channel 1 |

### API

| Method | Description |
|--------|-------------|
| `get_host(virtual)` | Returns the host string for a virtual channel |
| `get_physical_channel(virtual)` | Returns the physical channel number |
| `is_local(virtual)` | Returns `true` if mapped to `"local"` |
| `rewrite_command(cmd, virtual)` | Replaces the virtual channel number with the physical channel in the command string |
| `channels_for_host(host)` | Returns all virtual channels mapped to a specific host |
| `remote_hosts()` | Returns all unique non-local host strings |

---

## Content Sync Watchdog

### Purpose

Even with identical PTP clocks and synchronized PLAY commands, content can drift between nodes due to:
- OS thread scheduling jitter affecting frame delivery
- Disk I/O stalls during media decoding
- GPU pipeline depth differences
- Producer-internal buffering variations

The content sync watchdog detects and corrects this drift by continuously comparing the **expected** producer frame position (derived from the global frame clock) against the **actual** `frame_number()` reported by the producer.

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│ Watchdog Loop (dedicated thread, ~2ms sleep)            │
│                                                         │
│ For each tracked producer:                              │
│   expected = compute_expected_frame(global_frame)       │
│   actual   = producer->frame_number()                   │
│   drift    = expected - actual                          │
│                                                         │
│   if |drift| > threshold:                               │
│     stage->call(layer, {"seek", expected})               │
│     log WARNING "drift correction"                      │
└─────────────────────────────────────────────────────────┘
```

### Expected Frame Computation

$$\text{expected} = \begin{cases}
\text{elapsed} & \text{if infinite/unknown duration} \\
\text{elapsed} \bmod \text{duration} & \text{if looping} \\
\min(\text{elapsed}, \text{duration} - 1) & \text{if non-looping}
\end{cases}$$

Where $\text{elapsed} = \text{current\_global\_frame} - \text{start\_frame}$.

### Drift Threshold

The default threshold is **2 frames**. This means:
- 1 frame of drift is tolerated (normal jitter)
- 2+ frames triggers a seek correction

The threshold is configurable via `<content-sync-threshold>` in the config or at compile time.

### Tracking Modes

| Mode | How to activate | Behavior |
|------|----------------|----------|
| **Manual layer** | `CLUSTER TRACK 1-10` | Track a specific layer. Auto-queries producer for duration and loop state. |
| **Manual channel** | `CLUSTER TRACK 1` | Track all active layers on channel 1. Scans every 15 frames (~0.3s at 50fps). |
| **Config auto-all** | `<content-sync>true</content-sync>` | Track all layers on all channels automatically at startup. |

### Auto-Discovery

When a channel is tracked (either via `CLUSTER TRACK` or `<content-sync>true</content-sync>`), the watchdog periodically scans layers 0 through `max_layer` (default: 100) using `stage()->foreground(layer)`. For each layer with an active producer:

1. Reads `nb_frames()` for content duration
2. Reads `frame_number()` for current position
3. Queries `call(layer, {"loop"})` for loop state
4. Computes `start_frame` by back-calculating from the current position
5. Begins drift tracking

The scan uses `wait_for(0ms)` on the foreground future to avoid blocking the watchdog thread. Layers that are busy or empty are simply skipped.

### Content Change Detection

The watchdog automatically detects when content changes on a tracked layer and re-anchors without manual intervention:

| Signal | Detection | Action |
|--------|-----------|--------|
| Different `nb_frames()` | `current_nb != last_nb_frames` | New content loaded — re-anchor, re-query duration/loop |
| `frame_number()` reset to 0 | `current_fn == 0 && drift != 0` | Content restarted — re-anchor from current global frame |
| Producer becomes null | `!producer` | Layer cleared — mark paused, skip drift checks |
| Producer appears from null | `paused == true && producer != null` | New content on previously empty layer — re-anchor |

After a content change, a **5-frame cooldown** is applied before drift checking resumes, allowing the producer to stabilize.

---

## Producer Lifecycle Handling

The watchdog correctly handles all CasparCG producer lifecycle transitions:

### LOADBG (background load)

```
stage::load(layer, producer, preview=false, auto_play=false)
```

Producer goes to the **background** slot. `foreground()` still returns the old producer. The watchdog does not see the new producer — no action needed.

### LOAD (preview load)

```
stage::load(layer, producer, preview=true)
```

Producer goes to foreground **but is paused** (`frame_number()` frozen at 0). The watchdog's stall detection activates:

1. Auto-discovery sees the producer via `foreground()` and begins tracking
2. After 3 consecutive frames where `frame_number()` doesn't advance, marks as **stalled**
3. Drift checking is suspended — no spurious seek corrections

### PLAY (start playback)

```
stage::play(layer)
```

If after LOADBG: background → foreground transition. Content change detection fires (new `nb_frames`, `frame_number` reset). Watchdog re-anchors `start_frame` and begins drift tracking.

If after LOAD: `frame_number()` starts advancing. Watchdog detects movement after stall, logs "resumed", re-anchors `start_frame` from the current global frame.

### PAUSE

`frame_number()` stops advancing. After 3 stalled frames, watchdog suspends drift checking.

### RESUME

`frame_number()` starts advancing again. Watchdog detects movement, re-anchors, resumes tracking.

### EOF (end of non-looping clip)

Producer holds on the last frame (`frame_number() == duration - 1`). Stall detection triggers after 3 frames, drift checking suspended. `compute_expected_frame` also clamps at `duration - 1`, so even during the brief window before stall detection, `drift == 0` and no correction fires.

### STOP / CLEAR

Producer becomes null or empty. Watchdog marks layer as paused, skips it entirely.

### Stall Detection Details

```cpp
// 3 consecutive frames with no frame_number() change → stalled
if (current_fn == tp.last_frame_number && current_fn > 0) {
    stall_count++;
    if (stall_count >= 3) {
        paused = true;  // Suspend drift checking
    }
}
```

When the producer resumes (frame_number changes after being stalled):

```cpp
// Was stalled, now advancing → re-anchor
start_frame = current_global_frame - current_frame_number;
paused = false;
stall_count = 0;
```

---

## AMCP Commands

### CLUSTER STATUS

Returns the complete state of the cluster subsystem.

```
>> CLUSTER STATUS
<< 201 CLUSTER STATUS OK
<< MODE: master
<< PTP-STATE: locked
<< PTP-OFFSET-US: 42
<< PTP-DELAY-US: 150
<< FRAME: 1234567
<< PENDING-COMMANDS: 0
<< MEMBER: 192.168.1.10:5250 connected
<< MEMBER: 192.168.1.20:5250 connected
<< SYNC-CORRECTIONS: 3
<< SYNC-CH0: synced layers=4 max-drift=1 corrections=2
<< SYNC-CH1: synced layers=2 max-drift=0 corrections=1
```

### CLUSTER SCHEDULE

Schedule an AMCP command for frame-accurate execution.

```
CLUSTER SCHEDULE <command> [AT <frame>]
```

Without `AT`, uses `current_frame + sync_margin`:
```
>> CLUSTER SCHEDULE PLAY 2-1 clip1
<< 202 CLUSTER SCHEDULE OK 1234570
```

With explicit frame:
```
>> CLUSTER SCHEDULE PLAY 2-1 clip1 AT 1234600
<< 202 CLUSTER SCHEDULE OK 1234600
```

The scheduler automatically routes the command based on the virtual channel:
- Virtual channel maps to local → scheduled on local scheduler
- Virtual channel maps to remote host → forwarded via TCP relay with channel rewriting

### CLUSTER TRACK

Start drift monitoring for a layer or channel.

```
CLUSTER TRACK <channel>[-<layer>] [<duration> [LOOP]]
```

Track entire channel (auto-discover all active layers):
```
>> CLUSTER TRACK 1
<< 202 CLUSTER TRACK OK channel=1 (auto-discover)
```

Track specific layer (auto-queries producer for duration/loop):
```
>> CLUSTER TRACK 1-10
<< 202 CLUSTER TRACK OK duration=3750 loop=true
```

Track with explicit parameters:
```
>> CLUSTER TRACK 1-10 3750 LOOP
<< 202 CLUSTER TRACK OK duration=3750 loop=true
```

### CLUSTER UNTRACK

Stop drift monitoring.

```
>> CLUSTER UNTRACK 1        -- Untrack all layers on channel 1
>> CLUSTER UNTRACK 1-10     -- Untrack specific layer
<< 202 CLUSTER UNTRACK OK
```

---

## Configuration Reference

All options for the `<cluster>` block in `casparcg.config`:

| Element | Type | Default | Description |
|---------|------|---------|-------------|
| `<mode>` | enum | `disabled` | `disabled` \| `master` \| `client` \| `external` |
| **Sync settings** | | | |
| `<sync-margin>` | int | `3` | Frames of scheduling margin |
| `<bind>` | string | `0.0.0.0` | Local interface for PTP multicast |
| `<multicast-group>` | string | `224.0.1.129` | PTP multicast address |
| `<ptp-domain>` | int | `0` | PTP domain number (0–127) |
| `<sync-interval-ms>` | int | `125` | PTP Sync message interval (ms) |
| `<epoch-origin>` | string | *(none)* | ISO-8601 datetime for frame numbering epoch |
| **Content sync** | | | |
| `<content-sync>` | bool | `false` | Auto-track all layers on all channels |
| `<content-sync-threshold>` | int | `2` | Drift threshold in frames before correction |
| `<content-sync-max-layer>` | int | `100` | Highest layer index to scan per channel |
| **Master settings** | | | |
| `<channels>` | block | | Virtual-to-physical channel mappings |
| `<members>` | block | | List of `host:port` cluster members |
| **Client settings** | | | |
| `<master>` | string | | `host:port` of the cluster master |
| `<relay-port>` | int | `5250` | TCP port for relay listener |
| **Diagnostics** | | | |
| `<log-ptp-status>` | bool | `false` | Periodic PTP offset/delay logging |

### Channel Mapping Attributes

```xml
<channel virtual="1" host="192.168.1.10" physical="1" />
```

| Attribute | Description |
|-----------|-------------|
| `virtual` | Virtual channel number used in AMCP commands on the master |
| `host` | Target host (`"local"` for master's own channels, or `"ip:port"` for clients) |
| `physical` | Physical channel number on the target host |

---

## Configuration Examples

### Minimal Two-Node Cluster

**Master** (192.168.1.1):
```xml
<cluster>
    <mode>master</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <content-sync>true</content-sync>
    <channels>
        <channel virtual="1" host="local" physical="1" />
        <channel virtual="2" host="192.168.1.2:5250" physical="1" />
    </channels>
    <members>
        <member>192.168.1.2:5250</member>
    </members>
</cluster>
```

**Client** (192.168.1.2):
```xml
<cluster>
    <mode>client</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <content-sync>true</content-sync>
    <master>192.168.1.1:5250</master>
    <relay-port>5250</relay-port>
</cluster>
```

### Video Wall: 4 Machines, 4 Outputs

Master drives channel 1 locally, clients each drive one channel:

**Master**:
```xml
<cluster>
    <mode>master</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <sync-margin>3</sync-margin>
    <content-sync>true</content-sync>
    <content-sync-threshold>1</content-sync-threshold>

    <channels>
        <channel virtual="1" host="local"              physical="1" />
        <channel virtual="2" host="192.168.1.11:5250"  physical="1" />
        <channel virtual="3" host="192.168.1.12:5250"  physical="1" />
        <channel virtual="4" host="192.168.1.13:5250"  physical="1" />
    </channels>
    <members>
        <member>192.168.1.11:5250</member>
        <member>192.168.1.12:5250</member>
        <member>192.168.1.13:5250</member>
    </members>
</cluster>
```

**Each Client** (adjust IP):
```xml
<cluster>
    <mode>client</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <content-sync>true</content-sync>
    <content-sync-threshold>1</content-sync-threshold>
    <master>192.168.1.10:5250</master>
</cluster>
```

Then from the master's AMCP connection:
```
CLUSTER SCHEDULE PLAY 1-1 wall_topleft
CLUSTER SCHEDULE PLAY 2-1 wall_topright
CLUSTER SCHEDULE PLAY 3-1 wall_bottomleft
CLUSTER SCHEDULE PLAY 4-1 wall_bottomright
```

All four PLAYs are scheduled for the same future frame (`current + sync_margin`), ensuring simultaneous start across all machines.

### External PTP Grandmaster

Lock to a network PTP grandmaster (e.g., a Meinberg, Evertz, or Arista PTP master) instead of running CasparCG's own PTP master. All CasparCG nodes join as PTP clients:

```xml
<cluster>
    <mode>external</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <multicast-group>224.0.1.129</multicast-group>
    <ptp-domain>0</ptp-domain>
    <content-sync>true</content-sync>
</cluster>
```

In this mode, all nodes synchronize to the external grandmaster but operate independently (no command relay). Content sync still works because all nodes share the same PTP time base and epoch origin.

### High-Precision Live Event

Tighter sync margin and lower drift threshold for critical applications:

```xml
<cluster>
    <mode>master</mode>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <sync-margin>5</sync-margin>
    <sync-interval-ms>62</sync-interval-ms>
    <content-sync>true</content-sync>
    <content-sync-threshold>1</content-sync-threshold>
    <content-sync-max-layer>20</content-sync-max-layer>
    <log-ptp-status>true</log-ptp-status>
    <!-- ... channels and members ... -->
</cluster>
```

- `sync-interval-ms=62`: Doubles PTP measurement rate (8 measurements/sec vs 4)
- `sync-margin=5`: More headroom for command propagation
- `content-sync-threshold=1`: Correct at the first sign of drift
- `content-sync-max-layer=20`: Limit scan range for faster iteration
- `log-ptp-status=true`: Periodic logging for monitoring

### Multiple PTP Domains (Isolated Clusters)

Two independent clusters sharing the same network:

**Cluster A** (domain 0):
```xml
<cluster>
    <mode>master</mode>
    <ptp-domain>0</ptp-domain>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <!-- ... -->
</cluster>
```

**Cluster B** (domain 1):
```xml
<cluster>
    <mode>master</mode>
    <ptp-domain>1</ptp-domain>
    <epoch-origin>2026-01-01T00:00:00Z</epoch-origin>
    <!-- ... -->
</cluster>
```

PTP domain number filtering ensures that Cluster A's master ignores Cluster B's Sync/Follow_Up messages and vice versa.

---

## Timing Analysis

### End-to-End Latency Budget

For a command issued on the master to execute on a client at the correct frame:

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| AMCP parse on master | < 1ms | |
| TCP send (master → client) | < 0.5ms | LAN, TCP_NODELAY |
| TCP receive + parse on client | < 0.5ms | Line-buffered |
| Queue insertion | < 0.01ms | Priority queue push |
| **Total propagation** | **< 2ms** | |
| Scheduler sleep precision | < 0.5ms | Adaptive sleep + spin |
| Frame period at 50fps | 20ms | |
| **Sync margin at 3 frames** | **60ms** | 30× headroom |

### PTP Accuracy

On a dedicated LAN with managed switches:

| Metric | Typical Value | Worst Case |
|--------|---------------|------------|
| Clock offset after convergence | < 100µs | < 500µs |
| Mean path delay | 50–200µs | < 1ms |
| Frame number agreement | Exact | ±1 frame during lock acquisition |

The EMA filter converges within ~8 Sync intervals (1 second at default rate). After convergence, frame numbers are identical across all nodes provided the offset stays below 1 frame period (20ms at 50fps — well within the 1ms lock threshold).

### Content Sync Precision

| Metric | Value | Notes |
|--------|-------|-------|
| Watchdog check rate | Once per frame | Checks on new frame boundaries only |
| Detection latency | 1 frame | Drift detected on the next frame check |
| Correction latency | 1–2 frames | Seek command processed by stage |
| Scan interval | Every 15 frames | Auto-discovery of new producers (~0.3s at 50fps) |
| CPU cost per layer | ~2ns | One integer comparison |
| CPU cost per scan | ~50µs | `foreground()` future + wait_for(0) for 100 layers |

---

## Troubleshooting

### PTP-STATE: free-running

The client lost synchronization with the master.

- Verify the master is running and reachable: `ping <master_ip>`
- Check that UDP ports 319 and 320 are not blocked by firewall
- Verify both nodes use the same `<multicast-group>` and `<ptp-domain>`
- Check that the network supports multicast (managed switches may need IGMP snooping configured)
- Run `CLUSTER STATUS` on the master to confirm it's active

### PTP-OFFSET-US shows large values

The PTP offset is in microseconds. Values above 1000µs (1ms) indicate poor synchronization.

- Verify the network path: switches should be <= 3 hops, prefer direct connections
- Check for network congestion: PTP is sensitive to asymmetric delays
- Reduce `<sync-interval-ms>` to 62 for faster convergence
- Consider using a dedicated VLAN for PTP traffic

### SYNC-CH0: DRIFTED

The content sync watchdog detected drift exceeding the threshold.

- Check `max-drift` value: 2–3 frames is normal during content transitions
- Persistent drift may indicate I/O issues: check disk throughput and GPU utilization
- Reduce `<content-sync-threshold>` to 1 for tighter correction
- Monitor correction count: occasional corrections are normal, continuous corrections indicate a systemic issue

### Commands executing late

The command scheduler is logging "late by N frames" warnings.

- Increase `<sync-margin>` (try 5 or 7)
- Check network latency between master and clients
- Check for CPU contention on the client (the scheduler thread needs consistent scheduling)
- Verify `TCP_NODELAY` is effective: some Windows TCP stacks batch small writes

### Member: disconnected

The master cannot reach a client.

- Verify the client is running with `<mode>client</mode>`
- Check that `<relay-port>` matches between master's `<member>` address and client's config
- Verify TCP port 5250 (or custom port) is not blocked by firewall
- The relay auto-reconnects every 2 seconds — check client logs for connection attempts

### Best Practices

1. **Use a dedicated network** for PTP and relay traffic. Shared networks with heavy traffic introduce timing jitter.

2. **Set `epoch-origin` identically on all nodes.** Different epochs cause frame number disagreement — commands will execute at wrong times.

3. **Start the master first.** Clients that start before the master will be in `free-running` state until the master's Sync messages arrive.

4. **Use `content-sync=true`** for automated drift correction. Manual tracking with `CLUSTER TRACK` is available for fine control but requires the operator to track content changes.

5. **Monitor with `CLUSTER STATUS`** periodically. Watch for `free-running` PTP state, high drift values, or disconnected members.

6. **Keep sync margins reasonable.** 3 frames (default) works for most LAN setups. Increase to 5–7 for WAN or unreliable networks. Higher margins add latency to interactive commands.

7. **Use PTP domain isolation** when running multiple independent clusters on the same network. Each cluster should have a unique `<ptp-domain>` value.

8. **Pre-load content** on all nodes before issuing synchronized PLAY commands. The sync mechanism corrects timing drift, not missing media files.
