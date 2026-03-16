# DMX Lighting — ArtNet & sACN Consumers

CasparCG can drive DMX lighting fixtures directly from video output using the
**ArtNet** and **sACN** consumers. Both analyse each rendered frame, sample the
average colour of a configurable screen region per fixture, and transmit the
resulting DMX values in real time. This enables pixel-accurate live lighting to
match on-screen content — useful for virtual production LED walls, studio
ambience, and stage wash lighting.

---

## Table of Contents

1. [Protocol Overview](#1-protocol-overview)
2. [Consumer Configuration](#2-consumer-configuration)
3. [Fixture Definition](#3-fixture-definition)
4. [Fixture Types](#4-fixture-types)
5. [Sampling Regions & Coordinates](#5-sampling-regions--coordinates)
6. [ArtNet Reference](#6-artnet-reference)
7. [sACN Reference](#7-sacn-reference)
8. [Protocol Comparison](#8-protocol-comparison)
9. [Multiple Universes](#9-multiple-universes)
10. [Worked Examples](#10-worked-examples)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Protocol Overview

### ArtNet

ArtNet is a DMX-over-UDP protocol by Artistic Licence. It uses a fixed UDP
port (6454) and supports unicast and broadcast transport. It is widely
supported by conventional lighting consoles, pixel-mapped fixtures, and most
DMX/ethernet nodes.

- **Port:** UDP 6454
- **Transport:** Unicast or broadcast
- **Universes:** 0 – 32767
- **No priority system** — last sender wins
- **Header:** 18 bytes + 512 DMX bytes = 530 bytes per packet

### sACN (Streaming ACN / ANSI E1.31)

sACN is the ESTA/ANSI standard (E1.31) built on top of ACN. It is the
preferred protocol for modern LED processors (Brompton, NovaStar, Megapixel),
MA consoles, and ETC fixtures. Native **multicast** support means a single
UDP packet reaches every device in a universe simultaneously, with no need to
know each fixture's IP address.

- **Port:** UDP 5568
- **Transport:** Unicast or multicast (preferred)
- **Multicast address:** `239.255.<hi>.<lo>` per universe
- **Universes:** 1 – 63999
- **Priority system:** 1–200, higher value wins merge conflicts
- **Source name:** 64-character label visible in network analysers
- **Header:** ~126 bytes + 512 DMX bytes = 638 bytes per packet

Both protocols carry the same 512-byte DMX payload, so fixture definitions
are identical in both consumers.

---

## 2. Consumer Configuration

Consumers are added inside a `<channel>` block in `caspar.config`, alongside
other output consumers (screen, decklink, etc.). Multiple DMX consumers can
run on the same channel (e.g. for multiple universes).

```xml
<channel>
  <video-mode>1080p5000</video-mode>
  <consumers>

    <!-- ArtNet consumer -->
    <consumer>
      <type>artnet</type>
      <!-- ... options ... -->
    </consumer>

    <!-- sACN consumer -->
    <consumer>
      <type>sacn</type>
      <!-- ... options ... -->
    </consumer>

  </consumers>
</channel>
```

> **Requirement:** The channel must use **8-bit colour depth** (the default).
> 16-bit channels are not supported by either DMX consumer.

---

## 3. Fixture Definition

Fixtures are declared inside a `<fixtures>` element within the consumer block.
Each `<fixture>` element maps a rectangular screen region to a contiguous
block of DMX channels.

```xml
<fixtures>
  <fixture>
    <type>RGB</type>
    <start-address>1</start-address>
    <fixture-count>10</fixture-count>
    <fixture-channels>3</fixture-channels>
    <x>960</x>
    <y>100</y>
    <width>1920</width>
    <height>200</height>
    <rotation>0</rotation>
  </fixture>
</fixtures>
```

### Fixture parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `type` | string | ✓ | — | `DIMMER`, `RGB`, or `RGBW` |
| `start-address` | int | ✓ | — | DMX address of the first channel (1-based) |
| `fixture-count` | int | ✓ | — | Number of individual fixtures in the box |
| `fixture-channels` | int | | equal to type | Total DMX channels allocated per fixture; useful when a fixture has extra channels beyond colour (e.g. strobe, dimmer) |
| `x` | float | | 0 | Centre X of the sampling box in pixels |
| `y` | float | | 0 | Centre Y of the sampling box in pixels |
| `width` | float | | 0 | Width of the sampling box in pixels |
| `height` | float | | 0 | Height of the sampling box in pixels |
| `rotation` | float | | 0 | Rotation of the sampling box in degrees |

#### Address layout

When `fixture-count > 1`, the sampling box is divided equally along its
width axis, and each sub-region maps to consecutive DMX addresses:

```
fixture-count = 4, start-address = 1, fixture-channels = 3 (RGB)

  Box region    →  addresses
  sub-box 0     →  1, 2, 3    (R, G, B)
  sub-box 1     →  4, 5, 6
  sub-box 2     →  7, 8, 9
  sub-box 3     →  10, 11, 12
```

If `fixture-channels` is set higher than the minimum for the fixture type,
the extra channels receive value 0 (useful for 5-channel RGBW+strobe
fixtures where channels 5 is reserved):

```xml
<fixture-channels>5</fixture-channels>  <!-- RGBW at 1-4, channel 5 = 0 -->
```

---

## 4. Fixture Types

### DIMMER (1 channel per fixture)

Single-channel luminance fixture. The pixel colour is converted to a
perceived brightness value using the ITU-R BT.601 luma formula:

$$Y = 0.279 \cdot R + 0.547 \cdot G + 0.106 \cdot B$$

```xml
<type>DIMMER</type>
```

### RGB (3 channels per fixture)

Three-channel colour fixture. R, G, B are written directly to consecutive
DMX channels.

```xml
<type>RGB</type>
```

### RGBW (4 channels per fixture)

Four-channel fixture with a dedicated white channel. The white component is
extracted as the minimum of R, G, B and subtracted from each colour channel
to maximise efficiency:

$$W = \min(R, G, B)$$
$$R' = R - W, \quad G' = G - W, \quad B' = B - W$$

```xml
<type>RGBW</type>
```

---

## 5. Sampling Regions & Coordinates

Coordinates use the **pixel space** of the channel's video format. For a
`1080p` channel (1920×1080):

- Top-left corner: `x=0, y=0`
- Centre: `x=960, y=540`
- Bottom-right: `x=1920, y=1080`

The `x` and `y` values define the **centre** of the sampling box, not its
top-left corner. The `width` and `height` values extend equally in all
directions from that centre.

The `rotation` parameter rotates the sampling box around its own centre in
degrees (clockwise). This is useful for angled LED strips or when fixtures are
mounted at a tilt on the set.

```
Example: A strip of 8 RGB fixtures across the top edge of a 1920×1080 frame

  <x>960</x>      ← centre of the strip horizontally
  <y>30</y>       ← near the top edge
  <width>1920</width>
  <height>60</height>
  <fixture-count>8</fixture-count>

  Each fixture samples a 240×60 pixel region.
```

### Tips for virtual production LED walls

- Use the **full video format** dimensions as your coordinate space (e.g.
  1920×1080 regardless of the physical wall resolution).
- Map the LED sampling region to the screen area that corresponds to the
  physical wall.
- For a multi-panel rig, define one fixture group per panel, each with its
  own `<fixture>` element.
- Overlap slightly between adjacent fixture regions if you want smooth
  colour transitions across panel seam lines.

---

## 6. ArtNet Reference

### Consumer options

| XML element | Type | Default | Description |
|---|---|---|---|
| `universe` | int | `0` | ArtNet universe (0 – 32767) |
| `host` | string | `127.0.0.1` | Destination IP address (unicast or broadcast) |
| `port` | int | `6454` | UDP port (standard ArtNet port) |
| `refresh-rate` | int | `10` | DMX refresh rate in Hz (1 – n) |

### Minimal example

```xml
<consumer>
  <type>artnet</type>
  <universe>0</universe>
  <host>192.168.1.255</host>   <!-- broadcast to subnet -->
  <refresh-rate>25</refresh-rate>
  <fixtures>
    <fixture>
      <type>RGB</type>
      <start-address>1</start-address>
      <fixture-count>10</fixture-count>
      <x>960</x><y>540</y><width>1920</width><height>100</height>
    </fixture>
  </fixtures>
</consumer>
```

### Packet structure

```
Bytes  0–7   : "Art-Net\0" identifier
Bytes  8–9   : OpCode 0x5000 (ArtDMX)
Bytes 10–11  : Protocol version 14
Byte   12    : Sequence number (0 = disabled)
Byte   13    : Physical port hint (0)
Bytes 14–15  : Universe (low, high byte)
Bytes 16–17  : Data length (high, low byte)
Bytes 18–529 : 512 DMX channel values
```

### Broadcast vs unicast

Use the subnet broadcast address (e.g. `192.168.1.255`) to reach all ArtNet
nodes on a segment without enumerating individual IPs. For reliable delivery
to a specific node, use its direct IP.

---

## 7. sACN Reference

### Consumer options

| XML element | Type | Default | Description |
|---|---|---|---|
| `universe` | int | `1` | sACN universe (1 – 63999) |
| `host` | string | _(empty)_ | Leave empty for **multicast** (recommended); set to a unicast IP for directed delivery |
| `port` | int | `5568` | UDP port (standard sACN port) |
| `priority` | int | `100` | Source priority (1 – 200); higher value wins when multiple sources are active |
| `multicast-ttl` | int | `10` | IP multicast TTL / hop count; increase to 64+ if crossing managed switch boundaries |
| `refresh-rate` | int | `10` | DMX refresh rate in Hz (1 – n) |

### Minimal example (multicast)

```xml
<consumer>
  <type>sacn</type>
  <universe>1</universe>
  <!-- no <host> = uses multicast 239.255.0.1 for universe 1 -->
  <priority>100</priority>
  <refresh-rate>44</refresh-rate>
  <fixtures>
    <fixture>
      <type>RGBW</type>
      <start-address>1</start-address>
      <fixture-count>24</fixture-count>
      <fixture-channels>4</fixture-channels>
      <x>960</x><y>540</y><width>1920</width><height>1080</height>
    </fixture>
  </fixtures>
</consumer>
```

### Unicast example (specific LED processor IP)

```xml
<consumer>
  <type>sacn</type>
  <universe>1</universe>
  <host>192.168.10.50</host>  <!-- Brompton processor management IP -->
  <priority>120</priority>
  <refresh-rate>44</refresh-rate>
  <fixtures><!-- ... --></fixtures>
</consumer>
```

### Multicast addressing

Multicast destination address for a given universe:

$$\text{address} = 239.255.\lfloor\text{universe}/256\rfloor.(\text{universe} \bmod 256)$$

| Universe | Multicast address |
|---|---|
| 1 | 239.255.0.1 |
| 2 | 239.255.0.2 |
| 256 | 239.255.1.0 |
| 1000 | 239.255.3.232 |

Your network switch must have **IGMP snooping** enabled for multicast to work
reliably; without it, multicast frames flood all ports (which is often
acceptable in a closed studio network).

### Priority system

When multiple sACN sources send to the same universe (e.g. a backup console
and CasparCG), the receiver merges based on priority. CasparCG defaults to
100. Set a higher value (e.g. 120) to override background control:

```xml
<priority>120</priority>   <!-- takes precedence over console at 100 -->
```

### sACN packet structure (E1.31-2018)

```
Bytes   0–15  : Preamble (2) + Postamble (2) + ACN Identifier (12)
Bytes  16–37  : Root PDU  — Flags&Length + Vector 0x00000004 + CID (16 bytes)
Bytes  38-114 : Framing PDU — Flags&Length + Vector + SourceName (64)
                            + Priority + SyncAddr + Sequence + Options + Universe
Bytes 115–124 : DMP PDU  — Flags&Length + Vector + AddrType + Addressing
Byte       125: DMX Start Code (0x00)
Bytes 126–637 : 512 DMX channel values
```

The **CID** (Component Identifier) is a randomly generated UUID created once
when CasparCG starts. It persists for the lifetime of the process and
identifies this source to sACN receivers and network analysers.

---

## 8. Protocol Comparison

| Feature | ArtNet | sACN |
|---|---|---|
| Standard | Artistic Licence | ANSI E1.31 / ESTA |
| UDP Port | 6454 | 5568 |
| Universe range | 0 – 32767 | 1 – 63999 |
| Transport modes | Unicast, broadcast | Unicast, **multicast** |
| Priority / merge | No | Yes (1–200) |
| Source identification | No | Yes (64-char name + UUID) |
| Sequence numbers | Yes | Yes |
| Compatibility | Broad (legacy consoles) | Modern LED processors & consoles |
| Network overhead | Low (530 bytes) | Slightly higher (638 bytes) |

**Choose ArtNet when:** integrating with older consoles, conventional ETC/MA2
nodes, or simple unicast networks where broadcast is acceptable.

**Choose sACN when:** driving LED processors (Brompton, NovaStar, Megapixel),
MA3 / ETC Eos, or any setup where multicast, source merging, or network
visibility matters.

---

## 9. Multiple Universes

A single DMX universe carries 512 channels. For large rigs (hundreds of
RGBW fixtures), add multiple consumers on the same channel — one per
universe:

```xml
<consumers>
  <!-- Universe 1: LED wall, top half -->
  <consumer>
    <type>sacn</type>
    <universe>1</universe>
    <refresh-rate>44</refresh-rate>
    <fixtures>
      <fixture>
        <type>RGBW</type>
        <start-address>1</start-address>
        <fixture-count>128</fixture-count>
        <fixture-channels>4</fixture-channels>
        <x>960</x><y>270</y><width>1920</width><height>540</height>
      </fixture>
    </fixtures>
  </consumer>

  <!-- Universe 2: LED wall, bottom half -->
  <consumer>
    <type>sacn</type>
    <universe>2</universe>
    <refresh-rate>44</refresh-rate>
    <fixtures>
      <fixture>
        <type>RGBW</type>
        <start-address>1</start-address>
        <fixture-count>128</fixture-count>
        <fixture-channels>4</fixture-channels>
        <x>960</x><y>810</y><width>1920</width><height>540</height>
      </fixture>
    </fixtures>
  </consumer>
</consumers>
```

Each consumer runs its own independent send thread, so there is no
frame-blocking between universes.

---

## 10. Worked Examples

### Example A — Ambient strip behind a news desk (ArtNet)

20 RGBW LED fixtures in a horizontal strip behind the talent, sampling the
lower edge of the background video:

```xml
<consumer>
  <type>artnet</type>
  <universe>0</universe>
  <host>192.168.1.100</host>
  <refresh-rate>25</refresh-rate>
  <fixtures>
    <fixture>
      <type>RGBW</type>
      <start-address>1</start-address>
      <fixture-count>20</fixture-count>
      <fixture-channels>4</fixture-channels>
      <!-- sample the bottom 10% of the 1920×1080 frame -->
      <x>960</x><y>1026</y><width>1920</width><height>108</height>
      <rotation>0</rotation>
    </fixture>
  </fixtures>
</consumer>
```

---

### Example B — Virtual production ceiling wash (sACN multicast)

A grid of 64 RGBW ceiling fixtures, all receiving the average colour of the
entire frame to provide a fill light that matches the LED wall content:

```xml
<consumer>
  <type>sacn</type>
  <universe>3</universe>
  <priority>100</priority>
  <refresh-rate>44</refresh-rate>
  <fixtures>
    <fixture>
      <type>RGBW</type>
      <start-address>1</start-address>
      <fixture-count>64</fixture-count>
      <fixture-channels>4</fixture-channels>
      <!-- sample the full frame -->
      <x>960</x><y>540</y><width>1920</width><height>1080</height>
    </fixture>
  </fixtures>
</consumer>
```

---

### Example C — Mixed protocol rig (ArtNet + sACN on the same channel)

Older LED strip nodes on ArtNet, new Brompton processor on sACN, both driven
from the same CasparCG channel simultaneously:

```xml
<consumers>
  <!-- Legacy strip nodes -->
  <consumer>
    <type>artnet</type>
    <universe>0</universe>
    <host>192.168.1.255</host>
    <refresh-rate>25</refresh-rate>
    <fixtures>
      <fixture>
        <type>RGB</type>
        <start-address>1</start-address>
        <fixture-count>30</fixture-count>
        <x>960</x><y>540</y><width>1920</width><height>1080</height>
      </fixture>
    </fixtures>
  </consumer>

  <!-- Brompton LED processor -->
  <consumer>
    <type>sacn</type>
    <universe>1</universe>
    <priority>120</priority>
    <refresh-rate>44</refresh-rate>
    <fixtures>
      <fixture>
        <type>RGBW</type>
        <start-address>1</start-address>
        <fixture-count>128</fixture-count>
        <fixture-channels>4</fixture-channels>
        <x>960</x><y>540</y><width>1920</width><height>1080</height>
      </fixture>
    </fixtures>
  </consumer>
</consumers>
```

---

### Example D — Rotated fixture array

Fixtures mounted at a 45° angle on a diagonal set piece:

```xml
<fixture>
  <type>RGB</type>
  <start-address>101</start-address>
  <fixture-count>8</fixture-count>
  <fixture-channels>3</fixture-channels>
  <x>600</x><y>300</y><width>400</width><height>80</height>
  <rotation>45</rotation>
</fixture>
```

---

## 11. Troubleshooting

### No DMX output

- Verify the channel is running at 8-bit colour depth (default; check
  `caspar.config`).
- Check that the `<host>` resolves correctly. For sACN multicast, ensure
  `<host>` is empty (not set to `127.0.0.1`).
- Confirm the `<refresh-rate>` is ≥ 1.
- For sACN, the universe must be between 1 and 63999 (not 0).

### Fixtures receiving wrong colours

- Recheck `<x>`, `<y>`, `<width>`, `<height>` — these are pixel coordinates
  in the **channel's video format** space, **not** in the physical display
  resolution.
- Confirm the `<start-address>` is 1-based and does not overflow past
  channel 512 when multiplied by `fixture-count × fixture-channels`.

### sACN multicast not reaching fixtures

- Check that **IGMP snooping** is enabled on the managed switch.
- Increase `<multicast-ttl>` to `64` if the fixture is beyond one switch hop.
- As a diagnostic, switch temporarily to unicast by setting `<host>` to the
  fixture's IP.

### High CPU usage

- Reduce `<refresh-rate>` — 25 Hz is sufficient for most lighting rigs and
  imperceptible to the human eye.
- The colour sampling algorithm iterates over all pixels in each fixture's
  sampling region every refresh tick. Very large sampling regions (full-frame
  per fixture) at high refresh rates will increase CPU load.

### ArtNet nodes not responding to broadcast

- Some switches block UDP broadcast. Switch to unicast (`<host>` = node IP).
- Use the subnet-directed broadcast (e.g. `192.168.1.255`) rather than the
  global broadcast `255.255.255.255`.

### Multiple sACN sources fighting (flicker)

- Set a definitive priority: CasparCG defaults to `100`. If a console is also
  sending to the same universe, either set CasparCG higher (`120`) or lower
  (`80`) depending on which should be the master.
- If flicker persists, ensure both sources are not set to the same priority
  value — receivers may alternate between them.
