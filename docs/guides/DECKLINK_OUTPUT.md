# DeckLink SDI Output

> **State and measurements:** [`../features/decklink-output.md`](../features/decklink-output.md)
> **Implementation notes:** [`../architecture/DECKLINK_GPU_DIRECT_OUTPUT.md`](../architecture/DECKLINK_GPU_DIRECT_OUTPUT.md)
> **This document is how-to.** Per [`../README.md`](../README.md), measured figures live once in `features/`; a tolerance an operator acts on may appear here, the measurements behind it should not.

The primary SDI path. This fork rewrote it substantially — **2,382 lines against upstream across 37
files** — mostly to get the composited frame from the GPU to the card without a host round trip, and
to signal colour and HDR metadata correctly on the way out.

Upstream's DeckLink documentation still applies for the basics (device index, keyer, embedded audio,
latency). This guide covers **what this fork added**, and the traps that come with it.

---

## 1. The GPU output path — three orthogonal choices

The fork's central change: the frame can be read back, packed and delivered entirely on the GPU.
Three settings control it, and they are **independent of each other** — a common misreading is to
treat them as one "GPU mode".

```xml
<consumers>
    <decklink>
        <device>1</device>
        <gpu-readback-mode>auto</gpu-readback-mode>   <!-- how the frame leaves the mixer -->
        <gpu-transfer>auto</gpu-transfer>             <!-- how the packed frame reaches the card -->
        <gpu-pack>auto</gpu-pack>                     <!-- where v210/BGRA packing happens (OpenGL mixer) -->
    </decklink>
</consumers>
```

### `gpu-readback-mode` — how the frame leaves the mixer

| value | what it does |
| :--- | :--- |
| `auto` | **default.** Pick by what the mixer and hardware support |
| `cuda` | mixer texture → CUDA → V210 packed by a CUDA kernel |
| `vulkan` | packed by a Vulkan compute shader, then read back |
| `vulkan-dma` | as above, using a DMA path |
| `cpu` | read back to host memory and pack there — the fallback, and the slowest |

Also accepted as **`gpu-strategy`**, an older spelling kept for existing configs. Both name the same
setting; `gpu-readback-mode` is the one to use in new work.

### `gpu-transfer` — the final hop to the card

| value | what it does |
| :--- | :--- |
| `auto` | **default** |
| `dvp` | NVIDIA GPUDirect for Video — direct GPU-to-card transfer where the hardware allows |
| `copy` | an ordinary copy |

### `gpu-pack` — where packing happens, **OpenGL mixer only**

| value | what it does |
| :--- | :--- |
| `auto` | **default — and it means CPU here.** GL packing runs *only* when you write `gpu` explicitly |
| `gpu` | pack v210/BGRA in a GL compute shader |
| `cpu` | pack on the CPU with AVX2 |

**`auto` is not "pick the best" for this one setting.** `ogl_gpu_pack_eligible` returns false unless
`gpu-pack` is literally `gpu`, so leaving it alone keeps CPU packing. That is the opposite of
`gpu-readback-mode`, where `auto` does try the GPU first.

**And GL packing switches itself off for any subregion.** It requires `dest-x`, `dest-y`, `width`
and `height` to all be `0` on the **primary and every secondary port** — one destination offset
anywhere puts the whole consumer back on CPU packing, silently. (v210 group alignment; §3.)

**Leave all three on `auto` unless you are diagnosing something.** They exist to let you pin a path
when one misbehaves, and to make a measurement attributable.

> **A pinned mode is NOT a refusal — it falls back, with a warning.** This section said the opposite
> until 2026-08-27. Ask for `cuda` in a build without CUDA and you get
> *"CUDA gpu-readback-mode requested but CUDA not available, trying Vulkan"* and the Vulkan path; ask
> for Vulkan without Vulkan and you get CPU the same way. Each GPU strategy is also handed the CPU
> strategy as a runtime fallback, so it can still degrade after start-up.
>
> So pinning does **not** by itself make a measurement attributable — **read the log line**, which is
> the only thing that says which path actually ran.

---

## 2. Colour and HDR signalling

```xml
<decklink>
    <color-transfer>pq</color-transfer>       <!-- sdr | pq | hlg -->
    <hdr-metadata>
        <max-cll>1000</max-cll>
        <max-fall>400</max-fall>
        <min-dml>0.005</min-dml>
        <max-dml>1000.0</max-dml>
    </hdr-metadata>
    <vanc>
        <hdr-line>9</hdr-line>                <!-- VANC line for the HDR metadata block -->
    </vanc>
</decklink>
```

`<hdr-metadata>` is its own child block, spelled identically to the Vulkan-output and FFmpeg
consumers on purpose — the same four numbers describe the same mastering display.

`color-transfer` sets what the output **signals**. It does not convert the picture — the channel's
colour space and transfer determine the pixels, exactly as elsewhere in this fork.

**HDR metadata rides in VANC**, on the line given by `hdr-line`. Receivers differ about which line
they read; if a downstream device shows SDR from an HDR feed, that line number is the first thing
to check and it is not always the default.

### The rest of `<vanc>` — five keys beyond HDR

This guide documented `hdr-line` alone until 2026-08-27. The block carries **six** settings, and
several are unrelated to HDR:

| key | what it carries |
| :--- | :--- |
| `hdr-line` | the HDR mastering-display metadata block |
| `op47-line` | **OP-47 teletext / subtitles** |
| `op47-line-field2` | OP-47's second field, for interlaced signals |
| `op42-sd-line` | the SD fallback line for OP-42 (default `21`) |
| `scte104-line` | **SCTE-104 triggers** — ad insertion and splice messaging |
| `op47-dummy-header` | an optional dummy header string injected into OP-47 packets |

**There is no `<enable>`, for any of them.** Each feature turns on when **its own line number is
greater than zero**, and the whole VANC path turns on when the `<vanc>` block is merely *present*.
Two consequences: `<hdr-line>0</hdr-line>` silently disables HDR VANC, and omitting the block
entirely is how you disable VANC rather than setting something to false.

### Per-consumer OCIO view

```xml
<decklink>
    <ocio-display>sRGB</ocio-display>
    <ocio-view>Film</ocio-view>
</decklink>
```

**Both or neither.** Setting one without the other is refused rather than half-applied: a display
without a view is not a transform, and accepting one silently would render the channel's own view
while *looking* configured.

---

## 3. Genlock — waiting for the card's reference

```xml
<decklink>
    <wait-for-reference>auto</wait-for-reference>              <!-- auto | enable | disable -->
    <wait-for-reference-duration>10</wait-for-reference-duration>   <!-- seconds -->
</decklink>
```

| value | behaviour |
| :--- | :--- |
| `auto` | **default** — and it is also what any unrecognised value falls back to |
| `enable` (or `enabled`) | hold start-up until the card reports a locked reference |
| `disable` (or `disabled`) | start immediately, locked or not |

`wait-for-reference-duration` bounds the wait, in seconds.

**This is the only output-side reference setting.** There is no config block that nominates which
card supplies house reference — that is the card's own genlock input, set in Desktop Video. §5's
`<decklink-sync>` looks like it might be that and is not: it belongs to the *producer*.

---

## 4. Subregion output

A subregion copies a rectangle of the channel to a position on the DeckLink frame. All six numbers
are honoured, and since 2026-08-27 the **Vulkan compute readback places them on the GPU** — no host
round trip.

**On the OpenGL mixer, any subregion disables GL packing** (§1's `gpu-pack`): eligibility requires
`dest-x`, `dest-y`, `width` and `height` to be `0` on the primary *and every secondary* port. So a
subregion on one port moves the whole consumer to CPU packing without saying so.

```xml
<decklink>
    <subregion>
        <src-x>100</src-x>  <src-y>200</src-y>
        <width>640</width>  <height>360</height>
        <dest-x>114</dest-x> <dest-y>70</dest-y>
    </subregion>
</decklink>
```

| `gpu-readback-mode` | destination placement | measured |
| :--- | :--- | ---: |
| `auto` | **on the GPU** (resolves to `cuda`) | 62.92 dB |
| `cuda` | **on the GPU** | 62.92 dB |
| `vulkan` | **on the GPU** | 62.92 dB |
| `cpu` | honoured | 62.92 dB |
| `vulkan-dma` | **not possible** — coerced to `cpu`, with a warning | 62.92 dB |

Every mode gives the same picture; the difference is whether the readback stays on the GPU.

`vulkan-dma` is the one exception and it is a limit of the **mechanism**, not of effort: it copies
image→buffer with a single `region.imageOffset`, and a `VkBufferImageCopy` cannot express a
rectangle inside a larger frame. There is no shader in that path to place anything, so it falls back
to `cpu` and says so.

### Keep `dest-x` even

**An odd `dest-x` costs about 20 dB, on every readback mode including `cpu`.** Measured over the SDI
loopback, `640x360` from `(100,200)`:

| `dest-x` | CPU | Vulkan |
| :--- | ---: | ---: |
| 114 (even) | 62.92 dB | 62.92 dB |
| 115 (odd) | 42.79 dB | 42.79 dB |

This is **not** a bug in either path — they agree exactly. 4:2:2 pairs pixels horizontally, so an
odd destination x inverts the luma/chroma phase relative to the frame's grid. The tell is that all
three wire formats land within 0.6 dB of each other and 16-bit gains only +0.06 dB over 8-bit
instead of its usual +3.94: a precision-independent error swamping a precision-dependent one.

There is **no** 6-pixel alignment requirement, despite V210 packing six pixels per four words. The
shaders walk output groups and compute each from scratch, so a region boundary inside a group is
handled without any special case.

## 5. `<decklink-sync>` — an INPUT setting, listed here because nothing else documents it

**This is not an output option.** It supplies the default `SYNC_GROUP` / `SYNC_PEERS` for a
DeckLink **producer**, so several capture cards deliver frames as one synchronised set. It has no
effect on the consumer, and there is no config block that chooses which card carries house
reference — for output timing see `<wait-for-reference>` in §3.

> Renumbered 2026-08-27: this pointer said §3 while §3 was the subregion section, because
> `<wait-for-reference>` was referenced here and documented nowhere. It now exists.

```xml
<configuration>
    <decklink-sync>
        <device-1>
            <group>1</group>
            <peers>2</peers>
        </device-1>
    </decklink-sync>
</configuration>
```

`device-N` matches the producer's device index, and it is a **subtree**, not a scalar. Read at
`decklink_producer.cpp:1790-1806`, and only when the `PLAY` line did not already give
`SYNC_GROUP` — an explicit parameter wins, and a `group` of `0` means no grouping. It is also
skipped entirely for `device_index` 0.

The equivalent per-`PLAY` form, which is what the config defaults:

```
PLAY 1-10 decklink 1 SYNC_GROUP 1 SYNC_PEERS 2
PLAY 1-20 decklink 2 SYNC_GROUP 1 SYNC_PEERS 2
```

> **A config element whose key is assembled at runtime is invisible to every mechanical check.**
> This one is built by string concatenation —
> `L"configuration.decklink-sync.device-" + device_index` — so no audit that greps for a literal
> config name can find it, in either direction: a sweep for undocumented elements misses it, and a
> sweep for documented-but-unread elements reports it as unread. Both happened here.

---

## 6. Bring-up order that avoids the usual confusion

1. Start with **everything on `auto`** and a plain SDR channel. Confirm a picture on the card.
2. Set `color-transfer` and, if HDR, check the receiver actually reads `hdr-line` — change the line
   before suspecting the metadata.
3. Only then pin `gpu-readback-mode` if you are chasing performance, and change **one** of the three
   settings at a time. They are orthogonal, so changing two makes the result unattributable.
4. If you need a subregion, keep `dest-x` **even** (§4). `auto`, `cuda`, `vulkan` and `cpu` all
   place it correctly; **only `vulkan-dma`** falls back to the CPU path, and it says so. On the
   OpenGL mixer a subregion also disables `gpu-pack` (§1).

---

## 7. What is measured, and what is not

Verified by `sdi-output` (pixels, `--hdr-metadata` for the HDR block), `signalling` (colour and HDR
static metadata read back from the card — its ancillary-data checks live in `core/anc_check.py` and
are **not** a subcommand of their own), and `sdi-input` / `decklink-input-cost` for the input
direction. The figures live in
[`../features/decklink-output.md`](../features/decklink-output.md), which owns them.

**Not measured:**

* **The four output strategies are never compared against each other on one clip.** Each is
  measured; nothing asserts they agree, so a difference between `cuda` and `vulkan` readback would
  not be caught.
* **Most configuration axes are unswept** — which DeckLink options select genuinely different code
  and which the harness cannot reach is tracked separately, and several are unreachable from it.
* **`bluefish`**, the other SDI card, has no coverage and no guide.
