# DeckLink SDI Output

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
| `auto` | **default** |
| `gpu` | pack v210/BGRA in a GL compute shader |
| `cpu` | pack on the CPU with AVX2 |

**Leave all three on `auto` unless you are diagnosing something.** They exist to let you pin a path
when one misbehaves, and to make a measurement attributable. A pinned value that the hardware cannot
honour is a refusal, not a silent fallback — which is deliberate, because a silent fallback is how a
"GPU path" measurement ends up describing the CPU one.

---

## 2. Colour and HDR signalling

```xml
<decklink>
    <color-transfer>pq</color-transfer>       <!-- sdr | pq | hlg -->
    <vanc>
        <hdr-line>9</hdr-line>                <!-- VANC line for the HDR metadata block -->
    </vanc>
</decklink>
```

`color-transfer` sets what the output **signals**. It does not convert the picture — the channel's
colour space and transfer determine the pixels, exactly as elsewhere in this fork.

**HDR metadata rides in VANC**, on the line given by `hdr-line`. Receivers differ about which line
they read; if a downstream device shows SDR from an HDR feed, that line number is the first thing
to check and it is not always the default.

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

## 3. Subregion output — and a trap that will cost you a show

A subregion lets one channel drive part of a larger raster. On this fork's **GPU readback paths only
the source origin is honoured**:

| setting | GPU paths | effect |
| :--- | :--- | :--- |
| `src-x`, `src-y` | **honoured** | reach the shaders as push constants, and the DMA path as an image offset |
| `dest-x`, `dest-y` | **silently dropped** | |
| `width`, `height` | **silently dropped** | |

So the wire carries **the whole source from the origin, placed at 0,0**. If you need real
destination placement or cropping, use `gpu-readback-mode cpu`, which honours all six — and accept
its cost.

This is the single most surprising behaviour in the fork's DeckLink path: the config is accepted,
the output looks plausible, and four of the six numbers you set did nothing.

---

## 4. Genlock / sync devices

```xml
<configuration>
    <decklink-sync>
        <device-1>1</device-1>
    </decklink-sync>
</configuration>
```

Names the card whose reference is used for timing. Relevant when several cards are in one machine
and only one carries house reference.

---

## 5. Bring-up order that avoids the usual confusion

1. Start with **everything on `auto`** and a plain SDR channel. Confirm a picture on the card.
2. Set `color-transfer` and, if HDR, check the receiver actually reads `hdr-line` — change the line
   before suspecting the metadata.
3. Only then pin `gpu-readback-mode` if you are chasing performance, and change **one** of the three
   settings at a time. They are orthogonal, so changing two makes the result unattributable.
4. If you need a subregion, read §3 first and decide whether you can live with origin-only.

---

## 6. What is measured, and what is not

Verified by `sdi-output` (pixels, `--hdr-metadata` for the HDR block), `signalling` (colour and HDR
static metadata read back from the card), `anc-check` (ancillary data), and `sdi-input` /
`decklink-input-cost` for the input direction. The figures live in
[`../features/decklink-output.md`](../features/decklink-output.md), which owns them.

**Not measured:**

* **The four output strategies are never compared against each other on one clip.** Each is
  measured; nothing asserts they agree, so a difference between `cuda` and `vulkan` readback would
  not be caught.
* **Most configuration axes are unswept** — which DeckLink options select genuinely different code
  and which the harness cannot reach is tracked separately, and several are unreachable from it.
* **`bluefish`**, the other SDI card, has no coverage and no guide.
