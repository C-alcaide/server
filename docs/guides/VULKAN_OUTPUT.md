# Vulkan Output — direct display output

Presents a channel straight to a display through Vulkan, bypassing the window compositor. Built for
**LED processors and projectors driven from HDMI/DisplayPort** rather than SDI, with control over HDR
signalling, sync groups and EDID.

Read [§6](#6-what-works-on-this-platform-and-what-does-not) before planning an HDR install — on
Windows 10 the HDR path is unreachable, and the server does not refuse the configuration.

---

## 1. Minimal configuration

```xml
<consumers>
    <vulkan-output>
        <gpu>0</gpu>
        <device>1</device>
    </vulkan-output>
</consumers>
```

`gpu` is the adapter index, `device` the output on that adapter. Enumerate both with:

```
INFO VULKAN_OUTPUT
```

which lists every display it can drive — gpu index, output index, GPU name, display name, resolution
and whether the card is a `pro` or `consumer` tier.

Add `<identify-on-start>true</identify-on-start>` to make the output announce itself visually when
it starts, which is how you tell four identical screens apart.

---

## 2. HDR signalling

```xml
<vulkan-output>
    <transfer>pq</transfer>                       <!-- sdr | pq | hlg -->
    <display-peak-luminance>1000</display-peak-luminance>
    <edid-auto-hdr>true</edid-auto-hdr>
</vulkan-output>
```

| setting | meaning |
| :--- | :--- |
| `transfer` | what the output **signals**: `sdr`, `pq` or `hlg` |
| `eotf` | transfer function override — `linear`, `pq`/`st2084` |
| `display-peak-luminance` | the display's peak, in cd/m² |
| `edid-auto-hdr` | read the display's own HDR limits from its EDID and use those |
| `auto-tone-map` | tone-map operator for content brighter than the display |

**`edid-auto-hdr` overrides what you configured.** When it is on, the EDID's maximum luminance
replaces your `max-cll`. So a request for 1000 nits on a 600-nit panel becomes 600 — which is
usually what you want, and is worth knowing before you wonder why your number changed. Check the
effective value with `vulkan-output-signalling` (§7) rather than assuming.

**`<gamut>` is accepted and does nothing.** The per-consumer colour conversion pass is disabled
because the **channel's** colour space and transfer already determine every pixel in the
framebuffer; running a second conversion here would double-convert. Set the **channel's**
`<color-space>` instead. The server warns about this at startup, and reports `gamut-inert` in its
state.

---

## 3. Sync groups — driving several displays as one

```xml
<vulkan-output>
    <sync-group>1</sync-group>
</vulkan-output>
```

Outputs sharing a `sync-group` present together. Use it when several displays form one image — an
LED volume made of multiple processors, or a multi-projector blend — so a moving edge does not tear
across the seam.

On Quadro/RTX Pro hardware with a sync board, `INFO`'s state also reports the Quadro Sync status:
whether a board is present, whether timing is locked, whether house sync is connected, and whether
this output is master or slave.

---

## 4. Behaviour when the display disappears

```xml
<vulkan-output>
    <on-disconnect>retry</on-disconnect>   <!-- retry | hold | black -->
</vulkan-output>
```

| value | on cable pull or display power-off |
| :--- | :--- |
| `retry` | **default** — keep trying to re-acquire |
| `hold` | keep presenting the last frame |
| `black` | present black |

For a show, `retry` is almost always right: a display that comes back should resume without an
operator touching anything. `hold` is for cases where a frozen frame is less alarming than a flash.

---

## 5. EDID emulation

```xml
<vulkan-output>
    <edid-emulation>true</edid-emulation>
    <persist-edid>false</persist-edid>
</vulkan-output>
```

Presents a synthetic EDID to the GPU so an output can be driven at a mode the attached device does
not advertise — routine with LED processors, which often report a mode that is not the one you need.

**`persist-edid` writes it into the driver so it survives a restart.** Leave it `false` unless you
mean that: a persisted EDID outlives CasparVP, so a machine can be left claiming a display
capability that nothing on it is configured for any more.

**Nothing asserts that an EDID injection actually happened.** The request is made and the server
reports its resolved state; no check compares that against the driver.

---

## 6. What works on this platform, and what does not

**HDR is unreachable on Windows 10.** `VK_KHR_display` — the direct-scanout extension — needs
Windows 11 build 22000 or later. On Windows 10 the consumer falls back to a fullscreen-exclusive
path where **no HDR surface is offered**, so:

```
requested:  transfer = pq, gamut = bt2020, max-cll = 1000
delivered:  surface = bgra8, colour space = srgb_nonlinear, hw-hdr = false
```

The configuration is accepted, the picture is correct, and **nothing HDR is signalled**. The server
logs the fallback at startup; §7 is how you check it deliberately.

Two independent blockers on this rig, worth separating when planning: the OS version, **and** no HDR
display attached. Fixing one does not get you HDR.

---

## 7. Check what it is actually signalling

```
python cli.py vulkan-output-signalling --server <casparcg.exe>
```

Reports, per requested transfer, the surface format and colour space the swapchain **actually got**,
plus `hw-hdr`, MaxCLL/MaxFALL and whether `<gamut>` was inert. It gates the config round trip and
internal contradictions, and *names* rather than fails the platform degradations above.

This is the only way to see the difference between "HDR configured" and "HDR signalled" — a
distinction that leaves the picture looking correct either way.

State and figures live in [`../features/vulkan-output.md`](../features/vulkan-output.md);
implementation notes in [`../architecture/VULKAN_OUTPUT.md`](../architecture/VULKAN_OUTPUT.md).

---

## 8. Also available

| setting | meaning |
| :--- | :--- |
| `video-mode` | force a mode rather than taking the channel's |
| `delay`, `delay-ms` | present N frames or milliseconds late, for lip-sync against another path |
| `buffer-depth` | presentation queue depth |
| `display-name` | select the output by name instead of index |
| `display-blanker` | blank other outputs on the adapter |

`delay`, `delay-ms` and `buffer-depth` interact, and their combined behaviour is **not documented in
one place** — treat a combination of the three as something to measure on your rig rather than to
reason about.
