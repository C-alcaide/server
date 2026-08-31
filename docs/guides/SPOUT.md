# Spout — sharing video with other Windows applications

> **State and measurements:** [`../features/spout.md`](../features/spout.md)
> **This document is how-to.** Per [`../README.md`](../README.md), measured figures live once in `features/`; a tolerance an operator acts on may appear here, the measurements behind it should not.

Spout shares GPU textures between applications on the same machine, with no encode, no file and no
network. CasparVP is **both ends**: it can publish a channel as a Spout sender, and it can play
another application's Spout sender as a producer.

Typical use is a media server feeding Notch, Resolume, TouchDesigner, Unreal or a VJ tool — and the
reverse, pulling a generative scene from one of those into a CasparVP layer for compositing,
grading and projection.

Windows only.

---

## 1. Publish a channel (consumer)

```
ADD 1 SPOUT
ADD 1 SPOUT MyChannel
ADD 1 SPOUT MyChannel MAX_WIDTH 1920 MAX_HEIGHT 1080
```

| parameter | meaning |
| :--- | :--- |
| *(name)* | the **sender name** other applications will see. Optional — omitted, it is **`CasparCG Spout`** |
| `MAX_WIDTH` | maximum shared-texture width; the picture is **downscaled** to fit |
| `MAX_HEIGHT` | maximum shared-texture height; likewise |
| `BIT_DEPTH` | `16` publishes a 16-bit channel at 16 bits per component; `8` forces 8-bit even on a 16-bit channel. Omitted, the channel's own depth is followed — which for an 8-bit channel is 8 either way. **See the warning below before using 16** |

### 16-bit senders, and why they are opt-in

`ADD 1 SPOUT name BIT_DEPTH 16` publishes `DXGI_FORMAT_R16G16B16A16_UNORM` instead of
8-bit BGRA. Measured through a real receiver, a 16-bit channel at 1080p: **256 distinct
levels per component at 8-bit against 1920 at 16-bit** — 1920 being the test ramp's width,
so every column resolved and the fixture ran out before the depth did.

**Many Spout receivers assume RGBA8 and will not read a 16-bit sender at all.** That is
why this is opt-in rather than automatic: following the channel silently would break
working installations. Check your receiver before switching a live sender.

**The CPU fallback is always 8-bit**, whatever is asked. Spout's `SendImage` accepts only
8-bit RGBA/BGRA/RGB/BGR, so a channel that has fallen off the GPU path is truncated by the
SDK rather than by choice. `spout/published-depth` reports what the live path actually
sent — check it rather than assuming the request was honoured:

```
INFO 1                     # spout/published-depth, spout/requested-depth,
                           # spout/depth-truncated
```

### What Spout cannot tell a receiver

| property | reaches the receiver? |
| :--- | :--- |
| raster and bit depth | **yes** — carried in the sender info as a DXGI format |
| gamut (BT.709, BT.2020) | **no** — Spout has no field for it |
| transfer function (SDR, PQ, HLG) | **no** — Spout has no field for it |
| mastering / MaxCLL metadata | **no** — Spout has no field for it |

So an HDR channel published over Spout arrives as **pixels a receiver cannot interpret**.
The consumer reports `spout/color-space` and `spout/color-transfer` so a controller reading
server state can learn what was rendered, alongside `spout/color-signalled`, which is
**always false**. Those two describe what the server rendered and never what the receiver
was told; do not read them as signalling. A receiver that needs to know must be told out of
band.

Remove it like any consumer:

```
REMOVE 1 SPOUT
```

**Name the sender explicitly in any real setup.** The receiving application picks senders by name,
so an unnamed sender means whoever is configuring the other end has to discover what it was called —
and the name is the only thing tying the two applications together.

**`MAX_WIDTH` / `MAX_HEIGHT` are a bounding box, and the picture IS resampled to fit it.** The two
together give one aspect-preserving scale factor — whichever limit binds harder wins — applied
through swscale, with the result rounded **down to an even** width and height. Nothing is cropped and
nothing is letterboxed.

Two consequences:

* **Neither ever upscales.** A cap larger than the channel raster is ignored, so `MAX_WIDTH 3840` on
  a 1080p channel sends 1920×1080, not a stretched 4K frame.
* **A cap costs a resample per frame**, so leave both out unless the receiver actually needs a
  smaller texture. Without them the share is the channel's native raster and format-conversion only,
  which the consumer splits across four threads; with them it is single-threaded on the assumption
  that the output is small.

---

## 2. Receive another application's sender (producer)

Three spellings, all equivalent — use whichever your automation reads best:

```
PLAY 1-1 [SPOUT] SenderName
PLAY 1-1 spout://SenderName
PLAY 1-1 SPOUT SenderName
```

**Type them exactly as written: all three are case-sensitive, and they do not agree on case.**
AMCP upper-cases the command name only, never the parameters, so the producer sees what you typed.
`[SPOUT]` and `SPOUT` must be upper-case and `spout://` must be lower-case — `[spout]`, `spout`
and `SPOUT://Name` all match nothing.

**A mistyped spelling and an absent sender fail differently, and telling them apart saves the
guessing:**

| what you typed | what happens |
| :--- | :--- |
| a spelling that matches no producer | an explicit error — *"No match found for supplied commands. Check syntax."* — because no producer claims it and the registry then refuses |
| a correct spelling, sender not publishing | **no error, a layer with no picture** (§4). The sender may legitimately appear later |

So if the reply was an error, the problem is your syntax; if the layer is simply black, the problem
is at the other application.

The `spout://` form is convenient in playlists and rundowns where a single string has to carry the
whole source.

Once playing it is an ordinary layer: grade it, transform it, key it, route it into projection.

```
PLAY 1-1 spout://NotchOutput
MIXER 1-1 OPACITY 0.5
MIXER 1-1 FILL 0.25 0.25 0.5 0.5
```

---

## 3. Worked example — round trip through Notch

Send the programme channel out, bring the treated result back on a second layer:

```
ADD 1 SPOUT ProgrammeOut
PLAY 1-2 spout://NotchReturn
```

**Do not point a receiver at the sender on the same channel.** The channel would be reading its own
output while producing it — a feedback loop, and the shape of it depends on frame timing rather
than on anything you configured. Use a second channel if you need a loop deliberately.

---

## 4. Things worth knowing before a show

**No configuration elements.** Everything is per-command; there is nothing in `casparcg.config` to
set up, and nothing to restart for.

**Both applications must be on the same GPU.** Spout shares a texture through a D3D11 shared
handle, and such a handle is adapter-bound. On a two-GPU machine, a sender on one adapter and a
receiver on the other will fail rather than run slowly — check which adapter each application is
using before assuming the name is wrong.

**The sender exists only while the consumer is attached.** `REMOVE 1 SPOUT` makes it disappear from
every receiver's list immediately, which looks to them like the source vanished.

**A missing sender is not an error on this side.** Playing `spout://Name` when nothing is publishing
that name gives you a layer with no picture rather than a refusal, because the sender may legitimately
appear later. If a receiving layer is black, check the *other* application is running and publishing
under exactly that name — spelling and case included.

---

## 5. What is not covered

**No battery drives Spout**, in either direction. Nothing measures the picture through a send or a
receive, so the round trip is documented and unmeasured. State and any figures live in
[`../features/spout.md`](../features/spout.md).

**Alpha behaviour through the share is not established here.** Spout can carry an alpha channel and
CasparVP layers are premultiplied; which convention the shared texture ends up in has not been
measured, so it is not written down as advice.
