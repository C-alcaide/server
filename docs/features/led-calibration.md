# LED calibration — channel-master display LUT

> **State:** shipped
> **Modules:** **not a module** — the channel-master LUT stage declared in
> `src/core/mixer/image/image_mixer.h` and implemented in both
> `src/accelerator/{ogl,vulkan}/image/image_mixer.cpp`
> **Architecture:** none, deliberately — one 3D LUT applied to the composite; the structural facts are the two claims in §3, and both are measured
> **Guide:** [`../guides/LED_CALIBRATION.md`](../guides/LED_CALIBRATION.md)
> **Commands:** `CALIBRATION` (fork-only)
> **Coverage:** `calibration` — 1 LSB, 32/32, byte-identical on both mixers

A single **display-to-display 3D LUT applied to the final composited channel output**, so every
consumer attached to the channel receives the corrected pixels. Intended for whole-LED-wall
calibration, typically from a LUT solved by [OpenVPCal](https://github.com/Netflix-Skunkworks/OpenVPCal).
CasparVP *applies* the LUT; it does not perform the colorimetric solve.

> **Not every feature has a directory.** Calibration is a *command* feature living in the mixer, so
> any inventory built by walking `src/modules/` cannot see it. Worth knowing before trusting a
> completeness check that enumerates by module — `../features/README.md` carries the general form.

---

## 1. What is implemented today

```
CALIBRATION <channel> LUT <file.cube> [strength]
CALIBRATION <channel> CLEAR
CALIBRATION <channel> BYPASS <0|1>
CALIBRATION <channel> [INFO]
```

Whole-screen correction: one channel drives one LED screen and carries one calibration LUT.
Individual panels or tiles are **not** addressed — that remains the LED processor's job.

**Not the same thing as `MIXER LUT3D`**, which is per *layer*. This is per *channel*, after the
composite. §3 measures that difference rather than asserting it.

---

## 2. Verification — and why 1 LSB is defensible here

Gated at **1 LSB, 32/32 cases, byte-identical on both mixers** (`core/calibration.py`).

**Every expectation is computed from the frame the server itself rendered with no calibration LUT.**
The producer, the colour conversion, the OETF and the consumer are therefore identical on both sides
of the comparison and cancel exactly, so a disagreement can only be the calibration pass. That is
what makes a 1 LSB gate defensible without modelling the whole pipeline.

Eight cases per patch. Three — `strength 0`, `BYPASS 1` and `CLEAR` — are identity checks against a
frame the server produced, so a wrong model cannot satisfy them. All three return exactly **0.00 LSB**.

Two controls are asserted by the verdict rather than printed, because 32/32 against a model is
otherwise compatible with a command that did nothing:

| control | why it is needed | measured |
| :--- | :--- | ---: |
| the LUT must **move** the picture | else the model is applied to a baseline equal to the measurement | ≥ 61.0 LSB |
| the two LUTs must **differ** | `set_calibration_lut` invalidates the still-frame cache **by hand**; a stale fingerprint would replay the previous LUT's frame, and a close pair would hide it inside the gate | ≥ 27.0 LSB (needs ≥ 8.0) |

---

## 3. The two claims that distinguish it from `MIXER LUT3D`

Neither is visible over one layer through one consumer, so each is its own case and both are part of
the verdict.

**Applied once to the composite, not once per layer.** Two layers blended with `MIXER OPACITY`, and a
non-linear LUT separates the hypotheses: `LUT(blend(a,b))` against `blend(LUT(a), LUT(b))`. Both
models are built from **measured** frames — each layer alone, and the blend itself — so the server's
own blend appears on both sides and cancels; nothing assumes a blend domain or a rounding rule. The
two sit **17.9 LSB** apart, and the answer is **channel-master** on both mixers, at **0.35 LSB**
against **18.12 LSB** for the per-layer alternative.

**Every consumer sees it.** A LUT applied only to the IMAGE consumer would satisfy every other case,
since IMAGE is how they are all captured — so this reads the **screen** window instead, a consumer
that is not the capture path. **0.32 LSB** from the model, with the LUT moving the screen picture
**88 LSB** (one that moved nothing would make the claim unfalsifiable). No loosened gate: screen
holds 1 LSB on this rig.

---

## 4. Known gaps

Read §3 as *"channel-master, demonstrated on a second consumer"* rather than *"every consumer"*.

1. **DeckLink and NDI are not covered.** `cli.py consumer-view` can drive DeckLink looped back to a
   second card at a 12 LSB gate; this battery reads screen only, so "every consumer" is
   demonstrated on two of five.
2. **One layer composited** in most cases — the two-layer case exists only for the blend-domain
   claim in §3.
3. **`strength` is measured at 0 and 1**, not swept across the middle.
4. **No panel- or tile-level addressing**, by design — that is the LED processor's job, and nothing
   here would catch a config that expected otherwise.
5. **The solve is not ours.** A wrong LUT from OpenVPCal is applied faithfully, and nothing here
   would know.
