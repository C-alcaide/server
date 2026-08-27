# ACES Metadata File (AMF) support — design study

> **Status:** SHIPPED — verified 2026-08-27 — implemented in `180b2fb41`; the research below is why
> it looks as it does. Command at `AMCPCommandsImpl.cpp:5676`, covered by `cli.py amf`.
> **Falsifier:** `AMF` is registered — absence of that command would mean this status is wrong.

Kept in `plans/` rather than moved to `../deprecated/` because a source comment cites it; current
state is in [`../features/colour-grading-and-ocio.md`](../features/colour-grading-and-ocio.md).

Everything below was measured against the
pinned config with PyOpenColorIO 2.5.2, not reasoned about.

An AMF is the document that ties an ACES pipeline together: which input transform, which
look(s), which output transform. If the server could read one, configuring a channel for a
show would be one command instead of three, and the values would come from the show's own
metadata rather than from someone retyping them.

---

## The question that decides whether this is worth building

**Is the mapping from AMF transform IDs to OCIO objects mechanical, or a hand-maintained
table?** A table would be a second copy of knowledge with no oracle to check it against —
exactly what the OCIO integration exists to avoid, since the whole point is that OCIO *is*
the reference.

**It is mechanical.** OCIO configs carry a purpose-built field:

```yaml
looks:
  - !<Look>
    name: ACES 1.3 Reference Gamut Compression
    interchange:
      amf_transform_ids: |
        urn:ampas:aces:transformId:v2.0:Look.Academy.ReferenceGamutCompress.a2.v1
```

Reachable through the API as `getInterchangeAttribute("amf_transform_ids")` on `ColorSpace`,
`Look` and `ViewTransform`. **86 transform IDs resolve from the pinned studio config.**

> Two of my own readings were wrong on the way to that, and both are worth recording because
> they are the shape of mistake this measurement invites.
>
> **First**, `getDescription()` and `getAliases()` carry no transform IDs, so an initial check
> concluded there were none and that a hand-maintained table would be needed. Serialising the
> config found 134 occurrences of `urn:ampas`. *The API surface you happen to check is not the
> config.*
>
> **Second**, a resolution table built with `setdefault` while iterating colour spaces before
> view transforms reported "view_transform: 2". The real number is 11+ — the same ID appears
> on more than one kind of object, and taking the first hit hid exactly the relationship the
> output mapping depends on.

---

## The three hops, all verified

| AMF node | resolves to | drives |
| :--- | :--- | :--- |
| `inputTransform` | a **colour space** | `MIXER <ch>-<layer> OCIO "<space>"` |
| `lookTransform` | a **look** | `OCIO_LOOK <ch> "<look>"` |
| `outputTransform` | a **colour space** *and* a **view transform** | `OCIO_DISPLAY <ch> "<display>" "<view>"` |

Worked example, resolved end to end against the pinned config:

```
input   urn:…:CSC.Academy.ACEScct_to_ACES.a2.v1
          -> colorspace     ACEScct
look    urn:…:Look.Academy.ReferenceGamutCompress.a2.v1
          -> look           ACES 1.3 Reference Gamut Compression
output  urn:…:Output.Academy.Rec709-D65_100nit_in_Rec709-D65_sRGB-Piecewise.a2.v1
          -> colorspace     sRGB - Display                      (the DISPLAY)
          -> view_transform ACES 2.0 - SDR 100 nits (Rec.709)   (the VIEW)
```

**The output hop is the one that looked hard and is not.** `OCIO_DISPLAY` needs a
display *and* a view, while the AMF gives one ID. That ID appears on exactly one colour
space and one view transform, and the colour space's name is also a display name — so the
pair is determined, not guessed. No heuristic, no "pick the first view".

---

## How it would be verified

The same equivalence that gates `MIXER CDL_FILE`, and for the same reason: **applying an AMF
must render byte-identically to issuing the three commands by hand.** No colour model, no
tolerance — both sides are frames the server produced, and every transform involved is
already gated by `ocio`, `ocio-look` and `ocio-display`.

What that cannot catch is a wrong *resolution* — an AMF pointing at ACEScct that loaded
ACEScc would be self-consistent. So the battery needs a second axis: two AMFs differing in
exactly one node must render differently, by more than the gate. That is the same
"the id was honoured" control `cdl-file` uses for `.ccc` ids.

---

## What is still open

* **Which AMF elements to honour.** The format also carries clip metadata, archived
  `<aces:pipelineInfo>`, and optional inverse-output transforms. Only the three above have an
  obvious mapping onto commands that exist.
* **Atomicity.** Three settings applied from one file should not leave a channel half
  configured if the third fails. The natural shape is resolve-everything-then-apply, which is
  what `OCIO_LOOK` already does for its single value.
* **`<aces:lookTransform>` with a file reference** rather than a transform ID — AMF permits a
  CLF/CTF path. OCIO reads those formats, so it is tractable, but it is a different code path
  from ID resolution and would need its own case.
* **Where the AMF's look lands.** `OCIO_LOOK` is channel-level and requires `OCIO_DISPLAY`;
  an AMF sets both, so the ordering is forced — display first, then look.

## Cost

Small, and unusually well-bounded for the value: an XML parse (boost::property_tree is
already used for the server config), a resolution function beside `looks()`/`has_look()` in
`ocio_config.cpp`, one AMCP command that applies three existing setters, and a battery that
is mostly orchestration because the equivalence needs no model.
