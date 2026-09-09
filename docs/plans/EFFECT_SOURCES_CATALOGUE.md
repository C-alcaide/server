# Where to get effects — ISF, OFX, HTML, and the licensing that decides it

> **Status:** CATALOGUE, compiled 2026-09-09. **Nothing here has been loaded into this server.**
> Star counts, dates and licences are from the GitHub API and vendor pages on that date and go
> stale — a licence especially, since a repository can gain or change one.
> **Falsifier:** this file assumes `src/modules/isf` and `src/modules/ofx` exist and load
> third-party content. If either module goes, the corresponding section is dead.
>
> **This file is not legal advice.** It records what each source *states* about its own terms. For
> anything going to air, the licence in the file header wins over anything written here.

Companion to `WEBGPU_IN_THE_HTML_PRODUCER.md`, which covers only **open-source HTML/WebGPU
engines**. This one covers the rest: gratis-but-proprietary options, and the ISF and OFX
collections the fork can already load.

---

## 1. Read this first: the licence is the hard part, not the finding

Effects are easy to find and easy to use unlawfully. Two traps account for most of it.

**Shadertoy's default licence is CC BY-NC-SA 3.0 — *no commercial use whatsoever*.** Every shader
where the author declared nothing else is unusable in a paid show. Shadertoy's own terms say the
author owns the shader and chooses the licence, and that this default applies in the absence of
one. This matters here because Shadertoy is the largest single body of GLSL on the internet and
converting one to ISF is a ten-minute job — the conversion does not change the licence.

**"Free" from a vendor usually means "free during the trial".** Boris FX Continuum, for instance,
is a commercial OFX bundle whose effects are available at no cost only for the trial period. That
is not a free effect library; it is a demo.

So the taxonomy that matters is not open-source vs proprietary. It is:

| | commercial use | notes |
| :--- | :--- | :--- |
| **MIT / BSD / CC0** | yes | the only category to reach for without thinking |
| **GPL-2.0 / GPL-3.0** | yes, with obligations | fine for a *plugin* loaded over a stable C API; see §3 |
| **CC BY-NC / CC BY-NC-SA** | **no** | the Shadertoy default, and much of the ISF community site |
| **no licence file at all** | **no** | all rights reserved by default. Absence is not permission |
| **vendor "free"** | read the EULA | often trial-limited, often host-limited |

---

## 2. ISF — the strongest option, and it is already supported

The fork's `src/modules/isf` implements ISF 2.0 including multi-pass `PASSES` and `IMPORTED`
images, so it takes real-world shaders rather than only trivial ones. Guides:
`../guides/ISF_USER_AND_SHADER_GUIDE.md`, feature doc `../features/isf-and-openfx.md`.

| source | size | licence | commercial | notes |
| :--- | ---: | :--- | :--- | :--- |
| [Vidvox/ISF-Files](https://github.com/Vidvox/ISF-Files) | **327 `.fs` files** (371 entries) | **MIT** | **yes** | The standard collection: colour adjustment, blurs, distortion, geometry, stylize. 99★, pushed 2026-06-04. **Start here.** |
| [vidvox/isf](https://github.com/vidvox/isf) | spec | none stated | — | The specification itself, for writing your own |
| [editor.isf.video](https://editor.isf.video/) | community, large | **per shader** — CC *or* MIT | **check each** | The community site. CC-licensed entries cannot go in a paid show; MIT ones can. The licence is normally in a comment at the top of the shader |
| [Ethereios/ISF-shaders](https://github.com/Ethereios/ISF-shaders) | small | **CC0-1.0** | **yes** | 8★, "advanced ISF shaders with tunable parameters". CC0 is public domain — no attribution needed |
| [bareimage/ISF](https://github.com/bareimage/ISF) | 28★ | **NO LICENCE FILE** | **no** | Advertises "converted shaders" with persistent buffers. No licence means all rights reserved, and *converted* is the Shadertoy problem in §1. Read, do not ship |
| VJ-shop bundles (e.g. [headsta.sh](https://www.headsta.sh/shop)) | claims 868+ | unstated per item, includes "Shadertoy ports" | **no, until proven** | A bundle advertising Shadertoy ports is advertising CC-BY-NC-SA content unless each author relicensed. Treat the count as a warning, not a feature |

**The practical recommendation for ISF is short:** take the 327 MIT files from `Vidvox/ISF-Files`
as the working library, add CC0 sources freely, and treat everything else as needing a per-file
licence check before it touches a show.

## 3. OFX — free and open, with one licence nuance worth understanding

`src/modules/ofx` hosts OpenFX plugins; guide at `../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md`.

| source | size | licence | notes |
| :--- | ---: | :--- | :--- |
| [NatronGitHub/openfx-misc](https://github.com/NatronGitHub/openfx-misc) | 330★ | **GPL-2.0** | The big general collection — generators (CheckerBoard, ColorBars, ColorWheel, Constant, Solid), time effects (AppendClip, Deinterlace, FrameBlend, FrameHold), transforms, colour. Pushed 2025-04-29 |
| [NatronGitHub/openfx-io](https://github.com/NatronGitHub/openfx-io) | 28★ | **GPL-2.0** | Readers and writers. Pushed 2026-07-24 |
| [NatronGitHub/openfx-arena](https://github.com/NatronGitHub/openfx-arena) | 52★ | **GPL-2.0+**, some parts LGPL-2.1+ | ImageMagick-based: text, Magick, OCL, audio categories. Pushed 2026-07-16. GitHub reports NOASSERTION; the README states GPL2+ |
| [AcademySoftwareFoundation/openfx](https://github.com/AcademySoftwareFoundation/openfx) | 569★ | **BSD-3-Clause** | The API itself plus sample plugins. The standard is ASWF-governed, which is the same body as OpenColorIO — see `CLAUDE.md` on where numbers come from |

**The nuance:** these are **GPL-2.0**, and CasparCG is **GPL-3.0**, which are not compatible for
*linking into one binary*. That is not what happens here — an OFX plugin is a separate binary
loaded at runtime across a stable C ABI the host does not derive from, which is the arrangement
the OFX standard exists to define. It is the same shape as loading a GPL plugin into any host.
**Worth a lawyer's five minutes before shipping a bundled plugin set**, and not worth worrying
about for using them in-house.

**One warning carried from openfx-misc's own README:** it says its plugins should *not* be used in
DaVinci Resolve or Fusion, because some use OFX features Blackmagic's hosts do not implement.
Whether this fork's host implements them is **unknown and untested** — see §6.

## 4. HTML — the gratis-but-proprietary side

`WEBGPU_IN_THE_HTML_PRODUCER.md` covers the open-source engines (three.js, PlayCanvas, vgpu,
Babylon, and the 3DGS renderers). The proprietary-but-free category is **thinner than it looks**,
and mostly resolves into three things:

* **Free tiers and trials of commercial bundles** — Boris FX Continuum and similar. Trial-limited;
  not a library. Do not build a show on one.
* **Font and asset CDNs** — Google Fonts is the one that matters, and note the artifact CSP rules
  in this tree already assume it. Check each family's licence; most are OFL, which is permissive.
* **Effect *code* on sites with restrictive or unstated terms** — Shadertoy (§1), CodePen,
  Codrops demos. Individually excellent, individually needing a licence check, and by default
  **not** usable.

There is no free proprietary HTML effect library worth naming that the open-source options do not
already cover better. **The recommendation is to stay open-source for HTML.**

## 5. A different category entirely: asset generators

`playgrnd.tools` prompted this file and is worth separating out, because it is **not an effect
source at all**. It is a set of browser tools — "tiny tools for making weird, beautiful things" —
that run locally and **export a file**. Nothing is uploaded and, per the site, "exports are yours".

That makes it a **content generator**, not a runtime effect:

```
  runtime effect   an ISF / OFX / HTML thing the SERVER evaluates per frame, parameterised live
  asset generator  a tool that produces a PNG / MP4 you then PLAY as a clip
```

Both are useful and they are not substitutes. A generator cannot be keyframed by `MIXER`, cannot
respond to tracking, and cannot be a filter on a live layer. An effect cannot give you a
hand-tuned one-off texture in thirty seconds. **Sites of this shape belong in a media library
list, not an effects list** — and their export terms ("exports are yours") are a different
licensing question from an effect's source licence.

If a generator list is wanted it should be its own file, and its acceptance test is different:
what formats does it export, at what bit depth, and with what alpha.

## 6. What this catalogue does not establish

* **Nothing here has been loaded.** Not one ISF file or OFX plugin in this document has been run
  in this server. The ISF module implements `PASSES` and `IMPORTED`, so the 327-file collection
  *should* mostly work — "should" is the word doing the work.
* **The openfx-misc compatibility warning is unresolved.** Its README says some plugins use OFX
  features Blackmagic's hosts lack. Whether this fork's host implements them is untested, and the
  failure mode of a missing OFX suite is not documented here.
* **Licences were read from repository metadata and vendor pages, not from file headers.** For ISF
  in particular the per-file header is authoritative and often differs from the repository's
  declared licence, because the files are collected from many authors.
* **No per-shader inventory exists.** "327 `.fs` files" is a count, not a list of what they do, and
  nobody has checked which of them the fork's ISF implementation actually accepts.
* **The GPL-2 / GPL-3 plugin-boundary argument in §3 is a layman's reading**, offered so the
  question is visible rather than settled.
