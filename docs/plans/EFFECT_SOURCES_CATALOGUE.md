# Where to get effects — ISF, OFX, HTML, and the licensing that decides it

> **Status:** CATALOGUE, compiled 2026-09-09. **§6 is measured** — 327 ISF shaders and 147 OFX
> plugins were played into a real channel and judged from the captured picture. Everything outside
> §6 is desk research.
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
images. **But see §6.1: every shader that ships its own vertex shader currently fails to compile
it**, which takes out the whole blur/glow family — 38 of 327 — until that is fixed. Guides:
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
as the working library — **276 of them render today, measured** (§6.1) — add CC0 sources
freely, and treat everything else as needing a per-file licence check before it touches a show.

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
Measured in §6: nothing was rejected and 143 of 147 rendered, so whatever those
features are, they are present enough. Silent degradation is still untested.

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

---

### 5.1 Tool DIRECTORIES are a third thing again

`designminis.com` is not a tool, it is a **curated directory of ~48 tools by independent
designers** — colour pickers, easing editors, shadow and gradient generators, SVG builders. So
"checking it" means checking forty-eight separate things, each with its own author, its own terms
and its own free/freemium line. **A directory confers no licence on anything it lists.**

Sorted by what this server can actually do with the output, which is the only sort that matters:

| what it emits | examples from that directory | value here |
| :--- | :--- | :--- |
| **CSS / JS code** | Easings (cubic-bezier editor), Gradient Builder, Gradlab, Gradient studio, Dot Grid Builder, Springs Studio, Design System Studio | **Runtime effect.** Paste into an HTML template and it is evaluated per frame, parameterisable, keyframable |
| **SVG** | Shape Divider, Sinwaver, bookofshapes, Durves, GeoLogo, Spherium, Cursor Lab, HalftonePro | **Also runtime — see below.** The interesting category, and the non-obvious one |
| **raster / 3D / video** | Shatter, Refract, Prisma, Marbler, Weaver, Hatchery, paperlab, Orby, Frametic | Asset generator, exactly as §5 describes |

**SVG is not a baked asset here, and that is measured.** An inline SVG in an HTML template renders
*and animates* in the producer — verified 2026-09-09 with a page carrying a linear gradient and two
SMIL `<animate>` elements, captured three times through the IMAGE consumer: max channel delta 120
and 124 between frames, so the animation is genuinely advancing rather than a static first paint.

That makes an SVG generator worth more than a raster one for this pipeline: the output stays
resolution-independent, recolourable by CSS, and animatable — a *runtime* asset rather than a
picture of one. It is the one thing on that directory worth going out of the way for.

**One library rather than a web toy:** `ShaderGradient` (2402★) ships as `@shadergradient/react` on
npm, so it can be bundled into a template like any other dependency rather than copy-pasted.
**Licence discrepancy to resolve first:** npm declares **MIT**, its GitHub repository declares
**none**. Absence is not permission (§1), so establish which governs before it goes near a show.

**And the licence reality for the rest:** per-tool, unstated by the directory. CSS output is
low-risk — a gradient declaration is barely copyrightable — but generated *shader or JS* code is
not, and several of these are freemium, where the free tier commonly restricts commercial use.

### 5.2 Three checked individually, and one is worth real attention

| source | shape | licence | verdict |
| :--- | :--- | :--- | :--- |
| [`lersent001/orb`](https://github.com/lersent001/orb) | **WebGPU shader** + editor | **MIT** (×2, see below) | **The one worth pursuing** — §5.3 |
| [bookofshapes.com](https://bookofshapes.com) | SVG pattern generator | **explicitly commercial-OK** | Good. Runtime asset per §5.1 |
| [geometric-art.com](https://geometric-art.com) | PNG / MP4 exporter | **unstated** | Asset generator, and terms are silent — the §1 "no licence" case |

**bookofshapes has the clearest terms in this whole document**, which is worth saying because
almost nothing else here does: *"Whatever you generate on this site is yours to use. Commercial or
not, modified or as it comes."* Attribution not required. Two limits: a pattern *"can be part of
what you sell. It cannot be the thing you sell"* — so no reselling the collection — and three
patterns derived from the Joy Division cover and a Müller-Brockmann poster are excluded. Combined
with §5.1's measurement that SVG animates live in a template, this is a genuinely usable source.

**geometric-art.com** turns photos and video into geometric abstractions and exports PNG/MP4, all
on-device. Proprietary, no stated terms, EASY/PRO tiers. An asset generator whose licence is
silent — which §1 says means all rights reserved. Fine to play with, not to build a show on until
someone reads the terms.

### 5.3 `lersent001/orb` — a WGSL generator that already emits a key

648★, 73 forks, **MIT**, pushed 2026-08-29. Presented as a "liquid glass orb editor"; what matters
is what is in the repository root:

* **`effect.wgsl`, 1236 lines, self-contained.** One binding — `@group(0) @binding(0) var<uniform>
  u: Uniforms` — and **zero textures**. It is a pure procedural generator, so it cannot filter
  video, but it also depends on nothing but a uniform buffer.
* **63 uniform fields**, all floats and RGBA colours: `time`, `speed`, `radius`, `zoom`, `warp`,
  `exposure`, `style`, `edgeSoftness`, `edgeGlow`, the `metal*` / `ribbon*` / `particle*` groups,
  eleven colours and twelve palette stops.
* **`effect.metal`** beside it — the same effect already ported once, which says the author treats
  the shader as portable rather than as app internals.
* Two MIT licence files: `LICENSE` (© LerSent001) and `TOOLCRAFT_LICENSE.md` (© Pixel Point), so
  there is vendored third-party code. Both MIT; **keep both notices**.

**The detail that makes it immediately relevant here is the alpha.** Its final lines are explicit:

> *The ball's own edge, and nothing outside it — everything the effect does not paint must be
> exactly 0 so the page shows through.*

It returns `vec4(finalColor, finalAlpha)` with a considered `sphereAlpha`/`emissionAlpha` split.
**HTML is how this fork renders fill + key lower thirds** (§3 of `../features/html-gpu-direct.md`),
and a generator that already produces a correct key is rarer than one that looks good.

**Two routes, and the trade is real:**

| | cost | what you get |
| :--- | :--- | :--- |
| **A — run it in the HTML producer** | ~none; WebGPU works as of `5177908db` | Works today. Parameters from JS, reachable by the `INPUT` command family. Costs a CEF layer per instance |
| **B — port `effect.wgsl` to ISF** | a WGSL→GLSL port of 1236 lines | Native mixer execution, no browser, and the 63 uniforms map almost mechanically onto ISF `INPUTS` — giving `CALL 1-1 ISF SET <name>`, keyframing and the grading chain |

Route B is attractive precisely because the parameter set is flat floats and colours, which is
exactly ISF's `INPUTS` model. It is mechanical work, not clever work, and it ends with a fork of
someone else's shader to maintain.

**Not run.** Everything above is read from the repository; no part of it has been executed in this
server. Route A is a couple of hours to find out.

## 6. MEASURED — what actually loads and renders, 2026-09-09

Every shader and plugin below was played into a real channel on the OpenGL mixer at 720p25 and
**judged from the captured picture**, not from the AMCP reply. Inventories:
[`data/isf_inventory_2026-09-09.json`](data/isf_inventory_2026-09-09.json),
[`data/ofx_inventory_2026-09-09.json`](data/ofx_inventory_2026-09-09.json).

### 6.1 ISF — 314 of 327 render after a fix this sweep found

| | before the fix | after |
| :--- | ---: | ---: |
| **render** | 276 | **314** |
| no picture | 51 | 13 |
| **rejected** | **0** | **0** |

**Nothing was rejected.** All 327 loaded and played; the question was only whether a picture came
out.

**The 51 failures are almost entirely one defect.** Cross-tabulating against whether a shader
ships its own vertex shader:

```
                renders   no picture
  has .vs             0           38
  no  .vs           276           13
```

**Every one of the 38 shaders with a `.vs` fails, and every shader that renders has none.** The
cause is in the log:

```
[isf] vertex shader compile failed: 0(12) : error C1503: undefined variable "blurAmount"
                                    0(19) : error C1503: undefined variable "PASSINDEX"
                                    0(20) : error C1503: undefined variable "RENDERSIZE"
```

**The fork declares ISF's automatic variables and the shader's own INPUTS in the fragment stage
only.** ISF 2.0 makes `RENDERSIZE`, `PASSINDEX` and the declared inputs available in **both**
stages, so any shader supplying a custom vertex shader fails to compile it. This is why the
collection's entire blur/glow family is dark — those are the multi-pass shaders, and multi-pass
shaders ship a `.vs` to set up per-pass coordinates. **Fixed the same day.** Both stages now share one declaration block; the collection went
**276 → 314 of 327, +38, zero regressions**. Two cautions on that number: only the **22
single-pass** ones are verified working, and of the **16 multi-pass** ones exactly one —
`Multi Pass Gaussian Blur` — is confirmed to render incorrectly, with the rest unverified rather
than broken. The multi-pass engine itself was probed and is sound. See
`../features/isf-and-openfx.md` §5.0, including the false positive that a first reading produced.

The other **13** failures each need an input the test never supplied, and are not evidence of
anything broken: `FFT Spectrogram` and `Radial Spectrogram` (audio), `Cursor`, `Random Shape`,
`Circle Trails` (mouse/point2D), `Histogram Viewer`, `Duotone From Histogram`,
`Color Organ Polyphonic`, `Color Relookup`, `Optical Flow Generator`, `Random Characters`,
`Show Alpha`, `Tiny Date Time Overlay`.

### 6.2 OFX — 143 of 147 render, and the 4 that do not are explainable

Natron 2.6.0-alpha1's Windows build was used for compiled bundles, because **neither
openfx-misc nor openfx-arena publishes Windows binaries** — their releases carry no assets, and
Natron's own latest release is macOS-only.

The host initialised cleanly: `[ofx] OpenFX host initialised (image-effect API v1); discovered
167 plug-in(s)` — 147 unique ids after de-duplication.

| bundle | renders | no picture | total |
| :--- | ---: | ---: | ---: |
| Misc (openfx-misc) | 113 | 4 | 117 |
| CImg | 30 | 0 | 30 |
| **total** | **143** | **4** | **147** |

The four: `Solid` (renders a solid colour — flat *is* its output), `ConstantPlugin`, `FrameRange`
and `LayerContactSheetOFX` (need parameters or several inputs the test does not wire). **No
plugin was rejected, and openfx-misc's own warning about hosts lacking OFX features did not
bite** — see §7 for what that does and does not prove.

**Only 2 of the 6 bundles contributed anything.** `GMIC`, `Arena`, `IO` and `Shadertoy` produced
**zero** ids. Most likely their runtime dependencies — ImageMagick and friends, which Natron ships
outside `Plugins/OFX/` — rather than a host fault, but that is unverified.

### 6.3 Two flaws in the first version of this measurement

Recorded because both produced confident wrong numbers:

* **327/327 "rejected" against a working server.** The check was `reply.startswith("202")`, and the
  harness AMCP client **parses the status code off and raises on 4xx/5xx** — so every success
  returns a bare `PLAY OK`. Success is "it did not raise". This trap is already written down in
  this project's notes, and it still caught this sweep.
* **A flat colour is the wrong input for a filter.** Edge, halftone, dither and blur filters
  correctly produce nothing from a featureless image. Re-running the ambiguous cases against a
  textured pattern recovered **25 ISF and 4 OFX** entries that the first pass called broken.

## 7. What this catalogue does not establish

* **"Renders" means a picture came out, not that it is CORRECT.** No output was compared against
  a reference — a shader applying the wrong maths, the wrong channel order or the wrong gamma
  passes §6 exactly as a right one does. That is a much weaker claim than the 1 LSB gates
  elsewhere in this tree, and the gap is the whole distance between "loads" and "usable".
* **Each entry was judged from ONE frame** after ~1.1 s. Anything that animates in, accumulates
  over frames, or uses a persistent buffer may have been caught mid-warm-up.
* **The openfx-misc compatibility warning is only partly answered.** Nothing was rejected and 143
  of 147 rendered, so the features it warns about are evidently present enough to run these
  plugins. Whether any of them silently degrades is untested.
* **Four of the six OFX bundles loaded nothing** and the reason is assumed (missing runtime
  dependencies) rather than established.
* **Licences were read from repository metadata and vendor pages, not from file headers.** For ISF
  in particular the per-file header is authoritative and often differs from the repository's
  declared licence, because the files are collected from many authors.
* **No per-shader inventory exists.** "327 `.fs` files" is a count, not a list of what they do, and
  nobody has checked which of them the fork's ISF implementation actually accepts.
* **The GPL-2 / GPL-3 plugin-boundary argument in §3 is a layman's reading**, offered so the
  question is visible rather than settled.
