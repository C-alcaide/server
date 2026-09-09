# Where to get effects — ISF, OFX, HTML, and the licensing that decides it

> **Status:** CATALOGUE, compiled 2026-09-09. **§5.3, §5.6 and §6 are measured** — the orb effect was
> built and played as a layer (§5.3), and — 327 ISF shaders and 147 OFX
> plugins were played into a real channel and judged from the captured picture. The rest is desk
> research and says so.
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
as the working library — **314 of them render today, measured**, and the browsable list by
category is `docs/plans/data/isf_working_by_category.md` (§6.1) — add CC0 sources
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

**Route A RUN AND MEASURED, 2026-09-09.** Not the editor — a 87 KB bundle of the effect alone,
reusing the project's own `orbShaderSource` and `writeOrbUniforms` so the 136-float layout is
theirs rather than a reimplementation. Played as an HTML layer at 1280×720:

| | |
| :--- | :--- |
| renders | **yes** — the Siri-style liquid glass orb, spectral bands, clean edge |
| surface format | `bgra8unorm`, `alphaMode: "premultiplied"` |
| **key** | **correct.** Outside the orb is RGBA **(0, 0, 0, 0)**; centre opaque; **242 distinct alpha values**, so a real soft matte rather than a binary cutout; 10.4% frame coverage |
| **frame rate** | **22 fps steady state** into a 25 fps channel — see below |

**It does not quite hold rate.** 22 against 25 means roughly one frame in eight is a repeat, and by
§2 of `WEBGPU_IN_THE_HTML_PRODUCER.md` **nothing in the channel's timing will report that** —
`late_frames` stays 0 and the period stays nominal. Its 63 uniforms include a `radius`, so a
smaller orb is the obvious lever before anything cleverer. Measured at default `initialParams`,
one style, on an RTX A4000, with nothing else on the box.

Two practical notes for anyone repeating it: the exports are `orbShaderSource` (not
`shaderSource`), `stylePresets[name]` **is** the params object rather than wrapping one, and the
`../effect.wgsl?raw` import is Vite syntax that esbuild needs shimmed. And **the first run
reported 3 frames** — that was shader compilation, not the effect; a warm-up before the
measurement window is mandatory here.

### 5.4 Four more, checked individually

| source | shape | licence | verdict |
| :--- | :--- | :--- | :--- |
| [fluid](https://github.com/enonforetsam/fluid) / fluid.krackeddevs.com | **npm library** `fluid-bg` | **MIT**, 66★ | **Best of the four.** A runtime effect |
| [holocloth](https://github.com/dmitrykurash/holocloth) | Three.js app | **MIT**, 188★ | Interesting, but an app |
| sticky.ui8.dev | sticker/video exporter | **unstated** | Asset generator |
| ascii.krackeddevs.com | ASCII converter | **unstated**, no repo | **Redundant — you already have it** |

**`fluid-bg` is the one to take.** v0.4.0, **MIT**, **zero dependencies**, described as a
"dependency-free WebGL studio for generative backgrounds and live embed" — a native canvas rather
than an iframe, embeddable as an HTML tag, a React component or `npm i fluid-bg`. That is the same
shape as §3.1's vgpu and §3.2's three.js: bundle it, one `<script>`, done. It is **WebGL, not
WebGPU**, which on this server means it still needs `<enable-gpu>true` (§3.2's context table — a
default config has no WebGL either).

**holocloth** drapes an uploaded image or SVG over Verlet-simulated cloth with an iridescent foil
shader, and exports transparent PNG. MIT, 188★, Three.js WebGL 2 + React, shaders and physics
written from scratch. Two ways to read it: as an **asset generator**, it makes a transparent PNG of
a station graphic on cloth, which is immediately usable; as a **runtime effect** it would mean
lifting custom shaders and a physics loop out of a React app — the web-splat problem of §3.5 again.
The asset route is the sane one unless the cloth needs to move on air.

**sticky.ui8.dev** (UI8, built with Forge and three.js) exports images and 60 fps video with custom
and video stickers. Free, no repository, no stated licence. Asset generator, §1's unstated-terms
case.

**ascii.krackeddevs.com converts images to ASCII and exports txt/ans/png/gif — and you already
have this, better.** `ASCII Art` is in Vidvox's MIT collection and **renders** (§6.1), which means
it runs on the mixer over live video, takes parameters through `CALL 1-1 ISF SET`, and can be
keyframed. A website that exports a PNG cannot do any of that. The same applies to `CMYK Halftone`,
`RGB Halftone` and `Dither-Bayer`, all of which render. **Check the 314 first** — it is the
cheapest question in this document and it answers a surprising number of these.

### 5.5 A wider sweep — and the licence pattern is the finding

Searching for more of the same shape turned up one thing worth taking, one worth knowing about,
and a pattern that matters more than either.

| source | ★ | licence | verdict |
| :--- | ---: | :--- | :--- |
| [pmndrs/postprocessing](https://github.com/pmndrs/postprocessing) | 2852 | **Zlib** | **The gem. 35 ready-made effects for three.js** |
| [enonforetsam/fluid](https://github.com/enonforetsam/fluid) | 66 | **MIT** | §5.4 — drop-in WebGL background |
| [jonradoff/shadervine](https://github.com/jonradoff/shadervine) | 24 | **MIT** | WebGPU shader editor that **exports raw WGSL** |
| [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer) | 5324 | BSD-3 | Console GLSL sandbox — for *authoring*, not playout |
| [hydra](https://github.com/hydra-synth/hydra) | 2707 | **AGPL-3.0** | Live-coding video synth. AGPL needs legal thought before it goes near a product |
| [ybouane/liquidglass](https://github.com/ybouane/liquidglass) | **463** | **NO LICENCE** | Cannot use |
| [jeantimex/glass-effect-webgpu](https://github.com/jeantimex/glass-effect-webgpu) | 63 | **NO LICENCE** | Cannot use |
| [grigM/ISF-shaders-collection](https://github.com/grigM/ISF-shaders-collection) | 33 | **NO LICENCE** | **995 `.fs` files** — and unusable |

**`pmndrs/postprocessing` is the one to take.** 2852★, **Zlib** (permissive, MIT-shaped), pushed
today, **35 effects** — bloom, depth of field, glitch, chromatic aberration, scanlines, god rays,
outline, pixelation and the rest. It targets **three.js**, which §3.2 already measured rendering
into a channel at 1 LSB, so the integration question is answered before it is asked. This is a far
better return than any single-effect site in this document.

**The pattern, and it is the real finding: popularity is uncorrelated with usability.**
`ybouane/liquidglass` has **463 stars and no licence file**; `grigM/ISF-shaders-collection` has
**995 shaders and no licence file** — nearly three times Vidvox's collection, and all of it
off-limits. Meanwhile the usable things here are a 66-star library and a 24-star editor. §1 says
absence is not permission; this section is what that costs in practice. **Check the licence before
the star count**, every time.

**One to think about rather than dismiss:** `hydra` (2707★) is **AGPL-3.0**. That is not the same
question as the GPL-2 plugin boundary in §3 — AGPL's network clause is a different animal, and a
broadcast facility putting it in front of a service deserves a lawyer's opinion rather than mine.

### 5.6 `pmndrs/postprocessing` — MEASURED in a channel, and it is the best return here

35 effects, **Zlib**, 2852★. Run through the HTML producer 2026-09-09.

**It is WebGL, not WebGPU** — zero references to `WebGPURenderer` in the repository, and its
README uses `WebGLRenderer`. So it runs on a *different* path from §3.2's measured
`WebGPURenderer`, and like everything else needs `<enable-gpu>true`, since a default config has
neither WebGL nor WebGPU. Peer range `three >= 0.168 < 0.186`; r185 fits.

| stage | patch | vs identity | |
| :--- | :--- | ---: | :--- |
| **identity** (composer, no effects) | (204.0, 115.0, 51.0) | — | **0.0 LSB against the closed-form source** |
| BrightnessContrast | (255.0, 202.0, 109.0) | 140 | changed |
| ChromaticAberration | (204.0, 115.0, 51.0) | 140 | **unchanged at centre, 140 at the edges — correct for a radial effect** |
| Scanline | (152.9, 73.2, 32.5) | 255 | changed |
| DotScreen | (221.0, 79.6, 10.6) | 255 | changed, halftone visible |

**The identity row is the one that matters.** An `EffectComposer` with only a `RenderPass` is
**byte-exact** against the source colour, so the composer introduces no colour error of its own —
which is what makes every effect above it attributable. That check cost nothing and is the
difference between "the effects look right" and "the chain is clean".

Two things to set, both learned here: `renderer.outputColorSpace = THREE.LinearSRGBColorSpace`
(**`NoColorSpace` is valid on `WebGPURenderer` and throws on `WebGLRenderer`** —
`_getDrawingBufferColorSpace` has no config for it) and `toneMapping = NoToneMapping`, which is
postprocessing's own documented advice.

**Why this is the best return in the document:** 35 effects for one bundle step, on a host already
measured, including `LUT1DEffect`, `LUT3DEffect`, `GammaCorrectionEffect` and `ToneMappingEffect`
— which are the fork's own colour vocabulary, arriving in a browser layer.

**What is not established:** only four of the 35 were driven, none against a model — the gate was
"differs from identity", not "computes the right thing". `SSAOEffect`, `DepthOfFieldEffect` and
`GodRaysEffect` need real geometry and a depth buffer, which a flat quad does not provide. And
nothing was measured for frame rate.

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

* **"Renders" means a picture came out, not that it is CORRECT** — and that gap is now partly
  closed by a battery rather than a caveat. `cli.py isf-conformance` gates the *implementation*:
  a custom `.vs` reading `RENDERSIZE` and its own INPUTS (mutation-proved — reverting CasparVP
  `1a4121267` turns it black at 191 LSB while its five siblings still pass), one channel of an
  asymmetric colour inverted so no r/g/b permutation passes, and `pow(in,1/g)` at 1 LSB. 6/6 on
  **both** mixers with identical numbers. **What it still does not say is whether any particular
  third-party shader is right** — three fixtures are not 314 shaders, and that remains open.
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
