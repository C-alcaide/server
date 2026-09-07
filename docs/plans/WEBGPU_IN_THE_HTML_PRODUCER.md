# WebGPU in the HTML producer — what could run, and what including it costs

> **Status:** SURVEY. **Three items are measured and the rest are not.** `vgpu` 0.4.0 (§3.1),
> three.js r185 (§3.2) and PlayCanvas 2.22.0 rendering 3DGS (§3.3) were run in a channel on
> 2026-09-07; every other project here was
> assessed from its repository and documentation and **has never been started in this server**. Star counts and dates are from the
> GitHub API on 2026-09-07 and go stale.
> **Falsifier:** if `docs/features/html-gpu-direct.md` §2 stops listing three gates for WebGPU,
> this document's premise has changed and its effort estimates are void.

WebGPU works in the HTML producer as of `5177908db`. That turns the browser layer from "HTML and
WebGL" into a general GPU surface, and the question becomes which existing projects are worth
pointing at it. This file exists so the next person does not re-derive the answer.

---

## 1. Read this first: the producer is a SOURCE, not a filter

A page is rendered and its output is pulled into a layer. **The page cannot see the channel's
other layers**, and there is no route from a composited CasparCG frame back into a browser. Any
idea of the form "run this effect over my video in a template" does not work, and an earlier
version of this survey implied it did.

What a page *can* acquire on its own:

* a **capture device**, because `html.cpp` passes `--enable-media-stream` and
  `--use-fake-ui-for-media-stream`, so `getUserMedia` succeeds with no permission prompt;
* anything reachable over the network, with `--disable-web-security` already set;
* files, via `file://`.

So a matting or analysis model can process a **camera**, not the programme output. Routing the
channel out through an FFmpeg consumer and back into the page is possible and adds a round trip
of latency for something the mixer is better placed to do.

## 2. The three prerequisites, and the one that is not obvious

From `docs/features/html-gpu-direct.md` §2: `<enable-gpu>true`, the DXC pair beside the exe (fixed
in `5177908db`), and `<gpu-direct>true`. **The third is the surprise** — with GPU compositing on
and gpu-direct off, the channel composites a fully transparent frame rather than falling back.

Two further constraints that shape every choice below:

* **`gpu-direct` pins ONE adapter for every html producer in the server.** `CefInitialize` runs
  once per process, long before any channel exists, so `gpu-direct-adapter-luid` is a global
  decision on a two-GPU box, not a per-channel one.
* **`--enable-begin-frame-scheduling` is set**, so CEF paces to the channel. A page that cannot
  hold frame rate **paces the channel** rather than dropping frames quietly. Better than silent
  corruption, but it means "looks right" and "holds rate" are separate questions, and only the
  second one is a broadcast question.

And the fork-specific one: **HTML is how this fork renders fill + key lower thirds**, so the key
is an output of the page. Anything touching alpha or premultiplication has to be checked in both.

---

## 3. What including each one actually means

The useful axis is not popularity, it is **shape**: an embeddable JS library costs a bundle step;
an application costs a fork.

| project | ★ | licence | pushed | renderer | shape |
| :--- | ---: | :--- | :--- | :--- | :--- |
| [three.js](https://github.com/mrdoob/three.js) | 115231 | MIT | 2026-09-07 | WebGPU + WebGL2 | library |
| [BabylonJS](https://github.com/BabylonJS/Babylon.js) | 26035 | Apache-2.0 | 2026-09-04 | WebGPU + WebGL2 | library |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | 21791 | MIT | 2026-09-07 | WebGPU EP | library |
| [playcanvas/engine](https://github.com/playcanvas/engine) | 16663 | MIT | 2026-09-07 | WebGPU + WebGL2 | library |
| [transformers.js](https://github.com/huggingface/transformers.js) | 16290 | Apache-2.0 | 2026-09-07 | WebGPU + WASM | library |
| [supersplat](https://github.com/playcanvas/supersplat) | 9959 | MIT | 2026-09-07 | WebGPU + WebGL2 | **app** (editor) |
| [spark](https://github.com/sparkjsdev/spark) | 3582 | MIT | 2026-09-03 | **WebGL2 only** | library |
| [lygia](https://github.com/patriciogonzalezvivo/lygia) | 3428 | NOASSERTION | 2026-08-17 | shader source | snippets |
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | 2883 | MIT | 2025-10-19 | WebGL2 | library, **dormant** |
| [luma.gl](https://github.com/visgl/luma.gl) | 2467 | NOASSERTION | 2026-09-07 | WebGPU + WebGL2 | library |
| [vgpu](https://github.com/vercel-labs/vgpu) | 1862 | MIT | 2026-09-04 | WebGPU only | library |
| [web-splat](https://github.com/KeKsBoTer/web-splat) | 298 | Apache-2.0 | 2026-03-31 | WebGPU (wgpu) | **app**, Rust |
| [cables_dev](https://github.com/cables-gl/cables_dev) | 100 | MIT | 2026-09-07 | WebGL | node editor |

### 3.1 vgpu — measured 2026-09-07

Vercel's TypeScript WebGPU layer; WGSL as a typed module system. **Verified running in a channel
on 2026-09-07**: `init`/`surface`/`effect`/`frameLoop` over 37 frames, four asymmetric quadrant
colours landing in the right positions within 1 LSB on every channel.

*Including it:* `npm i vgpu`, bundle with esbuild to an IIFE, one `<script>`. `effect()` takes a
raw WGSL string, so no bundler plugin is needed for the simple case. **Effort: hours.**

*What it is actually for:* a nicer authoring surface for hand-written WGSL, plus — the more
interesting half — the **same API headless in Node with `pixelmatch`/`pngjs` image tests**. That
is a shader regression harness that needs neither a server nor a browser, which is the shape
`CasparCG-TestRunner` already has. Whether a third shader toolchain beside GLSL and SPIR-V earns
its keep is a real question and the honest answer is probably not, unless templates start
carrying substantial WGSL.

*Against it:* WebGPU only, no fallback. A vgpu template is blank on a default-configured server.

### 3.2 three.js — measured 2026-09-07, and one claim here was wrong

*Including it:* `npm i three`, bundle, one `<script>`. **Effort: hours**, confirmed — r185 through
`WebGPURenderer` + TSL rendered four asymmetric quadrants into a channel, 39 frames, worst **1
LSB** against the shader constants. Numerically identical to the vgpu run.

**Trap, and it is a colour-management one.** three.js sets `outputColorSpace = SRGBColorSpace` by
default and applies that OETF on top of whatever the shader wrote. Measured: linear `0.90` arrived
as **243** instead of 230, `0.20` as 124 instead of 51, `0.10` as 89 instead of 26 — every patch
~70 LSB light, in the right position with the right hue, which is exactly the shape of error that
reads as "looks a bit washed out" rather than as a fault. A template feeding a colour-managed
server must set:

```js
renderer.outputColorSpace = THREE.NoColorSpace;
```

With that one line the same page lands at 1 LSB. Any engine in §3 may have an equivalent default;
none of the others has been checked.

**The "it degrades gracefully" argument was wrong, and this is the more useful finding.** TSL does
compile one source to both WGSL and GLSL, and `WebGPURenderer` does fall back — it logs
*"WebGPU is not available, running under WebGL2 backend"* — and then it **throws**, because on a
default CasparVP config there is no WebGL either. Measured on the shipped default
(`<enable-gpu>` absent):

| context | `enable-gpu` false (default) | `enable-gpu` true + `gpu-direct` |
| :--- | :--- | :--- |
| `webgl2` | **false** | ANGLE / NVIDIA RTX A4000 / Direct3D11 |
| `webgpu` | api present, **no adapter** | working (nvidia / ampere) |

`--disable-gpu` takes WebGL with it, and `enable-webgl` is appended only *inside* the
`if (enable_gpu_)` branch, so it never applies when it would be needed. **On a default
configuration no GPU rendering context of any kind exists** — which is a statement about every
WebGL template as much as about WebGPU, and is worth knowing independently of this survey.

So the fallback protects against *a browser without WebGPU*. It does not protect against *this
server with its default config*, and nothing does: `<enable-gpu>true` is a prerequisite, not a
nicety.

### 3.3 PlayCanvas — 3DGS measured in a channel, 2026-09-07

Engine 2.22.0 ships a **compute-based WebGPU renderer for 3D Gaussian splats**, with automatic
streamed LOD and a WebGL2 fallback its authors describe as visually identical.

**Verified rendering 3DGS into a CasparCG channel.** `createGraphicsDevice(..., {deviceTypes:
["webgpu"]})` reported `deviceType: "webgpu"`, a `gsplat` asset loaded, and 36 frames rendered.

The oracle was **synthetic rather than a downloaded scene**, which is what makes the number mean
anything: a four-gaussian INRIA `.ply` written by hand, colours encoded through the SH DC relation
`c = 0.5 + C0·f_dc`, identity rotations, `log` scale and `logit` opacity. So the expected picture
is closed-form, and the four colours are asymmetric for the usual reason.

| splat | measured | expected | Δ |
| :--- | :--- | :--- | ---: |
| (−1,+1) | (228, 51, 26) | (230, 51, 26) | 2 |
| (+1,+1) | (25, 177, 64) | (26, 178, 64) | 1 |
| (−1,−1) | (38, 89, 215) | (38, 89, 217) | 2 |
| (+1,−1) | (203, 190, 38) | (204, 191, 38) | 1 |

**Worst 2 LSB** across the whole chain: ply → SH DC decode → gaussian rasterisation → WebGPU →
CEF compositing → the gpu-direct D3D11 shared texture → mixer → IMAGE consumer.

*Including it:* `npm i playcanvas`, bundle, one `<script>`. One bundler wrinkle — its sort workers
carry a Node dual-path, so esbuild needs `--external:node:worker_threads` for a browser build.
Tone mapping and gamma were set to `NONE` up front rather than discovered later, on the strength
of §3.2.

**What this does NOT establish, and it is the part that matters for playout:** four gaussians is
not a scene. Nothing here says anything about a real capture of one to ten million splats, about
the streamed-LOD path, about memory, or about holding 25/50 fps beside a mixer — and
`--enable-begin-frame-scheduling` means a scene that cannot keep up will **pace the channel**.
The correctness question is answered; the cost question is untouched.

*Still to do for it to be useful:* the camera is an ordinary `pc.Entity`, so driving it from
`PREVIZ`/`TRACKING` through AMCP is a bridge of maybe a day — that remains the interesting work
and is not done.

Note `supersplat` (9959★) is the **editor** built on the same engine, not a runtime to embed —
useful for preparing scenes, not for playing them out.

### 3.4 Spark — easiest splats, and it does not need WebGPU at all

Actively developed by World Labs; multiple 3DGS objects per scene, relighting, a splat shader
graph, and most formats (`.ply`, `.spz`, `.splat`, `.ksplat`, `.sog`).

**It is WebGL2 only** — a deliberate choice for ~98% device reach, and a code search for
`WebGPURenderer` in the repository returns **0**. So it would have run on the build *before* the
WebGPU work, and picking it means none of §2 applies. That is a point in its favour for shipping
and a point against it for exercising anything new.

`mkkellogg/GaussianSplats3D` is the older, more famous Three.js splat renderer and introduced
`.ksplat`; its author has said it is no longer actively developed and npm has not moved in a year.
Prefer Spark or PlayCanvas.

### 3.5 web-splat — the one that is NOT a template

Rust/wgpu, >200 FPS on a 3090 by its authors' measurement. `build_wasm.sh` does produce a
`wasm32-unknown-unknown` build through `wasm-bindgen` — but `src/lib.rs` exports exactly **one**
`#[wasm_bindgen]` function, `run_wasm(...)`, which hands control to a winit event loop.
`set_camera` is a Rust method on `WindowContext` and **is not exported to JS**.

So it can be started and pointed at a scene, and then it drives its own camera from mouse and
keyboard. Driving it from `TRACKING` — the entire reason it would be interesting — means adding
exports to a fork, plus a Rust toolchain and `wasm-pack` in the build, and re-forking on every
upstream pull. **Effort: medium, and ongoing.** PlayCanvas gets to the same place without a fork.

### 3.6 transformers.js / ONNX Runtime Web — narrower than it looks

WebGPU-backed RMBG-1.4 and MODNet give real-time matting in a page.

*Including it:* bundle plus **self-hosted model files** — hundreds of MB, otherwise pulled from
HuggingFace's CDN at runtime, which is not a thing to have in a playout path.

*The limit is §1:* this can matte a **camera**, not the programme output. And per-frame inference
at 25/50 fps on the same GPU as the mixer is a cost nobody has measured; every demo in the wild is
single-image. Treat any frame-rate claim as unproven until `coexistence` says otherwise.

### 3.7 The rest, briefly

* **BabylonJS** — the other mature WebGPU engine. No reason to prefer it over three.js here unless
  a specific feature decides it; listed so the choice is visible rather than accidental.
* **luma.gl** — the WebGPU/WebGL layer under deck.gl. Interesting only if data-overlay graphics
  become a requirement.
* **lygia** — a multi-language shader function library (GLSL/HLSL/Metal/WGSL), not a renderer.
  Worth knowing about when writing WGSL by hand. **NOASSERTION licence — check before shipping.**
* **cables** — a node-based patching environment, WebGL. Conceptually close to what
  `GRADING_NODE_GRAPH_STUDY.md` explores, and worth a look for that reason rather than for output.

---

## 4. Suggested order, if any of this is pursued

1. ~~**three.js**~~ — **done, 2026-09-07.** Hours of work as estimated, 1 LSB against the shader
   constants, and it found the `outputColorSpace` trap in §3.2. The "degrades gracefully"
   half of the argument was wrong and is corrected there.
2. ~~**PlayCanvas splats**~~ — **rendering, 2026-09-07**, at 2 LSB against a synthetic ply
   (§3.3). What remains is the camera bridge to `PREVIZ`/`TRACKING`, and a cost measurement
   on a real scene rather than four gaussians.
3. Everything else only if a specific show needs it.

And before any of them ships: **`coexistence`**. These are all additional tenants on the one
`VkDevice`/adapter the mixer, the decoders and the encode exporter already share, and that battery
exists precisely because a route measured alone says nothing about it running beside the others.

## 5. What this document does not establish

* **Nothing here except vgpu, three.js and PlayCanvas has been run.** Every other effort
  estimate is read off a repository, not measured, and the two that involve a camera bridge (PlayCanvas, web-splat) are the ones most
  likely to be wrong.
* **No frame-rate claim is ours.** Every FPS figure quoted is the project's own, on their
  hardware, outside CEF's offscreen path — which is exactly the variable that matters here.
* **No coexistence measurement exists** for any of them.
* **Alpha is unchecked.** Given fill + key, a page compositing with premultiplied alpha through
  `gpu-direct` is a question this survey did not open.
