# 3D Gaussian splatting — literature survey, September 2026

> **Status:** SURVEY ONLY — nothing here is implemented, prototyped or measured in this tree.
> Every number below is an author's own claim from a **preprint**, none peer-reviewed. No code in
> CasparVP touches 3DGS.
> **Falsifier:** `grep -rniE 'splat|3dgs' src/` returns **0** (checked 2026-09-07) — if it ever
> returns something, this document is no longer the whole story. Do not grep `gaussian`: that hits
> 14 times and every one is the *Gaussian blur* in `shader.frag` and `frame_transform.h`, which is
> unrelated.

Compiled 2026-09-07 from the arXiv API (`search_query=all:"gaussian splatting"`, sorted by
submission date). Included because 3DGS is the capture format most likely to arrive at a virtual
production server next, and because two of these papers describe problems this fork already solves
for raster video.

## How active the field is

The 60 most recent submissions span **2026-08-21 to 2026-09-04** — about four papers a day. Any
survey of it is stale within a fortnight, so this file is dated in its title on purpose. Re-run:

```bash
curl -sS -A "casparvp-research/1.0" \
  "https://export.arxiv.org/api/query?search_query=all:%22gaussian+splatting%22&start=0&max_results=60&sortBy=submittedDate&sortOrder=descending"
```

## The four that touch this codebase's concerns

### Colour and grading — [2609.03897](https://arxiv.org/abs/2609.03897)

*Reparametrizing 3D Gaussian Splatting for Real-Time Palette-based Color and Luminance Editing*
(cs.GR, 2026-09-03).

The closest thing here to `COLOR_GRADING.md`'s problem statement. Reparameterizes the spherical
harmonics of a **pretrained vanilla 3DGS** — no retraining — so they encode view-dependent palette
weights, solved jointly with the palette colours under an image-space sparsity loss. Gives
palette-based recolouring, **per-palette tone curves** for colour-aware luminance adjustment, and
pixel-level colour constraints. Luminance editing is a per-pixel weight shift along the achromatic
axis, which the authors show is equivalent to a palette-aware luminance edit.

Why it matters here: it is a *view-space* formulation, introduced specifically to fix a limitation
of primitive-space methods where alpha compositing breaks the edit. That is the same distinction
this tree draws between grading before and after the composite — see
`<working-space-composite>` in `../features/colour-grading-and-ocio.md`.

### Rasterization cost — [2609.03613](https://arxiv.org/abs/2609.03613)

*TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization* (cs.GR, 2026-09-03).

Replaces the globally sorted tile stream with shorter depth-local ranges rasterized front-to-back,
with selective repair where coarse ordering is not enough to match baseline compositing.

**The honest number is the interesting one.** Mean **1.44x raster-kernel** speedup on an RTX 4090,
but mean **1.069x end-to-end** frame speedup on the same card (1.094x on an RTX 1000 Ada), against
gsplat, matching its output to within numerical noise. The gap between kernel and frame is the
same lesson this tree keeps re-learning: a kernel win is not a frame win, and only the second one
is a feature. Most papers in this batch report the first number alone.

### Per-primitive storage — [2609.05255](https://arxiv.org/abs/2609.05255)

*Compact Neural Appearance Models for Efficient Gaussian Splatting* (2026-09-04).

Spherical-harmonic coefficients dominate per-primitive storage and memory traffic, and their
band-limited basis caps angular detail. Replaces them with compact per-primitive latent codes
decoded by a tiny shared MLP, with forward and backward passes **fused into a differentiable CUDA
rasterizer**, plus a WebGL viewer.

Relevant because the CUDA half would share a device with everything in
`../architecture/GPU_INTEROP_ARCHITECTURE.md`. Note what that costs: the `coexistence` battery
exists because routes measured alone say nothing about running beside each other, and
`av_vulkan_import.cpp` loses the device at four concurrent producers.

### Volumetric video — [2608.30184](https://arxiv.org/abs/2608.30184)

*ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation* (2026-08-31).

Argues that tracking long-range motion with individual Gaussian primitives is inherently unstable,
and organises them around **time-conditioned anchors** with a temporal windowing strategy that
activates only the anchors relevant to the queried time. Aimed squarely at long sequences, which is
what separates a clip a server could play from a demo.

## Also noted, not read closely

| paper | why it might matter |
| :--- | :--- |
| [LightBridge](https://arxiv.org/abs/2609.02543) | Feed-forward relighting of a complete 3DGS asset in one pass, no per-scene optimization. Obvious ICVFX relevance — an LED wall's content wants relighting to match the stage. |
| [Atlas](https://arxiv.org/abs/2609.02352) | City-scale on-device rendering via hierarchical memory offloading, temporal-aware LoD search and stereo rasterization. cs.AR, VR-targeted. |
| [KISS-GS](https://arxiv.org/abs/2608.26948), [Non-Uniform Quantisation](https://arxiv.org/abs/2608.28272), [CC-4DGS](https://arxiv.org/abs/2609.02184) | Compression cluster. Relevant only once there is a format to store. |
| [ABCD](https://arxiv.org/abs/2608.27735) | Constant-VRAM training for large radiance fields. |
| [GradRig](https://arxiv.org/abs/2609.05127), [As-Rigid-As-Possible](https://arxiv.org/abs/2608.29538) | Deformation and rigging of splat assets. |

## What this survey does not establish

* **No claim that any of this runs.** Nothing was built, benchmarked or even downloaded.
* **Author-reported figures only.** The TileGS end-to-end number is the single figure here that
  reads as if it survived contact with a real frame budget; treat the rest as upper bounds.
* **No integration path was assessed.** Where a 3DGS renderer would sit relative to the mixer, the
  composite and the colour pipeline is an open question, not a sketched design.
* **Preprints, all of them**, in a field publishing four papers a day. Several will not survive
  review, and at least one of the above will have been superseded before this file is next read.
