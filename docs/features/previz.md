# PREVIZ — 3D pre-visualisation

> **State:** shipped; **cost** measured, **picture** unmeasured
> **Modules:** **not a module** — `src/accelerator/ogl/image/previz_renderer.cpp`, `previz_scene.h`,
> `previz.frag` / `previz.vert`, with the Vulkan route through
> `src/accelerator/vulkan/image/previz_texture_bridge.cpp`
> **Commands:** 13 fork-specific AMCP commands
> **Architecture:** none — shares the ICVFX state the projection commands write; the two routes to
> it are §5.3 below
> **Guide:** [`../guides/PREVIZ_3D_MODULE.md`](../guides/PREVIZ_3D_MODULE.md)
> **Coverage:** `cli.py preview-cost --arm previz --arm previz_spout --arm previz_screen` — what previz costs the channel, at 1080p50 and at the VP workload; the floor those shares are measured against is `cli.py raster-capacity`. **No picture check of any kind** — see §4

Loads a 3D scene, maps channel output onto meshes in it, and renders a camera view of the result —
so a projection design can be checked without the venue. Screens, presets and camera positions are
addressable by name at runtime.

> **Read §4 first.** This is the largest surface in the fork with **no picture check**: 13 commands,
> five of them now driven by `preview-cost`'s previz arms and **none of them checked against a
> rendered pixel**. That combination — undocumented *and* undriven — is exactly what ICVFX was
> before an audit found a live colour defect in it, and this document exists because that
> correlation was measured rather than assumed. **This file closed the documented half.** Cost
> coverage arrived on 2026-08-31 and closes none of the ICVFX-class gap: a battery that gates on
> timing cannot see a channel exchange.

---

## 1. What is implemented today

Verified against `src/protocol/amcp/AMCPCommandsImpl.cpp` at the definition line given.

| command | shape | def |
| :--- | :--- | ---: |
| `PREVIZ SCENE` | `<path>` — loads a scene file | 4777 |
| `PREVIZ MAP` | `<mesh_name> <channel>` — needs ≥2 parameters, else `400` | 4847 |
| `PREVIZ UNMAP` | `<mesh_name>` | 4869 |
| `PREVIZ CAMERA` | `RESET` \| `OVERRIDE 1\|0` \| camera placement arguments | 4887 |
| `PREVIZ VIEW` | `CLEAR` \| `RESET` \| view arguments | 4938 |
| `PREVIZ INFO` | — query | 4985 |
| `PREVIZ SHOW` | `<mesh_name> [1\|0]` | 5011 |
| `PREVIZ GRID` | `[1\|0]` | 5028 |
| `PREVIZ WIREFRAME` | `[1\|0]` | 5041 |
| `PREVIZ GIZMO` | `[1\|0]` | 5054 |
| `PREVIZ PRESET` | `RECALL <name>` \| `LIST` | 5069 |
| `PREVIZ SCREEN` | `<name> REMOVE` \| `LIST` \| `ROTATION` \| `RESOLUTION` \| `CHANNEL` \| `EYEMODE` \| `ARCV` \| `ICVFX` | 5110 |
| `PREVIZ AUTOPROJECTION` | `[1\|0]` — defaults to **on** when no parameter is given | 5252 |

**Details that are not guessable from the names:**

- **`PREVIZ SCREEN` is a command family, not one command.** It dispatches on a second keyword —
  eight of them, including `ICVFX` and `EYEMODE`, which means per-screen ICVFX state is reachable
  from here as well as from `MIXER PROJECTION_ICVFX`. Two routes to one piece of state is the
  pattern `CLAUDE.md` says earns a diagram; §7 records that as owed.
- **`PREVIZ AUTOPROJECTION` with no argument turns it ON** — `ctx.parameters.empty() || at(0) != "0"`.
  Toggles that default to enabled are unusual in this command set and easy to trip over in a
  startup script.
- **The `[1|0]` commands are all "anything but `0` is on"**, not strict boolean parsing.
- **`PREVIZ MAP` requires two parameters and returns `400`**; most of the others accept an empty
  parameter list as a query or a default.

---

## What previz costs, and why the earlier figures in this file were misleading

**In the configuration anyone actually runs — previz on one channel visualising the
others — it costs almost nothing.** Five 1080p50 channels, previz active on N of them:

| previz active on | 0 | 1 | 2 | 3 | 5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| late frames / ~5012 | 0 | **2** | **2** | 6 | 14 |

Earlier revisions of this section reported previz costing 5-11% of ticks and named first
the bridge and then the render as the culprit. **Both readings came from a machine state
that could not be reproduced.** The same configurations that measured 267-1280 late frames
in one session measure 2-14 in another, with the same binary, while the previz-off control
holds at 0-1 throughout. Nothing has explained that, and it is the largest open question
about previz — larger than anything in the renderer.

### The render itself, measured directly

The channel's late-frame count is a threshold effect and useless for anything finer than
on/off. Timing `previz_renderer::render` directly is stable and is the instrument to use:

| | p50 | p90 | p99 | max |
| :--- | ---: | ---: | ---: | ---: |
| render, 2-mesh scene | **157-175 µs** | 202-268 | 415-948 | 1140-4090 |
| render, 60-mesh scene | **546-612 µs** | | | |
| — of which FBO bind/attach/viewport/clear | **3.4 µs** | 5.7 | 6.9 | 92 |
| — the draw loop | 108 µs | 145-158 | 258-318 | 474-2078 |
| — the ground grid, within that | ~15-25 µs | | | |

**p50 is reproducible to a few per cent between runs; nothing else here is.** The heavy
tail lives entirely in the draw loop, is the same at 2 meshes as at 60 — so it is a
per-render event, not per-mesh — and is what turns into late frames when it happens.

### Three micro-optimisations, all measured, none worth keeping

| attempt | p50, 2 meshes | p50, 60 meshes |
| :--- | :--- | :--- |
| baseline | 174-204 µs | 546-612 µs |
| cache `u_mvp`/`u_model` locations instead of `glGetUniformLocation` per mesh per frame | 172-221 | 554-650 |
| drain GL errors once per frame instead of once per screen mesh | (same change set) | (same) |
| skip the `glGetError` after `glBindTexture` | 103-120 | — |

None of them moves p50, and the first two were marginally *worse* at 60 meshes — the
switch guarding them costs about what they save. All reverted. Worth knowing so nobody
spends the afternoon again: the per-mesh cost is roughly **8 µs at 60 meshes**, and it is
not in the uniform lookups or the error checks.

### Where to look next, if previz ever needs to be faster

Not at micro-optimisation. The two open questions are the **100x instability between
sessions**, which dominates everything else and is unexplained, and the **heavy tail in
the draw loop**, which is per-render rather than per-mesh and was not localised further
than that. The fixed per-render cost outside the draw loop is 3.4 µs and has nothing left
in it.

## Red and blue were exchanged on every mapped screen (fixed 2026-08-29)

A channel playing `#20A0C0` appeared on its previz screen as `#BE9E1E` — the same colour
with **red and blue exchanged**. Visible on every mapped screen, on the Vulkan mixer, for
as long as the VK→GL bridge has existed.

| | mapped screen renders | source |
| :--- | :--- | :--- |
| before | `#BE9E1E` (190, 158, 30) | `#20A0C0` (32, 160, 192) |
| after | `#1F9FBF` (31, 159, 191) | `#20A0C0` — the 1/channel is previz's own shading |

**Cause.** The mixer's 8-bit attachment is declared `eR8G8B8A8Unorm` but *holds BGRA
bytes* — only the 8-bit shader path swizzles, which `vulkan::device` records in two
places. `previz_texture_bridge` copied it into another `R8G8B8A8` image and imported that
as `GL_RGBA8`, so GL read blue where red should be.

**Fix.** The destination image is now `B8G8R8A8`, and the copy is a **`vkCmdBlitImage`
rather than a `vkCmdCopyImage`** — that second half is the part that is easy to get wrong.
A copy moves *bytes* and is format-agnostic, so changing the destination's component order
would have changed nothing; a blit maps *components*, which is what reorders BGRA into
true RGBA for the importer. Same extents, so the filter never runs. The 16-bit path is
already RGBA and is untouched.

**Why it survived this long: nothing had ever looked at previz's output.** No battery
drives previz, and the defect is invisible unless you compare a mapped screen's colour to
the channel's — a grey or white test pattern is invariant under the exchange. It was found
by publishing the previz channel over Spout and reading the pixels back, which is now the
cheapest way to check previz at all:

```
PREVIZ 1 SCENE scene.obj
PREVIZ 1 MAP screen1 1
ADD 1 SPOUT previzcap MAX_WIDTH 512
```

then receive `previzcap` and compare the mapped quad against the channel's colour. The
scene needs only a two-quad `.obj` inside the server's media path.

## 2. How to drive it

Load a scene, map channel 1 onto a mesh, look at it:

```
PREVIZ SCENE "venue/stage.gltf"
PREVIZ MAP led_wall_main 1
PREVIZ SHOW led_wall_main 1
PREVIZ GRID 1
PREVIZ WIREFRAME 0
PREVIZ INFO
```

Screens and presets are enumerable, which is the fastest way to discover what a scene actually
contains:

```
PREVIZ SCREEN LIST
PREVIZ PRESET LIST
PREVIZ PRESET RECALL front_wide
```

Careful with this one — it enables autoprojection rather than querying it:

```
PREVIZ AUTOPROJECTION
```

No `<configuration>` elements; entirely runtime.

---

## 3. Design decisions, and what they cost

**The bridge lives in the OpenGL mixer.** `previz_*` commands reach the mixer through
`ogl_mix->...` (e.g. `unmap_mesh` at 4869), so this feature is tied to the OpenGL accelerator.
On a Vulkan channel the commands have no mixer to talk to. That is a real constraint on a fork
whose Vulkan mixer is the one under active development, and it is not stated anywhere else.

**Names rather than indices** for meshes, screens and presets. It makes show files readable and
survives a scene being re-exported with different ordering, at the cost of silent no-ops when a
name does not match — which is the failure mode to look for first when a command returns `202`
and nothing moves.

---

## 4. Verification — what is measured, and what is not

**Cost is measured, and since 2026-09-05 so is the picture — for `MAP`.** `preview-cost`'s three
previz arms drive `PREVIZ SCENE`, `MAP`, `SHOW`, `GRID` and `WIREFRAME` against a generated scene
(`core/previz_scene.py` — four named quads around a stage) and report what previz costs the
channel's tick; that is where the figures above come from. **They gate on timing and never look at
a pixel**, so on their own *"a previz that rendered the wrong channel onto every mesh, or nothing
at all, would pass every arm."*

`cli.py previz-picture` closes that for the mapping. Each mapped channel plays a **different
asymmetric colour** and the previz render is searched for each, which answers all three checks
below at once: a colour that is absent did not arrive, a colour found has its components in the
right order, and four distinct colours mean four distinct channels landed. When a colour is
missing its **permutations** are searched too, so an exchange is named as one rather than reported
as a mapping fault.

**Result, 2026-09-05, both mixers: 4/4 screens, and the pixel counts are identical between OGL and
Vulkan** — 346800 / 188232 / 188232 / 405674. So `PREVIZ MAP` delivers the right channel to the
right mesh with its components in order, and the two mixers agree, which is the parity §5.2 records
as unmeasured.

**Still not covered**: eight of the thirteen commands (`UNMAP`, `SCREEN`, `CAMERA`, `VIEW`,
`AUTOPROJECTION`, `GIZMO`, `PRESET`, `INFO`), check 3 below, spatial placement — four colours prove
four channels arrived on four meshes, not that `screen1` is the back wall — and colour accuracy,
since the quads are lit and projected so the tolerance is deliberately wide.

This section read *"Nothing. No battery in the harness references PREVIZ"* until 2026-08-31, and
was already false when the cost figures above were written into this file. The correction is worth
keeping visible: **a cost battery is not coverage of a feature**, and the two halves went stale in
opposite directions in one sitting.

The remaining gap is recorded as the finding it is rather than as a to-do. The measured correlation from the
2026-08-26 audit: of the fork's 58 AMCP commands, the ones carrying defects were the ones that
were both undocumented and undriven. ICVFX had a red/blue exchange on the OpenGL mixer that
survived because no battery drove it and because the natural way to test a white balance — equal
gains — is invariant under exactly that defect. **PREVIZ is the same shape, thirteen times over**, and it
reaches the same OpenGL renderer from both mixers.

What a first battery should do, in the order that would have caught the ICVFX class:

1. `PREVIZ MAP` a channel onto a mesh, render a view, and check the mesh is **not** the background
   colour — the cheapest possible "did anything arrive" check.
2. An **asymmetric** source colour through the mapping, compared per channel, because a symmetric
   one cannot see a channel exchange.
3. `PREVIZ SCREEN ... ICVFX` against `MIXER PROJECTION_ICVFX` for the same screen — two routes to
   one piece of state, which is where they can disagree.

### A mapped channel needs a consumer, or `PREVIZ MAP` succeeds and shows nothing

**Measured 2026-09-05, and it cost a fabricated defect.** The first run of `previz-picture`
reported 0 of 4 screens on both mixers with every command returning `202 PREVIZ OK` — the scene
loaded with 4 shapes, all four meshes reported *"Mapped mesh ... to channel N"*, and the render
came back as grey quads. Read alone that is *"`PREVIZ MAP` reports success and delivers nothing"*,
which is what it was about to be written up as.

**The cause was the fixture: channels 2–5 had no consumer.** A channel with no consumer does not
produce frames, so previz had nothing to sample. Confirmed by elimination rather than assumed —
with the source channels left consumerless, **20 seconds of settle still gave 0 px**, so it is not
a timing problem; adding a capture on each source channel makes the same run pass 4/4.

Two consequences:

* **For a battery**: `previz-picture` captures each mapped channel *before* asking previz about it,
  and an arm whose source control fails reports **"this arm says NOTHING about previz"** rather
  than a previz failure. A check that cannot tell "the mapping dropped it" from "the channel never
  ticked" will eventually report the second as the first.
* **For a client**: mapping a channel that nothing consumes gives a silent grey mesh, with `202 OK`
  on every command. A GUI driving previz must ensure its mapped channels are being consumed, and
  should not treat `PREVIZ MAP`'s `202` as evidence that anything will appear.

---

## 5. Known gaps

1. **Picture coverage exists for `MAP` only, since 2026-09-05** — `cli.py previz-picture`, §4.
   Arrival, component order and per-mesh identity are gated on both mixers. The other twelve
   commands still have no picture check, and §4's third check (`PREVIZ SCREEN ... ICVFX` against
   `MIXER PROJECTION_ICVFX`) is blocked on §5.3.
2. **The renderer is OpenGL on both mixers, and the parity that implies is unmeasured.** This item
   read *"OpenGL-only — either the Vulkan mixer grows the same bridge or the commands should
   refuse"*; the bridge exists (`vulkan/image/image_mixer.cpp:1204-1254`). A Vulkan channel
   composites in Vulkan, posts the result to the VK→GL bridge, renders the scene on the OGL thread
   and returns *that* as the channel output — so previz **replaces** the 2D output and skips the
   working-space composite. Two consequences nothing checks: a Vulkan channel running previz is
   doing a per-frame round trip through a second API, and its colour handling differs from the same
   channel with previz off.
3. **`PREVIZ SCREEN`'s eight subcommands are unenumerated** in any document, including this one:
   the list in §1 was recovered from a dispatch chain, and the parameter shape of each is not
   established. Reading the handler is currently the only way to know.
4. **Two routes to per-screen ICVFX state** (`PREVIZ SCREEN ... ICVFX` and
   `MIXER PROJECTION_ICVFX`) with no documented precedence.

---

## 6. Related commits

Not yet traced. The module predates this document and its history has not been read; that is
deliberate rather than lazy — a commit list here is only useful if each line says *why the commit
matters*, and inventing that from subject lines would be exactly the kind of claim this folder
exists to avoid. Tracked as work owed.

---

## 7. Diagrams

**Owed.** Two of the three criteria in `CLAUDE.md` apply: `PREVIZ SCREEN ... ICVFX` and
`MIXER PROJECTION_ICVFX` are two paths reaching the same state, and the scene → mesh → mapped
channel → camera view chain is an order that prose describes badly. Operator-facing, so a rendered
PNG from a script in `docs/diagrams/`.
