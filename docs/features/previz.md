# PREVIZ — 3D pre-visualisation

> **State:** shipped, unmeasured
> **Modules:** **not a module** — `src/accelerator/ogl/image/previz_renderer.cpp`, `previz_scene.h`,
> `previz.frag` / `previz.vert`, with the Vulkan route through
> `src/accelerator/vulkan/image/previz_texture_bridge.cpp`
> **Commands:** 13 fork-specific AMCP commands
> **Architecture:** none — shares the ICVFX state the projection commands write; the two routes to
> it are §5.3 below
> **Guide:** [`../guides/PREVIZ_3D_MODULE.md`](../guides/PREVIZ_3D_MODULE.md)
> **Coverage:** **none** — no harness battery drives any of these

Loads a 3D scene, maps channel output onto meshes in it, and renders a camera view of the result —
so a projection design can be checked without the venue. Screens, presets and camera positions are
addressable by name at runtime.

> **Read §4 first.** This is the largest **unmeasured** surface in the fork: 13 commands, none of
> them exercised by any battery. That combination — undocumented *and* undriven — is exactly what
> ICVFX was before an audit found a live colour defect in it, and this document exists because that
> correlation was measured rather than assumed. **This file closed the documented half**; the
> unmeasured half is unchanged.

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

## Previz costs the channels, and it is the RENDER — not the bridge

**This section previously said the bridge was the cost. That was wrong**, and the way it
was wrong is worth more than the conclusion: it rested on a single run.

The claim came from one measurement — "skip the VK→GL bridge entirely and the late frames
go to 0" — taken once, in a favourable machine state. Four fixes were then built on it
(deferring the submission, moving the fence wait to two different threads, and a full
double-buffer rewrite), none of which could be shown to help, which in hindsight is
exactly what should have been expected.

Re-measured properly, **interleaving the two conditions** so drift cannot favour either:

| | previz OFF | previz ON, **bridge removed entirely** |
| :--- | ---: | ---: |
| round 1 | 0 / 5012 | **267 / 5016** |
| round 2 | 1 / 5008 | **548 / 5011** |
| round 3 | 1 / 5011 | **331 / 5011** |

The bridge is absent from *both* columns of the right-hand condition — `post_channel`
returns before a slot is ever created — and the channels still run 5-11% of ticks long.
The only difference between the columns is that previz renders. **So the cost is the
previz 3D render on the OGL thread**: an extra render pass per channel per frame, whose
output becomes the channel's output.

Splitting the bridge's own two halves confirms it from the other side. Interleaved, all
four conditions land in the same band:

| bridge mode | round 1 | round 2 |
| :--- | ---: | ---: |
| none | 452 | 293 |
| GL samples the imported texture, no VK copy | 380 | 232 |
| VK copy runs, GL never samples | 368 | 223 |
| full | 856 | 229 |

Nothing separates them.

### What this measurement can and cannot answer

**Can**: previz on versus off. That comparison is enormous (0-1 against 230-1280) and
holds in every round, in every build, across hours.

**Cannot**: anything finer. The same code path measured 8-46 late in one sitting, 65-167
an hour later, and 293-548 later still. Only interleaved conditions mean anything here,
and even interleaved, differences smaller than about 3x vanish. Any future previz work
should establish that before believing a result — the failure mode is not noise around a
number, it is the number moving by 100x between sessions.

### What is left, and it is not in the bridge

The previz render itself is the thing to optimise: an extra pass per channel per frame,
and at five channels it puts 5-11% of ticks over budget. Nothing here has looked at
`previz_renderer` at all. The four bridge fixes are recorded in the commit history rather
than the tree; the double-buffer one is architecturally sound and produced a
pixel-identical picture, so it is worth reviving *if* the bridge ever turns out to matter,
but on this evidence it does not.

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

**Nothing.** No battery in the harness references PREVIZ. There is no picture check, no parity
check between mixers, and no check that a mapped mesh receives the channel's output at all.

That is recorded as the finding it is rather than as a to-do. The measured correlation from the
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

---

## 5. Known gaps

1. **No coverage at all** — see §4 for the first three checks worth writing.
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
