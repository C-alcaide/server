# PREVIZ — 3D pre-visualisation

> **State:** shipped, unmeasured
> **Modules:** `src/modules/previz` (bridge in `src/accelerator/ogl/image/image_mixer`)
> **Commands:** 13 fork-specific AMCP commands
> **Architecture:** none, deliberately — shares the ICVFX state the projection commands write -- see features/previz.md §on the two routes
> **Guide:** [`../guides/PREVIZ_3D_MODULE.md`](../guides/PREVIZ_3D_MODULE.md)
> **Coverage:** **none** — no harness battery drives any of these

Loads a 3D scene, maps channel output onto meshes in it, and renders a camera view of the result —
so a projection design can be checked without the venue. Screens, presets and camera positions are
addressable by name at runtime.

> **Read §4 first.** This is the largest undocumented and unmeasured surface in the fork: 13
> commands, none of them exercised by any battery. That combination is exactly what ICVFX was
> before an audit found a live colour defect in it, and this document exists because that
> correlation was measured rather than assumed.

---

## 1. What is implemented today

Verified against `src/protocol/amcp/AMCPCommandsImpl.cpp` at the definition line given.

| command | shape | def |
| :--- | :--- | ---: |
| `PREVIZ SCENE` | `<path>` — loads a scene file | 4777 |
| `PREVIZ MAP` | `<mesh_name> <channel>` — needs ≥2 parameters, else `400` | 4847 |
| `PREVIZ UNMAP` | `<mesh_name>` | 4869 |
| `PREVIZ CAMERA` | `RESET` \| camera placement arguments | 4887 |
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
gains — is invariant under exactly that defect. **PREVIZ is the same shape, thirteen times over,
and on the same OpenGL path.**

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
2. **OpenGL-only**, undocumented until now. Either the Vulkan mixer grows the same bridge or the
   commands should refuse on a Vulkan channel with a message saying why, rather than silently
   doing nothing.
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
