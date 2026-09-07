# Interactive previz — where it renders, and what a new client should assume

> **Status:** SCOPE, written while a **new client** is being designed (§7). Nothing here is
> implemented. The **latency numbers in §4 are measured**
> (2026-09-08, this box, 1080p25, OpenGL mixer); everything else is a design argument.
> **Falsifier:** `PREVIZ <ch> CAMERA` is registered at `AMCPCommandsImpl.cpp` as
> `previz_camera_command` and takes `x y z yaw pitch roll fov`. If that command or its
> parameter order changes, §3's conclusion — that no new server input path is needed — is void.

---

## 1. The correction that reframes the question

**Previz is not HTML, so the CEF interaction API is irrelevant to it.** `previz_renderer` lives in
`src/accelerator/ogl/image/previz_renderer.h` — it is the *OpenGL mixer's own* 3D renderer, and
`image_mixer.cpp` calls `previz_renderer_.render(target_texture, …)` straight into the channel's
target texture. No browser is involved at any point.

So reviving the interaction API removed in `efcd57858` would **not** make previz interactive. That
is a separate feature, for interactive HTML *pages*, and it is scoped separately in §8.

## 2. Nothing new is needed in Spout, and no new language is needed either

Worth stating plainly because it is the natural first assumption: **Spout does not become
interactive, and nothing about it changes.** The two directions are independent one-way flows.

```
  pixels     server ──── Spout (texture) ────▶ client        unchanged
  control    client ──── AMCP / control API ─▶ server        exists today
```

Neither transport needs to know about the other. What the two ENDS must agree on is the scene and
its coordinates — and they already do, through the `PREVIZ` command set and glTF (§6).

## 3. Interactive previz needs no new server input path — it already exists

`PREVIZ <ch> CAMERA x y z yaw pitch roll fov` sets the previz camera; with no parameters it
returns the current one. The camera is also published at
`/channel/{n}/mixer/previz/camera/position` on the control API's `/v1/events` WebSocket.

And the client is already equipped: `casparcg-360-client/amcp_client.py` carries a `_PREVIZ_ERRORS`
table mapping 400/403/404/502 to operator messages, so it speaks the command family today.

**A mouse drag in the client can therefore drive the server's previz camera with no server change
at all.** The whole question is whether the resulting latency is acceptable, which is §4.

## 4. The latency budget, measured

Measured 2026-09-08 on this box, 1080p25, OpenGL mixer, 20 samples:

| stage | measured |
| :--- | :--- |
| AMCP command round trip | **0.7 ms** median, 1.0 p90, 1.2 max |
| command → new camera published by the server | **38.4 ms** median, 39.2 p90, 39.7 max |

The second number is **one channel frame** (nominal 40 ms). The camera applies on the next frame
boundary, which is the correct and expected behaviour — not a queue.

Adding the parts not measured here, each bounded by a frame:

```
  client mouse event      ~0
+ AMCP send               0.7 ms          measured
+ server apply + render   1 frame         measured (38.4 ms at 25p)
+ Spout hop               ~1 frame        not measured
+ client paint            1 client frame  not measured
= 2-3 frames end to end   80-120 ms at 1080p25
                          40-60  ms at 1080p50
```

Against a client-side GL viewport responding in **under 16 ms**. That gap is the entire trade.

80–120 ms on a camera orbit is roughly remote-desktop feel: fine for "look around the venue",
irritating for precise placement. At 50p it halves and becomes broadly comfortable.

## 5. Do NOT go back to an embedded screen consumer

This decision was already made, on measured grounds, and is recorded in the client's own
`spout_preview.py` header. Embedding was done by HWND reparenting (`window_embedder.py`,
`QWindow.fromWinId` + `createWindowContainer`) and abandoned because:

* **it does not scale** — every embedded view is a full-resolution CasparCG swapchain presenting
  at vsync in another process, so N previews are N full presentations;
* **it is cross-process and deadlocks** — `window_embedder` carries its own comments about
  `SetWindowLong` and `QWindow.fromWinId` hanging the Qt UI thread when CasparCG's window thread
  is busy, guarded with `IsHungAppWindow`.

The Spout path exists precisely because of that, and the measurements behind it are in the same
header: `glReadPixels` at 1920×1080 costs 3.12 ms (19% of a 16.67 ms frame) and at 256×144 costs
0.088 ms, so the receiver never lets a pixel reach Python — it receives into a GL texture and
draws a quad.

**Nothing in the interactivity question changes any of that.** Going back to embedded windows
would re-adopt a scaling problem and a deadlock class in exchange for interactivity that §3 shows
is available over AMCP anyway. `window_embedder` should stay only as the degradation path it
currently is.

## 6. The three architectures, and the recommendation

| | interaction feel | fidelity to what the server renders | code to maintain |
| :--- | :--- | :--- | :--- |
| **A. Double rendering** (today) | <16 ms | **can diverge silently** | two renderers, two scene models |
| **B. Server-only** (Spout + AMCP camera) | 80–120 ms @25p | exact by construction | one renderer |
| **C. Client viewport as a controller** | <16 ms | exact, on the authoritative view | two, but the client one shrinks |

**A's real defect is not duplication, and it is not a missing shared language either — both of
those were wrong in the first draft of this file.** The two ends already speak one:

* the **client authors** the scene (`previz_scene_model.py`: `ScreenConfig`, `LEDPanelPreset`,
  `SceneNode`, `PrevizCamera`);
* it **pushes** it with `PREVIZ <ch> SCREEN …` field by field, plus `PREVIZ <ch> SCENE <path>` for
  a glTF/obj venue mesh — `stage_tab.py:1613` is literally *"Send PREVIZ AMCP commands to recreate
  all screens on the server"*;
* the **server publishes the resulting stage back** on the control API, which is what `api-stage`
  and `tracking-previz` read.

**The defect is that the return leg is never used.** The client references the control API
**zero** times — no WebSocket, no `/v1/` anywhere in it — so the push is one-way and
fire-and-forget, into an `AMCPClient.send()` that returns `None` by construction. Nothing compares
what the server ended up holding against what the client believes it sent, so the two pictures can
disagree and no code path notices.

That reframes the work. It is not "make them talk the same language"; it is **close the loop on a
language that already exists**. The reconciliation mechanism is already built and published — it
simply has no consumer.

**Recommended for the EXISTING client: C.** Keep its GL viewport, but change what it is *for*. It stops being an
independent preview and becomes a **manipulator**: it draws the stage schematically for hit-testing
and dragging, emits `PREVIZ <ch> CAMERA …` as the user moves, and the **Spout-received server
render becomes the authoritative picture** shown beside or behind it.

Why C over B, given B deletes 1850 lines:

* interaction stays sub-frame, so precise placement work stays pleasant;
* the client renderer no longer has to be *pixel-accurate*, only *spatially* correct — so it can
  lose its shaders, its material handling and most of its scene model over time, rather than in
  one change;
* it is the smallest step from where the code already is, and B remains reachable later if the
  measured lag turns out to be acceptable in practice.

Why not B outright: at 25p the lag is on the edge of usable, and the decision is easier to make
later with a real operator in front of it than now from a latency budget.

**For a NEW client the answer is different — see §7.**

**The one thing C must add that A does not have:** a *visible* statement of which view is
authoritative. If the manipulator and the Spout render disagree, the operator must be able to see
that they disagree rather than trust whichever is in front.

## 7. A NEW client changes the answer — and the constraint that decides it

§6 was written for the client that exists. A **new** client is being scoped, and it is green-field,
so there is no sunk renderer to protect. That is the right moment to make this call deliberately.

### 7.1 Previz cannot move. This is a constraint, not a preference

`image_mixer.cpp` says it, at the call site:

> *If previz is active, first do normal 2D compositing to post this channel's output to the
> texture store (so other screens — including screens mapped to **this** channel — can sample it),
> then render the 3D scene for the previz viewport.*

**Screen meshes sample live composited channel output, across channels, on the same GPU at the
same tick.** Nothing else in the system can do that:

* a **client-side** renderer would need every mapped channel as live video — which is the reason
  Spout exists, and why the 360-client's viewport can only draw a schematic;
* a **browser-based** renderer (PlayCanvas or three.js in an html producer, now that WebGPU works —
  `WEBGPU_IN_THE_HTML_PRODUCER.md`) cannot either, because **a page cannot see the channel's other
  layers**. That is the same §1 limit that rules out matting the programme output in a template.

So "where does previz render" is settled: **inside the mixer, permanently**. Any new-client design
that assumes otherwise is designing around a wall.

### 7.2 "Revamp the 3D engine" is the wrong frame

The engine is **1782 lines** (`previz_renderer.{h,cpp}` + `previz_scene.h`) and is not trying to be
a beauty renderer: grid, venue mesh, screens carrying real output, wireframe, gizmos. Previz has to
be **accurate**, not pretty — the questions it answers are about geometry, projection, calibration
and what is actually on the walls.

Writing a general 3D engine in C++ inside a media server is an open-ended commitment outside this
fork's competence. It would buy fidelity previz does not need while putting at risk the geometry
correctness it does.

**What is actually missing is narrower, and one item gates everything else:**

| gap | today | why it matters |
| :--- | :--- | :--- |
| **picking / hit-testing** | **nothing** — no raycast, no unproject, 0 references | the real gate on interactivity. Without it you get camera orbit and nothing more: no click-a-screen, no drag-a-screen, no gizmo the operator can grab |
| **textured materials** | `baseColorTexture` is never read; flat base colour only | most of why a venue model does not look like the venue |
| **camera control** | already works | 0.7 ms + one frame, measured in §4 |

So the work to scope is **picking first, then materials**. Interactivity then costs almost nothing,
because §3 already supplies the input path.

### 7.3 What the new client should be

**Thin on previz: pixels in, control out, and no second scene renderer.** The 360-client's 1854-line
viewport exists because there was no other way to get an interactive view. There is now. Building a
second renderer into a green-field client would be re-adopting the divergence problem of §6 with
full knowledge of it.

The one genuine caveat is a UX call rather than a technical one: fully thin means camera drag at
**80–120 ms** on a 25p channel (§4) — remote-desktop feel. Two cheap outs:

* **run previz on a 50p channel**, halving it to 40–60 ms; or
* give the client a **manipulator** — a schematic hit-test layer that responds locally and
  reconciles to the server render. Note *manipulator*, not renderer: no materials, no textures, no
  scene fidelity, and therefore nothing that can diverge in a way that misleads an operator.

**On "the server will now have an API":** the control API already exists and everything in §3–§4
was measured against it — `/v1/events` over WebSocket, a write path, and the stage published
including the previz camera. If the new work builds on that, the previz half is already there. If
it is something separate, this document's §3 conclusion still holds, since `PREVIZ CAMERA` is AMCP.

## 8. Separately: reviving the CEF interaction API

Not needed for previz. Worth scoping on its own for interactive HTML templates.

**What existed.** Added 2015-05-21 (`e050e99b7`, Helge Norberg), removed 2018-02-12 (`efcd57858`,
ronagy, "refactor: removed interaction API", 512 deletions across 31 files, **no reason recorded**).
It sits inside a run of removals by the same author in that period — `removed reroute`,
`removed image_scroll_producer`, `removed interlace patterns` — so *unused surface area cut during
the 2.2-era cleanup* is the fair reading, though the commit does not say so.

**What it was.** SFML window events in the screen consumer, normalised to 0–1, into a
`core::interaction_sink`, routed through `stage.cpp` to layer and producer, and turned back into
`CefMouseEvent` by the html producer. Two limits: **mouse only** — `interaction_event.h` had
`mouse_move_event`, `mouse_button_event` and `mouse_wheel_event` and never a keyboard event — and
**screen consumer only**, since no other consumer ever produced events despite all of them
carrying the sink.

**A revival should not reinstate that shape.** The old design needed a physical window with a real
mouse on it. The fork now has a control API with a write path, so the input should arrive as a
command instead — an AMCP or API verb carrying normalised coordinates to a layer, calling
`SendMouseClickEvent`/`SendMouseMoveEvent`/`SendKeyEvent` on the browser host. That is a smaller
change than the original (no `core/interaction/` subsystem, no sink on every consumer), it gets
the **keyboard the original never had**, and it works headless.

Estimate, deliberately rough: a day for the command and the CEF calls, and considerably longer to
decide what the security posture is — `--disable-web-security` is already set on that browser, and
adding remote synthetic input to it deserves more thought than this paragraph.

## 9. What this scope does not establish

* **The Spout hop and client paint were not measured**, only bounded at a frame each. The 80–120 ms
  figure is therefore an estimate with two measured terms and two assumed ones.
* **Nothing was measured at 1080p50**, where the budget halves and the answer may change.
* **No operator has tried it.** "80–120 ms is irritating for precise placement" is a judgement, not
  a finding, and it is the judgement the whole recommendation turns on.
* **Picking need not be server-side, and this scope does not prove it should be.** §7.2 lists it
  as a server gap because the engine has none — but the stage is published, so a client could
  raycast against the *same* geometry it already authored and send the result as a command. That is
  cheaper and keeps the engine small; it is worse if the venue mesh is large or if what the operator
  clicks must match what the server drew exactly. Nobody has costed either.
* **§7.2's gap table is read off the source, not exercised.** "No picking" is 0 references to
  raycast/unproject/hit-test; "flat base colour" is `baseColorTexture` never appearing. Neither was
  confirmed by rendering something and observing the absence.
* **"80–120 ms is remote-desktop feel" is still a judgement**, and §7.3 now rests a new-client
  decision on it. One operator with a mouse for ten minutes would settle it better than this file.
* **The cheapest real step is not any of the three architectures.** It is to make the client
  *read the published stage back* after it pushes, and show the operator when the server's stage
  differs from what it sent. That needs no renderer decision, no Spout change and no new command —
  only a WebSocket the client does not currently open — and it makes the divergence in §6 visible
  instead of theoretical.
* **The divergence in option A is argued, not demonstrated** — no one has shown the client and
  server previz disagreeing on a real scene. Doing so would strengthen the case considerably, and
  a `previz-parity` check comparing a Spout-received server render against the client's own
  viewport is the obvious way.
