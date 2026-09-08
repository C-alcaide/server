# Interactive previz — where it renders, and what a new client should assume

> **Status:** PARTIAL — verified 2026-09-08 — **the recommendation in §6 is superseded by a
> fourth architecture that is now implemented.** The server's own window is interactive at
> **1 frame** (§4.1, `previz-interact`, both mixers, mutation-proved). §4's 80–120 ms budget
> was and remains correct for the **cross-process** case; it was never the only case, and §6
> recommended a client-side manipulator against a number that does not apply to the route that
> shipped. §5, §7.1 and §7.2's *conclusions* stand; §7.2's gap table is closed at its first row.
> The HTML half (§8) is scoped and in progress.
> **Falsifier:** `previz_renderer::input` exists in
> `src/accelerator/ogl/image/previz_renderer.cpp` and `core::compute_pick` in
> `src/core/stage/stage_math.cpp`. If either disappears, §4.1 and the status above are void and
> §6's recommendation is live again.

---

> **Read §4.1 before §4–§7.** Everything from §4 onwards was written on the assumption that a
> gesture has to cross a process boundary to reach previz. It does not. The screen consumer's
> window already had a Win32 message pump on the render thread, mouse and keyboard messages
> already arriving at its `WndProc` and falling through to `DefWindowProc`, and the consumer's
> factory has received the whole `channels` vector since 2018 without ever dereferencing it. The
> architectural objection to server-side interaction was never real, and no measurement in this
> file had looked for it — §4 measured the round trip it assumed was mandatory.

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

## 4. The CROSS-PROCESS latency budget, measured

This section is about a gesture made in **another process**. For a gesture made on the server's own window, see §4.1 — the two differ by two orders of magnitude, and this budget's four terms all vanish.

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

### 4.1 On the server's own window it is 1 frame — measured 2026-09-08

**Shipped, both mixers.** The screen consumer's window takes the mouse and the keyboard: right-drag
orbits the view camera, the wheel dollies it, middle-drag pans it, left-click picks the screen under
the pointer, left-drag moves that screen in its own plane, arrows nudge, `Esc` deselects. Gated on
the `<interactive>` element that already existed and until now only hid the cursor.
`docs/features/previz.md` §2.2 is the reference; `previz-interact` is the coverage, 16/16 on ogl and
on vulkan.

| | this route | §4's route |
| :--- | :--- | :--- |
| **latency** | **1 frame**, measured on `Event.frame` from one posted wheel message to the published camera change | 2–3 frames, two terms measured and two assumed |
| **transport** | none — the same `poll()` that already pumped the window, on the render thread with the GL context current | AMCP send, Spout hop, client paint |
| **render cost** | none measurable: A/B/A/B, four channels at 1080p50, a drag held for 12 s at 49 messages/second — 0 late frames dragging against 1 idle, `consume_max` inside its idle spread | a second renderer in the client |
| **fidelity** | exact by construction: it *is* the server's render | exact for the picture, schematic for the manipulator |

**Why every term in §4's budget disappears rather than shrinks.** The events do not cross a process
boundary, a socket, or a frame queue. They are read in the same call that pumps the window and
applied through the same mutators an AMCP command would use — so a drag is indistinguishable
downstream from `PREVIZ SCREEN … POSITION`, and moving a screen recomputes its ICVFX projection on
the mapped channel in the same frame.

**What made it cheap was not new code but an unused seam.** Three things were already true and
none of them had been checked when this file recommended C:

* `win32_gl_window::WndProc` — the fork's own Win32/WGL window, not `sf::Window` — already ran a
  `PeekMessage`/`DispatchMessage` pump **on the render thread**, and mouse and keyboard messages
  already arrived there and fell through to `DefWindowProc`;
* every consumer factory already receives the full `channels` vector and its own
  `channel_info.index`, and has done since 2018 without dereferencing either;
* `video_channel` owns both the stage and the image mixer, so one method on it —
  `input(const input_event&)`, trying the mixer first and the stage second — is the whole dispatch
  layer, and `core::image_mixer` already had a `state()` virtual to put it beside.

The only genuinely new maths is `core::compute_pick`, 15 property checks at every server start
beside `compute_frustum`.

**What this does NOT do**, so the trade is legible rather than implied: it is interaction **at the
server**, on the machine with the window. A remote operator still goes through §4's budget, and a
client still cannot hit-test without either the published stage or a renderer of its own. §4.1
removes the *latency* argument for a client-side manipulator; it does not remove the *remote*
argument for one.

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

**But note exactly what this section's evidence covers, because §4.1 sits just outside it.** Both
measured objections are about **embedding the server's window inside a client's window in another
process** — N full-resolution swapchains, and `SetWindowLong`/`QWindow.fromWinId` hanging the Qt UI
thread. Neither is an argument against the server's own window being interactive **where it already
is**: there is one swapchain, which the consumer creates anyway, and no cross-process reparenting at
all. This section was cited in the first draft of this file as though it settled server-side
interaction generally. It does not. It settles reparenting.

## 6. The three architectures, and the recommendation

| | interaction feel | fidelity to what the server renders | code to maintain |
| :--- | :--- | :--- | :--- |
| **A. Double rendering** (today) | <16 ms | **can diverge silently** | two renderers, two scene models |
| **B. Server-only** (Spout + AMCP camera) | 80–120 ms @25p | exact by construction | one renderer |
| **C. Client viewport as a controller** | <16 ms | exact, on the authoritative view | two, but the client one shrinks |
| **D. The server's own window** (§4.1, **shipped**) | **1 frame**, ~20–40 ms | exact by construction — it *is* the render | one renderer, and **no client at all** |

**D was missing from this table, and its absence is what made C look best.** Not because it was
weighed and rejected: the question "what if the interaction happened where the render already is"
was never asked, and every number in the table above it describes a round trip. D is now measured
and shipped, and it is strictly better than B and C on every column — for an operator at the
machine. Its one limit is exactly the one the other three exist to solve: it is not remote.

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

**Recommended for the EXISTING client, WITH D shipped: C, for the remote case only.** D covers the
operator at the machine, so C's remaining job is narrower than this section originally gave it —
someone driving previz from another desk. Keep its GL viewport, but change what it is *for*. It stops being an
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
| ~~**picking / hit-testing**~~ | **CLOSED 2026-09-08** — `core::compute_pick`, 15 property checks at boot, driven from the window (§4.1) | it was the real gate, and this file was right that it was. Screens are picked as *planes*; the venue mesh is drawn and still not pickable |
| **textured materials** | `baseColorTexture` is never read; flat base colour only | most of why a venue model does not look like the venue |
| **camera control** | already works | 0.7 ms + one frame over AMCP (§4); **1 frame** on the window (§4.1) |
| **rotation and resize handles** | position only | resize is blocked lower down: `size` and `arc` have no mutator, because `add_screen_*` rebuilds a fresh `screen_meta` and would discard everything else on it |

So the work to scope was **picking first, then materials**, and the first half is done. Interactivity
did indeed cost almost nothing once picking existed — but not for the reason given here: the input
path that supplied it was the window's own (§4.1), not §3's command route.

**And "it does not need to be pretty" is a statement about today's priorities, not a ceiling.** The
growth path is a sequence of rungs, each gated on its own measurement rather than on a redesign:
picking (done) → textured materials → a rotation handle → shadows → IBL. Nothing about the current
1782 lines forecloses any of them, and none of them requires a general 3D engine — which remains
the thing not to build.

### 7.3 What the new client should be

**Thin on previz: pixels in, control out, and no second scene renderer.** The 360-client's 1854-line
viewport exists because there was no other way to get an interactive view. There is now. Building a
second renderer into a green-field client would be re-adopting the divergence problem of §6 with
full knowledge of it.

The one genuine caveat is a UX call rather than a technical one: fully thin means camera drag at
**80–120 ms** on a 25p channel (§4) — remote-desktop feel. Three cheap outs, and §4.1 added the
third:

* **run previz on a 50p channel**, halving it to 40–60 ms;
* give the client a **manipulator** — a schematic hit-test layer that responds locally and
  reconciles to the server render. Note *manipulator*, not renderer: no materials, no textures, no
  scene fidelity, and therefore nothing that can diverge in a way that misleads an operator; or
* **do the placement on the server's window** (§4.1) and use the client for everything else. For an
  operator at the machine this is not a compromise at all — it is the lowest-latency, highest-
  fidelity option available, and it needs no client code. It answers nothing for a remote operator,
  which is why the first two outs remain on the table.

**A new client should therefore not carry a renderer.** That conclusion is stronger after §4.1 than
before it: the reason to build one was interaction feel, and interaction feel is now available
without one wherever the operator is at the machine. What the client still owes is the **return
leg** — reading the published stage back, which is §9's cheapest real step and is unaffected by any
of this.

**On "the server will now have an API":** the control API already exists and everything in §3–§4
was measured against it — `/v1/events` over WebSocket, a write path, and the stage published
including the previz camera. If the new work builds on that, the previz half is already there. If
it is something separate, this document's §3 conclusion still holds, since `PREVIZ CAMERA` is AMCP.

## 8. Separately: reviving the CEF interaction API

Not needed for previz. Worth scoping on its own for interactive HTML templates — **and now
scoped**: it is Part 2 of the interaction plan, downstream of §4.1's `core::input_event` and
`video_channel::input`, which were built to serve both. The window is already a source; what Part 2
adds is the *sink* — `stage_base::input` hit-testing to the topmost layer that wants the event, and
`html_producer::input` posting it onto CEF's own thread.

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

**A revival should not reinstate that shape** — and the reason is the *routing*, not the window.
This paragraph originally said the old design's defect was needing "a physical window with a real
mouse on it"; §4.1 shows that window is in fact the best input surface the server has. What was
wrong with the 2015 shape is that it put an `interaction_sink` on **every** consumer to reach
**producers**, and that five defects rode along with it: the SFML→CEF button cast swapped right and
middle, `e.modifiers` was never set so no in-page drag ever worked, there was no keyboard at all,
the window size was captured at construction, and `SendMouseMoveEvent` was called off CEF's UI
thread. So the input should reach the browser as a plain `input_event` from either source — the
window or a command — an AMCP or API verb carrying normalised coordinates to a layer, calling
`SendMouseClickEvent`/`SendMouseMoveEvent`/`SendKeyEvent` on the browser host. That is a smaller
change than the original (no `core/interaction/` subsystem, no sink on every consumer), it gets
the **keyboard the original never had**, and it works headless.

Estimate, deliberately rough: a day for the command and the CEF calls, and considerably longer to
decide what the security posture is — `--disable-web-security` is already set on that browser, and
adding remote synthetic input to it deserves more thought than this paragraph.

## 9. What this scope does not establish

* **This section was the warning that turned out to matter most.** Two of its bullets — the
  unmeasured Spout hop, and picking not needing to be server-side — were the load-bearing
  uncertainties under §6's recommendation, and the recommendation was made anyway. What settled it
  was neither: it was a route this file never listed (§4.1). **A scope document's "what this does
  not establish" section is not a disclaimer; it is a list of the places the conclusion can be
  wrong.**
* **The Spout hop and client paint were not measured**, only bounded at a frame each. The 80–120 ms
  figure is therefore an estimate with two measured terms and two assumed ones. Still true, and
  still the number for the cross-process case.
* **Nothing was measured at 1080p50**, where the budget halves and the answer may change.
* **No operator has tried it.** "80–120 ms is irritating for precise placement" is a judgement, not
  a finding, and it is the judgement the whole recommendation turns on.
* ~~**Picking need not be server-side, and this scope does not prove it should be.**~~ **Decided
  2026-09-08, and this bullet had the trade right.** Picking went server-side, and the deciding
  argument is the one named here: what the operator clicks matches what the server drew, exactly,
  because it is the same geometry and the same camera. The cost the bullet worried about was
  measured and is not there — `compute_pick` against screen planes on the render thread moves no
  frame period and produces no late frame (§4.1). The venue-mesh case is still open, and is still
  the case where a client-side raycast would win: `compute_pick` tests `screen_meta` quads only.
* **§7.2's gap table was read off the source, not exercised** — and its first row has since been
  exercised the hard way, by implementing it. "Flat base colour" is still only
  `baseColorTexture` never appearing in the source, unconfirmed by rendering anything.
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
