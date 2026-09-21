# Plan: a drift correction the audience cannot see

> **Status:** UNIMPLEMENTED — 2026-09-21 — **PARKED, scoped only.** Nothing here is built. The
> design below is a decision list, not a task list, and §3's leading option removes most of the
> work the idea appeared to need.
> **Falsifier:** `prepare_standby`, `swap_to_standby`, `standby_layer_index` — if any of these
> appears under `src/`, this plan is no longer unimplemented and this line is the thing to fix
> first.

---

## 1. The problem

`content_sync` detects drift correctly and then corrects it like this:

```cpp
std::vector<std::wstring> seek_params = {L"seek", std::to_wstring(expected)};
auto seek_future = channel->stage()->call(tp.layer_index, seek_params);
```

A **bare seek on the live layer**. The frame it lands on is right; the way it arrives is a
visible jump. On a video wall the entire proposition is that the audience cannot tell the
machines apart, and a snapping correction is the one moment they can — so the correction is most
visible exactly when the system is doing its job.

This is not a rare path. The watchdog corrects whenever drift exceeds `drift_threshold_`, counts
it in `total_corrections_`, re-anchors `start_frame` and holds a 15-frame cooldown. It is
designed to fire.

## 2. Where the idea comes from, and the half not to take

[`FiLORUX/casparcg-auto-sync`](https://github.com/FiLORUX/casparcg-auto-sync) — MIT, 702 lines of
Node.js, an external controller. It is a **much weaker thing than this module** and worth being
clear about why, so nothing else gets borrowed by accident: its `t0` is wall-clock on the
controller, and it measures drift by polling `CALL <ch>-<layer> FRAME` over TCP, so the
measurement's own jitter sits inside the control loop. We have PTP, a relay and scheduled frame
execution. Nothing about its clock is worth having.

**Its one good idea is that it never touches the visible layer.** It keeps a standby layer with
the same clip — `LOADBG … SEEK <frame> LOOP`, then `PAUSE`, then `OPACITY 0` — and swaps active
for standby inside a single `DEFER`/`RESUME` bracket, so both opacity changes land in one render
cycle. Cut, or crossfade over N frames.

**And the half to leave behind:** their standby is parked at the frame that was due when the
PRELOAD was issued, and the swap happens a network round trip later. Their correction therefore
carries an error equal to its own preload latency — which is precisely the quantity a drift
corrector exists to remove. Re-derive against our clock; do not port the sequence.

## 3. The design, and why it is smaller than it looks

**A cut needs no second layer.** `core::layer` already has a background slot:

```cpp
void load(spl::shared_ptr<frame_producer> producer, bool preview, bool auto_play = false);
void play();                       // promotes background to foreground
spl::shared_ptr<frame_producer> background() const;
bool has_background() const;
```

`play()` is the atomic promotion, on a frame boundary, inside the stage. So the whole of their
two-layer, opacity-juggling, `DEFER`/`RESUME` apparatus collapses to: **load the corrected
producer into the background, then `play()` on the chosen tick.** They needed two layers because
an external controller wanting a *crossfade* needs both visible at once; we do not need a
crossfade to fix one frame of drift.

That reduces the work to the part that is actually interesting: **which frame to park at.**

We schedule on the tick, so unlike an external controller we know the frame the swap will land
on. The standby is parked at the frame due at **T**, not at the frame due when the load was
issued:

```
T        = current_global + lead
park_at  = T - tp.start_frame          // the content frame due at T
```

`lead` must be **measured, not guessed** — it is how long a `load` takes to become playable for
this producer and raster, and it differs per codec. A lead that is too short means the swap
happens before the background is ready; too long and the correction is late and the drift has
moved on. This is the number the whole design rests on, and it is the first thing to measure.

## 4. Four decisions, none of them free

1. **Does the background slot survive what else uses it?** `LOADBG` from an operator, a
   transition producer, and `auto_play` all touch the same slot. A watchdog quietly consuming it
   would break an operator's prepared next clip — silently, because nothing reports that the
   background was replaced. **This is the decision the rest depends on.** If the answer is no,
   the second layer comes back and question 2 becomes much heavier.
2. **The cost of a second producer.** The corrected producer must decode to be playable, so for
   `lead` frames there are two decoders on that layer. `replay`'s cost arm found one recording
   spending an eighth of a frame budget; a second decoder of the same content is larger. It must
   be released after the cooldown, and the peak — not the mean — is what matters on a wall node
   already carrying four channels.
3. **What happens when the background is not ready at T.** A correction that silently does not
   happen is *worse* than a visible one, because the operator now believes the wall is locked. It
   has to either retry with a longer lead or fall back to the bare seek and say so in the log.
4. **Whether `frame_number()` still means what the watchdog thinks** after a background swap.
   `tp.start_frame` is re-anchored after a correction today; the swap changes which producer
   object answers, and the GPU-direct producers were reporting their DECODE clock until
   `2229346f3`. Re-anchoring against the wrong clock would make the corrector chase itself.

## 5. The oracle comes before the code

**"The correction is invisible" is not checkable by asserting the picture is right afterwards** —
a bare seek also ends up right. That is the `MIXER EXPOSURE` shape from the root `CLAUDE.md`,
reached from the other side: the thing that must be true is about the frames *during* the
transition, and the end state is identical either way.

The property is: **no frame between the last pre-correction frame and the first post-correction
frame shows content belonging to neither.** That needs consecutive frames in order, which means
`frame_hashes` over a recording rather than the IMAGE consumer — the IMAGE consumer is one `ADD`
per frame and cannot sample consecutive frames, which is the same constraint `html-cadence`
documents.

Two things that would make the check vacuous, both already paid for elsewhere in this tree:

* **A fixture whose frames are identical** cannot distinguish a clean swap from a jump, a repeat
  or a dropped frame. It needs a burnt-in frame marker — `frame_pin.py` exists and
  `producer-params` already uses it to assert that `position = N` returns frame N.
* **Recording needs the consumer removed with the same arguments the `ADD` carried.**
  `PLAYBACK_AND_RECORDING_GUIDE.md`, *"`REMOVE` needs the same arguments as the `ADD`"*: a bare
  `REMOVE 1 FILE` answers `404` and finalises nothing, and killing the server instead leaves a
  48-byte `ftyp` stub that reads exactly like a recorder that never ran.

And the correction has to be **provoked deterministically** rather than waited for. Forcing
`drift_threshold_` low and nudging one node is the obvious route; a test that waits for natural
drift measures the clock, not the corrector.

## 6. What would make this not worth doing

Stated now, because the answer may be no and that is a legitimate outcome:

* If **question 1 resolves against us** and a second layer is required, the cost is a reserved
  layer index that must be invisible to `CLEAR`, to the tree, to `state_leaves` and to anything
  enumerating layers — a much larger surface than the correction it buys.
* If **`lead` turns out to be large** for the codecs actually used on a wall, the corrector
  becomes slow enough that drift moves during the preparation, and a visible-but-immediate
  correction may genuinely be better than an invisible-but-late one.
* If in practice **drift corrections are rare** on a PTP-locked cluster, this is polish on a path
  that fires once a week. Nobody has counted: `total_corrections_` is published and has never
  been read over a long run, and that measurement costs nothing compared to the work below.

**That last one is the cheapest next step and should come first** — run a cluster for a day and
read `total_corrections_`. A correction mechanism nobody sees firing does not need to be
invisible.

## 7. What this plan does not establish

Nothing here has been built, measured or prototyped. `lead` is unmeasured, the background-slot
question is unanswered, and the correction rate on a real cluster is unknown. The `cluster`
battery (8/8, both mixers, two nodes on one box) **cannot see any of this** — it touches no pixel
during a correction, so it would pass unchanged whether the swap is seamless or not.
