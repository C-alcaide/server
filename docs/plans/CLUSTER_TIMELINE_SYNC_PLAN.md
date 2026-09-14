# Plan: a timeline that stays in sync across a cluster

> **Status:** RESEARCH — 2026-09-14 — **PARKED, blocked on hardware: this needs TWO MACHINES.**
> Everything measurable on one box has been measured and is in §1; §2 is why that is not the
> answer. **Do not re-measure on one box** — a second fixture there will produce the same table
> of zeroes for the same structural reason, and reading it as "no drift" is the mistake this
> plan exists to prevent
> **Falsifier:** `timeline_clock_source`

Two cluster nodes can be told to **start** a timeline on the same frame today, and that is
measured. Nothing keeps them together afterwards. This is what was measured, what could not be,
and the three shapes the fix could take.

---

## 1. What is true today, measured

**Starting together works and is frame-exact.** `TIMELINE` is a registered AMCP channel command
and the relay carries AMCP text, so `CLUSTER SCHEDULE TIMELINE 1 PLAY <name> AT <frame>` reaches
every node. `cli.py cluster` gates it: both nodes report `LAST-EXEC-TARGET` == `LAST-EXEC-ACTUAL`
== the scheduled frame, at 1080p59.94 and again at 2160p50 under a real 4K decode.

**Staying together is implemented by nothing.** `transport::position_at(frame, per_frame)` is
handed `video_channel::frame_counter_` — a plain `uint64_t` initialised to 0 and incremented once
per tick. Neither `video_channel.cpp` nor `stage.cpp` contains a single reference to `frame_clock`
or to the cluster module. The PTP clock and the timeline never meet.

**The content-sync watchdog does not cover it.** It tracks `tracked_producer` — producer
transitions and seek corrections per channel and layer. A timeline's position is not among them.

### Measured on one box, two nodes, 2026-09-14

| measurement | result |
| :--- | :--- |
| two timelines, started on the same cluster frame, sampled for 175 s | within **±1 frame** throughout; no trend |
| each channel's `frame` against its own `CLUSTER STATUS` FRAME, 230 s | **0 frames slipped**, 0 ppm, on both nodes |
| the two channels against each other over 230 s | 13791 frames each, **0 apart** |

---

## 2. Why those numbers do not answer the question — and that is the finding

**Both measurements are structurally incapable of showing the drift they were pointed at.** Two
processes on one machine share one hardware oscillator, so their channel ticks *cannot* differ in
rate, and the PTP clock they are compared against is disciplined from that same oscillator. The
mechanism under investigation is absent from the fixture by construction.

The ±1 frame in the first row is also the measurement's own floor: the published position is
frame-quantised, and the sampling gap between two AMCP round trips is up to a frame. A reading of
"±1 frame" there is indistinguishable from zero.

**So the honest statement is not "there is no drift" but "this box cannot produce any".** Recorded
because the opposite conclusion is the easy one to draw from a table of zeroes, and because it
bounds the problem from below rather than answering it.

### What decides the real case

A channel's tick is paced **by its consumers** (`has_synchronization_clock()`), not by a timer the
cluster controls. So the question is what those consumers are locked to:

* **Genlocked SDI on every node** — the channel ticks already share a reference, the timelines
  stay aligned, and this plan is unnecessary for those deployments. *This is the normal broadcast
  case and should be stated first in any advice to an operator.*
* **Free-running** (screen consumer, NDI, a software clock) — each node follows its own crystal.
  Two commodity oscillators at ±50 ppm can differ by 100 ppm: **~0.36 s/hour, about 21 frames/hour
  at 59.94**, one frame every ~3 minutes. That is derived from the hardware tolerance, **not
  measured here**, and is flagged as such.

**Measuring it needs two machines.** No fixture on one box will do, and a plan that pretends
otherwise will produce another table of zeroes.

---

## 3. The three shapes a fix could take

| | approach | what it costs |
| :--- | :--- | :--- |
| A | **Timeline advances on the cluster `frame_clock` when clustered**, on `frame_counter_` otherwise | The smallest change that fixes the stated problem, and it decouples the timeline from the thing that renders it: the position would come from PTP while the *pictures* still come out at the channel's tick, so a node whose channel is slow would skip timeline positions rather than fall behind. That may be exactly right — it is what a genlock failure should look like — but it must be a deliberate answer, not a side effect |
| B | **Discipline the channel tick itself** to the PTP clock when clustered and not otherwise genlocked | Fixes every time-dependent thing at once, the timeline included, and is what an operator means by "in sync". It is also much larger: the tick is paced by consumers today, and taking that over interacts with `has_synchronization_clock()`, the audio cadence, and every consumer that paces |
| C | **Periodic re-anchor** — leave the timeline on `frame_counter_` and have the cluster re-issue the transport anchor when divergence exceeds a threshold | Cheapest, and the same shape as the content-sync watchdog already in the module (`drift_threshold`, seek corrections). Corrects rather than prevents, so the picture jumps by the accumulated error at each correction — acceptable for a long show, wrong for a video wall |

**A is the recommendation to evaluate first**, because it is the smallest change that addresses
what was asked, and because C's machinery already exists and can be added later without
conflicting with it. B is a genlock feature wearing a timeline feature's clothes and should be
planned separately if it is wanted.

---

## 4. What to do before writing any code

1. **Decide whether this is a real deployment scenario.** If every clustered node is genlocked —
   which is the usual case for a wall — the drift is zero for a reason that has nothing to do with
   the timeline, and the right deliverable is a paragraph in `CLUSTER_SYNC.md` saying so rather
   than a code change.
2. **If it is, get two machines and measure**, at 59.94 free-running, for at least an hour. The
   number wanted is frames-per-hour of separation. Everything in §2 is a bound or a derivation.
3. **Then pick from §3 against that number**, because the three differ mostly in how they behave
   at a divergence the measurement has not yet supplied.

---

## 5. Out of scope

* `at_frame` on the control API, which has the same root cause — `control-api.md` gap 5 records
  it: *"`at_frame` is this server's own frame counter, so two servers cannot yet be told to change
  together."* A fix for either should probably serve both, which is an argument for doing the
  design once.
* The content-sync watchdog's coverage of timelines (approach C's prerequisite).
* Genlock itself.
