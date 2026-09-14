# Plan: the producer control surface reaches the API — a declared transport contract

> **Status:** SHIPPED — 2026-09-14 — P1-P5 all landed; §4 marks each phase and §6 what is still owed
> **Falsifier:** none — SHIPPED carries its outcomes in §4, which marks each phase

Arose from the replay work of the same day (`docs/features/replay.md` §4.1), which found a module
whose entire interaction is scrubbing and whose entire control surface is AMCP.

---

## 1. The finding this rests on, measured rather than asserted

```
producers implementing frame_producer::call()          13
producers implementing frame_producer::parameters()     2   (isf, ofx)
/v1 routes onto call()                                  0
```

`run_action` (`api_action.cpp`, `queue_verb`) is a **fixed switch over seven stage verbs** —
`play`, `stop`, `pause`, `resume`, `preview`, `clear`, `clear_transforms`. Nothing anywhere
forwards to a producer's own `call()`. So the control API can *watch* any producer and *drive*
almost none of them.

**And the thirteen are not thirteen different vocabularies.** Extracted from each `call()`:

| producer | transport verbs it implements |
| :--- | :--- |
| `ffmpeg_producer` | `loop` `speed` `in`/`start` `out` `length` `seek` `pingpong` |
| `notchlc_producer` | `loop` `speed` `in` `out` `length` `seek` `pingpong` |
| `prores_producer` | `loop` `speed` `in` `out` `length` `seek` `pingpong` |
| `hap_producer` | `loop` `speed` `in` `out` `length` `seek` `pingpong` |
| `replay_producer` | `loop` `speed` `in` `out` `seek` (+ `export`) |
| `gst_producer` | `seek` `position` `length` `play` `pause` `resume` |
| `dmx_producer` | `seek` only — **this row was WRONG when written**, and the correction is the useful half: `in` and `out` here are ARGUMENTS to `SEEK` (`SEEK in`, `SEEK out`), and the survey grep found the words in the argument position. Same shape as the `PREVIZ MAP` note in `CLAUDE.md`, where a grep for a two-word form matched the wrong half of the grammar. **Read the dispatch, not the strings** |
| `image_scroll_producer` | `speed` |

**Eight producers share one transport vocabulary, hand-written eight times, reachable only over
AMCP.** There is already a de-facto interface here; it has simply never been declared, so nothing
can address it generically — not a client, not a timeline, not a binding.

The remaining five are not transport at all and are out of scope below: `isf` and `ofx` already
declare `parameters()`; `remotewall` has configuration verbs (`BINDIP`, `PORT`, `CODEC`);
`html` and `flash` carry template/CG verbs.

---

## 2. What already exists, which is why this is cheap

**`core/producer/producer_params.h` was built for exactly this** and its own header says so:

> *"Declaring them here makes them describable and writable through the same machinery a mixer
> field uses, which is also the prerequisite for BINDING one to a live source — a binding
> resolves a target path to a descriptor and a setter, and neither existed for a producer."*

A `param_desc` declared by a producer already gets, with **no protocol-layer work at all**:

| | mechanism | verified by |
| :--- | :--- | :--- |
| publication in the tree | `api_tree.cpp`, the `params` node under `foreground` | `producer-params` |
| typed JSON write with range + arity checking | `api_value.cpp` → `stage::set_param` | `producer-params` |
| a **binding** target | binding resolves path → descriptor + setter | `binding-*` |
| a **timeline keyframe** target | `timeline-targets`, gated as a 1 LSB picture | `timeline-targets` |
| forwarding through every wrapper | `frame_producer_registry`, `separated_producer`, `sting_producer`, `transition_producer` — **all four already forward `parameters()`**, checked 2026-09-14 | — |

That last row matters more than its size: the `destroy_producer_proxy` trap in `CLAUDE.md` is a
virtual dropped by a hand-written forwarder, and it would have made a parameter vanish under a
transition with no symptom. It is already closed.

`fields::value_type` covers `boolean`, `integer`, `real`, `enumeration` — every type this plan
needs — and `fields::access_t` has a **write-only** state, whose comment says *"a trigger is
write-only, and a boolean `writable` cannot say so."*

**So the work is declaring parameters in producers, not building API machinery.**

---

## 3. Decisions

| | decision | why |
| :--- | :--- | :--- |
| D1 | **A SHARED, NAMED transport contract in `core`, not per-producer names.** A header declaring the canonical name, type, range and semantics of each transport parameter, which each producer builds its `param_desc` rows from | Eight producers declaring eight spellings of "speed" reproduces the AMCP problem one layer up: a client would special-case per producer, which is the thing being fixed. The contract is the deliverable; the rows are the easy part |
| D2 | **`seek` becomes `position`, a `read_write` integer — state, not an action** | Writing a frame number *is* setting the playhead, and the getter already exists as the published `file/frame`. As state it is scrubbable, **bindable and keyframable**; as an action it is none of those. This is the single highest-value line in the plan |
| D3 | **Parameter names match the AMCP verb wherever the verb is already state** — `loop`, `speed`, `in`, `out`, `length`, `pingpong` | A second vocabulary for the same quantity is a migration cost with no benefit. `position` (D2) is the one deliberate rename, because the AMCP name describes the gesture and the parameter describes the quantity |
| D4 | **No generic `/v1/.../call` route in this plan** | It would be AMCP over HTTP — stringly-typed, undiscoverable, unvalidated — and it competes with the `parameters()` route rather than completing it. §7 states when it becomes the right answer |
| D5 | **`EXPORT` stays on AMCP** | A long-running job with a busy state (`400 EXPORT BUSY`) and no progress reporting. It needs a job resource, which is a separate design; bundling it would smuggle an unplanned one in |
| D6 | **Parameters are pulled on request, never published per tick** | `CLAUDE.md`'s per-tick publication cost: ~600 leaves per channel per tick is the ceiling on this box, and eight transport parameters × the layers on a channel would spend it on data nobody is watching. `describe_params` is already on-demand; keep it that way |
| D7 | **The published state leaf and the parameter coexist deliberately** | `file/frame` is the *monitoring stream* (cheap, per-tick, already carried by OSC); `position` is the *addressable control*. They are the same number reached for different reasons, and the alternative — removing the leaf — would break every existing OSC client |
| D8 | **A producer declares only what it actually implements** | `replay` has no `pingpong`, `image_scroll` has only `speed`. A contract that forces every row onto every producer produces parameters that accept a write and do nothing, which is the `MIXER EXPOSURE` class |

### The one design risk, stated rather than discovered later

**A `read_write` `position` on a producer that is also advancing by itself is a feedback
hazard.** A timeline keyframing `position` fights the producer's own per-frame advance, and the
result depends on which writes last.

That is also exactly what a scrub is, so it cannot simply be refused. The resolution follows the
node-graph precedent: while a timeline or binding **owns** the parameter, it is authoritative and
the producer's own advance is suppressed; on release the producer resumes from wherever the
playhead was left. `timeline-targets` already measures the producer-parameter release path —
*"a producer parameter [is released] by writing back a value CAPTURED on entry"* — so the
ownership machinery exists and this plan must decide whether `position` releases to the captured
value (wrong: it would jump backwards) or holds where it was left (right). **`position` releases
to where it was left, and it is the first parameter for which those two differ**, which is why
this paragraph exists.

---

## 4. Sequencing — each phase is a commit with its own gate

### P1 — the contract, and `replay` as its first implementation — **SHIPPED 2026-09-14**

* `src/core/producer/transport_params.h` (new): the canonical rows — name, `value_type`,
  `access_t`, unit, bounding, and the one-line semantic each. No behaviour, no dependencies
  beyond `producer_params.h`.
* `replay_producer::parameters()` (new): `position`, `speed`, `loop`, `in`, `out` built from the
  contract, getters and setters delegating to the same members `call()` already writes — so
  AMCP and the API reach one value by construction rather than by agreement.
* `call()` keeps every verb, unchanged. **Nothing is deprecated in this plan.**
* **Gate:** `producer-params` gains a replay arm — describe, round-trip, and *the picture moves*.
  The last part is what makes it more than a value check: write `position` and the decoded
  frame must change, which is the only oracle that can distinguish a parameter that seeks from
  one that stores a number. `replay` (6/6) must stay green, both mixers.

**What P1 found, and the two places it departed from this plan.**

* **The gate went into `replay` rather than `producer-params`**, which this plan named. The
  oracle has to be a picture, and a picture oracle needs a recording whose frames DIFFER —
  `replay`'s fixture was a flat colour, against which *"write `position` and the picture
  changes"* is unfailable, and building a two-colour recording inside `producer-params` would
  have duplicated the whole recording half of this battery. The fixture now records colour A,
  then colour B on air, and the check is that scrubbing between them returns both.
* **`GET /v1/value/.../params/{name}` answered `unknown_path` for a path that accepted a PUT.**
  The read route served the state snapshot only, and `api_value.cpp`'s own comment stated the
  assumption — *"which is where the producer's own `state()` publishes it, so reads need nothing
  new"* — which held for exactly as long as `isf` and `ofx` were the only two producers with
  parameters, because both double-publish. D6 makes `replay` the first that does not. Fixed in
  the same commit: `read_value` asks the producer, as the write route and the tree already did.
  **One reader, enumerated before fixing** (`CLAUDE.md`'s rule), with one consequence left
  standing: a parameter a producer does not publish emits no CHANGE EVENTS over `api-events`,
  which is the honest price of D6 and is recorded in §8.

**Why replay first rather than ffmpeg:** it is the module whose natural client is a scrub bar, it
is the one this plan came out of, and — since 2026-09-14 — it is the only one of the eight with
a battery at all. Starting where the coverage is is the difference between a measured change and
a plausible one.

### P2 — `ffmpeg_producer`, the reference implementation — **SHIPPED 2026-09-14**

The highest-traffic producer and the one whose `call()` the other three copied. `loop`, `speed`,
`in`, `out`, `length`, `position`, `pingpong` — the full contract, and the row that proves the
contract is general rather than shaped around replay.

* **Gate:** a `producer-params` ffmpeg arm, plus **`timeline-targets`**, because this is the first
  transport parameter that a timeline can keyframe and §3's ownership decision is only testable
  here. And a `flat-decoded` run, because `in`/`out`/`length` touch the trim path.

### P3 — the GPU-direct trio: `notchlc`, `prores`, `hap` — **SHIPPED 2026-09-14**

Three near-identical `call()` implementations. Mechanical once P2 exists, and the phase where the
contract earns its keep — if it needs bending for any of the three, the bend belongs in P1's
header rather than in the producer.

* **Gate:** `producer-params` arms for each; the existing decode batteries for the trim path.

### P4 — the stragglers: `gst`, `dmx`, `image_scroll` — **SHIPPED 2026-09-14**

`gst_producer` is the interesting one: its `play`/`pause`/`resume` are genuine **actions** on the
producer rather than on the stage, and the stage already has verbs by those names. Decide whether
they map onto the existing stage verbs or need their own — this phase is small but it is not
mechanical, and it is the one that tells us whether D4 holds.

**What P2-P4 found.**

* **The GPU-direct trio needed no adaptation at all**, which is the strongest evidence the
  contract is general: `hap`, `cuda_prores` and `cuda_notchlc` keep their transport state under
  eight identically-named members, so ONE implementation applies to all three verbatim. Two of
  them were written from the third, which is exactly the duplication the contract replaces.
* **`ffmpeg` needed one thing exposed.** `file_duration()` — the MATERIAL length, as against
  `duration()`'s clip length — existed on the impl and was published inside `file/frame`, but had
  no public accessor. `length` means the material, so it is public now.
* **`dmx`'s survey row was wrong** (see the table above), and `gst` settled D4 in its favour: its
  `PAUSE`/`RESUME` are imperative and stay on `CALL`, because a stage `pause` pauses the LAYER
  and these pause the PIPELINE. A real difference, an argument for a producer action route one
  day, and not an argument for modelling an action as a value.
* **And a HARNESS defect that cost real coverage.** `producer-params` gated its OFX arm with
  `if not ofx_plugins: rep.add(...); return rep` — a RETURN, so on a box with no OFX plugins
  installed every check after it silently did not run, including six `/v1/catalog` checks, while
  the headline read 18/19 as though that were the whole story. The catalogue section now sits
  above the gate with only its two plug-in-naming checks guarded. 18/19 → 30/31 on this box, and
  the eleven newly-running checks were all green — so this cost coverage rather than hiding a
  defect, which is the good version of that discovery and not a reason to have left it.

### P5 — docs — **SHIPPED 2026-09-14**

`docs/features/` gains the contract's own page or a section in the producer doc that owns it;
`OPERATIONS_GUIDE.md` gains the parameter paths beside the AMCP verbs they mirror; the harness
`CLAUDE.md` command table gains the new arms; `CHANGELOG.md` carries the behaviour change (a
producer parameter is a new writable surface). Per `CLAUDE.md`, each phase updates its docs **in
its own commit** — P5 is only what is left over.

---

## 5. Measurement — what each gate can and cannot see

| gate | sees | **cannot** see |
| :--- | :--- | :--- |
| `producer-params` | the descriptor, the round trip, and the PICTURE at 1 LSB | whether a timeline can drive it |
| `timeline-targets` | a keyframed producer parameter, and its release | the transport semantics under ownership — **§3's release decision needs its own arm** |
| `replay` | the module end to end, including that recording survives a parameter write | anything about the other seven producers |
| `flat-decoded`, the decode batteries | that `in`/`out`/`length` still trim correctly | the API surface |

**The failure this plan is most exposed to** is the `MIXER EXPOSURE` class: a parameter that
describes correctly, round-trips correctly, and drives nothing. A descriptor check and a
round-trip check both pass for it, which is why every phase's gate includes a **picture or a
decoded frame**, and why P1's arm is specified as *"write `position` and the decoded frame must
change"* rather than *"`position` reads back what was written"*.

---

## 6. What is still owed, and what was explicitly out of scope

**Owed: a PICTURE gate for the GPU-direct trio.** `ffmpeg` and `replay` are measured to the
decoded frame; `hap`, `cuda_prores` and `cuda_notchlc` are verified only as DECLARING the
contract. That check catches a producer that declares nothing — the realistic failure when five
producers are given the same treatment in one pass — and cannot catch one that declares a row it
does not drive, which is the `MIXER EXPOSURE` class. Closing it needs a frame-identifying fixture
in each codec, and there is no NotchLC or Hap encoder on this box to build one with. Stated in
the battery's own text rather than left for a reader to infer.

**Owed: `gstreamer` and `artnet` are declared but undriven** by any battery, for the same reason
and with less excuse.

### Explicitly out of scope

* **`isf` and `ofx`** — they already declare `parameters()`, and `producer-params` covers them.
* **`remotewall`, `html`, `flash`** — configuration and CG verbs, not transport. A CG API is its
  own plan and a bigger one.
* **`EXPORT`** (D5), and any job/progress resource.
* **Deprecating any AMCP verb.** Every `call()` verb keeps working. The contract adds a second,
  typed route to the same members; it does not replace the first.
* **A generic `/v1/.../call` route** (D4) — see below.

---

## 7. When the generic `call` route becomes the right answer

D4 defers it rather than rejecting it. It becomes correct when **a producer has verbs that are
genuinely imperative and genuinely per-producer** — `html`'s CG operations are the live candidate,
and `remotewall`'s configuration verbs are a weaker one. At that point the choice is between a
typed route per family and one stringly-typed route for everything, and the honest comparison
needs a family that has actually been designed.

What would make it the *wrong* answer permanently: if P1–P4 show the transport contract absorbing
most of what `call()` carries, the residue is small enough that a per-family route costs less than
the discoverability a generic one gives up.

**Re-read this section after P4 with the residue counted**, rather than deciding it now on eight
producers' worth of intuition.

---

## 8. What this does not fix

**No change events for a transport parameter.** D6 keeps parameters out of the per-tick snapshot,
and `api-events` computes its diff over that snapshot — so a client polls `/v1/value` or the tree
for a parameter's value and is not told when it moves. For `position` on a playing producer that
is arguably right (it moves every frame, and `file/frame` is already published for exactly that
monitoring purpose); for `speed` and `loop` it is a real gap. The fix is not "publish them" —
that is the cost D6 refuses — but a change-event source that reads descriptors rather than the
snapshot, which nothing needs yet.


The control API still cannot express **a gesture**. Scrubbing is a stream of `position` writes,
and eight of those a second over HTTP is a different conversation from one keyframed ramp. The
timeline is the answer for anything rehearsed; for live scrubbing the open question is whether
the events surface (`api-events`) should carry writes in the other direction, and that is
untouched here and deliberately so.
