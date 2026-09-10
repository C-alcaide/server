# Timeline — one time model, one resolver, one owner per parameter

> **State:** **shipped**, 21 commits, 2026-09-10. A document animates any addressable parameter —
> mixer field, audio volume, producer parameter, previz camera — on the **channel's own clock**;
> bindings, documents and operator writes coexist under one published ownership rule; releasing a
> parameter gives the operator's value back exactly. One document can drive several channels off
> one playhead, and it can start clips. §22 is what is **not** measured.
> **Commands:** AMCP `TIMELINE <ch> PLAY|PAUSE|STOP|SEEK|RATE|LOOP|CHASE|GO|NEXT|PREV|INFO|LIST`
> and `HOLD`/`RELEASE <ch-layer> <field>`. **No `TIMELINE LOAD`** — a document is JSON and arrives
> over the control API. Every transport verb is also `POST /v1/timeline/{name}/{verb}` (§18.1)
> **API:** `PUT`/`GET`/`DELETE /v1/timeline/{name}`, `GET /v1/timeline`,
> `GET /v1/timeline/{name}/resolved?at=`, `POST /v1/timeline/{name}/{verb}`, and
> `{"op":"timeline"}` inside `POST /v1/batch`. `PUT /v1/value/{path} {"hold":true}` takes a
> parameter for the operator
> **Modules:** **not a module** — `src/core/timeline/` (`time`, `curve`, `expression`, `model`,
> `resolver`, `transport`, `timeline_store`) and `src/core/address/` (one resolver over five
> registries), with the tick in `src/core/producer/stage.cpp`, the JSON codec in
> `src/protocol/http/api_timeline.cpp` and the commands in
> `src/protocol/amcp/AMCPCommandsImpl.cpp`
> **Replaces:** `src/modules/keyframes/` (the `KEYFRAMES` command family, removed in §8) and the
> OFX producer's private `OFX KEY` engine (removed in §17). Both with a `CHANGELOG` measurement
> **Coverage:** ALL ON BOTH MIXERS. Five boot self-tests (`time_self_test`,
> `target_self_test`, `curve_self_test`, `resolver_self_test`, `transport_self_test`), and
> sixteen batteries — `api-timeline` (§5), `timeline-tween-survives`,
> `timeline-ramp`, `timeline-clock` (§6, §7), `timeline-resolved` (§9), `timeline-stack` with an
> inverted `binding-owner` (§10), `timeline-step` (§11, a **picture** check),
> `timeline-targets` (§12, a **1 LSB picture** check), `timeline-cue` (§13),
> `timeline-transport` and `timeline-loop` (§14), `timeline-seek-compile` (§15),
> `timeline-chase` (§16), `api-atframe` (§18), `timeline-crosschannel` (§19),
> `timeline-media` (§20), `timeline-cost` and `publication-cost` (§21). Plus `conformance`
> 100/100 within 1 LSB and
> `grading` 48/48, because this edits the tick

---

## 1. Time

**Internally, time is a 64-bit count of flicks, 1/705 600 000 s.** Everything else — frames,
timecode, seconds, bars — is a presentation raster derived from it.

**Why an integer.** A resolver compares times constantly: *does this object's start equal that
one's end*, *which of two instances on a layer started later*. Last-started-wins collision is decided
by exactly such a comparison, and with `double` seconds (the old `KEYFRAMES` unit) equality is an
epsilon question — 0.48 s on a 25p channel is frame 11.9999. With frames at the channel rate, one
document cannot play on a 25p and a 50p channel, and a 59.94 end cannot be compared to a 25 start
without rounding.

**Why this integer.** 705 600 000 = 2⁹·3²·5⁵·7². It is exactly divisible by the frame period of
every format this server ships — 24, 25, 30, 48, 50, 60, 100, 120 and every ×1000/1001 rate — and
by every audio rate (44.1k, 48k, 96k):

| rate | flicks per frame |
| :--- | ---: |
| 25 | 28 224 000 |
| 29.97 (30000/1001) | 23 543 520 |
| 59.94 (60000/1001) | 11 771 760 |
| 120 | 5 880 000 |

So any time in any of those rasters round-trips through a flick count without loss, and a comparison
between two of them is an integer compare. **This is rational time with the denominator chosen
once** — OpenTimelineIO's `RationalTime` becomes `RationalTime(flicks, 705600000)` on export,
exactly. Nanoseconds were the alternative and are inexact at every NTSC frame boundary.

**What is not a time base.** *Drop-frame* is a labelling of a frame count (SMPTE 12M: frame numbers
0 and 1 are skipped at the start of every minute except every tenth, so `00:10:00;00` lands on frame
17 982 of 29.97) and never a base. *Tempo* is a **remap**, `bars → flicks` through
`tempo{bpm, beats_per_bar}` — at 120 bpm 4/4 one bar is two seconds — so a musical timeline and a
video timeline can share one document with different units per object.

**The wire keeps seconds.** A time on `PUT` is a JSON number rounded to the nearest flick;
`{"frames": n}`, `{"tc": "hh:mm:ss:ff"}` and `{"bars": b}` are accepted as literals and converted
against the document's declared `rate` and `tempo`. Every time whose denominator divides the flick
rate survives the round trip exactly, so `0.48` on a 25p channel is frame 12.

**Where it is checked.** `core::timeline::time_self_test()` runs at every server start beside
`binding_math_self_test`, and asserts the one property everything rests on: `flicks_per_frame` has
zero remainder for every rate in the format table plus 29.97 and 48 (which the table does not
carry but LTC chase and imported media will). It also checks the frame round trip at five frame
counts including past an hour, the floor at one flick before a boundary and for negative time, the
`0.48 s → frame 12` wire example, the two SMPTE drop-frame anchors (`00:01:00;02` at frame 1800,
`00:10:00;00` at frame 17 982), and the tempo remap. **The mutation that proves it can fail:**
change `flicks_per_second` by one and the 59.94 exactness check fails at boot.

*Functions:* `flicks_per_frame`, `from_seconds`/`to_seconds`, `from_frames`/`to_frames` (floor),
`to_timecode(t, fps, drop_frame)`, `from_bars`/`to_bars` — all in `src/core/timeline/time.h`.

---

## 2. Addressing — which parameter a curve drives

A timeline object, a binding and an operator write all have to name a parameter, and until this
commit each of them named it differently. `BIND` took a bare registry name and validated it with
an if/else that knew two of the five registries. `KEYFRAMES` took a frozen 193-name table with
degrees baked into it. The HTTP write path split a channel-qualified URL into segments and
consulted the tables itself. One question, three parsers, and a timeline about to be the fourth.

`core::address::parse` is the one place the registries are consulted. The grammar is the binding
one, because that is the short form an authored document wants — the channel and layer come from
the object that owns the target, so they are not in the string:

| address | registry |
| :--- | :--- |
| `opacity` | `fields::find` — the image half of the layer transform |
| `fill_translation.0` | the same, component 0 of an arity-2 row |
| `volume` | `fields::audio_fields()` — **the audio half**, new in this commit |
| `producer/brightness` | `frame_producer::parameters()` on the layer's foreground |
| `previz/camera/position.1` | `fields::find_camera_field`; `view_camera` for the viewport one |
| `previz/screen/wall/position.0` | `fields::find_screen_field` on the named screen |

A leading `mixer/` is accepted and ignored, so an address copied out of the published state tree
or out of an HTTP path resolves unedited.

**What `parse` decides and what it cannot.** Two of the five registries are static tables, so a
path into them comes back resolved to a `field_meta*`. The other three need live state the header
must not depend on: a producer parameter exists only while that producer is on that layer, and a
screen exists only if the channel's previz renderer has one. So `parse` reports the **kind** and
the key and leaves the live half to the stage, which holds both. Answering "valid" here and
failing at the write would put two answers to one question in two places, which is the thing this
file exists to undo.

**Addresses are the TABLE's names, not the struct's.** A screen's X is `position.0`, because the
screen table declares one vec3 row over `pos_x`/`pos_y`/`pos_z`. `previz/screen/wall/pos_x` does
not resolve, and `target_self_test` asserts that it does not — the first version of that test
asserted the struct member and aborted the boot, which is what the test is for.

### 2.1 Audio is addressable now — `mixer/volume`

`volume` and `immediate_volume` were the last mixer parameters no table described. `MIXER
1-10 VOLUME` set them and `audio_transform` held them, and because no registry row existed, every
table-driven surface was blind at once: the control-API tree did not list them, `PUT` answered
`unknown_path`, `MIXER FIELD volume` answered 403, the state publisher's change test compared only
the image half of the transform, and `BIND 1-10 volume` was refused. Six mechanisms knew about
`opacity` and one knew about `volume`.

`fields::audio_fields()` is that table, in the same row type as the image one. Two rows:

| row | type | compose | animatable |
| :--- | :--- | :--- | :--- |
| `volume` | real, ≥ 0 | **multiply** — two layers of gain are a product, which is what `audio_transform::operator*=` already does | yes, continuous |
| `immediate_volume` | boolean | or | **no** — `kf_names` is deliberately null |

`immediate_volume` says *how* a volume change is applied (ramp the intra-frame samples, or jump),
not what the volume is. Animating it would mean animating the ramping policy at 25 Hz, which is
not a quantity an operator wants a curve on, so it carries no keyframe name.

**Where it is checked.** `address::target_self_test()` at boot; `api-roundtrip` asserts both rows
**by name** on both mixers, because everything else in that battery discovers its fields from the
tree and so cannot fail for a field nobody declared; `binding-lfo` gains an arm that binds an LFO
to `volume` and fits the published stream against the same sine as the brightness arm.

**Not audible.** Every check above proves `volume` is a stored, published, round-tripping,
bindable number. Whether the audio mixer applies that gain to samples needs a recording and
`volumedetect`, which is F5 of this plan and is owed.

---

## 3. The curve — the interpolation engine

`core::timeline::curve` is the old `KEYFRAMES` engine, moved into core and re-pointed at the
address space. The design study says keep the engine (`L26`) and break the commands (`L171`), and
this is where that line falls: the per-path index with its binary search, the hold-before and
hold-after rules, the segment easing and the shortest-path angular wrap are the same algorithm,
because they were right. Three things change.

**Time is `flicks`.** A segment's fraction comes from two integers, so `local == key.time` is
exact and a key placed at frame 12 of a 59.94 channel is at frame 12 forever. The old engine
compared `double` seconds with a 1 ms tolerance, which called two keys 0.9 ms apart one key.

**Keys are address-space paths.** `opacity`, `fill_translation.0`, `volume`,
`producer/brightness`, `previz/screen/wall/position.0` — not the 193 frozen KEYFRAMES names. So a
curve drives anything §2 resolves and nothing has to be added to a second table first. The
degrees-to-radians conversion the frozen table performed dies with it: a path writes the
registry's own units, exactly as `PUT` does.

**Kind comes from the registry, through a lookup the caller supplies.** `curve` has to
interpolate a producer parameter and a screen property as readily as a mixer field, and those
live in registries `core/timeline` must not depend on — so "is this path angular" is asked of a
function rather than answered by an `#include`. `kind_of` is the default, over `core::address`,
and it **derives** `discrete` for a boolean, an integer or an enumeration rather than having the
table declare it twice: those cannot be half-way between two values whatever a `kind` column
says, and the column exists to tell angular from plain, which is a question only reals have.

### 3.1 The per-kind modulus, which is the trap this rewrite introduced

The old engine wrapped angles at 360 and only at 360, because the frozen table converted degrees
to radians on the way *in*. With paths writing registry units there are two kinds and they need
two moduli: a `kf_kind::angular` row holds degrees and wraps at 360, an `angular_rad` row holds
radians and wraps at 2π. Using 360 for both is silent — a radian rotation never reaches the wrap
threshold, so it takes the long way round instead of the short one and looks like a deliberate
spin. `curve_self_test` asserts both, and asserts that `angle` really is `angular_rad` in the
registry, so the two halves of the claim are connected rather than each checked against a lambda.

### 3.2 One easing table

`common/tweener`'s **43** names are authoritative. The `KEYFRAMES` map's 35 go: 31 of them are
tweener names already, and the four that are not survive as aliases so no saved document stops
working — `ease`, `easein` and `easeout` (that map's shorthands for the cubic family) and
`easeinelestic`, a long-standing CasparCG spelling. An unknown name is **refused**, not defaulted
to linear: the old behaviour warned once and animated linearly forever, so a document with a typo
animated differently from the one its author wrote and the log said so once, at startup, months
ago.

### 3.3 Where it is checked

`curve_self_test()` at boot, and it is the stronger of the two gates: every engine-level mutation
tried against it aborts the boot with the failing rule named. **Shown failing first:** flipping
the sign of the angular wrap failed *"350 -> 10 degrees passes through 360, not 180"* before the
server finished starting.

`KEYFRAMES` runs on this engine now, through a seconds-and-frozen-names adapter, so the swap
shipped with the old command still working rather than in the same commit that removes it. The
adapter had its own battery — temporary, deleted with the command family, and its last green run
is the evidence that what was removed worked. It drove `SET`, `GET`, `STATUS`, `ARM` and `DISARM` and fitted the published stream against
a three-key piecewise ramp whose middle key was deliberately *not* the linear midpoint: **8/8 on
both mixers, max |error| 0.0000 over 99 frame-stamped samples.** Shown failing first by halving
the adapter's time base, which the boot self-test could not see because it did not run the
adapter: two named failures, the steps landing at exactly twice the authored rate.

Both the adapter and that battery are **gone now** — the command family was removed one commit
later, once the replacement had been measured. §8 has the deletion.

That battery also measured two things about the old command worth writing down, because they are
what D2 replaces:

* **the clock is the producer's, not the arm's.** A layer already playing is already that far
  into its keyframes the instant it is armed. With a two-second document the first readable
  sample was at t = 1.56 s — three quarters of the way through, before anything could observe it.
  The battery's document is twelve seconds long for that reason.
* **`SEEK` is not observable.** `KEYFRAMES 1-10 SEEK 1.0` sets the position and the next tick
  recomputes it from `producer->frame_number()`, so it is overwritten before any read can see it:
  measured at 0.23, the document's end, against the 0.55 the seek asked for.

---

## 4. The model, the grammar, and the resolver

Three files, all pure: `model.h` says what a timeline is, `expression.*` says when an object is
active, `resolver.*` turns the second into absolute times. Nothing here touches a clock, a
channel or the stage, which is what lets `GET …/resolved?at=` answer for any position without
running the show — the client draws what the server computed rather than reimplementing the
collision rules and disagreeing (`L28`).

### 4.1 What a timeline is

A tree of objects, each with an `enable` **expression** rather than a start and a duration. Groups
nest and carry their own `time_remap`, so a sequence can be slowed without touching its contents.
Two properties are worth naming because a straight port of a video NLE would not have them:

**An object's layer may be empty.** A group, or a bare object, with no layer is a **transparent
anchor**: it occupies time, other objects reference its start and end, and it writes nothing.
That is what makes `#interview.end` expressible without inventing a layer to hang the interview
off — `O22` of the study asks for it, and the cue-stack work needs it.

**Time is local and composed.** A key inside a group inside a group is at its own local time; the
remaps compose outermost-first to place it. Nothing stores an absolute time except the resolver's
output, so moving a group moves its contents for free.

### 4.2 The grammar, v1

| written | means |
| :--- | :--- |
| `12.5` | seconds |
| `300f` · `00:00:12:00` · `00:01:00;02` · `4bars` | frames, timecode, drop-frame timecode, bars — against the document's `rate` and `tempo` |
| `#interview` | that object's **start**; the bare form means `.start` |
| `#interview.start` · `.end` · `.duration` | explicit |
| `.lowerthird.start` | the **earliest** start of every object carrying that class |
| `.lowerthird.end` | the **latest** end of them |
| `#interview.end + 5` | one offset, a literal, `+` or `-` |
| `(#a.end) + 5` | parentheses, so a client can round-trip its own formatting |
| `1` *(in `while` only)* | always active |
| `#interview` *(in `while`)* | active exactly while that object is |

**`while` is a different vocabulary, and it has its own parser.** In a time position `1` is one
second; in a `while` position it is *always*. One parser accepting both read every `end: "1"` as
always-on, which resolved to time zero and gave every repeating object a zero-length span — found
by the resolver's repeating case, a long way from the cause. `parse_while_expr` also refuses a
class, because a class's earliest start and latest end belong to different objects and there is no
single span to be active for.

**Two parse rules that are not obvious and both cost a round of the self-test:**

* **The offset comes off before anything classifies the expression.** Splitting it only after the
  leading `#` was seen made `(#a.end) + 5` unparseable — it starts with `(` and ends with `5`, so
  the paren-stripper left it alone and the classifier read the whole string as a literal.
* **A sign is an operator only when space-separated.** Without that rule `#lower-third` parses as
  `#lower` minus `third`, and an id with a hyphen in it is the most natural id there is.

`5 + 3` is **refused** rather than answered: arithmetic between two literals is something the
author expects to be added, and v1 does not do it.

**Deferred, each because it needs something v1 has not got:** `$layer` references (the resolver
would have to know what is playing), reference-to-reference arithmetic (needs an expression tree
rather than one offset), boolean `& | !` in `while` (needs a predicate evaluator), `*` and `/`
(need units), and `seamless`.

### 4.3 What `resolve` does

Flatten the tree composing the remaps · index by id and by class, refusing a duplicate id ·
order the objects by dependency, depth-first, reporting a cycle **by object** · evaluate in that
order, so `#a.end + 5` is asked only once `a` has an end · expand a `repeating` spec into its
instances · stable-sort by start.

Four decisions inside that are choices rather than mechanics:

**Last-started wins, after `priority`.** An operator firing a cue expects it to take over from
whatever was running, and every product surveyed agrees. Earliest-start would make a long
background object permanently shadow every cue fired during it. The sort is *stable*, so two
simultaneous starts fall through to document order — a client-controlled tie-break rather than the
standard library's.

**An open end is open, not a large number.** Last-started-wins has to tell "runs until told" from
"runs until 10:00", because a later object with a definite end must not be shadowed forever by an
earlier one that is merely unfinished. A dependent of an open end **does not resolve**, and that
is not an error either: firing the trigger re-resolves the document. The triggers a document waits
on are published, so the transport knows which GO means something.

**A disabled object still resolves its references.** It produces no instance and yet `#its.end`
still answers, because switching an object off must not break the show around it — which deleting
it would.

**Over-determined and disagreeing is an error.** Any two of start, end and duration derive the
third; all three given with `end != start + duration` is a document whose author believes two
contradictory things, and answering with one of them is how a show goes wrong in a way nobody can
debug.

**`one_at_a_time` is computed, not authored.** A child of such a group starts where the previous
sibling ended, rather than the document carrying `#prev.end` on every child — otherwise inserting
one cue in the middle means rewriting every expression after it.

### 4.4 Where it is checked

`resolver_self_test()` at boot, which runs `expression_self_test()` first: ten resolver cases and
about fifty grammar cases, over the `#a.end` chain (and the same document declared **backwards**,
which must land identically), a class reference with an offset, priority against last-started, a
transparent anchor with three dependents, a group at rate 2 with the local-time inverse asserted,
`repeating` count 3, a three-cue `one_at_a_time` stack, an open end before and after its trigger
fires, a disabled object's references, and five named failures — a cycle, a duplicate id, an
unknown reference, an end before its start, and the over-determined case.

**Shown failing first.** Inverting the tie-break to earliest-start aborts the boot naming *"during
the cue, LAST-STARTED wins — not the long background object"*. **One** named failure, not the two
predicted: the self-test stops at the first, and structurally only that check discriminates —
the priority case passes either way, because priority outranks the tie-break in both directions.
Two more were found by the self-test during development rather than by prediction: the
parenthesised-offset parse, and `local_at` double-counting the composed offset, which put a child
of a group offset by ten seconds at local time −18 s.

**No battery when this landed, and none was possible.** `resolve` was not reachable from outside
the process until the routes existed (§5), so the boot self-test was the whole gate — run on
every start rather than on demand. `api-tree`, `api-roundtrip`, `binding-lfo` and the
legacy-keyframes battery were run to show nothing moved. It is gated by
**`timeline-resolved`** now (§9), which compares the resolver's answer with the tick's at every
position.

**The JSON codec is not in `core`.** The plan put `json.*` in `core/timeline`, which would drag
Boost.JSON into a target that has a precompiled header — `protocol_http` deliberately has none
for exactly that reason, and its `boost_prelude.h` records the four seconds per translation unit
it costs. So the codec is in `protocol_http/api_timeline.*` beside the routes, where Boost.JSON
already compiles, and the document **type** is in core, which is what AMCP and the HTTP layer
both need.

---

## 5. The document on the wire

```
GET    /v1/timeline                     the documents that are loaded
PUT    /v1/timeline/{name}              store one
GET    /v1/timeline/{name}              the document as stored, with its resolution and faults
DELETE /v1/timeline/{name}              remove it
GET    /v1/timeline/{name}/resolved?at= instances, and the owner of each layer at `at`
```

A document is **server-wide**, not per-channel: one may drive several channels, so its name is
not qualified by a channel index. `DELETE` is the only one in this API, and it is here because a
timeline is the first thing the API *owns* — everything else it writes is a property of something
the server already had, and there is no meaning to deleting an opacity.

### 5.1 Units on the wire

**Seconds, as a JSON number.** `{"frames": 300}`, `{"tc": "00:00:12:00"}` (or `00:01:00;02` for
drop-frame) and `{"bars": 4}` are accepted as literals and converted at PUT against the
document's own `rate` and `tempo`, so nothing downstream has to know which form the author used.
A **string** is an expression and goes to the grammar in §4.2. A read comes back in seconds,
which is the canonical form — a client that sends `{"frames": 300}` gets `12` back.

`{"trigger": "go"}` is the one expression form that is an object rather than a string, because a
trigger name is not a time and giving it string syntax would add a third sigil to a grammar that
has two.

`while` takes `true` or `"1"` or `"#id"`. A bare **number** is refused with a message saying
what to send: a number in every other position in the document is a time, and `while: 1` meaning
"always" is exactly the ambiguity §4.2's separate parser exists to remove.

### 5.2 An invalid document is stored

`timeline_invalid` rather than `bad_request`, and the document is kept. A half-authored show is
the normal state of a document being edited, and a client cannot show the author their mistake if
the server threw the document away — so `GET` returns what was sent with the faults attached, one
per object, carrying `object`, `expression` and `reason`. Nothing evaluates a document whose
resolution failed, so an invalid one is inert rather than dangerous. `docs/faults.yaml` lists the
usual causes.

A **name in the path that disagrees with the `name` in the body** is `bad_request`. Guessing
which one is right would store the document under a name the client does not expect.

### 5.3 What makes a PUT observable

The store's revision counter is mixed into each stage's **structure fingerprint**, so a document
appearing, changing or being deleted bumps `channel/{n}/stage/structure_revision`. That matters
more than it looks: without it a PUT is invisible to anything walking the tree, and a client
displaying a show would have to poll `/v1/timeline` on a timer to find out that it had been
edited. The counter rather than the documents themselves — the question is "has anything changed
since I looked", and hashing a large document every tick on every channel to answer it would be
the wrong trade.

### 5.4 Where the codec lives, and why not where the plan said

`api_timeline.cpp` is in `protocol_http`, not `core/timeline`. Boost.JSON is compiled from source
into exactly one translation unit and `protocol_http` deliberately has **no precompiled header**
because of it — `boost_prelude.h` records the four seconds per translation unit that Beast and
Boost.JSON cost, and `CLAUDE.md` records that a header inside a PCH needs the full
touch-everything-and-delete-the-PCH sweep on every edit. `core` has a PCH. So the document
**type** is in core, which is what the stage and AMCP both need, and the JSON is here.

### 5.5 Where it is checked

**`api-timeline`, 25/25 on both mixers.** Its reference document is built so that a wrong answer
cannot look right: a transparent anchor with three dependents; two objects sharing a class that
do *not* nest, so `.lt.start` and `.lt.end` come from **different** members and a resolver that
took one member's span fails one of the two checks; two objects overlapping on one layer, so the
owner at t=20 tests last-started-wins rather than document order; and a duration given in frames,
so the derived end tests the rate conversion.

**Shown failing first:** removing the store's revision from the structure fingerprint fails *"a
PUT moves structure_revision"* — `2 -> 2` — and leaves the other 24 passing.

**And it found a real defect on its first run, in commit 1's code.** Thirty seconds came back as
`29.999999999999996`. `to_seconds` divides, which is exact in IEEE arithmetic, and **this tree is
built with `/fp:fast`**, under which MSVC may replace a division by a constant with a
multiplication by its reciprocal — and 1/705,600,000 is not representable. It reached the wire
because `time_self_test`'s round-trip check allowed one flick of error, which is far more than
this. `to_seconds` now splits into integer seconds plus a remainder, so the fractional term is
multiplied by zero for any whole number of seconds, and the self-test asserts exact equality for
seven whole-second values.

---

## 6. The transport

**The clock is the channel's frame counter, never a producer's.** That is the single largest
behavioural difference from the `KEYFRAMES` engine, which took its time from
`producer->frame_number()` on the animated layer. The consequences of that were not edge cases:

| the old clock | what it did |
| :--- | :--- |
| a **colour** layer | frame number 0 forever, so an animated grade on a colour fill never moved |
| an **empty** layer | pinned t = 0, so a document animating a layer before its clip loaded started over when the clip arrived |
| a **paused** clip | froze the animation with it, so a hold on a still frame stopped the grade ramping under it — the one time an operator most wants it to keep going |
| `SEEK` | unobservable: the next tick recomputed the position from the producer and overwrote it |

`position_at(frame)` advances `flicks_per_frame × rate` from an **anchor pair**, and the playhead
is derived rather than stored. Storing it and adding every tick would accumulate whatever the
last rate change rounded, and would make `position_at` unanswerable for any frame but the current
one. Both matter: an hour of playback lands on exactly an hour, and a client can predict where a
document will be at a frame it has not reached.

Three details that are choices:

* **`rate` is a rational.** `RATE 1/3` covers one second in three exactly. A double would drift,
  and a slowed group nested in a slowed document would compound the drift.
* **The elapsed frame count is signed.** `frame - anchor` in unsigned arithmetic after a backwards
  seek wraps to near 2⁶⁴ and puts the playhead ten billion years out.
* **The loop wrap is a floor-modulo.** `%` truncates toward zero, which for a negative rate gives
  a position *before* the region's start on every backwards pass.

**`rate 0` is refused**: it is `pause` under another name, and two ways to do one thing is two
things to keep consistent.

### 6.1 Stop > Pause > Run, within one tick

Several clients may act in the same frame, and a stop that loses to a play is a show that keeps
running after somebody stopped it. So a whole tick's commands are applied together: the
positional ones (seek, rate, loop, go) in arrival order, then **the strongest single run-state
command**. WATCHOUT ranks them the same way.

The first implementation sorted by rank and applied every one, which is wrong in the obvious way
— whichever rank goes last wins, so stop-then-play left it *playing*. `transport_self_test`
said so at boot, naming the rule.

### 6.2 AMCP drives it; AMCP does not load it

```
TIMELINE <ch> PLAY|PAUSE|STOP <name>
TIMELINE <ch> SEEK <name> <seconds>
TIMELINE <ch> RATE <name> <n>
TIMELINE <ch> LOOP <name> <from> <to>   ·   TIMELINE <ch> LOOP <name> OFF
TIMELINE <ch> GO <name> [trigger]
TIMELINE <ch> CHASE <name> ON|OFF|OFFSET <s>|FREEWHEEL <n>|REGION <a> <b>|REGIONS OFF
TIMELINE <ch> NEXT <name>   ·   TIMELINE <ch> PREV <name>
TIMELINE <ch> INFO <name>   ·   TIMELINE <ch> LIST
```

**There is no `TIMELINE LOAD`.** A document is JSON, and the JSON façade is the one that speaks
JSON — `PUT /v1/timeline/{name}`. The codec lives in `protocol_http`, which is a **sibling** of
the AMCP library rather than something below it, and linking it there to obtain a parser would
invert exactly the dependency `api_context.h` states the design is keeping apart. AMCP's
tokenizer is also the wrong shape for a document: `KEYFRAMES SET` had to wrap its JSON in
parentheses because the tokenizer splits on spaces, and that wart is not worth reproducing. So
the document arrives over HTTP and either façade drives it — the same split the rest of the API
already has.

A verb answers **202 = queued**, applied on the next tick under the rank above. A document that
declares a different channel is `404` from this one rather than a silent no-op.

## 7. The tick

```
1. tweens_[*].tick(1)                     the operator's constants advance (unchanged)
2. evaluate_bindings(dt)                  unchanged; still writes the constant through patch()
3. evaluate_timelines(frame)              per document this channel owns:
                                            transport.apply_all(pending)   stop > pause > run
                                            a GO fired -> retrigger, re-resolve
                                            per layer: active_on(pos) -> the object
                                            content, then curves.interpolate(local)
                                            -> timeline_overlay_[layer]
4. resolve_drivers()                      constant + overlay -> resolved_[layer]
5. route ordering -> receive              draw_frame::push(raw, effective_transform(layer))
6. publish                                mixer/* EFFECTIVE, layer/M/driver/<path>,
                                            layer/M/constant/<path>, timeline/<name>/{state,
                                            position, rate, revision, active/<layer>}
```

### 7.1 An overlay, not a write

**Nothing but the operator ever touches `tweens_`.** That is D3, and it is the whole reason this
shape differs from the engine it replaces. `KEYFRAMES` wrote by *replacing* the layer's tween with
a zero-duration one built from the interpolated values, and that made four things impossible at
once: an in-flight `MIXER <duration>` was collapsed; an operator's explicit write during an
animation was lost rather than remembered; a holding timeline leaked its last value forever after
it ended; and nothing could give a field back, because nothing remembered what it had been.

Here the constant lives in `tweens_`, the timeline publishes into `timeline_overlay_`, and one
`resolve_drivers()` per tick composes them into `resolved_`. **Ending a driver is then just
clearing an overlay** — the constant is still there, untouched, and comes back by construction
rather than by being restored. Two of the stack's five ranks exist so far: timeline over constant.
Bindings still write the constant directly and move up to their own overlay later.

`resolved_` holds an entry only for a **driven** layer, so the cost is proportional to what is
animated rather than to the number of layers.

### 7.2 What is published, and why the effective value

`mixer/*` carries the **effective** value, not the constant. It is what a client reads to draw a
slider, and publishing the constant while the picture showed the driver's value would make the API
disagree with the screen — which is the one thing a control surface cannot recover from. The
constant is still reachable, under `layer/M/constant/<path>`, and only where it *differs*, so a
client can show "you set 0.5, the timeline is at 0.35" and has both numbers.

`layer/M/driver/<path>` names what is writing a field, published unconditionally for a driven
layer. There is no descriptor default for "who owns this", so an absent key means *nobody* —
which is exactly the fact a client needs. Without it a client sees a value move and not what
moved it, which is how an operator ends up dragging a slider that snaps back every frame with no
explanation.

`timeline/<name>/{state, position, rate, revision, ok, active/<layer>}` per tick per document.
An invalid document publishes `ok: false` and a fault count and is **inert** — it is stored so a
client can show the author the faults, and evaluating a half-authored show would be the wrong
reading of "stored".

**Stop releases; pause does not.** A stopped document owns nothing, so every layer it drove falls
back to its constant on that very tick. Pause re-anchors and keeps ownership, which is D4 — a
paused document holding a still frame is exactly when an operator wants the value held rather
than reverted.

### 7.3 Where it is checked

**`timeline-ramp`, 15/15 both mixers.** A colour layer's brightness ramped 0.2 → 0.8 over four
seconds with `easeinquad`, fitted on the frame clock; the picture captured at a **paused**
position and compared to brightness × 255; `driver/` and `constant/` read back; and the release,
twice — when the object's own span ends, and again on `STOP`.

Two fixture decisions worth naming. The easing is `easeinquad`, which **shares both endpoints with
linear**, so a resolver that ignored easing passes a two-point check and fails this fit by eight
times its gate. And the operator's constant is **0.5, between the authored endpoints**, so
"released" cannot be confused with "held at an endpoint".

The gate is *derived*: 1.5 frames of the model's own steepest slope. A hand-picked 0.01 failed at
0.0119, which is one frame of slope — the fit cannot do better with an integer origin frame and a
PLAY that landed mid-tick, so the gate was failing for a reason that had nothing to do with the
feature.

**`timeline-clock`, 7/7 both mixers.** The same document over a layer with **no producer**
(play-then-stop, so the layer exists with an empty foreground) and over a **paused clip** — and it
gates that the paused clip's own frame number really is still, without which the check would pass
on a producer clock too and prove nothing.

**Shown failing first, twice, and both mutations are the old design rather than a synthetic
error:**

* **the clock taken from the producer.** `timeline-clock` fails 4 of 7, and `timeline-ramp` cannot
  get past its first real check — 13 samples where 101 are needed.
* **the old lossy writer** — composing into `tweens_` instead of `resolved_`. `timeline-ramp` fails
  exactly the four release checks: the object's end does not release, `constant/brightness`
  disappears, `STOP` leaves 0.35 where the operator set 0.5, and the **picture** shows 89 instead
  of 127. The other eleven pass, which is what makes those four the discriminating ones.

`transport_self_test` at boot covers the arithmetic: exact advance over an hour, pause and resume
without a jump, seek both ways, rate 2, rate 1/3 and rate −1, a loop region never left in either
direction, and the rank.

---

## 8. `KEYFRAMES` is gone

The command family and `src/modules/keyframes` were deleted once the replacement had been
measured, which is why it was two commits rather than one: a deletion whose replacement has not
been proved is a deletion nobody can defend.

**What went:** eleven files; the `add_subdirectory`; ten `shared_ptr<void>` virtuals on
`stage_base` and their overrides on `stage` and `stage_delayed`; the tick's evaluation block; the
five `kf_*` maps; the **change filter** `if (values == last) continue;`, which broke its own
precedence by skipping the tick on which a value happened not to move; and the **`CALL SEEK`
sniff**, which read the seek out of a producer call to guess a media time and was the only reason
`SEEK` half-worked.

**What survived:** the interpolation engine, as `core/timeline/curve.cpp` (§3), and the legacy
names in each field descriptor, published as **`keyframe_names`** — so a client that stored one
can still find out which path it refers to. They are a **migration aid and not an animatability
signal**: the tree publishes `animatable` for that, because reading the name list that way told a
client a mixer field could be driven by a deleted command and told it nothing at all about a
producer parameter. The frozen 193-name check went with the module, and nothing
replaces it: a path is validated against the **live** registry at PUT, which is what a frozen list
was standing in for.

`docs/features/keyframes.md` is now a redirect with the command-by-command mapping, kept rather
than deleted because a doc that vanishes leaves a reader with a dead link and no explanation.

**Measured:** `KEYFRAMES 1-10 SET (...)` answers **AMCP 400 with the line echoed back**, which is
this protocol's reply for a command it does not know. The measurement was taken with the
temporary battery that had been driving it, on its last run before deletion — so the same tool
reported both that the family worked and that it is gone.

**F11 is closed.** The plan asked whether anything depended on `KEYFRAMES`, and the honest answer
before this series was *nobody knows*: the harness had never sent the command. It now has a green
run recorded against the old engine and a green run recorded against the new one through the same
verbs, and then the family removed. That is the most that can be said without a user to ask.

---

## 9. `/resolved` and the tick agree

`GET /v1/timeline/{name}/resolved?at=<seconds>` carries, per layer, the object that owns it, that
object's own **local** time, and the **values** every animated path holds there. The values are
what make the endpoint worth having: without them a client knows *which* cue owns a layer and has
to interpolate the curve itself to draw the parameter — and a client's own interpolation is a
second implementation of the easing, the per-kind angular modulus and the discrete hold, which is
where it comes to disagree with the server about what is on air.

Both sides run the same `curve::interpolate` on the same local time, so they must agree to
floating-point noise rather than to a tolerance.

**`timeline-resolved`, 9/9 both mixers, worst difference 0.000e+00 over 67 positions.** The
method is worth stating because it is what makes the comparison meaningful: `/resolved?at=p` is a
question about a *position* and the tick publishes at a *frame*, so the document is played once
with the value stream captured against `Event.frame`, the transport's origin frame is anchored
from the stream against the document's own first key, and each sample's frame is turned back into
the position the tick must have been at. Nothing compares a wall-clock instant to anything.

The fixture has **two objects on two layers with different easings and different spans**, and the
second one starts late — its curve is authored 0..8 in local time while it runs 2..10 in the
document, so at position 5 it is 3 s into its own curve. Reading an object's curve at the
*document's* position rather than its own is the commonest way to draw a timeline wrong, and it
has its own check.

**The count is part of the gate.** `worst` starts at zero, so a run that compared nothing would
report a perfect match — the false-green shape this project has recorded five times over. At
least 40 positions must have been compared.

**Shown failing first, twice:**

* **the document's position instead of the object's local time** — the late-starting object's
  value is wrong and its own check fails, while the layer whose object starts at zero still
  agrees. That is precisely why the fixture has two layers with different starts.
* **one frame late** — the position-by-position check fails at 1.54e-2, which is one frame of the
  steepest segment, and the late-start value check goes with it. A gate of one frame's tolerance
  here would have passed both.

---

## 10. Ownership — who has a parameter, and what happens next

| rank | writer | lives in | ends by |
| :--- | :--- | :--- | :--- |
| 1 | **dominant** — `PUT {"hold": true}`, `HOLD <ch>-<layer> <field>` | `drivers_[layer].dominant` | `RELEASE` |
| 2 | **binding** | `drivers_[layer].binding` | `UNBIND` |
| 3 | **timeline** | `drivers_[layer].timeline` | the object's span ends, or `STOP` |
| 4 | **constant** — `MIXER`, `PUT`, a preset recall, a `MIXER <duration>` tween | `tweens_[layer]` | never; always stored |

Every boundary in that list is a decision somebody could have made differently.

**A binding outranks a document** because a binding is a *live* input — an audio level, a
tracker, a fader — and a document is authored ahead of time. Every product surveyed gives the
live thing the parameter, and an operator whose fader stopped working because a show was running
would not accept the opposite. Per-binding `mode ∈ {replace, multiply, add}` is where WATCHOUT's
`tweenValue * masterDim` fits later; v1 is `replace`.

**`HOLD` outranks both** because it is the operator saying "this one is mine now". A show needs
an escape hatch: a document animating a grade is the normal case, and a document animating the
grade on the shot that has just gone wrong is the case where somebody has to be able to stop it
*without stopping the show*. PIXERA calls it Dominant and binds it to a key. **A hold takes what
is on air**, not the operator's last typed value — snapping to a number from minutes ago is the
opposite of "stop it where it is".

**Nothing but the operator writes rank 4.** That is what makes every step down the stack
lossless rather than restored: each driver is its own overlay above one constant, so removing a
driver is clearing an overlay and the constant was never touched.

### 10.1 A write to a driven field is remembered, not refused

`field_bound` is retired. It used to refuse a write to a bound field, reasoning that a write
applied and then overwritten one tick later succeeds and does not last, which is worse than a
refusal. **That reasoning was right about the old write path and wrong about what to do.** On
that path the operator's value and the binding's went to the same place — the layer's tween — so
they genuinely could not coexist, one had to lose, and losing silently was the bad outcome.

They no longer share a place. The write lands in the constant, is kept, and takes effect the
moment the driver ends. The reply says which:

```json
{"path": "…/mixer/opacity", "value": [0.11], "previous": [0.3],
 "effective": false, "shadowed_by": "binding:1",
 "stack": "binding:1,timeline:show/lt1"}
```

`effective` is **absent** rather than `true` when nothing shadows the write, so a client that
never looks for it behaves exactly as it did before the field existed.

**One site keeps `field_bound`, for a different reason than the code originally gave:** a driven
**producer parameter**. A producer parameter has no constant on the stage — the value lives
inside the producer and a binding writes it through the producer's own setter — so there is
nowhere to remember an operator's write, and it really would be applied and overwritten. That is
a gap in the ownership stack rather than a policy, and it closes when producer parameters get
their own overlay. `docs/faults.yaml` says so.

### 10.2 What is published

`layer/{m}/driver/<path>` is the effective owner. `layer/{m}/stack/<path>` is every rank that
wanted the path, strongest first, comma-separated — `"binding:1,timeline:show/ramp"`.

**The stack is not decoration.** Without it, `UNBIND` looks like it will hand the parameter back
to the operator when in fact a document underneath takes it, and a client cannot warn anybody.
With it, the client can say what happens next. It is the same question `field_bound` used to
answer with a refusal, answered with information instead.

### 10.3 Where it is checked

**`timeline-stack`, 16/16 both mixers.** A document ramping `brightness` and a binding parked at
0.65 contending for it, then a `HOLD` above both, then each rank removed in turn. Three fixture
decisions: the binding's parked value sits *inside* the ramp's range, so "the binding has it"
cannot be confused with "the ramp is at an endpoint"; the operator's constant is a fourth
distinct number, so the final release is unambiguous; and the `HOLD` is checked *again* after the
binding is removed, because while the binding was live the held value was also what a hold that
did nothing would read.

The check that separates an overlay from a write: after `UNBIND`, the document takes the
parameter **at its own position** — it kept running while it was shadowed — so the value is
compared against the ramp at that moment, not against the value it had before. A mechanism that
froze a shadowed driver would resume from where it was when the binding took over.

**`binding-owner`, 8/8 both mixers, inverted.** It used to assert the refusal; it now asserts the
write is accepted, reports `effective: false` and `shadowed_by`, does not reach the picture, and
**lands on `UNBIND` with no further write**. Its value is 0.11, which is none of 0.3 (what was
set before the binding), 0.6 (the binding's value) or 1.0 (the default), so a revert to any of
those is distinguishable. `tools/linux_smoke.py` is inverted the same way.

**Shown failing first, twice:**

* **the rank inverted** (binding applied before the timeline) fails three `timeline-stack`
  checks, starting with "a binding OUTRANKS the document".
* **the binding writing the constant as well** — the old writer — fails
  `binding-owner`'s "UNBIND lands the write made during the binding" (0.6 instead of 0.11) and
  `timeline-stack`'s final release (0.65 instead of 0.42). One check in each, and they are the
  two that describe losslessness.

**And the read forms had to move with it.** `MIXER 1-10 VOLUME` and `MIXER FIELD <name>` read
through `get_current_transform`, which returned the *constant*. The moment bindings became an
overlay that stopped being what is on air, and `binding-lfo`'s "both facades agree" check caught
it. It returns the effective transform now — every caller is a reader, and a writer gets the
tween handed to its closure.

---

## 11. Step keyframes, and `rebase`

### 11.1 Two keyframe mechanisms, and why

A curve interpolates, and most parameters want that. An **enumeration does not**: half-way
between `normal` and `screen` is an ordinal that names some third blend mode, and half-way
between two LUT filenames is nothing at all. So a document has two kinds of keyframe:

| | `keys` | `keyframes` |
| :--- | :--- | :--- |
| what | numeric **curves** | **step** values |
| for | anything that can be a number | enums, booleans, names, files |
| between two keys | interpolated, with easing | the earlier value **holds** |
| changes | continuously | **at** the key |

A path in both: the **curve wins**. A step keyframe's time must be a **literal** and a `PUT`
refuses an expression there, rather than accepting it and treating it as "never" — an expression
would need the resolver, and the resolver works on objects rather than on keys inside them.

The order per tick is `content`, then `keyframes`, then `keys`: what the object sets on entry,
then what its steps have reached, then its curves.

### 11.2 `rebase` — starting from what is on air

`"rebase": true` on an object makes its **first segment** start from the value the parameter had
when the object took over, instead of from the authored first key. ossia calls it Tweening,
Hippotizer a floating keyframe, and the reason is the same in both: an object that takes a
parameter over mid-show should *move* it from where it is. Jumping to where the author happened
to be sitting when they wrote the first key is a visible cut on a grade, and not cutting is the
whole point of an authored ramp.

**Only the first segment.** Once the second key is passed the authored curve is authoritative
again — applied to every segment it would leave a rebased object permanently offset from what its
author wrote, which is a different feature and not this one.

**The capture is from the EFFECTIVE transform**, so an object taking over from another driver
starts from what was on air rather than from the operator's constant underneath it. And **entry
is detected by comparing an identity string**, `<document>/<object>#<repeat>`: nothing tells the
tick that an object began, so it compares the active instance to the one it saw last tick. A
string rather than a pointer, because the resolution is rebuilt on every re-resolve. The identity
carries the repeat index, which is what makes a `repeating` object rebase on every repetition
rather than only the first.

### 11.3 Where it is checked

**`timeline-step`, 8/8 both mixers, and it is a PICTURE check.** Every other timeline battery
reads the value stream, and a value stream cannot tell a `blend_mode` that stored from one that
is composited — `blend_mode` is the one animatable parameter whose whole effect *is* the
composite. Two flat colour producers, no decode and no resampling, and both mixers blend in
display space by default (which is what `blend_domain` establishes), so `screen` is exactly
`1-(1-a)(1-b)` on 0-1 display values. **Measured 204/176/184 against a model of 203.9/175.8/183.8.**

The colours are asymmetric in all three channels and the two expected results share no
component: `normal` gives 192/128/64 and `screen` gives 204/176/184. A red/blue exchange, a
channel-order fault or the wrong blend mode all land somewhere that is neither.

**Shown failing first:** dropping the time comparison, so every step keyframe applies from the
object's entry — the shape a naive implementation has. Three of eight fail, including the picture
before the key.

**`rebase` has an arm in `timeline-ramp`** (21/21 both mixers). The operator's constant is 0.5
and the document's first key is 0.2, so a rebase and a non-rebase differ by **0.3 at entry**,
thirty times the fit's own tolerance. It also checks that the second key is still reached
*exactly* and that the segment after it is the authored one — `0.8 → 0.2`, read three quarters
along — because a capture that leaked past the first key would offset the rest of the curve.

The maths is checked at boot as well: `curve_self_test` covers the rebased first segment, the
authored second, an unnamed path left alone, and the no-capture case. **Shown failing first:**
rebasing the *last* segment instead of the first aborts the boot naming the entry rule.

---

## 12. The two targets that are not on the transform

A `producer/<name>` path and a `previz/camera/<field>` path resolve like any other (§2), and a
document can drive both. What is different is that **neither has a constant.**

Every other target is a field of the layer's `frame_transform`: the operator's value lives in the
stage's own tween, an overlay sits above it, and releasing a driver costs nothing because nothing
was overwritten. A producer parameter's value lives **inside the producer** and a camera's
**inside the renderer**, so there is nowhere for an overlay to sit above.

So they are applied by their own pass, `apply_live_targets`, after `resolve_drivers` — same
overlays, same rank, a different destination — and released differently:

| | driven by | released by |
| :--- | :--- | :--- |
| a **producer parameter** | the producer's own setter | writing back a value **captured on entry** |
| a **previz camera field** | the renderer's mutator, through a bridge the shell injects | **nothing** — it stays where the document left it |

**The asymmetry is deliberate and it is measured, not assumed.** Restoring a previz field would
need its value read back per tick, and reading the renderer from the stage is the synchronous
round trip the bridge's whole shape exists to avoid. `timeline-targets` asserts the camera stays
put after `STOP`, so the behaviour is a checked property rather than something for a reader to
discover.

### 12.1 Write-on-change, which is F1 answered rather than deferred

Every previz mutator re-applies the mesh transform and calls `update_projections()`. Writing an
unchanged value every tick would recompute the projection fifty times a second for nothing, so
the pass compares against **what it last wrote** — not against what the renderer holds, since
reading it back is the round trip being avoided. The plan flagged this as F1, "calling
`set_stage_field` every tick from the stage executor is unverified"; it is answered by
construction.

**Cameras only in this build.** A camera's position, rotation and fov are settable from their own
values alone. A screen's are not: `size` and `arc` have no mutator at all (they are set when the
screen is created, and re-creating it would discard every other property), and the mutators that
do exist need a whole `screen_meta` read back out of the renderer. Both facts are already
recorded in the HTTP bridge; the timeline inherits them.

**A wider stage field needs all its components in the document.** The renderer's mutator takes
the whole vector and the stage **refuses a partial write** rather than guessing the rest, so
`previz/camera/position` with three values works and `position.0` alone does not.

### 12.2 Where it is checked

**`timeline-targets`, 12/12 both mixers.**

The producer arm **gates the picture at 1 LSB**, which is possible only because it borrows
`producer-params`' fixture shader: a flat fill computed from the shader's own parameters, no
decode and no resampling, so `level × 255` is a closed-form model rather than a previous capture.
Measured **159 against 159.4** while driven, and **94 against 94.3** after the release. A
parameter that stores and does not reach the shader is the `MIXER EXPOSURE` class, which is why
that check is a pixel and not a value.

The camera arm gates the value at two positions, that a paused document holds it without
stalling the channel, and the release asymmetry above. It does **not** gate a picture and cannot:
where a camera is pointing is A16 — no battery in this project looks at that, which
`previz-picture` records.

**Found by this battery on its first run:** `STOP` restored nothing. The release was written into
the per-layer branch of the evaluation loop, and three of the four ways a driver can end — the
document stopped, deleted, or turned invalid — never reach that loop. It is a sweep over the
whole entry map now.

---

## 13. Cue stacks

A **group** is a cue stack, rather than a new object type. `one_at_a_time` runs its children in
sequence, each starting where the previous ended, and the sequence is **computed** — inserting a
cue in the middle does not mean rewriting every expression after it.

| flags | behaviour |
| :--- | :--- |
| `one_at_a_time` + `auto_play` | runs straight through: a sequence |
| `one_at_a_time` alone | **every cue after the first waits for a GO**: a stack |
| `loop` | **refused at PUT**, with a reason — see below |

**The first cue does not wait.** The group's own start is when the stack begins, and needing a GO
to start as well would mean two operator actions to start a show.

**A GO takes the next cue NOW, whenever it is pressed.** Cue N starts at the Nth firing at-or-after
the *group's* start — not at the first firing after the previous cue *ended*, which is what the
first implementation did and which meant a GO pressed while a cue was still running did nothing.
`resolver_self_test` said so at boot. A GO pressed early therefore starts the next cue while the
current one's span is still open, and last-started-wins gives it the layer — which is again what a
lighting desk does.

**A waiting cue has no position at all**, not a position of zero. A position of zero would put it
on air. It is absent from `instances` and the trigger it waits on appears in `pending_triggers`,
so a client can grey out a button that would do nothing.

`TIMELINE <ch> NEXT|PREV <name>` move the playhead between cue starts. They are **not** transport
verbs: the transport is pure and takes a position, so the stage reads the resolution, works out
the position and issues an ordinary seek — which is what keeps `transport_self_test` a table of
numbers rather than a table of documents. `NEXT` looks strictly *after* the playhead and `PREV`
strictly before it **with a one-frame guard**, because without the guard `PREV` pressed just after
a cue started lands on that same cue and looks like it did nothing. Neither is a silent no-op:
past the last cue, `NEXT` answers **404**.

**Group `loop` is refused rather than ignored.** Looping a cue stack means child *i mod n* repeats
forever, so the resolution has no end — and the resolution is the finite list `/resolved` hands a
client to draw. The refusal message points at the transport's loop region, which does what most
shows want and exists today. A flag accepted and silently ignored is the failure this whole API is
built to avoid.

### 13.1 Where it is checked

**`timeline-cue`, 14/14 both mixers.** Three cues with distinct brightnesses *and* distinct
durations, so "cue 2 is on air" cannot be confused with "cue 1 is still running", with a default,
or with a stack that used one cue's length for all three. It gates that only the first cue has a
position before any GO, that the pending trigger is published, that **one GO advances exactly one
cue**, that `NEXT`/`PREV` move strictly past the playhead, that `NEXT` past the end is a 404, and
that group `loop` is refused with a reason.

`resolver_self_test` covers the counting at boot: nothing fired, one GO, two GOs, and a GO from
*before* the stack began (which must not advance it).

**Shown failing first:** making any firing advance any cue — the "has anything fired" test —
aborts the boot at *"and the third still waits"*. One GO doing three cues' work is the failure
that shape has.

**And an existing self-test case had to be corrected**, which is the more interesting finding: the
`one_at_a_time` case predated the distinction and did not set `auto_play`, so it asserted
back-to-back sequencing. The moment cue stacks learned to wait, that fixture threw `invalid
unordered_map key` — a test that had encoded the only behaviour there was.

---

## 14. The transport, measured through the tick

`transport_self_test` already checks `position_at` against a table of numbers — exact advance
over an hour, rate 1/3 landing on a whole second, a loop region never left. It runs in
microseconds and **it cannot see whether the tick asks it the right question.** A channel that
passed the wrong frame number, asked once every two ticks, or evaluated before applying the
pending commands would leave every one of those numbers correct and the picture wrong.

So two batteries measure the resulting value stream on the frame clock.

**`timeline-transport`, 12/12 both mixers.** Rate 2, rate 1/3, rate −1, rate 0 refused, and
pause/resume. **The slope per frame is the discriminating quantity**, not the endpoints: two
rates reach the same two values, so a check on where it got to passes on either. `abs(rate)` — a
plausible mistake — gives the positive slope and *still descends from a seek*, which a direction
check would pass and the slope check fails.

The resume is **modelled on the frame clock**: the value must have advanced by exactly the
elapsed frames since the resume, and not by the frames that passed during the pause. A fixed
tolerance failed here against a correct engine, because the fixture's own settle is a second of
show at rate 1.

**`timeline-loop`, 9/9 both mixers.** The region is `[20, 45)` of a sixty-second ramp — **not
starting at zero and not the whole document**, so it cannot be confused with no region or with a
restart. **Every sample must be inside it**, not merely "it wrapped": an off-by-one at the
boundary is one sample outside, and a wrap count would miss it. The value must **re-ramp** on
each pass, because a loop that moved the playhead without re-evaluating the objects would hold
whatever it had at the region's end. And it runs **backwards**, where a truncating modulo puts
the playhead *before* the region's start on every pass — a defect invisible while the rate is
positive. `LOOP OFF` is the control: without it, "inside the region" could be true of a document
that never got there.

### 14.1 What the mutations showed, and which gate caught them

**`abs(rate)`** and **a truncating modulo** both abort the boot: `transport_self_test` is the
stronger gate for the arithmetic and catches them before a battery can start. That is worth
stating rather than glossing — the batteries are not there to re-check the arithmetic.

So the mutation that shows *they* discriminate had to be in the **tick**: asking
`position_at(frame_number / 2)`. `timeline-transport` cannot even get a stream (0 samples where
50 are needed) and `timeline-loop` fails three checks including its own control.

### 14.2 A consequence of the rank, found by a fixture

`stop > pause > run` within one tick means **a client that stops and immediately plays in the
same frame gets a stop.** Three AMCP sends take well under a frame, so an unseparated fixture
measures a stopped document: `timeline-loop`'s reverse arm read zero samples and
`timeline-transport`'s pause arm read a position of `0.000000` for exactly this reason. Both now
leave a tick between a `STOP` and what follows, and both say why.

It is the rank working as specified rather than a defect — but it is a real consequence for a
client, and it is written here because it took two fixtures to notice.

---

## 15. `on_end` and seek compilation

An object ends in one of two ways, and the difference is intent rather than mechanism.

| `on_end` | what happens | why |
| :--- | :--- | :--- |
| `release` *(default)* | the parameter goes back to whatever was underneath | the constant was never touched, so it costs nothing |
| `commit` | the object's **final value is baked into the constant** | "this is the new normal" — a cue that moves a grade and leaves it there |

The committed value is the curve's value **at the object's end**, not at the position the tick
has reached: an object that ended two frames ago commits the value it finished on, not the value
it would have had if it had kept running. It is written through
`tweened_transform::patch`, so an in-flight `MIXER <duration>` on another field of the same layer
keeps interpolating — which is the whole reason `patch` exists.

**`commit` on a producer parameter or a previz field is a no-op**, because neither has a constant
to write into (§12). Said here rather than left to be discovered.

### 15.1 Why seeking then needs a compilation

Playing a show to 100 s runs every cue in order, so the committed ones have left their marks.
Jumping to 100 s runs none of them. Without a compilation **the same position reached by playing
and by seeking looks different**, and an operator checking a cue by seeking to it is looking at
the wrong picture.

`SEEK t` replays every `on_end: commit` object whose end is at-or-before `t`, **in end order**.
That is the whole compilation, and why it is short is the interesting part: **a `release` object
leaves no state behind**, so only committed ones have anything to replay. End order matters
because two objects committing the same path must land in the order they would have.

A `STOP` compiles to a seek to zero, since that is what it does to the playhead.

### 15.2 Where it is checked

**`timeline-seek-compile`, 10/10 both mixers.** Three objects on one layer, every value distinct
and none of them the operator's:

```
A   0..2 s    commit    brightness → 0.30
B   4..6 s    commit    brightness → 0.70
C   8..10 s   release   brightness → 0.50      (operator's constant: 0.15)
```

So each state says something specific. After A only, 0.30. After the whole document, **0.70** —
because C releases and its 0.50 must not persist, which is the check a "commit everything"
implementation fails. Seeking to 7 s must give 0.70 and not 0.30, which is what makes the replay
*order* checkable. And **the constant is reset to the operator's value before each seek**, so the
compilation has to do something rather than finding the state already correct.

**Shown failing first, twice.** Removing the compilation fails the two seek checks — both read
the operator's 0.15 where playing gave 0.30 and 0.70. Committing every object regardless of
`on_end` fails exactly one: the released object's 0.50 persists.

**And the fixture had to be corrected**, in a way worth keeping: a seek while **stopped**
positions the playhead and drives *nothing*, because a stopped document owns no parameter. The
first version expected the object under the playhead to be driving, which is to say it was
asserting that a stopped document plays. There is now a check for the real behaviour, and the
object-is-driving checks `PAUSE` first.

---

## 16. Timecode chase

A show that must not slip against house time follows LTC. **The frame counter is still the
clock** — chase is a *correction* on top of it, not a different clock, which is what keeps every
other transport property in §6 true while chasing.

```
TIMELINE <ch> CHASE <name> ON | OFF
TIMELINE <ch> CHASE <name> OFFSET <seconds>
TIMELINE <ch> CHASE <name> FREEWHEEL <frames>
TIMELINE <ch> CHASE <name> REGION <from> <to>   ·   CHASE <name> REGIONS OFF
```

Sub-verbs rather than one positional argument list, because the four settings are set
independently in practice: an operator adds a hot region without restating the offset.

### 16.1 The three parts

**The offset.** House time is a time of day; a document starts at zero. `OFFSET -36000` runs a
document whose zero is house 10:00:00. Following the timecode also **re-anchors** the
free-running playhead to the chased position, so leaving a hot region does not jump back to
wherever the free clock had drifted to.

**The freewheel.** When the signal goes, the position keeps running on the frame counter for a
declared number of frames, and then the transport **pauses**. Pause rather than stop, and rather
than running on for ever:

* a dropout of a few frames is ordinary — a cable, a switcher cut — and stopping the show for one
  would be worse than the dropout;
* running on for ever is worse still, because the show drifts against house time with **nothing
  saying so**, which is the one thing chase exists to prevent. Pausing holds the last known-good
  position, so the picture freezes rather than sliding.

`timeline/<name>/{chasing, freewheeled}` are published, so an operator whose show has stopped can
see *why*.

**Hot Regions** (PIXERA's). Outside every declared window the transport runs free and ignores the
timecode; inside one, the timecode takes control. Empty means "the whole document", which is the
simple case and the default — reading an empty list as "never chase" would make `CHASE ON` do
nothing until a region was declared.

This is the difference between a feature an operator can use and one they cannot. A show is
usually a few timed sequences with interactive stretches between them; without hot regions, chase
either drags the interactive parts along with house time or has to be switched on and off by hand
at every boundary.

The regions are tested against the **document's own position**, not against house time: an author
points at a region on their own timeline, and testing house time would make a document's regions
depend on when the show is run. The region gates *whether* to chase, not *what* to chase to.

### 16.2 Where the house timecode comes from

A bridge the shell injects, for the reason the previz writer is one: `LTCInput` lives in
`modules/ltc` and `core` does not link the modules. That inversion is exactly what forced the ten
`shared_ptr<void>` virtuals the removed `KEYFRAMES` module needed.

**`is_valid()` is checked first and the frame is returned only when it is true.**
`get_current_frame_number` answers whatever it last held otherwise, and a chase that followed a
stale frame would look like a working chase on a dead cable — which is the failure chase exists to
make visible. Absent is the honest answer, and the freewheel decides how long to tolerate it.

The source is asked once per tick **per chasing document**, so a server with no LTC and no chase
pays nothing.

### 16.3 Where it is checked, and what cannot be

**`transport_self_test` covers the arithmetic at boot**: the offset, the re-anchor, a jump in
house time followed at once, the freewheel boundary, the pause after it, the recovery when the
signal returns, and hot regions in both directions. It can do all of that because it injects a
house position directly.

**`timeline-chase`, 12/12 both mixers, covers the half that needs a running server** — and the
half it covers is decided by a measured fact rather than a preference: **there is no LTC signal on
this machine, and `is_valid()` is false without one.** Its system-clock fallback supplies a
timecode *string* for display and does not make the input valid, so the bridge reports the house
timecode as absent.

That makes *following* a timecode unmeasurable here, and makes the more interesting half
measurable: **what chase does when the signal is not there.** It pauses after the freewheel and
holds; a long freewheel keeps the show running through the dropout and a short one does not; and
**hot regions still gate it**, so with the LTC cable pulled the interactive stretches outside the
regions keep working and only the timed sequences stop. That last one is a property an operator
would want to know before a show, and it is checkable *because* there is no signal.

**Shown failing first, three times.** Hot regions not gating, and a freewheel that never expires,
both **abort the boot** — the self-test is again the stronger gate for the logic. So the mutation
for the battery is in the tick: never calling `chase_position` fails five of its twelve.

**Not measured, and recorded rather than implied:** accuracy against a real house clock, the
offset end to end, and the recovery after a dropout. F9 of the plan records that the system-clock
mode's rate predictability is unverified; this does not verify it and does not depend on it.

---

## 17. The OFX producer's private keyframe engine is gone

`CALL … OFX KEY` and `CALL … OFX CLEARKEYS` are removed, along with the engine behind them: a
per-parameter keyframe map, a tweener per parameter, an interpolator, and an `apply_animation`
call on six render paths.

It went **after** §12 rather than with it, for the same reason `KEYFRAMES` went after the tick
landed: a document could not animate a producer parameter until §12 shipped, so removing this
first would have left a gap.

**What the private engine could not do**, all four of which a document does:

* it was clocked by the **plug-in's own frame number**, so it had the whole family of faults
  §6 lists — a paused clip froze it, an empty layer pinned it at zero;
* it had **no notion of ownership**: an operator's `OFX SET` during an animation was silently
  overwritten on the next frame, and nothing said so;
* it could not **give a parameter back**. There was no capture, so a finished animation left the
  parameter wherever it stopped;
* it was reachable **from AMCP only** — no description, no read-back of what was animating.

A document is more to type for a single ramp. That is the trade, and it is stated in
`OPENFX_USER_AND_PLUGIN_GUIDE.md` §1.4 beside the replacement rather than left implicit.

**The reply cannot be measured on this machine, and the attempt is worth recording** because it
is the shape of false-green this project keeps hitting. Driving `CALL 1-10 OFX KEY scale 0 0`
against an **ISF** producer answered `202 CALL OK` — the `CALL` fell through to the wrapped
producer, which knows nothing about `OFX` and returned an empty reply. Nothing was measured; the
"OK" was the fall-through. Instantiating the OFX producer whose `call` handles the verb needs an
OFX bundle, and there is none here — the same gap `producer-params` reports on its own OFX arm,
which has never run.

So the removal is supported by the source and the build, and the **reply** is not measured. What
*is* measured: `timeline-targets` **12/12 both mixers** for the replacement path — a document
driving a producer parameter, with the picture at 1 LSB — and `producer-params` **18/18 on its ISF
arm** for the `SET` path that stays. Pointing `--ofx-plugins` at a bundle directory would close
both this and the older gap in one run.

---

## 18. Starting two channels on one frame

A show that spans two channels is two documents, and until now nothing could start them
together. Both mechanisms below exist because a transport verb is not a field write, so neither
of the API's two existing frame-accurate paths carried one.

### 18.1 `POST /v1/timeline/{name}/{verb}`

The transport over HTTP, with the same verb table AMCP's `TIMELINE` drives:

```
POST /v1/timeline/show/play
POST /v1/timeline/show/seek      {"at": 12.5}
POST /v1/timeline/show/rate      {"rate": 0.5}
POST /v1/timeline/show/loop      {"from": 20, "to": 45}   or  {"off": true}
POST /v1/timeline/show/go        {"trigger": "go"}
POST /v1/timeline/show/next      POST /v1/timeline/show/previous
```

**The channel is not in the path.** A document declares which channel drives it, and it is the
one thing in this API that is not a property of something under a channel — so restating the
channel here could only create a second source of truth whose one possible contribution is to
disagree. The batch op accepts an explicit `channel` and *checks* it against the document for
exactly that reason.

`chase` is not here: it is configuration rather than an action, so it has no queue and no frame
to land on. It stays on AMCP in this release, and the route says so rather than 404ing.

### 18.2 `at_frame` on a verb

```
POST /v1/timeline/show/play   {"at_frame": 6280}
```

The command is held in the document's pending list until **that channel's** frame counter
reaches 6280, and is then applied by that tick. So two clients that never talk to each other can
start two channels on one instant with no batch between them, which is what a show controller
actually has.

Three properties, each of which is a decision rather than a consequence:

* **The hold lives in `transport::apply_all`**, not in the stage. A command that is not due yet
  must not be visible to `stop > pause > run` — a `STOP` scheduled for fifty frames' time
  outranking a `PLAY` due now would be a show that will not start. Putting the partition in the
  caller would put that rule one function away from the rank it has to agree with. It is also
  then covered by `transport_self_test` at boot, which in this tree is consistently the stronger
  gate.
* **A frame already past fires NOW**, because the alternative is waiting for a counter that has
  gone by, which is forever. This is deliberately the *opposite* of what `/v1/batch` does with a
  stale `at_frame`, where the reply is a refusal — and the difference is not an inconsistency: a
  batch firing late applies stale **field values** over whatever has happened since, and a
  document starting late merely starts late.
* **The frame is echoed in the reply.** A client that scheduled something needs to know which
  frame the server understood, and a spread between two channels is only diagnosable against the
  frame they were both told.

### 18.3 `{"op": "timeline"}` in a batch

```json
{"at_frame": 6280, "ops": [
  {"op": "timeline", "name": "show-left",  "verb": "play"},
  {"op": "timeline", "name": "show-right", "verb": "play"},
  {"op": "set", "path": "/channel/1/stage/layer/10/mixer/opacity", "value": 1.0}
]}
```

The verb parser is **shared** with the route (`parse_transport_verb`), so the two cannot drift
into disagreeing about what `rate` means. `at_frame` on an *op* is refused: a batch pins one
frame for everything it carries, and an op naming its own would break the single guarantee a
batch makes.

**The op goes through the batch's `stage_delayed` like every other op**, which is why
`timeline_command` is on `stage_base` rather than only on `stage`. Reaching past the delayed
stage to the real one would not merely land on the wrong frame — the delayed stage is holding
that channel's executor, so the call would block the HTTP thread against a lock it is itself
responsible for releasing.

One limit stated rather than hidden: a timeline op that the channel refuses **after** the batch
has landed is reported and not rolled back. The store is mutable, so a `DELETE` can arrive
between validation and apply; by the time that is knowable the field writes are already on the
stage, and the honest answer is "the batch landed and this op did not".

### 18.4 What is measured

`api-atframe` **20/20 both mixers**, extended with a timeline arm that drives both mechanisms:

| | ogl | vulkan |
| :--- | :--- | :--- |
| one batch, two channels | frames 562 / 562 | 560 / 560 |
| two independent POSTs at frame N | 628 / 628 | 626 / 626 |

The gate is the **spread**, at most one frame, for the same reason the battery's field-write arm
gates at one: the two channels tick on their own threads, so an apply landing between them
publishes on N for one and N+1 for the other. Both mechanisms measured 0 here.

Worth noting what the numbers show beyond the gate: the two-POST arm published on **exactly** the
frame it named, while the batch arm publishes two frames later. The scheduled command is applied
by the tick that *is* frame N; a batch's ops are applied from the HTTP thread and land on the
tick after.

**Not measured: the picture.** Both documents are driving before either channel renders again,
and proving that needs a capture per channel on a named frame, which nothing in this harness can
take.

**Mutations, both of them chosen to survive the boot** — the self-test now gates the `at_frame`
arithmetic, so a mutation there aborts the boot and says nothing about the battery:

* the route accepting `at_frame` and never passing it on → **2 named failures**, and the
  documents started 38 frames early (590 against a named 628);
* `stage_delayed::timeline_command` left at the base class's "no such document" → **2 named
  failures**, the other two, with nothing driven after the batch.

---

## 19. One document, several channels

A show spanning two channels was two documents until now, and nothing kept them together. A
document declares **one home channel** that owns its transport; every other channel it addresses
is a **guest**.

```json
PUT /v1/timeline/span
{"channel": 1, "rate": 25, "objects": [
  {"id": "left",  "layer": "1-10", "enable": {"start": 0, "end": 8},
   "keys": [{"at": 0, "values": {"brightness": 0.2}}, {"at": 8, "values": {"brightness": 0.8}}]},
  {"id": "right", "layer": "2-10", "enable": {"start": 0, "end": 8},
   "keys": [{"at": 0, "values": {"brightness": 0.8}}, {"at": 8, "values": {"brightness": 0.2}}]}
]}
```

`TIMELINE 1 PLAY span` starts both. `TIMELINE 2 PLAY span` is a **404**.

### 19.1 One playhead, published once

| | home channel | guest channel |
| :--- | :--- | :--- |
| takes transport commands | yes | **no — 404** |
| advances the position | off its own frame counter | reads the home channel's |
| chases house timecode | yes | inherits `chasing` |
| compiles a seek | yes | its own release path only |
| re-resolves on a GO | yes | picks up the new resolution next tick |
| publishes `position`, `rate` | **yes** | **no** |
| publishes `state`, `active` | yes | yes |
| publishes `follows` | no | **yes** — the home channel's index |

**Why a snapshot rather than a shared transport.** The transport is mutated per tick — the chase
correction, the anchor on a re-rate — and each channel ticks on its own thread. Sharing the object
would put a mutex inside the frame path of every channel and make one channel's chase visible to
another's arithmetic. A `playhead` published once per tick by the owner and read once per tick by
each guest is four scalars, replaced whole, so it cannot be raced into an inconsistent state.

**Why not a transport each.** Two channels would then each have a position and a run state for one
show, and any difference between them — a chase correction, a dropped frame, a command that
reached one executor first — would be a split brain a client can see and nothing can reconcile.
There is one playhead per show by construction, which is also why only one channel publishes it:
two `position` values for one document would be two numbers a client has to choose between,
differing by up to a frame for reasons that are not a fault.

### 19.2 A bare layer number is the home channel's

`"layer": "10"` means the *document's own* channel. A guest resolving it too would drive layer 10
on every channel the document happens to address, which is the opposite of what an author writing
an unqualified layer means.

This is guarded at **two sites** — the gate that asks whether a document concerns this channel at
all, and the per-layer resolution inside it — and §19.4 records what that costs a mutation test.

### 19.3 What a guest deliberately does not have

Each of these is absent by design rather than by omission, and each has a reason that is not
"not yet":

* **no command queue.** `timeline_command` refuses a document this channel does not own, so a
  guest has nothing queued and no second run state to reconcile.
* **no seek compilation.** The constants a `SEEK` replays are the ones on the channel whose
  operator set them. A guest's own constants are handled by its own release path.
* **no re-resolve on a GO.** The home channel re-resolves and swaps the store's pointer, so a
  guest picks up the new resolution on its next tick — within one frame, and stated as such.
* **`entries_` and `live_captures_` are per channel.** A guest's rebase capture and its producer
  parameter restore are about *its* layers, so they belong to its own stage.

### 19.4 What is measured

`timeline-crosschannel` **17/17 both mixers**. The two layers get **mirrored** ramps over the same
8 s, so `up + down` is exactly 1.0 at every position — which means comparing the sum frame by
frame measures the two channels' positions **against each other with no model in between**:

| | ogl | vulkan |
| :--- | :--- | :--- |
| frames carrying both channels | 99 | 93 |
| worst \|up + down − 1\| | 0.003000 | 0.003000 |
| gate (1.5 frames of the combined slope) | 0.009000 | 0.009000 |
| `active` first published | home 206, guest 207 | home 204, guest 204 |

The gate is **derived**: one frame late reads 0.006000, so 1.5 frames is 0.009. A guest running
its own transport drifts without bound and fails on the first pass; a guest one frame behind is
inside what the design promises and passes.

**Found by the battery on its first run.** The guest reported a separate `unstarted` state for a
never-played document, and the check asserting it **failed against the intended behaviour** —
because the home channel publishes a playhead every tick *including while stopped*, so "never
played" is a transient of at most one frame that no client can rely on seeing. The state name was
removed rather than the check weakened: a name that is always wrong is worse than no name.

**Mutations, and the second one is the interesting result:**

* the guest evaluating at position 0 instead of the playhead's → the guest published **1 sample
  instead of 93**, failing "the GUEST channel drove". A frozen guest is not a subtly wrong one.
* a guest resolving a bare layer number, at **one** site → **not caught, 17/17**. The check is
  defended twice, so no single-site mutation reaches it. Mutating **both** sites fails exactly
  the one check. Recorded rather than smoothed over: the check gates the *pair*, and a reader who
  removes one guard will get a green run.

**Not measured: the picture.** Both channels are driving before either renders again, and proving
they change on the same frame needs a capture per channel on a named frame, which nothing in this
harness can take.

Also green because this edits the tick: `conformance` **100/100 within 1 LSB** and `grading`
**48/48 inside their gate**, both mixers, plus the ten other timeline batteries and the six API
ones. `grading` needed `--sequential`, for a reason that is not this change: its parallel mode
could not START, because `env::ensure_writable` probes with a FIXED filename and three servers
sharing one `build/shell` race on it. Recorded in the harness `CLAUDE.md` rather than treated as
a finding — a battery that cannot start has measured nothing.

---

## 20. A document starts a clip

`clip`, `action` and `preroll_frames` were parsed, echoed back by `GET`, and read by nothing.
A document could grade a layer and put nothing on it.

```json
{"id": "vt1", "layer": "1-10", "enable": {"start": 3, "end": 12},
 "clip": "bars", "action": "play", "preroll_frames": 25,
 "keys": [{"at": 0, "values": {"brightness": 0.4}},
          {"at": 9, "values": {"brightness": 0.9}}]}
```

`action` is one of `play` `load` `pause` `resume` `stop` `clear` (default `none`). With a `clip`,
`play` loads and starts it and `load` leaves it in the background — load and play are different
cues, and a document that conflated them could not preload a next item. Without a `clip` the
action drives whatever the operator already had on the layer, which is the form a document uses
to start something somebody else cued.

### 20.1 Preroll is the feature, not an optimisation

A producer build opens a file and decodes: about 40 ms for a local clip on this box, unbounded
for a network source. Building on the cue frame puts that latency between the GO and the picture
with nothing the operator can do about it. `preroll_frames` is the operator **saying** how much
warning the source needs, and it defaults to 25 — one second, which is the answer for a local
file.

Four properties, each a decision:

* **the build runs OFF the stage executor.** A decode in the tick drops frames on every cue. The
  factory is a shell-injected bridge for the same reason the previz writer and the timecode
  source are: building needs the producer registry and a `frame_producer_dependencies`, which
  `core` cannot assemble.
* **the tick never blocks on a build.** It polls with `wait_for(0)` and moves on, so a slow
  source costs one atomic read per frame rather than stalling the channel.
* **a clip that is not ready does not hold the cue.** The instance starts with nothing on the
  layer, the fault is already published, and the action fires on the tick the build lands. Late
  is a visible mistake an operator can see and fix; a stalled channel is not.
* **one build per instance, and a repeat gets its own.** Sharing a producer across repeats would
  share its playhead, so the second pass would start where the first finished — exactly the class
  of defect the timeline exists to remove.

A build is never retried inside one instance: a clip that cannot be built will not build a second
time either, and retrying per tick would hammer a missing path fifty times a second. A re-PUT is
how an operator fixes it, and that re-resolves and gives new keys.

### 20.2 A bad clip is refused at PUT

Unlike an unresolvable expression, which is **stored** with a fault. The difference is what the
author can do with the result: an expression fault leaves the rest of the document usable and can
be seen against what they wrote, whereas a document that will not put a picture up is not worth
storing under a name a show will trigger.

**The check is the build.** "Does the file exist" is not the question — a clip may be a colour, an
HTML page, a device, a route or a stream, and only the registry knows which factories would take
it. So the PUT builds the producer and discards it, and the answer is whatever it threw. The
refusal carries one detail per bad object naming the object and the reason.

The producer is **not** reused at preroll. One a PUT built would be seconds or hours stale by the
time its cue arrived, and a stale producer on air is worse than a second build.

### 20.3 What the stage publishes

```
channel/1/stage/timeline/show/media/vt1/clip      "bars"
channel/1/stage/timeline/show/media/vt1/ready     true
channel/1/stage/timeline/show/media/vt1/build_ms  39.8
channel/1/stage/timeline/show/media/vt1/on_air    true
channel/1/stage/timeline/show/media/vt1/error     "..."   (only when the build failed)
```

`build_ms` closes **F10**, which the plan carried as "producer build latency unmeasured". It is
**reported and not gated**: a gate here would be gating this box's disk.

### 20.4 What is measured

`timeline-media` **14/14 both mixers**.

| | ogl | vulkan |
| :--- | :--- | :--- |
| `ready` at position (object starts at 3.0) | 2.04 | 2.08 |
| on air at position | 3.08 | 3.16 |
| `build_ms` for a local clip | 39.8 | 40.3 |
| `preroll_frames: 0`, on air at | 1.12 | 1.20 |

**The discriminating check is the ORDERING**, not the picture: `ready` has to be observable while
the position is still short of the object's start. Building on entry also puts a picture up, so
nothing else separates the two.

**The battery found two defects in this code on its first run**, both mine and both in the same
commit:

* **`preroll_frames: 0` never fired.** The window's upper bound was the instance's *start*, so
  with a zero window the only tick that could start a build was the one where the position
  equalled the start exactly — the build then took a frame, and the next tick's `pos > start`
  closed the door forever. The bound is the instance's **end** now, which also means a document
  seeked into the middle of a cue builds its clip, as an operator scrubbing a show expects.
* **the action fired only on ENTRY**, with a comment claiming a late build would still fire on a
  later tick. The comment was false — nothing called it again. It now runs every tick the
  instance is active and self-guards with `fired`, so it happens exactly once but on whichever
  tick the producer exists.

**Mutations:**

* building on entry rather than ahead → **exactly one** named failure, `ready` at 3.2 against a
  start of 3.0, with the other thirteen green. That is the docstring's point measured: both put a
  picture up.
* the PUT-time clip check skipped → **three** named failures, and it exposed a vacuous check of
  my own. "The refusal names the object" was matching the id anywhere in the reply, and a
  *successful* PUT echoes the resolution, which names every object — so it passed with the
  refusal removed. It now reads the structured `details` array.

**Not measured:** whether the picture is the right frame **of** the clip. That needs a
frame-pinned capture against a known frame of a marker clip, which `api-readiness` does not do
either. And there is no slow-source fixture, so a build that is slow for network reasons is
unmeasured.

Also green, both mixers: `conformance` **100/100 within 1 LSB**, `grading` **48/48** (with
`--sequential`, see §19.4), `api-readiness` 7/7, and the eleven other timeline batteries.

---

## 21. What it costs, and F7 closed the wrong way round

§3's cost paragraph predicted one `frame_transform` copy per **driven layer** per tick plus N
slot writes, against three copies per **binding**. That was **F7**, flagged as a prediction
because a prediction in a design document reads exactly like a measurement to the next person.

`timeline-cost` measures it, no code change. Arms on **one binary in one run**, because
comparing two mechanisms across builds compares two machines: 0, 8 and 32 keyed fields per
channel over four channels, the same counts as **bindings**, 8 of each together on disjoint
fields, and 32 keyed all on **one** channel to separate a per-channel cost from a total one.

### 21.1 The numbers

Four channels at 1080p50, ~2010 frames per arm, two passes A/B/A/B:

| arm | ch1 leaves | ogl late | vulkan late |
| :--- | ---: | ---: | ---: |
| nothing driven | 52 | 0 | 0 |
| 8 keyed / channel (32 total) | 91 | 0 | 0 |
| **32 keyed / channel (128 total)** | **187** | **0** | **0** |
| 8 bound / channel (32 total) | **188** | 0 | 0 |
| **32 bound / channel (128 total)** | **596** | **259** | **279** |
| 32 keyed all on one channel | 187 | 0 | 0 |
| 8 keyed + 8 bound, disjoint fields | 227 | 0 | 0 |

**`ch1 leaves` is the denominator, and reading the table with it changes the finding.** It is
what `channel/N/stage/state_leaves` publishes — how many keys that channel puts into
`monitor::state` each tick — and it is **identical on both mixers**, as a publication should be.

**32 keyed fields publish 187 leaves and cost nothing. 8 bindings publish 188 and cost nothing.
32 bindings publish 596 and cost 13%.** So the expense tracks the **total leaf count**, not which
mechanism produced it — at equal tree size the two are equally free. What the mechanism decides
is how many leaves you buy per unit:

| | leaves per unit |
| :--- | ---: |
| a keyed field | **4.2** — its mixer value, plus `driver/`, `constant/` and `stack/` |
| a binding | **17.0** — the same layer rows plus a 13-key `binding/{id}/` sub-tree |

**About 4×, and that is the whole of it.** An earlier version of this section said 13 against 1,
which was the `binding/{id}/` sub-tree measured against a keyed field's *value* alone — it
ignored that a keyed field also publishes its driver, constant and stack rows. The 4× is
measured; the 13× was arithmetic on an undercount.

The frame period sits at 40.00 ms in every arm and is **not** the discriminator: it is paced by
the consumer's own clock and cannot rise until the thread overruns. The late count and
`consume_max` are what move.

**A timeline is decisively cheaper than the same number of bindings**, and the margin is not
marginal: 128 driven fields cost nothing where 128 bindings cost 13 percent of frames. But the
reason is the leaf count above rather than anything about bindings as such — **the ceiling is a
published tree of roughly 600 leaves per channel per tick**, and a binding reaches it four times
faster than a keyed field does.

**Which is why 128 is not quotable as a limit on its own** — and the measurement has now been
taken. `publication-cost` walks a channel from idle to fully dressed, reading
`channel/N/stage/state_leaves` at each step:

| a channel with… | stage leaves | % of the 596 that breaks |
| :--- | ---: | ---: |
| nothing on it | 7 | 1.2% |
| one clip | 23–27 | ~4% |
| three layers (a composite) | 50 | 8.4% |
| + a grade, five fields on each layer | 68 | 11.4% |
| + two previz screens and cameras | 68 | 11.4% |
| **+ an ISF producer with parameters** | **78** | **13.1%** |
| *+ 32 bindings — the pathological reference* | *653* | *110%* |

**A fully dressed realistic channel publishes 78 leaves: an eighth of what costs frames.** So
the container fix below is not worth building, and that is a measurement rather than a guess.
What would change it is roughly **eight times** the publication of a dressed channel — which is
what 32 continuously-live bindings is, and nothing else here comes close.

Two things the arms established that are worth carrying:

* **`state_leaves` counts the STAGE's map, not the channel's.** Adding previz moved it not at
  all (68 → 68) because previz publishes under `mixer/previz` — `video_channel.cpp` assigns
  `state["mixer"]["previz"] = image_mixer_->state()`. The battery's growth gate failed on that
  first run, correctly, against a wrong expectation rather than a server defect.
* **Every stage leaf is inserted twice per tick.** The channel builds its own `monitor::state`
  and `state["stage"] = stage_->state()` goes through `state_proxy::operator=(const state&)`,
  which loops inserting every leaf under a new prefix — into a larger map holding the mixer,
  previz and output as well. So the stage's count is an undercount of what a channel pays, and
  the channel-level total is not published.

### 21.2 The mechanism F7 named is wrong, and the real one is the state publication

F7 predicted the difference was **copy count** — one `frame_transform` copy per driven layer
against three per binding. Three mutations, each on one variable, settled it. The first two found
nothing, and the third found all of it:

| mutation | bind-32 late, ogl | vulkan | verdict |
| :--- | ---: | ---: | :--- |
| *baseline, five runs* | 90–251 | 284–288 | — |
| `resolve_drivers` copies and stores **per write** (F7's shape) | — | no change | ruled out |
| the same, **20× per path** — 2560 copies/tick at the 128 arm | — | no change | ruled out |
| the per-binding **mutex + source map lookup, 20×** | — | 269 | ruled out |
| **the per-binding state publication REMOVED** | **0** | **0** | **it is this** |

**The null results were read on VULKAN, and neither baseline is as tight as first written.**
Across six runs of unmutated binaries the `bind-32` arm spans **90–259 on ogl** and **254–308 on
Vulkan** — a 2.9× spread against roughly ±10%. So Vulkan supports a null result and ogl does not,
but only within that ±10%: what the two nulls establish is that **20× the resolve work and 2560
extra locks and lookups per tick do not matter**, which is a large effect ruled out rather than a
small one measured. Only the last mutation — a flat **zero on both mixers** — is unambiguous.
Why ogl is that variable is unexplained and is a different question.

So **F7's conclusion holds and its stated reason does not.** The write path is shared with the
timeline and is free at 20× the work; the mutex and the string-keyed source lookup are free at
2560 extra of each per tick. Removing what a binding *publishes* takes 128 bindings from 10–14%
of frames late to **zero, on both mixers**.

**Why publishing costs that much**, three things compounding in `monitor::state`:

* **17 published leaves per binding against 4.2 for a keyed field**, measured (§21.1). Thirteen
  of a binding's are its own `binding/{id}/` sub-tree — `layer`, `target`, `component`, `source`,
  `min`, `max`, `in_min`, `in_max`, `gain`, `lag`, `curve`, `value`, `broken` — none of which a
  keyed field has an equivalent of. At 32 bindings that is **596 leaves on the channel against
  187** for the same number of keyed fields.
* **`data_map_t` is a `boost::container::flat_map`**, a sorted vector, so each new key is a
  binary search plus a memmove of everything after it. Insert cost therefore grows with the size
  of the whole channel's tree, not with the number of bindings.
* **each write rebuilds its path by concatenation at every level.** `state["binding"][id]["min"]`
  allocates `"binding"`, then `"binding/5"`, then `"binding/5/min"`, and `state_proxy::operator[]`
  returns by value, copying the key again.

**The obvious fix does not work, and the reason is worth knowing before anyone tries it.**
`monitor::state state;` is constructed **fresh every tick** and swapped into `state_` at the end,
so publishing the eleven static values only when they change would make them *vanish* from the
tree on every other tick — a client reading `binding/5/min` would find it once and never again.

Three that do work, smallest first:

1. **`reserve()` the tick's state from the previous tick's size.** **Done, and it produced no
   measurable change**: Vulkan's `bind-32` read 254, 279 and 308 with it against 284 and 288
   without, which is inside the ±10% spread either way. It is kept anyway, on the mechanism
   rather than on a number — rebuilding a container of known size without reserving is a real
   reallocation series, and four lines that cannot change behaviour are worth it. **It is not a
   fix**, exactly as predicted: it removes the reallocations and leaves the memmoves, and the
   memmoves are the cost.
2. **Build the map by APPEND-AND-SORT instead of insert-with-memmove**, inside
   `monitor::state`. This is the one to do if any of them is done. Every tick writes roughly the
   same key set in whatever order the publication code runs, and each write memmoves to keep the
   vector sorted; appending to a plain vector and flattening once — sort, dedupe last-wins, then
   `flat_map(boost::container::ordered_unique_range, first, last)` — turns O(n²) in memmoves into
   one O(n log n) sort. **No API change, no consumer change, sorted iteration preserved** (which
   is why `flat_map` is there), and it helps *every* publication site rather than bindings: the
   same quadratic sits under mixer fields, producer parameters and the previz tree. The risk is
   duplicate keys within a tick, which `data_[key] = v` currently absorbs and append-then-sort
   must handle explicitly, plus `merge()`, `operator=(const state&)` and `begin()`/`end()` all
   needing the flatten to have happened.
3. **Publish the eleven static binding values only when the binding set changes.** Recorded here
   because it is the obvious next idea and it **does not work as stated**: `merge()` loops and
   does `data_[key] = value` per key, so merging a cached sub-tree into each fresh tick costs the
   same inserts and buys nothing. It only helps if the *consumers* read two states, which means
   changing both the OSC fan-out and the API tree walk — a larger change than option 2 for a
   benefit option 2 gets for free.

**None is worth building, on the measurement in §21.1.** A fully dressed realistic channel
publishes **78** stage leaves against the **596** that costs frames — an eighth. The working
figure is unchanged (32 live bindings cost nothing however they are spread, which covers a whole
MIDI control surface), and the thing a client generates hundreds of is keyed parameters, which
measured 0 at 128.

**Option 2 is the one to build if that ever changes**, and the trigger is specific: a channel
publishing past roughly 400 stage leaves, which `publication-cost` reports on every run. Nothing
in a plausible show approaches it — you would need eight times a dressed channel's publication,
and 32 continuously-live bindings is the only thing measured here that does.

**The battery is not vacuous**, and the `bind-32` arm is the reason: it reports a real cost on
the same measurement path in every run. An instrument that never moves cannot be told from a
broken one; this one moves — and it is what let three mutations rule three sites in or out.

**What it therefore cannot see:** a regression that made the resolve pass ten times slower.
That pass is free at 20×, so nothing here would notice. Stated rather than left implicit.

### 21.3 The traps this battery inherited

All three from `binding-cost`, whose first version paid for them:

* **the targets come from the server's own tree**, filtered to scalar reals that are writable
  and carry **no `enables`**. A field with an `enables` switches its subsystem on when written,
  so keying `blur_radius` would add a blur pass to every frame and the arm would be measuring
  render work attributed to the timeline.
* **one layer per channel, identical in every arm.** `binding-cost`'s first version spread 32
  bindings over four layers and reported a real-looking 19% that was twelve extra layers.
* **A/B/A/B on one binary**, so drift over the run shows as disagreement between the passes
  rather than as a difference between the arms.

---

## 22. Known gaps

Everything on this page is measured on both mixers. This section is what is **not**, and it is
here so the numbers above are read as what they are.

### 22.1 Not measured, though the feature ships

| what | why not, and what would close it |
| :--- | :--- |
| **Whether two channels' PICTURES change on the same frame** (§18.4, §19.4) | Both are driving before either renders again, so the state is provably aligned. Proving the *picture* needs a capture per channel on a named frame, which nothing in this harness can take. |
| **Whether a clip's picture is the right FRAME of the clip** (§20.4) | Needs a frame-pinned capture against a known frame of a marker clip. `api-readiness` does not do this either, so it is a harness capability gap rather than a timeline one. |
| **A slow-source build** (§20.4) | `build_ms` is 40 ms for a local file. There is no fixture that builds slowly, so the "a clip that is not ready does not hold the cue" path is exercised only by `preroll_frames: 0`. |
| **Where the resolve pass's cost goes** (§21.2) | It has none at 20× the work, so a regression making it ten times slower would pass `timeline-cost`. Nothing measures the pass itself. |
| ~~Where the BINDING path's cost goes~~ (§21.2) | **ATTRIBUTED** — it is the state publication. 17 leaves per binding against 4.2 per keyed field, into a `flat_map` rebuilt whole each tick; removing it takes 128 bindings to 0 late frames on both mixers, and at equal leaf counts (187 vs 188) the two mechanisms are equally free. `reserve()` is done and made no measurable difference. |
| ~~How many leaves a REALISTIC show publishes~~ (§21.1) | **MEASURED** by `publication-cost`: a fully dressed channel publishes **78** stage leaves against the **596** that costs frames, so §21.2's fixes are not worth building and the trigger to revisit is ~400. What is still not measured is the **channel-level** total — every stage leaf is re-inserted into the channel's own map each tick, and that count is not published. |
| **Following a real house timecode** (§16) | `LTCInput::is_valid()` is false without a signal and there is no LTC generator on this box, so chase is covered for what it does with **no** signal. F9 stands: the system-clock fallback's rate predictability is unverified, and nothing depends on it. |
| **The `OFX KEY` refusal's reply** (§17) | Driving it against an ISF producer answered `202` because the `CALL` fell through to the wrapped producer. Instantiating the OFX producer needs a bundle, and there is none here. |
| **Audio beyond the value stream** | `mixer/volume` is addressable, writable, publishable, bindable and keyable, and every check reads the published value. A recording plus `volumedetect` would prove it is audible. F5. |

### 22.2 Deliberately not in v1

Each with the hook that makes it cheap later, because "not yet" and "not ever" are different
answers and a reader is entitled to know which:

* **MTC and Art-Net timecode** — no decoder exists. `LTCInput`'s shape is the hook.
* **A live tempo axis** (BPM, TAP, RESYNC). `tempo` is a PUT-time remap only (§1). A running
  tempo clock is a transport mode, and Ableton Link is the standard answer rather than an
  invented model.
* **OTIO import and export** — the mapping is trivial (`RationalTime(flicks, 705600000)`) and it
  is a client feature.
* **DMX channels as tracks** — nothing addressable exists yet. A producer or consumer exposing
  universe channels as `param_desc` rows makes it free.
* **ffmpeg filter parameters** — avfilter option strings are outside the address space and are
  set on the decode thread. `filter_param_tween.h` carries a comment saying so.
* **A follow-the-media clock** — `time_source: layer_media` per object is cheap to add. The
  channel clock is the default because §6 is the list of faults the producer clock had.
* **Blind edit and frozen-instance reset** — the store holds one document per name.
* **Outbound and bidirectional bindings, and WATCHOUT-style formula blend** — `mode` is declared
  in the model and only `replace` is implemented.
* **Bezier tangents** — easing families only, 43 of them.
* **`seamless`, `$layer` refs, ref±ref arithmetic, boolean `while`** — the grammar refuses them
  by name rather than mis-parsing them (§4).
* **Persistence** — the client owns the definition. A document lives until it is replaced or
  deleted.
* **A filler chain for missing media** — the PUT refuses instead (§20.2).

### 22.3 Two things that are stated behaviour, not gaps

Both look like defects to a client that has not read this page, which is why they are here:

* **`stop > pause > run` applies WITHIN one tick.** A client that stops a document and
  immediately plays it in the same frame gets a stop. It cost two fixtures and about half an
  hour hunting a server defect that was not there (§14.2).
* **`commit` on a producer parameter or a previz field is a no-op.** Neither has a constant to
  bake into, so `on_end: commit` has nothing to write. §15 says so where the verb is defined,
  rather than leaving it to be discovered.
