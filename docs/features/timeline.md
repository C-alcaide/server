# Timeline — one time model, one resolver, one owner per parameter

> **State:** **in progress** — commits 1–10 of 19 shipped. **The timeline runs in the tick**: a
> document animates any layer on the channel's own clock, and releasing it gives the
> operator's value back. **`KEYFRAMES` is removed** (§8). Nothing
> below §2 exists in the server yet; the plan is `~/.claude/plans/zesty-skipping-engelbart.md` and
> each section here lands with the commit that builds it.
> **Commands:** none yet. The `TIMELINE` family and `HOLD`/`RELEASE` arrive with commit 7; the
> addressing in §2 is reached today through the commands that already exist — `MIXER FIELD`,
> `BIND`, and `PUT /v1/value`
> **Modules:** **not a module** — `src/core/timeline/` (the time base today; model, resolver and
> transport to come) and `src/core/address/` (one target resolver over the four registries), with
> the commands in `src/protocol/amcp/AMCPCommandsImpl.cpp`
> **Replaces:** `src/modules/keyframes/` (the `KEYFRAMES` command family) and the OFX producer's
> private `OFX KEY` engine — both removed when the resolver lands, with a `CHANGELOG` measurement
> **Coverage:** `time_self_test()` and `address::target_self_test()` at boot (§1, §2), and
> `api-roundtrip` and `binding-lfo` for the audio rows §2.1 adds, `api-timeline` for §5,
> **`timeline-ramp`** and **`timeline-clock`** for §6 and §7, **`timeline-resolved`** for §9, and
> **`timeline-stack`** plus an inverted **`binding-owner`** for §10.
> The remaining `timeline-*` batteries do
> not exist yet and are named in the plan rather than here, because a battery named in a doc is a
> command a reader will try to run

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

**No battery, and none is possible yet.** `resolve` is not reachable from outside the process
until commit 7 adds the routes; the boot self-test is the whole gate for this commit, and it is
run on every start rather than on demand. `api-tree`, `api-roundtrip`, `binding-lfo` and
the legacy-keyframes battery were run to show nothing moved.

**The JSON codec is not here.** The plan put `json.*` in `core/timeline`, which would drag
Boost.JSON into a target that has a precompiled header — `protocol_http` deliberately has none
for exactly that reason, and its `boost_prelude.h` records the four seconds per translation unit
it costs. The codec lands with the routes in commit 7, in `protocol_http`, where Boost.JSON
already compiles. The document **type** is in core, which is what AMCP and the HTTP layer both
need.

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

**What survived:** the interpolation engine, as `core/timeline/curve.cpp` (§3), and the `kf` names
in each field descriptor, published as `keyframe_names` — so a client that stored one can still
find out which path it refers to. The frozen 193-name check went with the module, and nothing
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

*§11 Known gaps — arrives with commit 19.*
