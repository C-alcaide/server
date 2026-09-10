# Timeline — one time model, one resolver, one owner per parameter

> **State:** **in progress** — commits 1–5 of 19 shipped: the time base, addressing, the engine,
> and the model, grammar and resolver. Nothing runs in the tick yet. Nothing
> below §2 exists in the server yet; the plan is `~/.claude/plans/zesty-skipping-engelbart.md` and
> each section here lands with the commit that builds it.
> **Commands:** none yet. The `TIMELINE` family and `HOLD`/`RELEASE` arrive with commit 6; the
> addressing in §2 is reached today through the commands that already exist — `MIXER FIELD`,
> `BIND`, and `PUT /v1/value`
> **Modules:** **not a module** — `src/core/timeline/` (the time base today; model, resolver and
> transport to come) and `src/core/address/` (one target resolver over the four registries), with
> the commands in `src/protocol/amcp/AMCPCommandsImpl.cpp`
> **Replaces:** `src/modules/keyframes/` (the `KEYFRAMES` command family) and the OFX producer's
> private `OFX KEY` engine — both removed when the resolver lands, with a `CHANGELOG` measurement
> **Coverage:** `time_self_test()` and `address::target_self_test()` at boot (§1, §2), and
> `api-roundtrip` and `binding-lfo` for the audio rows §2.1 adds, and `keyframes-legacy` for the
> engine in §3 through the command it still drives. The remaining `timeline-*` batteries do
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
adapter has its own battery, **`keyframes-legacy`** — temporary, and deleted with the command
family. It drives `SET`, `GET`, `STATUS`, `ARM` and `DISARM` and fits the published stream against
a three-key piecewise ramp whose middle key is deliberately *not* the linear midpoint: **8/8 on
both mixers, max |error| 0.0000 over 99 frame-stamped samples.** Shown failing first by halving
the adapter's time base, which the boot self-test cannot see because it does not run the adapter:
two named failures, the steps landing at exactly twice the authored rate.

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
`keyframes-legacy` were run to show nothing moved.

**The JSON codec is not here.** The plan put `json.*` in `core/timeline`, which would drag
Boost.JSON into a target that has a precompiled header — `protocol_http` deliberately has none
for exactly that reason, and its `boost_prelude.h` records the four seconds per translation unit
it costs. The codec lands with the routes in commit 7, in `protocol_http`, where Boost.JSON
already compiles. The document **type** is in core, which is what AMCP and the HTTP layer both
need.

---

*§5 Precedence, §6 Transport, §7 API and AMCP, §8 Verification, §9 Known gaps — arrive with
commits 6–19.*
