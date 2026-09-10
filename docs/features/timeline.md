# Timeline — one time model, one resolver, one owner per parameter

> **State:** **in progress** — commits 1–3 of 19 shipped (the time base, and addressing). Nothing
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
> `api-roundtrip` and `binding-lfo` for the audio rows §2.1 adds. The `timeline-*` batteries do
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

*§3 Model, §4 Precedence, §5 Transport, §6 API and AMCP, §7 Verification, §8 Known gaps — arrive
with commits 5–19.*
