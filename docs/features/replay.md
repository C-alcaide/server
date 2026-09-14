# Replay — instant-replay recording and playback

> **State:** shipped; measured since 2026-09-14
> **Modules:** `src/modules/replay`
> **Commands:** none of its own — a `replay` consumer, a `replay` producer with `CALL` subcommands, and since 2026-09-14 the transport contract as control-API parameters
> **Architecture:** none, deliberately — the ring buffer and disk layout are conventional; nothing about the shape needs explaining beyond the guide
> **Guide:** [`../guides/REPLAY_MODULE_USAGE.md`](../guides/REPLAY_MODULE_USAGE.md)
> **Coverage:** `replay` — 11/11 both mixers (recording, retention, growing-file playback, the transport parameters, and cost)

Records a channel continuously to a segmented store, then plays any point in that recording back —
including while it is still being written — so a moment can be reviewed or exported without
stopping the record. The sports-replay pattern.

Operator detail is in [`../guides/REPLAY_MODULE_USAGE.md`](../guides/REPLAY_MODULE_USAGE.md). This
document is the state and the coverage.

---

## 1. What is implemented today

| piece | file |
| :--- | :--- |
| Consumer — continuous record | `replay_consumer.cpp` |
| Producer — playback from the store | `replay_producer.cpp` |
| Segmented storage | `replay_segmented_storage.cpp` |
| Extended index | `replay_extended_index.h` |
| Export | `replay_producer.cpp`, on its own `export_thread_` |

**`LIVE` is the interesting mode** (`replay_producer.cpp:471`, and again at 747): the producer
follows the write head rather than a fixed position, so playback tracks the recording as it grows.
That is the feature that makes it instant replay rather than playback of a finished file.

**`EXPORT` is asynchronous and says so when busy.** `replay_producer.cpp:551-566` accepts
`EXPORT [output] ([input] [start] [end])…` and returns **`400 EXPORT BUSY`** when one is already
running — a real refusal rather than queueing or silently replacing, which is the right behaviour
for an operation that writes a file.

Playback also accepts `LENGTH` and `LOOP`, and the consumer offers `OMT_HQ` / `OMT_LQ` quality
selections.

---

## 2. How to drive it

```
ADD 1 REPLAY "match_01" OMT_HQ
PLAY 2-1 "match_01" LIVE
CALL 2-1 EXPORT "highlight_01.mov" "match_01" 1200 1500
```

Exact parameter lists are in the operator guide.

---

## 3. Design decisions, and what they cost

**Segmented storage rather than one growing file.** A segment boundary is a safe place to start
reading, which is what makes `LIVE` playback of an open recording possible at all, and it bounds the
damage if a recording is interrupted. The cost is an index to keep consistent with the segments —
`replay_extended_index.h` exists for that, and it is the part where a crash mid-write would hurt.

**`EXPORT` refuses rather than queues.** One export at a time, reported as `400 EXPORT BUSY`. For an
operator that is better than an invisible queue, because the failure is immediate and legible.

---

## 4. Verification — what is measured, and what is not

**`cli.py replay`, 5/5 on both mixers, 2026-09-14.** The module's first coverage in its life, and
it found four defects on its first run — §4.1. Both segmentation parameters are per-recording, so
`?segment=1&max_duration=4` gives one-second segments and a four-second buffer and the retention
path, which with the defaults needs **24 hours** to run once, rolls in about ten seconds. A battery
that needed a day to exercise its own subject would never be run.

| check | what it holds | measured |
| :--- | :--- | :--- |
| a recording STARTS soon after `ADD` | `ADD` returning `OK` is not the recorder running: the consumer declares `needs_cpu_frame_data()` and `output::do_send` skips it on any tick whose frame has no host image | first segment **0.5 s** after `ADD`, both mixers |
| a recording writes segmented files | without it every check below is vacuous — they all describe a recording, and a recorder that wrote nothing would satisfy "the timeline is not too long" perfectly | 2→4 segments over 12 s |
| retention DELETES rather than growing forever | the companion to the one below, and the reason it means anything | levels at **4** of a 4 s buffer; unbounded would be 12 |
| a recording still being written plays back | the growing-file case IS the feature | `file/length` 79, `file/fps` 25 |
| **the reported TIMELINE does not exceed what is on disk** | the writer deletes and the reader must notice, or an operator seeking to "the start" lands in material deleted minutes ago | **89 frames against 125 on disk** (ogl), 94 (vulkan) |
| **the transport parameters are declared, and `position` SEEKS** | the fixture records colour A then colour B ON AIR, so scrubbing between them returns two DIFFERENT colours. A flat-colour recording would make this check unfailable — every frame the same, and a `position` that stored a number and seeked nothing would pass | at frame 25 the picture reads (65,127,193); at 131, (193,128,66). dR **+128**, dB **−127** |
| `length` is read-only and the write is REFUSED | an access flag nothing checks is a comment | `PUT length` → `not_writable` |
| `CALL SPEED` and the API parameter are one value | the two facades delegate to the same member, so they cannot drift | `CALL 1-10 SPEED 0.5` → `.../params/speed` reads 0.5 |
| recording does not cost the channel its cadence | a second server at **1080p50** — twice the flush rate and twice the bytes — because the record path is on the thread that paces the channel. Gated on LATE FRAMES rather than on the load: a load figure is a number to track, a late frame is the cadence actually missed | `consume_load` **0.127 → 0.0475** across the §5 gap-0 fix; 0 late frames either side |

**The last one is the check this battery exists for, and the first version of it could not fail.**
It read the timeline two seconds after `PLAY`, and the reader's error is proportional to *how long
the producer has been open* — so the deliberately un-pruned build reported 82 frames against a
fixed build's 78 and a 125-frame gate, and came back **5/5 green on the mutation the check was
written to catch**. Holding the producer open for 20 s while the writer keeps rolling makes the
same mutation report **624 frames against 125** — five times the buffer that exists. The lesson is
`docs/`-wide: *a check whose observation window is shorter than the defect's growth time is a check
that cannot fail*, and it looks exactly like a passing one.

**Still not measured:** interrupted-recording recovery, `400 EXPORT BUSY`, and `LIVE` mode's
freshness and tearing. §5 keeps them. Cost is measured — it is the sixth check, and closing §5's
gap 0 is what it was added for.

### 4.1 What the first run found — four defects, and a fifth in the battery

1. **The factory refused every name the guide tells an operator to type.** `create_producer`
   probed for a file called `<name>.mav`, which no recording has ever contained — the writer makes
   a directory `media/<name>/` holding `<name>.mav.NNN` and `<name>.idx.NNN`. So `PLAY 1-10
   "my_recording"` answered the generic **`File not found.`** for the whole life of the module, and
   only the undocumented `PLAY 1-10 "my_recording.mav"` spelling reached the producer, because that
   one is accepted on the suffix without any file having to exist. **Every playback example in the
   guide was one of the broken spellings.** The probe now accepts both layouts, matching on the
   `.idx` for the same reason the prune below does.
2. **The reader never noticed a segment being deleted.** `ReplaySegmentedReader::Refresh()`
   extended the last segment and appended new ones; there was no `erase` anywhere in the file. A
   rolling buffer's timeline therefore grew without bound.
3. **And the producer could only ever lengthen.** `if (current_len > duration_)` kept the
   high-water mark, so even a reader that pruned correctly could not have shrunk the reported
   length. Two defects in series, which is why neither was visible from outside.
4. **One rejected frame deleted the replay consumer, silently.** `send()` returned `false` on any
   `VMX_EncodeBGRA` failure, and `output::do_send` reads a false future as "this consumer has
   failed" and erases it from `consumers_` **with no log line of any kind**. An instant-replay
   buffer is exactly the consumer that must not do that — it is armed for hours against a moment
   nobody can schedule. It now logs once and keeps recording.

**And the fifth was in the battery, which is the one worth remembering.** Its first version globbed
`media/<name>.mav.*` — the flat layout — read 0 segments for 74 seconds and reported *"the recorder
writes nothing"*. 85 segments had in fact been written one directory down, and retention had
correctly pruned them to four. A defect was then hunted through `output.cpp`'s CPU-readback
handshake and a consumer guard reverted to isolate it, all against a recorder that was working
perfectly. **A battery that cannot see its subject reports its own blindness as the subject's
failure**, and it reads exactly like a real finding. The guide (§"File Format") had the layout
right the whole time.

---

## 5. Known gaps

0. ~~**THE RECORD PATH RUNS ON THE CHANNEL THREAD, AND FLUSHES TWICE PER FRAME.**~~
   **Closed 2026-09-14, with the number that made it worth closing.**

   `replay_consumer::send()` encoded the frame, copied it, and called
   `ReplaySegmentedWriter::WriteFrame` inline; `WriteFrame` does `fwrite` + **`fflush`** for
   the payload and again for the index entry. `output.cpp` waits on the consumer's future and
   measures that wait as the channel's remaining headroom — *"send() blocks until the device
   accepts the frame. Its size relative to the frame period is the remaining headroom."* So a
   50p channel made a hundred forced flushes a second **on the thread that paces every layer
   on it**, and a slow or busy volume stalled the whole channel rather than only the
   recording.

   Measured at 1080p50 with `cli.py replay`'s cost arm:

   | | `consume_load` idle | `consume_load` recording | late frames added over 20 s |
   | :--- | :--- | :--- | :--- |
   | before, OpenGL | 0.00027 | **0.127** | 0 |
   | after, OpenGL | 0.00031 | **0.0475** | 0 |
   | after, Vulkan | 0.00028 | **0.041** | 0 |

   `send()` now copies the frame into a **bounded** queue and returns; a worker does the
   encode, the payload assembly and both flushes. Bounded is the point — an unbounded queue
   converts a disk that cannot keep up into memory growth, which fails later and worse. The
   shape is `ffmpeg_consumer`'s, which this note already cited as the precedent.

   **A full queue drops a frame and never returns `false`.** Blocking would put the disk's
   worst case straight back on the channel, which is the defect being removed; returning
   `false` would be worse still, because `output::do_send` reads a false future as "this
   consumer has failed" and erases it silently (§4.1, defect 4). A dropped frame is one frame
   missing from a recording whose index carries real timestamps, so playback stays correct
   across the gap — and the count is published as `dropped_frames`, which is the only signal
   there is, since an always-ready future cannot carry back-pressure and never could.

   **What is left, and why it is not a defect:** an eighth of the budget became a
   twenty-fifth, not zero. The remaining cost is the **copy** — 8.3 MB per frame at 1080p50,
   on the channel thread. It is not avoidable without pinning a mixer readback buffer for as
   long as the queue is deep, which trades a measured 4% for an unmeasured stall on the
   mixer. Stated rather than fixed.

1. ~~**No coverage.**~~ **Closed 2026-09-14** — `cli.py replay`, 5/5 both mixers, §4. This was
   the stated reason gap 0 was not simply fixed: a 2,900-line module with no battery is the worst
   possible place to start a threading change, and the refactor would have been unfalsifiable.
   **That reason is now gone, and gap 0 was closed the same day** — the battery above drives
   exactly the path it moved, and its cost arm is what makes the before/after attributable
   rather than plausible.
2. **Interrupted-recording recovery is unverified** — the case the segmented design exists for.
3. **No cost measurement.** Continuous recording plus live playback on one channel has no measured
   overhead, so there is no guidance on how many replay channels a machine sustains.

---

## 6. Related commits

The module predates this document and its history is not traced. From 2026-09-14:

* `replay: the front door opens, the timeline stops lying, and a bad frame no longer kills the
  recording` — the four defects in §4.1, with the module's first battery.
* `replay: the record path comes off the channel thread` — gap 0 above, 0.127 → 0.0475.

---

## 7. Diagrams

Deferred. The write-head-versus-read-head relationship in `LIVE` mode is genuinely worth a picture,
but per this folder's rule a diagram is a claim too — and this one would illustrate timing that
nothing measures. Worth drawing once §4.1 exists.
