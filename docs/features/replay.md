# Replay — instant-replay recording and playback

> **State:** shipped; measured since 2026-09-14
> **Modules:** `src/modules/replay`
> **Commands:** none of its own — a `replay` consumer, a `replay` producer with `CALL` subcommands
> **Architecture:** none, deliberately — the ring buffer and disk layout are conventional; nothing about the shape needs explaining beyond the guide
> **Guide:** [`../guides/REPLAY_MODULE_USAGE.md`](../guides/REPLAY_MODULE_USAGE.md)
> **Coverage:** `replay` — 5/5 both mixers

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

**The last one is the check this battery exists for, and the first version of it could not fail.**
It read the timeline two seconds after `PLAY`, and the reader's error is proportional to *how long
the producer has been open* — so the deliberately un-pruned build reported 82 frames against a
fixed build's 78 and a 125-frame gate, and came back **5/5 green on the mutation the check was
written to catch**. Holding the producer open for 20 s while the writer keeps rolling makes the
same mutation report **624 frames against 125** — five times the buffer that exists. The lesson is
`docs/`-wide: *a check whose observation window is shorter than the defect's growth time is a check
that cannot fail*, and it looks exactly like a passing one.

**Still not measured:** interrupted-recording recovery, `400 EXPORT BUSY`, `LIVE` mode's freshness
and tearing, and cost. §5 keeps them.

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

0. **THE RECORD PATH RUNS ON THE CHANNEL THREAD, AND FLUSHES TWICE PER FRAME.** Found by audit
   2026-09-14, unmeasured, and stated here rather than fixed because §5.1 is the reason to
   measure it first.

   `replay_consumer::send()` encodes the frame, copies it, and calls
   `ReplaySegmentedWriter::WriteFrame` inline, then returns `make_ready_future(true)`.
   `WriteFrame` does `fwrite` + **`fflush`** for the payload and again for the index entry. So a
   50p channel makes a hundred forced flushes a second, on the thread that paces the channel.

   **That is the channel's headroom being spent**, and `output.cpp` says so where it measures it:
   *"send() blocks until the device accepts the frame. Its size relative to the frame period is
   the remaining headroom."* A slow or busy volume therefore stalls every layer on that channel,
   not only the recording — and because `send()` returns an already-ready future, the channel can
   never observe the consumer struggling. It always reports instant success, so there is no
   back-pressure signal at all.

   **The flushing itself is not the mistake** — it is what makes the live-playback-while-recording
   case work, since the producer reads the file as it grows. The question is which thread pays for
   it. Every other file consumer in this tree buffers: `ffmpeg_consumer` has
   `frame_buffer_.set_capacity(realtime_ ? 1 : 64)` and a worker thread. The same shape here would
   keep the flush and move it off the channel, and would let a full queue become the honest
   back-pressure the future is supposed to carry.

   **`consume_max_ms` is already published per consumer**, so this is measurable today without
   writing anything new — which is what makes it a gap with a number rather than an opinion.

1. ~~**No coverage.**~~ **Closed 2026-09-14** — `cli.py replay`, 5/5 both mixers, §4. This was
   the stated reason gap 0 was not simply fixed: a 2,900-line module with no battery is the worst
   possible place to start a threading change, and the refactor would have been unfalsifiable.
   **That reason is now gone, so gap 0 is the next thing to do here** — and the battery above
   drives exactly the path it would move (continuous record plus live playback on one channel),
   so a regression in it would be visible.
2. **Interrupted-recording recovery is unverified** — the case the segmented design exists for.
3. **No cost measurement.** Continuous recording plus live playback on one channel has no measured
   overhead, so there is no guidance on how many replay channels a machine sustains.

---

## 6. Related commits

The module predates this document and its history is not traced. From 2026-09-14:

* `replay: the front door opens, the timeline shrinks, and a bad frame no longer kills the
  recording` — the four defects in §4.1, with the module's first battery.

---

## 7. Diagrams

Deferred. The write-head-versus-read-head relationship in `LIVE` mode is genuinely worth a picture,
but per this folder's rule a diagram is a claim too — and this one would illustrate timing that
nothing measures. Worth drawing once §4.1 exists.
