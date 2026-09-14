# Replay — instant-replay recording and playback

> **State:** shipped, unmeasured
> **Modules:** `src/modules/replay`
> **Commands:** none of its own — a `replay` consumer, a `replay` producer with `CALL` subcommands
> **Architecture:** none, deliberately — the ring buffer and disk layout are conventional; nothing about the shape needs explaining beyond the guide
> **Guide:** [`../guides/REPLAY_MODULE_USAGE.md`](../guides/REPLAY_MODULE_USAGE.md)
> **Coverage:** **none**

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

**Nothing.** No battery records to a replay store or plays one back.

What makes this gap uncomfortable, in order:

1. **`LIVE` mode reads a file that is being written.** That is the highest-risk behaviour in the
   module and the one most sensitive to buffering and segment-boundary timing. A single-server
   check is entirely possible — record for a few seconds, play `LIVE`, assert the played frame is
   recent and the picture is not torn.
2. **The index and the segments can disagree.** Nothing verifies that an interrupted recording
   leaves a readable store.
3. **`400 EXPORT BUSY`** is a stated behaviour with no test; two overlapping `EXPORT` calls should
   produce it.

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

1. **No coverage.** §4 lists three single-server checks, the first of which covers the feature's
   distinguishing behaviour. **This is why gap 0 is not simply fixed**: a 2,900-line module with
   no battery is the worst possible place to start a threading change, and the refactor would be
   unfalsifiable until the checks exist.
2. **Interrupted-recording recovery is unverified** — the case the segmented design exists for.
3. **No cost measurement.** Continuous recording plus live playback on one channel has no measured
   overhead, so there is no guidance on how many replay channels a machine sustains.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Deferred. The write-head-versus-read-head relationship in `LIVE` mode is genuinely worth a picture,
but per this folder's rule a diagram is a claim too — and this one would illustrate timing that
nothing measures. Worth drawing once §4.1 exists.
