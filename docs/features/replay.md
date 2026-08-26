# Replay — instant-replay recording and playback

> **State:** shipped, unmeasured
> **Modules:** `src/modules/replay`
> **Commands:** none of its own — a `replay` consumer, a `replay` producer with `CALL` subcommands
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
| File operations, including export | `replay_file_operations.cpp` |

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

1. **No coverage.** §4 lists three single-server checks, the first of which covers the feature's
   distinguishing behaviour.
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
