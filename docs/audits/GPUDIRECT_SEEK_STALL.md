# GPU-direct decode did not resume after a seek — fixed

With `configuration.ffmpeg.producer.gpu-direct-decode` enabled, a clip stopped
producing frames after any seek. Looping was where it showed up, because the loop
wrap *is* a seek: position advanced to the end, wrapped to 0.00, and stayed there
for ever, on both mixers, with nothing in the log. An explicit `CALL 1-10 SEEK 20`
behaved identically, which is what ruled out anything loop-specific.

Three defects, all in `av_producer.cpp`. None is in the mixer or the bridge. The
first is the stall itself; the other two were masked by it and only became
observable — and only became measurable — once it was gone.

## 1. The decoder was fed no packets after a reset

`schedule()` routed a packet to a decoder only if its stream index appeared in
`sources_`, the filter-graph routing table. The GPU-direct video decoder is
deliberately absent from it — `reset()` skips registering it, because its frames
go straight to the mixer instead of through the graph — so after a reset it was
fed nothing and every video packet was dropped on the floor.

It worked at all before the first seek only by accident of ordering: the initial
`reset()` runs at producer start, *before* `gpu_direct_video_` is set, so the
video stream got registered that one time. Every later `reset()` — and
`seek_internal()` ends in one — dropped it again.

Nothing was logged because the run loop still saw `progress`: `schedule()` kept
popping packets and discarding them, so the "Waiting for video frame..."
debounce never fired. When the input reached real end-of-file the decoder raised
eof, `buffer_eof_` went true, and the loop wrap was then suppressed by its own
`frame_count_ > 2` guard — `frame_count_` was still 0, because no frame had been
produced since the seek. Hence a silent, permanent freeze at 0.00.

Packets are now routed by decoder for the one decoder that has no filter source
by design.

## 2. `flush()` never woke the hardware queue

`Decoder::flush()` set `flush_requested_` and notified `input_cond` and
`output_cond` — but not `hw_output_cond`. In GPU-direct mode the decode thread's
usual resting place is precisely that wait: `hw_output` runs full, because
nothing drains it faster than the decoder fills it. The flush request was
therefore never observed. `flush()` spent its full 500 ms timeout, logged
`Decoder flush timed out - continuing anyway`, and cleared `flush_requested_`
itself — so the flush never happened at all. `avcodec_flush_buffers` was not
called and `hw_output` kept its pre-seek surfaces, which the reader then served.

Symptom: roughly a second of pre-seek video played after every SEEK before the
new position appeared. Measured on the 90-second clip, seeking to 40.00 s and
then to 10.00 s, sampling `<time>` every 0.3 s:

```
before  +0.00:10.00  +0.30:10.00  +0.60:45.20  +0.90:45.48  +1.20:45.80  +1.50:10.04  +1.80:10.32
after   +0.00:10.04  +0.30:10.32  +0.60:10.64  +0.90:10.92  +1.20:11.24  +1.51:11.52  +1.81:11.84
```

The `hw_output_cond` wait also now carries `abort_` in its predicate and is
notified from `~Decoder`, matching the software queue beside it.

This one hid behind the first: with no frames produced after a seek at all, a
seek that served the wrong second of video was not observable.

## 3. A loop wrap ignored the IN point

A seek lands on the keyframe at or before the target, so something has to
discard the frames between the two. On the software path the filter graph does
it: `reset()` rebuilds the spec with `fps=...:start_time=`, which drops
everything before the target. GPU-direct frames never enter that graph, so their
only equivalent is the run loop's drop-to-target, and that runs off
`current_seek_target_`.

Only the explicit-seek call site set it. The loop wrap did not. So a clip
looping on an IN point that is not a keyframe restarted from the keyframe on
every wrap. `PLAY … SEEK 40 LENGTH 40 LOOP` on a clip keyframed every 25 wrapped
to 1.00 s instead of 1.60 s, on both mixers, and the exact landing frame varied
run to run with timing. It is now set in `seek_internal()`, which covers every
caller rather than one — the same reason the trim lives in `reset()` on the
software side — and gated on `gpu_direct_video_`, so the software path is
untouched.

**This was found after the first two fixes were committed, and the first round
of verification missed it.** Two reasons, both worth not repeating:

- The loop test used a clip looping from frame 0, where the IN point *is* the
  keyframe and the correct and incorrect behaviours coincide. Any loop test
  whose IN point is a keyframe cannot see this class of defect. `loop_inpoint.py`
  uses `SEEK 40 LENGTH 40 LOOP` on a clip keyframed at 25 for that reason;
  checked against the previous build, where it correctly reports the wrap
  dipping to 1.00/1.04 on both mixers.
- Fixing the stall is what exposed it. Before, the producer froze after the
  first wrap and went on serving the last correctly-trimmed frame from the
  initial play — so the picture was right *because* nothing was decoding. That
  is also why the picture matrix looked stable beforehand: a frozen producer is
  trivially reproducible.

## Already fixed earlier — do not re-do or revert

Commit `1423934ec` fixed two genuine defects in the same area, both still needed:

- **EOF never reached the reader.** The decoder marks end-of-file with a sentinel
  in the software `output` queue, which the video filter turns into
  `video_filter_.eof`. GPU-direct frames bypass the filter graph, so the flag
  stayed false and `buffer_eof_` never became true. Mirrored from the decoder
  where the reader pulls hardware frames — mirrored, never latched, and gated on
  `flush_pending()`.
- **`hw_output` was never flushed.** Every seek cleared the software `output`
  queue beside it and left the hardware queue holding pre-seek surfaces. Now
  cleared alongside — which, until fix 2 above, only ran when the decode thread
  happened not to be parked.

## Verification

Harnesses in `d:/Github/CasparCG-TestRunner/gpudirect/`, plus the scratch
`seek_stall.py` / `seek_long.py` probes described by the tables below. A consumer
must be attached (`ADD 1 NDI NAME "x"`) or the channel skips compositing and
nothing is exercised.

| case | gl software | gl gpu-direct | vk software | vk gpu-direct |
|---|---|---|---|---|
| `PLAY … LOOP`, 26 s | 6 wraps | **6 wraps** | 6 wraps | **6 wraps** |
| `PLAY …`, no loop | holds 4.00 | **holds 4.00** | holds 4.00 | **holds 4.00** |
| mid-clip `SEEK 40` | plays on | **plays on** | plays on | **plays on** |
| `SEEK 40 LENGTH 40 LOOP`, IN 1.60 s, keyframe 1.00 s | min 1.64 | **min 1.64** | min 1.64 | **min 1.60** |

That last row is defect 3. On the previous build the two GPU-direct cells read
min 1.04 and min 1.00 — the keyframe — with 21–22 samples below the IN point.

`Waiting for video frame` appears zero times in any of those logs, and
`Decoder flush timed out` — two per run before the fix, one per seek — is gone.

**CPU**, `cpu_matrix.py`, four layers of 1080p25, each mixer against its own
software path, on the 90-second clips, never looped:

| clip | gl sw | gl gpu-direct | vk sw | vk gpu-direct |
|---|---|---|---|---|
| H.264 | 1.84 | 1.24 | 1.90 | 1.14 |
| HEVC | 1.85 | 1.24 | 1.96 | 1.16 |
| VP9 | 1.91 | 1.23 | 1.94 | 1.17 |

The GPU-direct figures are the ones to compare: 1.23–1.26 on OpenGL and 1.14–1.17
on Vulkan, against 1.22–1.26 and 1.14–1.16 recorded in `489b02fbc`. Unchanged.
The *software* baseline is noisy by about ±10 % between runs — HEAD itself
measured 2.00, 1.84 and 1.80 cores for the same OpenGL software row — so the
percentage saving moves with it and is not a stable figure to hold a change to.
Measure the absolute GPU-direct cost, or A/B within one run.

**Picture**, `matrix_isolated.py` on both mixers: **6/7 identical, up from 4/7**,
the same six on each mixer. `m_h264_8_prog` and `m_vp9_8_prog` now match the
software path byte for byte where they used to differ.

Those two were **not** colour differences. `489b02fbc` recorded three clips
differing between on and off (91.5 / 70.9 / 72.8 dB) and called them
pre-existing; two of the three were defect 3 — the harness pins each clip with
`SEEK 40 LENGTH 1 LOOP`, which is a wrap on every frame, so GPU-direct was
capturing a frame near the keyframe while software captured frame 40. On a
static source that reads as a small, evenly spread pixel difference rather than
as an obviously wrong picture, which is how it passed for a precision artefact.
Confirmed by reading `<time>` at the moment of capture: GPU-direct reported
1.12 s / 1.08 s (frames 28 / 27, keyframe at 25) against software's 1.64 s, and
the capture hash tracked the reported position exactly. It now reports 1.64 s on
both mixers, stable across 15 runs, hash-identical to software.

`m_hevc_10_prog` still differs, and that one is real: both paths land on 1.64 s
and both are bit-stable across runs, so it is a genuine deterministic difference
in the picture, not a frame mismatch. Pre-existing and out of scope here.

> The earlier note in this file claiming `iso_*_on_m_vp9_8_prog.png` was
> inherently unreproducible "at HEAD as well" was wrong, and wrong in an
> instructive way: it rested on a single HEAD run. The instability was defect 3,
> and HEAD looked stable only because its producer was frozen. One sample cannot
> establish that something is stable — if a capture is going to be called flaky,
> count the distinct hashes over at least half a dozen runs of each build.

**Late frames are not affected**, though the first measurement said otherwise.
`cpu_matrix.py` reported 9–15 late frames on the OpenGL GPU-direct row against
1–3 for the previous build, reproducibly, which looked like a CPU saving bought
with dropped frames. It was not. `<late_frames>` is cumulative from channel
start and `cpu_matrix.py` reads it once at the end, so it counts producer
startup, which varies with machine state — the high readings came after ~40
server launches, the low ones straight after a fresh build. A paired A/B
alternating two prebuilt binaries round by round, differencing the counter
across the measurement window, gives **0 late frames in the window for both**
over five rounds each, and 1.225 vs 1.239 cores. Alternate the arms; three runs
of one then three of the other cannot separate a real difference from a drifting
machine.

## Ground rules that earned their place here

- **Instrument before theorising.** A `grep` for `Decoder flush timed out`
  settled defect 2 in seconds, after the mechanism had been guessed at wrongly
  twice. Defect 3 was found by reading `<time>` at the moment of capture rather
  than reasoning about pixel differences.
- **A frozen producer looks like a stable one.** Every "pre-existing, stable"
  property measured while the stall was present is suspect, because a producer
  that decodes nothing repeats itself perfectly. Two of the three recorded
  picture differences turned out to be this. Re-measure baselines after fixing a
  stall; do not carry them over.
- **Loop tests must loop on a non-keyframe IN point.** Looping from frame 0
  cannot distinguish "wraps to the IN point" from "wraps to the keyframe",
  because they are the same frame. That blind spot cost a commit.
- **Check a probe fails on the broken build before trusting it to pass.** The
  IN-point probe was run against the previous binary first and does report the
  defect there; a test that passes on both builds proves nothing.
- **Difference a cumulative counter across the window, and alternate the arms.**
  `<late_frames>` counts from channel start, so reading it once at the end
  measures startup as much as the run, and back-to-back arms confound the
  comparison with machine drift. That combination produced a convincing,
  reproducible late-frame regression that did not exist.
- **Measure on a clip longer than the run, and do not loop it, when measuring
  CPU.** A stalled producer costs less than a working one, so a looping benchmark
  reports the stall as a saving. That produced plausible, documented-looking
  numbers once already.
- **Read the log to decide which path ran**, never infer from timings. A silent
  fallback and a path with no benefit are indistinguishable in a CPU figure.
- **Compare each mixer against its own software path**, and prefer absolute
  GPU-direct cost to a percentage — see the ±10 % baseline noise above.

## Build and environment

```
cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\
        VC\Auxiliary\Build\vcvars64.bat\" -vcvars_ver=14.50 &&
        cmake --build d:\Github\CasparVP\build --target casparcg"
```

Use **BuildTools** with the `-vcvars_ver=14.50` pin; the Community install has a
different toolset and mixing them fails confusingly. Before every link, kill
leftover servers or it fails with `LNK1168` — CEF spawns the executable as helper
subprocesses that hold the file:

```
Get-Process | Where-Object {$_.Path -like "*CasparVP*casparcg.exe"} |
    Stop-Process -Force
```

Also kill them between back-to-back harness runs: port 5260 stays bound by the
dying server and the next run attaches to it, which surfaces as
`ConnectionResetError` rather than anything to do with the build.

Another session may be working in this repository: stage files by name when
committing, never `git add -A`.
