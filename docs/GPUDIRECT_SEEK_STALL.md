# GPU-direct decode did not resume after a seek — fixed

With `configuration.ffmpeg.producer.gpu-direct-decode` enabled, a clip stopped
producing frames after any seek. Looping was where it showed up, because the loop
wrap *is* a seek: position advanced to the end, wrapped to 0.00, and stayed there
for ever, on both mixers, with nothing in the log. An explicit `CALL 1-10 SEEK 20`
behaved identically, which is what ruled out anything loop-specific.

Two defects, both in `av_producer.cpp`. Neither is in the mixer or the bridge.

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

**Picture**, `matrix_isolated.py` on both mixers. 25 of 28 captures are
byte-identical to the pre-change build, and the verdict table is identical:
`m_h264_8_prog`, `m_hevc_10_prog` and `m_vp9_8_prog` differ between on and off on
both mixers, which is exactly the pre-existing set recorded in `489b02fbc`
(91.5 / 70.9 / 72.8 dB). The differences are unchanged in character: 0.007 % of
pixels in a 16×21 box for H.264, ~1–2 % spread evenly for the other two.

> `iso_*_on_m_vp9_8_prog.png` is **not reproducible run to run** on a single
> build — back-to-back matrix runs of the same binary produce different hashes
> for it and for nothing else. Do not read a changed hash on that one capture as
> a regression; it is why the "byte-identical to the software path" bar reads
> 4/7 or 5/7 depending on the run, at HEAD as well.

## Ground rules that earned their place here

- **Instrument before theorising.** Both defects above were found by measurement
  — a `grep` for `Decoder flush timed out` in the run log settled the second one
  in seconds, after the mechanism had been guessed at wrongly twice.
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
