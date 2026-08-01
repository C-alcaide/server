# GPU-direct decode does not resume after a seek

## The defect

With `configuration.ffmpeg.producer.gpu-direct-decode` enabled, a clip stops
producing frames after any seek. Looping is where it shows up in practice,
because the loop wrap *is* a seek.

```
PLAY 1-10 <4-second clip> LOOP
```

Position advances to the end, wraps to 0.00, and stays at 0.00 for ever.
Nothing is logged when it fails. Both mixers. It predates the Vulkan bridge
(`489b02fbc`) — verified by reproducing it on the OpenGL path.

**It is not about looping.** An explicit `CALL 1-10 SEEK 20` mid-clip behaves
identically: the position moves to the seek target and then stops. That is the
sharpest fact available and it narrows the search a long way — the question is
why hardware decoding does not resume after a flush, not anything to do with
loop points.

## Already fixed — do not re-do or revert

Commit `1423934ec` fixed two genuine defects in the same area. Both are verified
and neither is the remaining problem:

- **EOF never reached the reader.** The decoder marks end-of-file by pushing a
  sentinel into the software `output` queue, which the video filter turns into
  `video_filter_.eof`. GPU-direct frames bypass the filter graph, so the flag
  stayed false for ever and `buffer_eof_` never became true. Now mirrored from
  the decoder at the point the reader pulls hardware frames.
- **`hw_output` was never flushed.** It is emptied in exactly one place,
  `pop_hw()`. Every seek cleared the software `output` queue beside it and left
  the hardware queue holding pre-seek D3D11 surfaces. Now cleared alongside.

Proof the EOF fix works, on a clip played **without** `LOOP`: position used to
jump to 0.00 and stop, which is wrong because there is nothing to wrap to; it now
holds at 4.00, the end of the clip. Any change that breaks *that* has gone
backwards.

## Two wrong turns already taken

Both were caught by measurement. Do not repeat them.

1. **Latching the flag.** `video_filter_.eof = true` once, left set, made the clip
   wrap and then sit at 0.00 permanently — the wrap clears the decoder's eof but
   a latched flag still says end-of-file. It must mirror, not latch.
2. **Ignoring the asynchronous flush.** `seek_internal` requests a flush and
   returns; the decode thread performs it later. In that window the decoder still
   reports the end it reached *before* the seek, so acting on it asks for the wrap
   again. The mirror is now gated on `flush_pending()`.

## Where to look

The symptom is that `pop_hw()` returns nothing after a flush, for ever. Either
frames are not being produced, or they are produced and discarded. Establish
which **before** theorising further — instrument, do not reason. Every wrong turn
in this area so far came from reasoning about code that looked correct.

A useful first measurement: counters on packets fed to the decoder, frames
pushed to `hw_output`, and frames popped, sampled before and after a seek. If
pushes stop, the decode side is stuck; if pushes continue and pops do not
consume them, the reader is discarding.

Candidates, roughly in order of suspicion:

1. **Frames produced but discarded against the seek target.** The run loop has a
   fast-forward-to-target mechanism (`current_seek_target_`, `last_dropped_frame`)
   which drops frames until the target is reached. If the D3D11 frames' timestamps
   are not what that comparison expects, every frame is dropped and nothing ever
   arrives. This is the leading candidate because it explains the total silence:
   dropping is not an error and logs nothing.
2. **The decode thread is blocked.** It waits on `hw_output_cond` for queue space
   and on `input` for packets. Check it is not parked in either after the flush.
3. **`avcodec_flush_buffers` on a D3D11VA decoder** may need the hardware frames
   context re-primed before it produces surfaces again; the software path has no
   equivalent requirement.
4. **`gpu_direct_mode_` cleared on the decoder** during the flush, so frames go to
   the software queue while the reader still waits on the hardware one. The
   existing stand-down logs when it fires and nothing was logged, which makes this
   unlikely — but the flag is worth confirming rather than assuming.

## Reproducing

```
ffmpeg -f lavfi -i "testsrc2=size=1920x1080:rate=25" -frames:v 100 \
       -c:v libx264 -pix_fmt yuv420p -g 25 media/loop4s.mp4
```

Play it with `gpudirect_on.config` (OpenGL) or `gpudirect_on_vk.config` (Vulkan),
attach a consumer, and poll `INFO 1` for `<time>` once a second. Working looks
like the software control: the position wraps repeatedly. Broken is a position
that stops changing.

**A consumer must be attached** (`ADD 1 NDI NAME "x"`) or the channel skips
compositing entirely and nothing is exercised.

Compare against three controls, all of which must still pass afterwards:

| case | expected |
|---|---|
| software path, LOOP | wraps repeatedly |
| GPU-direct, no LOOP | holds at the end, 4.00 |
| GPU-direct, explicit mid-clip SEEK | plays on from the target |

## Verification bar

- The looping case wraps repeatedly on **both** mixers, for at least four wraps —
  one wrap proves nothing, since the current build manages one before stopping.
- The three controls above still pass.
- Re-run `gpudirect/cpu_matrix.py`: the 39–41 % saving must be unchanged, and the
  harness already fails any row whose log contains `Waiting for video frame`.
- Re-run the picture matrix: GPU-direct must stay byte-identical to the software
  path on the same mixer.

## Ground rules

These are not general advice; each one is a mistake already made in this work.

- **Measure on a clip longer than the run, and do not loop it, when measuring
  CPU.** A stalled producer costs less than a working one, so a benchmark that
  loops reports the stall as a saving. That artefact produced plausible,
  documented-looking numbers once already.
- **Read the log to decide which path ran**, never infer from timings. A silent
  fallback and a path with no benefit are indistinguishable in a CPU figure.
- **Instrument before theorising.** Four separate explanations in this area were
  plausible, consistent with the code, and wrong.
- Keep the software path and the OpenGL path working and verified. This is a
  producer used on air.

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

Harnesses are in `d:/Github/CasparCG-TestRunner/gpudirect/`. Another session may
be working in this repository: stage files by name when committing, never
`git add -A`.
