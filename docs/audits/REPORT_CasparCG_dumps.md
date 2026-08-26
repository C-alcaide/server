# CasparCG Crash-Dump Forensic Report

**Subject:** CasparCG server 2.4.3.0 crashes — night of **2026-06-17**
**Machine:** HRC playout workstation (`SERVIDOR_4`)
**Dumps analyzed:** `casparcg.exe.{8040, 4784, 17316, 22168}.dmp`
**Binaries:** bundled `bin\` (exact 2.4.3.0 build; `casparcg.exe` ImageBase `0x140000000`)
**Method:** Offline parse with Python `minidump`; native disassembly with `pefile` + `capstone`. No CasparCG PDBs, so application frames resolve as `module+RVA`; Windows/`ntdll` frames matched against WER bucket metadata.
**Report date:** 2026-06-19

> These are **DumpType 1** minidumps: faulting-thread stacks, registers, and the loaded-module list — **no heap pages**. They show *where* each crash tripped and *which* code path led there, but not the corrupted block's contents. Full heap forensics requires a DumpType=2 dump on the next crash (see README).

> **Update 2026-06-20:** follow-up reproduction testing has refined the thread-count interpretation below. The 548–603 thread figure is now understood to be **largely config-confounded** (it scales with CPU cores × concurrent producers × resolution) and is **not, by itself, evidence of a leak** — controlled testing with **local-file** churn showed threads, memory, and handles all reclaim cleanly. **However, a later test that injected network faults into an HLS source produced the first positive leak signal of the investigation: a monotonic accumulation of OS *thread handles* (~1,000/min) under HLS reconnect churn — see Addendum §8.** This corroborates upstream issue [#1549](https://github.com/CasparCG/server/issues/1549) ("more frequent with HLS"). The decisive heap-corruption evidence remains the **identical fault signature and identical culprit frame** across the two crashes. See the dated **Addendum (§7, §8)** for the full follow-up analysis and what it means for the proposed fixes.

> **RESOLVED 2026-06-21 — handle leak root-caused and fixed (see Addendum §9).** The thread-handle leak of §8 is **not** an ffmpeg teardown bug. It is CasparCG's own **`boost::stacktrace`** (MSVC **dbgeng/windbg** backend): every thrown exception captures a stack trace, and every `CASPAR_LOG_CURRENT_EXCEPTION` symbolizes it via dbgeng, which **attaches to the process and leaks a handle to every thread** on each call. The leak is **source-agnostic** — it scales with the *logged-exception rate*, not with HLS specifically; HLS was merely the easiest way to generate a fast exception storm on the bench. Any reconnect-prone network protocol (RTMP/RTSP/SRT/UDP/NDI/…) or even local-file error path drives the same leak, just at a lower rate. The §8.3 "decoder-rebuild" attribution and the `threads=1` lead were **measurement artifacts** (dbgeng was duplicating handles to the decode threads). Fix = stop capturing/symbolizing per-exception stacks (two files: `src/common/except.h`, `src/common/log.h`); validated **flat** under an 8-min faulty-HLS soak with full multi-threaded decode. Crash dumps are unaffected.
>
> **⚠️ This does NOT explain the 2026-06-17 production crashes.** Those occurred on a **local-file, stream-free** server — exactly the §7 condition where the leak was measured at **zero net drift**. The production crashes are `0xC0000374` **heap corruption** (a cross-module bad free, §1–§4), a **different mechanism** from a handle leak, and they remain **open and unproven**. The §8/§9 leak and the production crashes are most likely **two separate problems**; do not treat this fix as a fix for the local-file crashes.

---

## 1. Executive summary

The four CasparCG dumps fall into two groups that tell one coherent story:

1. **Two heap-corruption crashes (`0xC0000374`)** — dumps **8040** (19:47) and **22168** (23:18) — are **provably the same bug**. They share a byte-identical fault signature and the **same CasparCG application culprit frame** (`casparcg.exe+0x28a044`) on the faulting thread, 3.5 hours apart. This is one reproducible teardown/free code path, not random damage.
2. **Two access-violation crashes (`0xC0000005`)** — dumps **4784** (21:50) and **17316** (21:52) — fault inside `ntdll`'s heap free/walk routine (`ntdll+0x4d5a0`). These are **downstream symptoms**: the heap was already corrupted and the next allocation/free tripped over the damage.

Two pieces of **new quantitative evidence** emerge from the stacks that the WER metadata alone could not show:

- **Thread explosion: 548–603 live threads** in the process at crash time (a 6-channel server should run a few dozen), nearly all parked in `ntdll` waits. This is hard evidence of **thread accumulation** from per-loop decoder-thread creation and detached teardown threads. *(See §7 Addendum 2026-06-20: subsequent testing shows thread count is largely config-driven and not by itself proof of a leak; this figure is retained as context, but is no longer presented as the primary evidence.)*
- In dump 22168, the corrupting thread shows **`tbbmalloc.dll` and `avutil-57.dll` frames** beneath the CasparCG frames — a **cross-module free through `tbbmalloc_proxy` in the ffmpeg path**, directly supporting the mixed-allocator corruption theory.

**Conclusion:** the dumps confirm and strengthen the root-cause attribution — a reproducible CasparCG ffmpeg-teardown/free path corrupts the shared heap; the access violations are secondary heap-walk failures; and the thread count quantifies the churn/accumulation the proposed fixes target.

---

## 2. Per-dump findings

| Dump | PID | Time | Exception | Fault site | Faulting RIP | Threads |
|------|-----|------|-----------|-----------|--------------|---------|
| **8040** | 0x1f68 | 19:47 | `0xC0000374` heap corruption (noncontinuable) | `ntdll+0xf92a9` | `ntdll+0x9cc14` | **548** |
| **22168** | 0x5698 | 23:18 | `0xC0000374` heap corruption (noncontinuable) | `ntdll+0xf92a9` | `ntdll+0x9cc14` | **552** |
| **4784** | 0x12b0 | 21:50 | `0xC0000005` access violation | `ntdll+0x4d5a0` | — | 1 (degenerate) |
| **17316** | 0x43a4 | 21:52 | `0xC0000005` access violation | `ntdll+0x4d5a0` | `ntdll+0x4d5a0` | **603** |

### 2.1 Dump 8040 — heap corruption (19:47)
- Exception `0xC0000374` **NONCONTINUABLE**. Fault address `ntdll+0xf92a9` (matches WER offset `0xf92a9`). `ExceptionInformation[0] = ntdll+0x1627f0`.
- Faulting RIP `ntdll+0x9cc14` — **exactly** the WER bucket `PCH_43_FROM_ntdll+0x9CC14`. This is `ntdll`'s heap-corruption reporting path (`RtlReportFatalFailure` family).
- **548 threads**; census: 548 `ntdll`, 6 `win32u`.
- **Faulting-thread stack scan shows the application culprit `casparcg.exe+0x28a044`** (at sp+0x0188), sandwiched between the `ntdll` heap-error frames — i.e. the CasparCG code that invoked the free/destruct that `ntdll` then flagged.

### 2.2 Dump 22168 — heap corruption (23:18)
- **Signature identical to 8040**: fault `ntdll+0xf92a9`, RIP `ntdll+0x9cc14`, `ExceptionInformation[0] = ntdll+0x1627f0`, noncontinuable.
- **Same culprit `casparcg.exe+0x28a044`** at the same stack offset (sp+0x0188).
- Deeper on the same stack: **`tbbmalloc.dll+0xd5ea`**, **`casparcg.exe+0x250ffe`**, and **`avutil-57.dll`** frames (`+0xd0c79`, `+0xdd888`, `+0xd1bd3`).
- **552 threads**; census: 531 `ntdll`, 7 `vcruntime140`, 6 `win32u`, 3 `casparcg`, 3 `avcodec-59`, 1 `nvoglv64`, 1 `msvcrt`.

### 2.3 Dump 4784 — access violation (21:50)
- Exception `0xC0000005`, continuable. Fault `ntdll+0x4d5a0` (matches WER). `ExceptionInformation = [0x0, 0x0]`.
- **Degenerate capture: only 1 thread, no thread context.** Taken at 21:50–21:51 as Windows **FaultTolerantHeap** engaged (`ffffbaad` in the event log). Not independently diagnostic, but consistent with a heap already failing.

### 2.4 Dump 17316 — access violation (21:52)
- Exception `0xC0000005`, continuable. Fault and RIP both `ntdll+0x4d5a0`.
- **603 threads**; census: 595 `ntdll`, 7 `win32u`, 1 `avcodec-59`.
- Faulting window is entirely `ntdll` heap-free frames (`ntdll+0x718f5 / +0x71853 / +0x717fe`) with no readable CasparCG frame in the captured window — i.e. the crash lands **inside** the heap free routine walking an already-corrupt structure.

---

## 3. Disassembly of the culprit offsets

Disassembling the bundled `casparcg.exe` (ImageBase `0x140000000`, 1025 imports):

- **`casparcg.exe+0x28a044`** — the key application culprit, present on **both** heap-corruption faulting stacks. The surrounding code is a run of cleanup calls to internal routines followed by a tail-call:
  ```
  0x28a02f: call  sub_28a404
  0x28a036: call  sub_2b75c4
  0x28a03e: call  sub_2b75ca
  0x28a044: sub   rsp, 0x28          <-- return address lands here
  0x28a048: call  sub_28a644
  0x28a051: jmp   0x289ec8
  ```
  This is the shape of a **destructor / exception-unwind funclet** (repeated member cleanups + tail-call to a sub-object teardown). It is consistent with an ffmpeg-producer teardown path, **but cannot be named without PDBs** — stated here as structural evidence only, not a confirmed symbol.
- **`casparcg.exe+0x250ffe`** — resolves to the instruction after `call qword ptr [rip+...]` → **`KERNEL32.dll!TlsGetValue`** (the CRT/allocator thread-local lookup). Part of the thread-local/cleanup path; not independently meaningful.

---

## 4. Correlation and interpretation

1. **The two heap crashes are the same defect.** Identical `ntdll` signature *and* identical CasparCG culprit frame, 3.5 hours apart → one reproducible code path, not random heap noise.
2. **The access violations are downstream.** Both fault inside `ntdll`'s heap free/walk (`+0x4d5a0`) — the classic secondary failure mode once a heap is corrupted.
3. **The corrupting free is in the ffmpeg/allocator path.** The `tbbmalloc` + `avutil-57` frames beneath the CasparCG culprit in dump 22168 show a cross-DLL free routed through `tbbmalloc_proxy` — the mixed-allocator vector.
4. **Thread accumulation is real and large.** 548–603 live threads is far above a healthy steady state and quantifies the per-loop decoder-thread + detached-teardown-thread churn.

These four points are mutually consistent and align with the original investigation's root-cause attribution (ffmpeg producer teardown/rebuild churn under a mixed allocator).

---

## 5. Recommendations

- **Reduce teardown/thread churn** — bound the producer-destroyer thread pool and reuse ffmpeg decoders across loop wraps instead of destroying/recreating decoder threads and filter graphs every iteration. (This is what the proposed fixes target. The decoder-reuse fix is directly relevant to the **HLS thread-handle leak** in §8, which rides on the per-loop/per-reconnect decoder rebuild.)
- **Investigate the HLS reconnect thread-handle leak (§8).** Under HLS network faults the process accumulates handles to **already-exited threads** (live threads stay ~380 while thread handles climb past 13,000). Audit the reconnect/decoder-rebuild path for a thread created per reconnect whose handle is never `join()`ed/`detach()`ed/closed; verify avcodec multi-thread decoder teardown on the abnormal/reconnect path.
- **Capture a full dump next time.** Set `DumpType=2` for `casparcg.exe` (regkey in the README) so the corrupted heap block can be walked and the exact freed object identified.
- **Validation signal after fixes:** flat process RSS, a stable thread *handle* count under HLS reconnect soak, and absence of `0xC0000374` recurrence.

---

## 6. Limitations

- DumpType 1: no heap pages → the corrupted block itself cannot be inspected; the culprit is identified by faulting-stack code path, not by the damaged allocation.
- No CasparCG PDBs → `casparcg.exe+0x28a044` / `+0x250ffe` are RVAs; the destructor/unwind interpretation is structural (disassembly shape), not a resolved symbol.
- Stack frames beyond the faulting frame are recovered by a heuristic qword scan of stack memory (clamped to the containing segment), not a true unwind; candidate return addresses are validated by resolving into loaded modules.

---

## 7. Addendum — follow-up reproduction testing (2026-06-20)

After this report was first published, a controlled reproduction effort was run against the **exact unfixed 2.4.3.0 binary** from the bundled `bin\` directory, driven over AMCP with a topology matched to the production SERVIDOR1 show (5 program channels in sync, channel 6 = channel 1 routed to an HD output, screen consumers on channels 1–2, a distinct clip per channel). Process threads, RSS, OS handles, and GDI/USER object counts were sampled every 30 s throughout.

### 7.1 What the testing showed

- **No resource leak under sustained churn.** Across ~3,800 mid-show `MIX` producer replacements (≈760 full 5-channel transitions) over ~28 minutes, every metric oscillated within a **bounded band and returned to its baseline**:
  - **Handles**: trough flat at **2,916–2,931** — i.e. *zero* net drift after ~1,900 producer teardowns. A per-producer handle leak would have shown ~1,900 handles of growth; none was observed.
  - **Threads**: baseline **~379**, spiking to **~487** only during the overlap of a MIX transition, then settling back. No upward creep.
  - **RSS**: bounded **~3.2–3.9 GB**; **GDI** flat at 32; **USER** flat at ~4,820.
- **Coordinated end-of-broadcast teardown** (REMOVE all screens + CLEAR all channels, repeated) likewise produced a clean sawtooth (threads collapsing to ~106, then rebuilding) with **no crash**.
- **Passive looping** had already been confirmed not to reproduce the crash; this testing extends that negative result to **active teardown and mid-show churn** as well.

### 7.2 Refined interpretation

1. **The crash is not a slow leak / resource exhaustion.** Both teardown churn and mid-show churn reclaim fully. The earlier "thread accumulation" reading is **superseded**: thread count is largely a function of CPU-core count × concurrent producers × resolution, and is *config-confounded* rather than a leak indicator.
2. **The crash is consistent with a rare race / bad-free** — a one-shot heap corruption that occurs only on a specific teardown interleaving. This matches both the dump evidence (a single, reproducible cross-module free at `casparcg.exe+0x28a044` with `tbbmalloc` + `avutil-57` frames) and the operational reality that **most show days do not crash**.
3. **The corruption mechanism (cross-module free under the mixed `tbbmalloc_proxy` allocator) is unchanged and remains the core finding.** What changed is the *driver*: not gradual accumulation, but a low-probability race in the teardown free path.
4. **A trigger outside pure AMCP churn is plausible.** The synthetic tool deliberately never touches the **DeckLink** consumers (to protect SDI hardware), so it cannot exercise a reference-signal-loss or device-reconfiguration teardown. An external event (e.g. genlock/reference loss or a network drop forcing an internal consumer reconfigure) reaching the same buggy free path is a credible explanation for the rare production crashes and is the next avenue to test against a production log captured around a real crash time.

### 7.3 What this means for the proposed fixes (PRs #1755 / #1756)

The fixes remain worthwhile, but their **justification is reframed from "leak plug" to "race-window / frequency reduction"**:

| Fix | Original framing | Reframed in light of 2026-06-20 testing |
|-----|------------------|------------------------------------------|
| **Bounded producer-destroyer teardown** | Stops "thread explosion" / exhaustion | Thread-exhaustion rationale is **weakened** (no leak observed). Still valuable as a **concurrency limiter**: fewer simultaneous cross-module frees ⇒ smaller window for the race. |
| **ffmpeg decoder reuse across loop wraps** | Cuts per-loop decoder create/destroy churn | **Strengthened indirectly.** It removes the most frequent execution of the exact teardown free path implicated in the corruption, lowering crash probability — though it reduces *frequency*, it does not by itself *close* the race. |
| **Screen-consumer / GPU-free teardown hardening** | Addresses the 67% end-of-broadcast fingerprint | Could not be tripped synthetically; if the real trigger is a DeckLink reconfiguration, additional hardening of the **consumer-reconfiguration** path may be needed beyond what the current PRs cover. |

**Honest limitations of the validation:** because **neither** the unfixed nor the fixed build crashes under synthetic load, a bench A/B ("unfixed crashes, fixed does not") was **not** achievable. The fixes should therefore be treated as **low-risk hardening that reduces the concurrency and frequency of the implicated teardown free path**, with **validation deferred to production soak** (crash-rate reduction across show days), not a reproduced bench failure. None of the fixes are contradicted by this testing; the thread-count argument in §1/§2 should simply not be relied upon as primary evidence.

---

## 8. Addendum — HLS reconnect thread-handle leak (2026-06-20)

The §7 testing used **local files** and found clean reclaim. A follow-up test added the one production-relevant variable the local-file test was missing — **a network source that fails** — and produced the **first positive leak signal of the entire investigation.**

### 8.1 Why HLS, and how it was tested

Upstream issue [#1549](https://github.com/CasparCG/server/issues/1549) reports the **same** `0xC0000374` heap-corruption signature (no error logs, present from 2.3.x → master, graphics-independent, both SDI and HLS) and explicitly notes it is **"more frequent with HLS."** No real flaky HLS source was available, so a **fault-injecting HLS server** was built (`flaky_hls_server.py`): it serves a normal VOD playlist but, per `.ts` segment request, randomly (a) sends a partial body then resets the socket (truncated read), (b) returns 503/404, or (c) stalls then closes (read timeout). CasparCG was pointed at it and driven with the same churn harness.

Under faults the server log showed exactly the expected reconnect storm: *"Stream ends prematurely"*, *"keepalive request failed … Error number -10053 … retrying with new connection"*, *"Error when loading first segment"*, and constant `.ts` re-opens on fresh `hls@`/`http@` contexts — i.e. **repeated demuxer/decoder teardown and rebuild driven by the network faults.**

### 8.2 The measurement

Handle sampling of the **same unfixed 2.4.3.0 binary** (PID 443292):

| Condition | Handle trough behaviour |
|-----------|-------------------------|
| Local-file churn (§7) | **Flat** 2,916–2,931 (zero net drift over ~1,900 teardowns) |
| **HLS fault churn** | **Monotonic climb** 2,987 → 8,615 in ~6 min (**≈1,000 handles/min**), continuing past **13,000** |

The leaked handles were typed without Sysinternals by reading the live handle table (`NtQuerySystemInformation(SystemExtendedHandleInformation)`) and mapping `ObjectTypeIndex` via in-process probe objects. Result:

- **The runaway type is `Thread` (object-type index 8).** At 418 min uptime the process held **~7,000 → 13,000 Thread handles while only ~380 threads were actually alive** — i.e. thousands of handles to **already-exited threads that were never closed.** Event (~1,300) and Semaphore (~590) were mildly elevated; everything else was flat.

This is a true handle leak (handles to dead threads), **distinct** from the live-thread "explosion" discussed in §1/§7, and it appears **only** under the HLS reconnect path.

### 8.3 Where it comes from (code attribution)

The leak rides on the **decoder/demuxer teardown-and-rebuild rate**, which the HLS faults drive far higher than normal playback:

- In the **running 2.4.3-stable** build, every loop-wrap/seek and every reconnect-induced EOF runs `av_producer::seek_internal()` → **`decoders_.clear()` + full recreate** (`src/modules/ffmpeg/producer/av_producer.cpp`). Each `Decoder` (a) spawns a `boost::thread` worker and (b) opens an avcodec context with `threads=0` (**auto = N CPU cores**), spawning N frame-thread workers. Per rebuild that is ~(1 + N) threads created and destroyed; the HLS fault stream multiplies the rebuild rate, so any per-teardown thread-handle that is not released accumulates quickly.
- The producer teardown itself runs on a **detached** `std::thread` in `~ffmpeg_producer` (`ffmpeg_producer.cpp`); `detach()` releases that handle, so it is part of the churn but not itself the leak.
- **Next step to name the exact line:** audit the reconnect/abnormal-EOF teardown path for a thread whose handle is not `join()`ed/`detach()`ed/closed (candidate sites: the avcodec multi-thread decoder teardown on the error/reconnect path, the `Decoder` worker join, and the `Input` read thread join). Setting `configuration.ffmpeg.producer.threads = 1` and re-measuring would confirm whether the leaked handles are the avcodec frame-thread pool (rate should drop ~N-fold).

### 8.4 What this means for the fixes

- The **ffmpeg decoder-reuse fix (PR #1756, "Fix 4b")** flushes decoders **in place** across a loop-wrap/seek instead of `decoders_.clear()` + recreate. Because the HLS leak rides on the per-loop/per-reconnect rebuild, this fix is expected to **substantially reduce the thread-handle accumulation rate**, not merely the heap-race frequency — a more direct benefit than §7.3 credited it with.
- This finding gives the fixes a **concrete, measurable validation signal** that §7 lacked: under an HLS reconnect soak, the **unfixed build leaks Thread handles (~1,000/min) and the fixed build should hold them flat** — a bench A/B that *is* achievable, unlike the heap crash itself.
- It also independently corroborates **#1549's "more frequent with HLS"** observation with a mechanism: HLS reconnects hammer the exact teardown/rebuild path implicated in the corruption.

**Limitation:** the thread-handle leak is established; whether it is the *same* defect that ultimately trips the `0xC0000374` heap corruption, or a parallel symptom of the same over-exercised teardown path, is not yet proven. The exit-on-first-death harness remains armed to capture a full-heap dump if the HLS soak trips the crash.

---

## 9. Addendum — HLS thread-handle leak root-caused and fixed (2026-06-21)

The §8 leak was tracked to its source. It is **not** in the ffmpeg producer at all — it is in CasparCG's exception/logging machinery.

### 9.1 Root cause: `boost::stacktrace` over the dbgeng backend

- `CASPAR_THROW_EXCEPTION` (`src/common/except.h`) attaches a `boost::stacktrace::stacktrace()` to **every** thrown exception (cheap — captures return addresses only).
- `CASPAR_LOG_CURRENT_EXCEPTION` (`src/common/log.h`) then **symbolizes** that trace via boost's default MSVC backend, **dbgeng/dbghelp** (`windbg`). Each symbolization call **attaches the debug engine to the process and opens a handle to every thread** in it — and those per-thread handles are **never released**.
- The HLS fault stream is a high-rate exception generator: each truncated read / 503 / timeout → an ffmpeg error → `CASPAR_THROW_EXCEPTION` → logged → one dbgeng walk → one leaked handle **per live thread**. With ~380 live threads and a steady throw rate this produces exactly the **~1,000 handles/min** of §8.2.

### 9.2 How the §8.3 attribution was corrected

The earlier "decoder-rebuild / `threads=1`" lead was a **confound**: the leaked handles were dbgeng *duplicating* a handle to every thread, so reducing the avcodec frame-thread count (`threads=1`) only reduced the *number of threads dbgeng cloned a handle to* — it never touched the real leak. Distinct-TID accounting (separating duplicate-handle inflation from genuine dead-thread leaks) showed the leaked handles pointed at **live** threads across **all** modules evenly — the signature of a debugger walking the whole process, not an ffmpeg teardown. Confirming experiments:

- `dbgeng.dll` + `dbghelp.dll` were loaded; **`CLEAR`-all-channels stopped handle growth dead** (no throws ⇒ no symbolization).
- A `BOOST_STACKTRACE_USE_NOOP` rebuild **unloaded dbgeng** and held handles **flat** under identical faulty churn.

### 9.3 The fix (validated)

Two-file surgical change on branch `CasparVPV` (FFmpeg path left fully multi-threaded — no perf compromise):

- **`src/common/except.h`** — stop attaching a stacktrace on throw (removed `stacktrace_info` capture from `CASPAR_THROW_EXCEPTION`).
- **`src/common/log.h`** — `CASPAR_LOG_CURRENT_EXCEPTION` logs exception type + message + file:line only; no per-exception dbgeng symbolization. The `get_stack_trace()` helper is kept for *manual* diagnostic use.

**Validation:** 8-minute faulty-HLS soak (5-channel churn, full multi-threaded decode + filter) → OS handles **flat** (~3,000, oscillating, no upward trend), **0** zombie threads. Compare §8.2's 2,987 → 8,615 climb on the unfixed binary under the same load. The crash-dump path (`main.cpp` `safe_dump_to/from_dump`) is **separate and untouched** — it captures addresses signal-safely and symbolizes once at next startup, so it does not leak and full backtraces are preserved.

### 9.4 Scope of the fix (source-agnostic)

The leak is in the **exception-logging path**, not in any specific producer, so the fix is **not HLS-specific**. The leak rate is `~(live thread count) × (logged-exception rate)`; HLS reconnect storms merely maximize the second term. The same leak is driven by **any** exception-generating source — RTMP/RTMPS, RTSP, HTTP progressive, UDP/RTP, SRT, MPEG-TS, NDI, and even local-file error paths (decode errors, missing/corrupt media) — only at a lower throw rate, hence slower accumulation. A clean local-file show throws few exceptions, which is why §7's local-file soak stayed flat.

### 9.5 Relationship to the `0xC0000374` production crashes — **likely a separate problem**

This fix resolves the **§8 thread-handle leak**, but it should **not** be assumed to fix the 2026-06-17 production crashes:

- **Different environment.** Production ran **local files with no streams** — the §7 condition under which the handle leak measured **zero net drift**. A stream-free box does not generate the exception storm the leak depends on, so the leak was almost certainly *not* a significant factor there.
- **Different mechanism.** The crashes are `0xC0000374` **heap corruption** (a cross-module bad free under the mixed `tbbmalloc_proxy` allocator, §1–§4). Handle exhaustion presents as failures to create threads/events/handles, **not** as a corrupted heap block. These are distinct failure modes.
- **Status: open.** The heap-corruption root cause remains **unproven**; the `DumpType=2` full-heap capture (README) is still the way to close it.

**Weak, unproven bridge:** *if* the production server threw exceptions at a low background rate (e.g. DeckLink reference-loss / device reconfiguration, or occasional local-media decode errors), each logged throw would still leak ~(live-thread-count) handles, accumulating slowly over a long show day. That path is **speculative**, would manifest as handle exhaustion rather than `0xC0000374`, and was **not** reproduced in §7. It is plausible — not established — that reduced per-exception overhead also slightly narrows the teardown race window. Treat the §1–§4 heap corruption as **still open**.
