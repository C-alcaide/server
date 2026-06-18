# Crash Investigation Report — CasparCG & HRC_PLAYOUT_CG

**Machine:** HRC playout workstation (`C:\APP_VIDEOREPORT\HRC_H2025T20`)
**Incident date:** Night of **2026-06-17** (first event ~19:47)
**Report compiled:** 2026-06-18
**Data sources:** Windows Event Log (Application + System), Windows Error Reporting (WER) archives, CasparCG console log, HRC application logs (`C:\APP_VIDEOREPORT\log`), CasparCG public source repository.

---

## 1. Executive summary

During the night of 2026-06-17 the playout system suffered a series of crashes that began at **19:47** and recurred through **23:18**. The Windows Event Log shows two distinct but **linked** failures:

1. **CasparCG server (SERVIDOR_4) is the primary cause.** It crashes with **heap corruption** (`0xc0000374`) and **access violations** (`0xc0000005`), always reported inside `ntdll.dll` (the Windows heap manager). This is an **internal memory-management fault in CasparCG's ffmpeg media pipeline**. The original theory was that it is triggered specifically when many producers are torn down at once during `CLEAR`/`REMOVE` — but a per-crash analysis of the console log (§5.1) shows that **only 2 of the 6 crashes that night involved any teardown at all**; the other 4 occurred during steady-state looping playback or during DeckLink startup, with no operator action. The corruption originates in the **continuous teardown/rebuild churn of the ffmpeg producers** (every loop iteration destroys and recreates decoder threads and filter graphs — §8.1), of which the `CLEAR`/`REMOVE` destructor path is only one entry point. **Mixed-allocator corruption, the unpatched OS, and the old GPU driver are credible co-factors (§8.2).**
2. **HRC_PLAYOUT_CG is a secondary, cascading failure.** It is the .NET controller that drives CasparCG over TCP. When the server dies, HRC's `Svt.Network` client loses the connection mid-command, **hangs** ("window is idle"), then crashes with a **`System.NullReferenceException`**.

**This is not a hardware fault** — the System log shows zero disk/driver/GPU errors and no memory exhaustion (the machine has ~128 GB RAM). It is also **not a one-night event**: the same pattern has recurred for ~2 years (**94 crash/hang events since August 2024**).

---

## 2. Timeline of Windows events — night of 2026-06-17

All entries from the **Application** event log. Times are local (24h).

| Time | Event ID | Source | Process | Failure detail |
|------|----------|--------|---------|----------------|
| **19:47:11** | 1000 | Application Error | **casparcg.exe** 2.4.3.0 | Faulting module `ntdll.dll` 10.0.18362.657, exception **`0xc0000374` (STATUS_HEAP_CORRUPTION)**, offset `0x00000000000f92a9`, PID `0x1f68`. Report `ba9d9744-61f1-4a1e-8a24-750c69282e4d`. |
| 19:47:16 | 1001 | Windows Error Reporting | casparcg.exe | APPCRASH, bucket `1450794868407642640` (type 4), `StackHash_ddd9`, `PCH_43_FROM_ntdll+0x9CC14`. |
| **21:04:34** | 1002 | Application Hang | **HRC_PLAYOUT_CG.exe** 1.0.0.0 | **AppHangB1** — "Top level window is idle", PID `4148`. Report `c8fceea5-2631-43ba-b51d-21e824522bb6`. |
| 21:04:34–35 | 1001 | Windows Error Reporting | HRC_PLAYOUT_CG.exe | AppHangB1 (Critical), bucket `1416056936973609814`. |
| **21:50:59** | 1000 | Application Error | **casparcg.exe** 2.4.3.0 | `ntdll.dll`, exception **`0xc0000005` (access violation)**, offset `0x000000000004d5a0`, PID `0x12b0`. Report `bff4cfb6-e030-4f66-ab99-6aa738f78fe6`. |
| 21:51:00 | 1001 | Windows Error Reporting | casparcg.exe | **FaultTolerantHeap** triggered (`ffffbaad`) — Windows detected repeated heap faults and engaged the FTH shim. |
| 21:51:06 | 1001 | Windows Error Reporting | casparcg.exe | APPCRASH `c0000005` / `ntdll.dll`. |
| **21:51:07** | 1002 | Application Hang | **HRC_PLAYOUT_CG.exe** | AppHangB1 — "window is idle", PID `0x22f0`. Report `ee6d470c-5754-445a-af3e-c659ce4e51b8`. |
| **21:52:46** | 1000 | Application Error | **casparcg.exe** 2.4.3.0 | `ntdll.dll`, exception **`0xc0000005`**, offset `0x4d5a0`, PID `0x43a4`. Report `3e741b33-b561-47cb-becc-19349db518c0`. |
| **21:53:31** | 1000 | Application Error | **HRC_PLAYOUT_CG.exe** 1.0.0.0 | Exception **`0xc0000005`**, module `unknown`, offset `0x0dda7009`, PID `0x2dd0`. Report `8b3b11b4-5b67-4d89-9575-76e4fdf93e9d`. |
| 21:53:36 | 1001 | Windows Error Reporting | HRC_PLAYOUT_CG.exe | **CLR20r3** — faulting assembly **`Svt.Network` 1.1.0.0** (`503df37c`), exception **`System.NullReferenceException`**, IL offset `0x35`. |
| **23:18:04** | 1000 | Application Error | **casparcg.exe** 2.4.3.0 | `ntdll.dll`, exception **`0xc0000374` (heap corruption)**, offset `0xf92a9`, PID `0x5698`. Report `49688ffd-8de7-4f9c-b81f-a9a2f783d530`. |
| 23:18:10 | 1001 | Windows Error Reporting | casparcg.exe | APPCRASH, `StackHash_78ed`, `PCH_43_FROM_ntdll+0x9CC14`. |

### Restart correlation (HRC application log `HRC_2026-06-17.log`)
Each crash is followed by an automatic restart of HRC:
- `19:47:41` — INICIANDO SISTEMA (after 19:47 casparcg crash)
- `21:04:39` — APP STARTING (after 21:04 hang)
- `21:51:11` — APP STARTING (after 21:50–21:51 crashes)
- `21:53:36` — APP STARTING (after 21:53 crash)

---

## 3. Exception code reference

| Code | Meaning | Interpretation here |
|------|---------|---------------------|
| `0xc0000374` | `STATUS_HEAP_CORRUPTION` | The process heap was already corrupted; `ntdll`'s heap manager detected it and aborted. The corruption is caused by earlier code (ffmpeg producer teardown), **not** by `ntdll` itself. |
| `0xc0000005` | `ACCESS_VIOLATION` | Invalid memory access — the typical secondary symptom once a heap is corrupted; the process dies at a random spot. |
| `AppHangB1` (1002) | Application not responding | The GUI thread stopped pumping messages ("Top level window is idle") — HRC stalled waiting on the dead server connection. |
| `CLR20r3` | Unhandled .NET exception | Managed crash; here a `System.NullReferenceException` inside the `Svt.Network` CasparCG client library. |

**Why `ntdll.dll` appears as the "faulting module":** `ntdll` hosts the Windows heap implementation. When corruption is detected, the crash is *reported* against `ntdll`, but the *cause* is whichever component corrupted the heap beforehand — in this case CasparCG's media pipeline.

---

## 4. System log & resource check (causes ruled out)

- **System event log, 19:30–00:00:** **No** Error, Critical, or Warning entries. → No disk, storage, display-driver/TDR, or hardware faults.
- **Resource-Exhaustion-Detector:** **No** events. → The system did not run out of RAM or page file.
- **Installed RAM:** `130791 MB` (~128 GB) per WER metadata. → Memory pressure is not a factor.

**Conclusion:** the crashes are software defects, reproducible, not environmental/hardware.

---

## 5. WER report deep-dive (first CasparCG crash, 19:47)

Source: `C:\ProgramData\Microsoft\Windows\WER\ReportArchive\AppCrash_casparcg.exe_cc3cb11f3e6445534357457482d535adcf969dff_89c6c0d0_0ee67219-b56d-4652-8774-1f269e599249\Report.wer`

> **Note:** the full crash **memory dumps** (`.dmp`) were already purged by Windows from `...\WER\Temp` by the time of investigation. Only the `Report.wer` metadata survived. See §9 for enabling persistent dumps.

**Faulting executable:** `C:\APP_VIDEOREPORT\HRC_H2025T20\SERVIDORES\SERVIDOR_4\casparcg.exe`, version `2.4.3.0`, build timestamp `2025/02/26`.

**Key loaded modules at time of crash** (identifies the subsystems in play):
- **ffmpeg media pipeline:** `avcodec-59.dll`, `avformat-59.dll`, `avfilter-8.dll`, `avutil-57.dll`, `swscale-6.dll`, `swresample-4.dll`, `postproc-56.dll`, `avdevice-59.dll`
- **Intel TBB allocator:** `tbbmalloc.dll`, `tbbmalloc_proxy.dll` — replaces the global `malloc`/`free`. Heap corruption while this is active almost always indicates a **double-free or buffer overrun** in application code, not an allocator bug.
- **CEF / HTML producer:** `libcef.dll`, `chrome_elf.dll`
- **Output / graphics:** `DeckLinkAPI64.dll` (Blackmagic), `nvoglv64.dll` (NVIDIA OpenGL), `OpenAL32.dll`, `glew32.dll`, `sfml-*-2.dll`, `FreeImage.dll`

**OS:** Windows 10 **Pro for Workstations**, build **1909** (`10.0.18363` / `ntdll 10.0.18362.657`), patch level **March 2020** (`190318-1202`). This is years out of support.

### Correlation with the CasparCG console log
Immediately before the 19:47 crash the CasparCG log shows several producers being destroyed and then `clear`/`remove` commands:
```
19:47:02  ffmpeg[...v10_TO_GENERICO_CITY.mp4] Destroyed.
19:47:02  ffmpeg[...v7_PLASMA CENTRO T20.mp4] Destroyed.
19:47:02  ffmpeg[...v7_MT_t19.mov] Destroyed.
19:47:02  ffmpeg[...BANNER_T20_V2.mp4] Destroyed.
19:47:02  ffmpeg[...v8_VW_GENERICO_T20_DER.mp4] Destroyed.
19:47:03  Received: clear 10-100  -> ERROR
19:47:03  Received: CLEAR  10     -> ERROR
19:47:03  Received: REMOVE 2-2 SCREEN -> REMOVE OK
```
> **Correction to the original interpretation:** the `clear 10-100` and `CLEAR 10` commands **both returned `ERROR`** — channel 10 does not exist (the server has only 6 channels), so the **bulk `CLEAR` never executed**. What actually ran was the routine pattern of 5 looping clips being re-`PLAY`ed simultaneously (each `PLAY` over an existing layer destroys the outgoing producer) plus a single `REMOVE … SCREEN`. So even this crash is **not** the "mass simultaneous `CLEAR`" the heap-corruption-on-teardown theory describes — it is the normal loop-replace pattern.

### 5.1 Per-crash analysis — the teardown theory only explains ~2 of 6 crashes
The 2.4.3 sessions begin at console-log line 6453. (Earlier sessions in the same log file are CasparCG **2.5.0** and are unrelated to this incident.) **Every crash leaves no error trace** — the log simply stops mid-stream and a restart banner follows. Reconstructing the final seconds before each death:

| Crash time | What was happening immediately before the crash | Teardown involved? |
|------------|--------------------------------------------------|--------------------|
| **19:47:03** | 5 looping clips re-`PLAY`ed at once (outgoing producers `Destroyed`) + `REMOVE 2-2 SCREEN`. Bulk `CLEAR` returned `ERROR` (did not run). | Partial (loop-replace) |
| **21:04:30** | Idle looping playback; last command was a `MIXER … ROTATION` *query*. | **No** |
| **21:50:54** | `PLAY 3-10 / 4-10` loop-replace. | Partial |
| **21:52:43** | Idle looping playback; only periodic `Latency` warnings. | **No** |
| **21:53:21** | **During DeckLink `ADD`/init** (a crash-recovery restart) — no producers were even playing yet. | **No** |
| **23:17:48** | Idle looping playback; only periodic `Latency` warnings. | **No** |

**Four of the six crashes happened during steady-state looping playback (or during DeckLink startup), with no `CLEAR`/`REMOVE` and no operator action.** A single "heap corruption from destroying producers quickly" mechanism cannot account for those. The common factor across all six is **the ffmpeg producer's continuous heap/thread churn**, which runs during normal looping just as much as during teardown (§8.1).

---

## 6. HRC_PLAYOUT_CG crash detail (downstream effect)

Source: `...\ReportArchive\AppCrash_HRC_PLAYOUT_CG.e_..._c29a5cc1-bc3a-4a82-8ec7-5a78ea61bac6\Report.wer`

- Type: **`CLR20r3`** (unhandled .NET exception)
- Faulting assembly: **`Svt.Network` v1.1.0.0** (the SVT/Open CasparCG **AMCP client library** used by HRC to control the server over `127.0.0.1`)
- Exception: **`System.NullReferenceException`**, IL offset `0x35`
- Executable build timestamp: `2025/09/08`

**Failure chain:**
1. CasparCG (SERVIDOR_4) corrupts its heap and dies.
2. HRC's `Svt.Network` TCP socket drops mid-command.
3. HRC's UI thread stalls → **AppHangB1** ("window is idle").
4. The library dereferences a now-null connection object → **`NullReferenceException`** → process crash.

HRC does not currently guard against, or auto-recover from, a dropped server connection.

---

## 7. Historical context — this is chronic

Aggregation of all retained Application-log crash/hang events (IDs 1000/1002) for these two processes:

| Count | Process | Failure type |
|------:|---------|--------------|
| 41 | HRC_PLAYOUT_CG | Hang (1002, "window is idle") |
| 24 | HRC_PLAYOUT_CG | Access violation `0xc0000005` |
| 17 | casparcg | **Heap corruption `0xc0000374`** |
| 9 | casparcg | Access violation `0xc0000005` |
| 2 | HRC_PLAYOUT_CG | Other |
| 1 | casparcg | Hang |
| **94** | **Total** | (retained log range) |

- **Oldest retained event:** 2024-08-28 10:10:23
- **Newest retained event:** 2026-06-18 07:36:35

CasparCG **only ever fails with heap corruption / access violations** (consistent with the internal memory bug). HRC predominantly **hangs and null-refs** (consistent with losing the server connection).

### Pre-existing HRC defects (independent of the server)
Earlier in the same day's HRC log (16:44–16:57), unrelated to the server crashes:
- Repeated **`Colección modificada; puede que no se ejecute la operación de enumeración`** — a .NET `InvalidOperationException` caused by modifying a collection while it is being enumerated (a thread-safety bug).
- **`El proceso no puede obtener acceso al archivo 'Thumbnail.txt'`** — file-in-use contention.

These are latent stability issues worth fixing but were not the trigger for the night's outage.

---

## 8. Root cause analysis

> **Note on version:** the crashing build is **official `2.4.3-stable` (commit `fd9168c`)** — confirmed from the console-log banner. The code references below are against the upstream `CasparCG/server` `v2.4.3-stable` tree, **not** the local CasparVP fork, which does not run on this machine.

### Primary (CasparCG) — the ffmpeg pipeline's heap/thread churn, not specifically mass teardown
A **memory-management defect in CasparCG 2.4.3.0's ffmpeg producer pipeline**. The original theory (corruption only under a mass concurrent `CLEAR`/`REMOVE`) is **too narrow**: §5.1 shows 4 of 6 crashes occurred with no teardown at all. The defect lives in the **ffmpeg teardown/rebuild machinery**, which is exercised continuously during normal looping — not just at `CLEAR`. Under corruption the process dies as `0xc0000374`, or once the heap is already damaged, as a random `0xc0000005`.

### 8.1 Why steady-state looping is itself a heap-churn path (code-verified)
In `src/modules/ffmpeg/producer/av_producer.cpp` (v2.4.3), **every loop iteration re-seeks**, and `seek_internal()` destroys and rebuilds the decode pipeline:
```cpp
// run() loop, on each wrap-around:
if (loop_ && frame_count_ > 2) { frame = Frame{}; seek_internal(start); }
...
void seek_internal(int64_t time) {
    ...
    decoders_.clear();   // destroys each Decoder -> interrupt()+join() its boost::thread
    reset(time);         // rebuilds video/audio Filter graphs (avfilter_graph_alloc/free)
}
```
With 5–6 clips looping every ~20 s, the server is **continuously destroying and recreating decoder threads, codec contexts and filter graphs on the realtime producer thread** — a constant stream of cross-DLL `malloc`/`free` and thread create/join. A memory-safety bug here crashes during *idle looping* exactly as observed at 21:04, 21:52 and 23:18, with no operator action. There is also a genuine data race in the `Decoder` constructor — the worker lambda reads `thread.interruption_requested()` while the `thread` member is still being assigned:
```cpp
thread = boost::thread([=]() {
    while (!thread.interruption_requested()) { ... }   // reads 'thread' mid-assignment
});
```
The asynchronous-teardown destructor path (§9) is therefore **one entry point into the same fragile machinery**, not a separate root cause.

### 8.2 Alternate / contributing causes (independent of producer count)
These can corrupt the heap regardless of how producers are cleared, and the original report ruled them out too quickly:

- **Mixed allocators (strong candidate).** `tbbmalloc_proxy.dll` replaces the global `malloc`/`free`, but `DeckLinkAPI64.dll`, `nvoglv64.dll` (NVIDIA OpenGL), `libcef.dll` and `FreeImage.dll` allocate/free across DLL boundaries. Memory allocated by one heap and freed by another corrupts the process heap and surfaces later as `0xc0000374` in `ntdll`, **unrelated to producer-teardown volume**.
- **GPU driver.** NVIDIA **552.86** is loaded as `nvoglv64.dll`. Driver-side heap/handle bugs can corrupt the process heap **without** a TDR — the System log rules out *TDR*, not in-process driver corruption.
- **Unsupported OS.** Windows 10 **1909 / build 18363**, last patched **March 2020**. The Windows heap manager (`ntdll`) itself is years out of bugfix/security patches. **`ntdll.dll` version comparison:**
  - **This machine:** `ntdll.dll` **10.0.18362.657** (1909 base `18362`, March 2020).
  - **Latest Windows 10 (22H2, KB5094127, June 2026, OS build 19045.7417):** `ntdll.dll` **≈ 10.0.19041.7417** (the 2004–22H2 family shares the `19041` base; the 4th field tracks the update revision/UBR).
  - **Delta:** an entire feature-family jump (`18362` → `19041`) plus ~6 years and thousands of servicing revisions of heap-manager fixes. Because every crash is reported inside `ntdll`'s heap code, running a heap manager this far out of date is a genuine variable — not necessarily *the* cause, but it removes a large body of upstream heap-robustness fixes that could mask or mitigate the application bug.
- **The restart loop is its own failure mode.** At 21:51→21:53 there were three restarts in ~2.5 minutes, and one crash occurred *during DeckLink init*. A killed process does not cleanly release the DeckLink devices or the GL context, so the next instance can crash grabbing hardware left in an inconsistent state — a cascade, not the root cause.
- **DeckLink config mismatch.** Every channel logs `Failed to enable external keyer`; the config requests external keying on devices/wiring that reject it. Not a crash by itself, but it shows the DeckLink configuration is fighting the hardware on this box.
- **Ruled out:** DeckLink driver 14.3+ ABI break (upstream issue #1593) affects only **≤ 2.4.1**; **2.4.2 / 2.4.3 / 2.5.0 contain the fix**, so it is **not** a factor on this 2.4.3 build.

### Secondary (HRC)
`Svt.Network` does not survive the server dropping the TCP connection: it hangs the UI thread and then throws an unhandled `NullReferenceException`.

### Version / fix-availability check
- **CasparCG `2.4.3` (2025-02-26) is the final 2.4.x release** — there is no newer 2.4.x to move to.
- The only newer release is **`2.5.0` (2025-12-10)**, a large rebuild (ffmpeg 5.1 → 7.0/7.1, CEF update, VS2022 + newer MSVC runtime, AVX2 recommended).
- **No specific upstream commit was found that fixes this heap-corruption bug**, and both the asynchronous/detached teardown design and the per-loop `seek_internal()` teardown/rebuild (§8.1) are **still present in current `master`**. Therefore it **cannot be confirmed** that 2.5.0 fixes this exact issue — it must be validated empirically.

#### Verification: does 2.5.0 actually change the heap behaviour? (source-compared against `v2.5.0-stable`)
The relevant CasparCG code was diffed against the 2.5.0 tag. The conclusion is that **2.5.0 changes the heap *environment* but not the suspect *code paths*:**

| Element | 2.4.3 | 2.5.0 | Effect on heap behaviour |
|---------|-------|-------|--------------------------|
| `~ffmpeg_producer` teardown | `std::thread(...).detach()` per producer | **Byte-for-byte identical** | **Unchanged** — still one detached thread per producer |
| Per-loop `seek_internal()` (`decoders_.clear()` + filter rebuild) | present | **Identical** | **Unchanged** — same continuous churn (§8.1) |
| `Decoder` ctor thread-member data race | present | **Identical** | **Unchanged** |
| `AVProducer::Impl::~Impl` (implicit destruction order) | present | **Identical** | **Unchanged** |
| Allocator | `tbbmalloc` + **`tbbmalloc_proxy`** (older TBB) | **Still links `tbbmalloc` + `tbbmalloc_proxy`**, upgraded to **oneTBB 2022.3.0** | **Architecture identical** — the proxy still overrides global `malloc`/`free`, so cross-DLL frees from DeckLink/NVIDIA/CEF/FreeImage remain a corruption vector; only the allocator *version* is newer |
| ffmpeg | 5.x (avcodec-59 / avutil-57 / avfilter-8 …) | **7.0.2** (avcodec-61 / avutil-59 / avfilter-10 …) | **Different** — new internal buffer pools / allocation patterns; may shift *timing* of corruption but does not fix the CasparCG-side bug |
| Compiler / CRT | VS2019-era MSVC runtime | **VS2022 MSVC runtime, `/arch:AVX`** | **Different** — new CRT heap/iterator behaviour; AVX build (hence the AVX2-capable-CPU requirement) |

**Verdict:** the earlier wording (“may incidentally change the heap behaviour”) is confirmed but should be read narrowly. The heap *context* changes (newer `tbbmalloc`, ffmpeg 7.0.2, VS2022 CRT), which can move or mask the symptom, but **every CasparCG code path identified as suspect in §8.1–9 is unchanged in 2.5.0, and the mixed-allocator architecture is fully retained.** So 2.5.0 is **not a targeted fix** for this defect; any improvement would be incidental and **must be proven by testing**, not assumed.

---

## 9. Where a code fix would be needed (CasparCG)

Three structural locations in the CasparCG source (`CasparCG/server`), in priority order:

1. **`src/modules/ffmpeg/producer/ffmpeg_producer.cpp` (`~ffmpeg_producer`, ~L94)** — the destructor spawns an **unbounded, detached `std::thread`** to perform the blocking teardown. Under mass `CLEAR`/`REMOVE` this creates *N concurrent teardown threads* freeing ffmpeg resources on the shared heap simultaneously. **Primary fix:** route teardown through a single bounded worker instead of detaching an unlimited number of threads.
2. **`src/core/producer/frame_producer_registry.cpp` (`destroy_producer_proxy`, ~L80)** — the asynchronous-destruction scheduler. Contains `CASPAR_VERIFY(destroyer->size() < 8)` which can **throw from a destructor** during a large simultaneous clear. This is the right place to enforce a bounded teardown queue.
3. **`src/modules/ffmpeg/producer/av_producer.cpp` (`~Impl`, ~L788)** — relies on implicit member-destruction order to join decoder/input threads. Should **explicitly quiesce the pipeline in a defined order** (abort input → join run thread → stop decoder threads → free filters → free decoders) under lock.
4. **`src/modules/ffmpeg/producer/av_producer.cpp` (`seek_internal` / `Decoder` ctor)** — the per-loop teardown path (§8.1). Every loop wrap calls `decoders_.clear()` + filter rebuild on the realtime thread, and the `Decoder` worker lambda reads `thread.interruption_requested()` while the `thread` member is still being assigned (a data race). This is the path implicated in the **4 idle-playback crashes** that the teardown theory does not explain, and is the most important area to harden (capture the thread handle by value before the lambda, and avoid rebuilding the whole pipeline on every loop).

### 9.1 Proposed code changes (illustrative)

> These patches are **illustrative** — they convey the intended change and must be compiled and tested against the current source tree (line numbers/APIs may shift between versions). **Fix 4a (the `Decoder` data race) and Fix 4b (stop rebuilding the pipeline every loop) target the path behind the 4 idle-playback crashes and are the highest priority.** Fix 1 (bounding teardown) addresses the mass-`CLEAR`/`REMOVE` path; Fixes 2 and 3 are additional hardening.

**Fix 1 — bound the ffmpeg teardown instead of detaching unbounded threads**
`src/modules/ffmpeg/producer/ffmpeg_producer.cpp`

```cpp
// --- BEFORE -------------------------------------------------------------
~ffmpeg_producer()
{
    std::thread([producer = std::move(producer_)]() mutable {
        try {
            producer.reset();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
    }).detach();                       // unbounded: one new thread PER producer
}

// --- AFTER --------------------------------------------------------------
namespace {
// Single shared worker that performs the slow ffmpeg teardown off the
// realtime thread, WITHOUT spawning an unbounded number of concurrent
// threads. This serializes producer destruction so a mass CLEAR/REMOVE
// no longer races many ffmpeg/tbbmalloc frees against each other.
caspar::executor& ffmpeg_producer_destroyer()
{
    static caspar::executor destroyer(L"ffmpeg_producer_destroyer");
    return destroyer;
}
} // namespace

~ffmpeg_producer()
{
    ffmpeg_producer_destroyer().begin_invoke([producer = std::move(producer_)]() mutable {
        try {
            producer.reset();
        } catch (...) {
            CASPAR_LOG_CURRENT_EXCEPTION();
        }
    });
}
```
*Notes:* `caspar::executor` (from `common/executor.h`) is a single background thread with its own queue, so teardown stays off the realtime path but is **serialized** rather than parallel. If some parallelism is desired, use a small fixed-size pool (e.g. 2–4 workers) instead of one — never an unbounded thread-per-producer.

**Fix 2 — do not throw from a destructor during mass CLEAR/REMOVE**
`src/core/producer/frame_producer_registry.cpp` (`~destroy_producer_proxy`)

```cpp
// --- BEFORE -------------------------------------------------------------
CASPAR_VERIFY(destroyer->size() < 8);   // can THROW inside a destructor

// --- AFTER --------------------------------------------------------------
// Throwing from a destructor while many layers are being cleared at once is
// unsafe. Apply backpressure via logging instead of aborting.
if (destroyer->size() >= 8) {
    CASPAR_LOG(warning) << L"Producer destroyer backlog: " << destroyer->size();
}
```

**Fix 3 — explicit, ordered teardown of the producer pipeline**
`src/modules/ffmpeg/producer/av_producer.cpp` (`AVProducer::Impl::~Impl`)

```cpp
// --- BEFORE -------------------------------------------------------------
~Impl()
{
    input_.abort();
    try {
        if (thread_.joinable()) { thread_.interrupt(); thread_.join(); }
    } catch (boost::thread_interrupted&) {}
    video_executor_.reset();
    audio_executor_.reset();
    // decoders_, filters_, sources_, input_ freed implicitly afterwards
    CASPAR_LOG(debug) << print() << " Joined";
}

// --- AFTER --------------------------------------------------------------
~Impl()
{
    // 1) Stop feeding the pipeline.
    input_.abort();

    // 2) Join the run loop FIRST, so nothing else touches decoders_/filters
    //    while we tear them down.
    try {
        if (thread_.joinable()) { thread_.interrupt(); thread_.join(); }
    } catch (boost::thread_interrupted&) {}

    // 3) Stop the filter executors before freeing the graphs they reference.
    video_executor_.reset();
    audio_executor_.reset();

    // 4) Destroy the pipeline explicitly, in a defined order, while fully
    //    quiesced — instead of relying on implicit member-destruction order.
    sources_.clear();
    video_filter_ = Filter{};
    audio_filter_ = Filter{};
    decoders_.clear();   // joins each Decoder's worker thread

    CASPAR_LOG(debug) << print() << " Joined";
}
```
*Note:* relies on `Filter` being default-constructible/assignable (it already is — `reset()` reassigns it). Confirm against the target source.

**Fix 4 — close the `Decoder` constructor data race and stop rebuilding the whole pipeline on every loop (the idle-crash path)**
`src/modules/ffmpeg/producer/av_producer.cpp` (`Decoder` ctor and `seek_internal`)

This is the **highest-priority hardening** because it targets the path behind the **4 idle-playback crashes** (§8.1) that the teardown theory does not explain.

*4a — remove the data race: the worker lambda must not read the `thread` member while that member is still being assigned. Use an explicit abort flag instead.*
```cpp
// --- BEFORE -------------------------------------------------------------
// Worker reads 'thread' (the member it is being assigned to) -> data race.
thread = boost::thread([=]() {
    while (!thread.interruption_requested()) { /* decode loop */ }
});

// --- AFTER --------------------------------------------------------------
// Member of Decoder:
//   std::atomic<bool> abort_{false};
// Destructor sets abort_ = true; (in addition to interrupt()/join()).
thread = boost::thread([this]() {
    try {
        while (!abort_.load(std::memory_order_relaxed)) { /* decode loop */ }
    } catch (boost::thread_interrupted&) {}
});
// The lambda now reads only 'abort_' and 'this', never the 'thread' object
// it is stored into, so there is no read of a partially-assigned member.
```

*4b — avoid tearing down and rebuilding the entire decode pipeline on every loop wrap. Reuse the existing decoders/filters and just seek, instead of `decoders_.clear()` + full `reset()` each iteration.*
```cpp
// --- BEFORE -------------------------------------------------------------
void seek_internal(int64_t time) {
    ...
    decoders_.clear();   // destroys+recreates every Decoder thread each loop
    reset(time);         // reallocates the whole avfilter graph each loop
}

// --- AFTER --------------------------------------------------------------
void seek_internal(int64_t time) {
    ...
    // Flush the existing decoders/filters and seek the demuxer, but KEEP the
    // Decoder threads and filter graphs alive across a normal loop wrap.
    input_.seek(time);
    for (auto& [index, decoder] : decoders_)
        decoder.flush();            // avcodec_flush_buffers, no thread churn
    flush_filters();                // reset filter state without realloc
    // Only fully rebuild (decoders_.clear() + reset(time)) when the source
    // graph actually changes (resolution/codec/stream-map change), not on a
    // plain loop boundary.
}
```
*Notes:* 4a is a small, low-risk correctness fix and should be applied first. 4b is a larger change that removes the **continuous** cross-DLL `malloc`/`free` and thread create/join churn that runs during normal looping; it needs careful testing of seek-accuracy and A/V sync but is what actually eliminates the idle-crash pressure. Both must be validated against the target source tree.

### Impact on broadcast behavior (important)
Applying these fixes **does not change on-air behavior**. CasparCG already separates:
- **Phase 1 — visual switch (real-time):** `CLEAR`/`REMOVE` drops the layer's producer pointer and the output goes black on the next frame. *Untouched by the fix.*
- **Phase 2 — teardown (background):** reclaiming memory/threads of the discarded producer. *Only this is changed* — from "free everything in parallel instantly" to "free serialized / bounded in the background."

Result: outputs still go black simultaneously and immediately; only the **invisible background cleanup** is staggered (which is exactly what removes the corruption).

---

## 10. Recommendations

### CasparCG (root cause)
- **Mitigation (no code change):** staggering `CLEAR`/`REMOVE` only helps the ~2 teardown-related crashes; it will **not** prevent the 4 idle-playback crashes (§5.1). A more effective no-code mitigation is to **reduce continuous loop churn** — e.g. avoid having many short clips (~20 s) loop indefinitely, since each loop re-tears-down and rebuilds its ffmpeg pipeline (§8.1). Where possible, use longer/static assets or pause idle layers.
- **Capture a full crash dump first (see §9 diagnostics).** Because the cause is now known to be broader than teardown, a dump is the only way to confirm *which* path (loop churn, mixed allocator, driver) is corrupting the heap before investing in a fix.
- **Validate `2.5.0` in a staging/test environment** with the real playout pattern (note its AVX2 recommendation and required newer MSVC runtime). Do not assume it fixes the bug without testing.
- **Update the NVIDIA driver** (552.86) and **patch/upgrade the OS** (see below) before drawing conclusions — both are credible heap-corruption co-factors (§8.2).
- If confirmed reproducible, **report upstream** with the full crash dump and reference the code locations in §9.

### HRC_PLAYOUT_CG (resilience)
- Wrap `Svt.Network` calls in try/catch with **automatic reconnect**, and null-guard the connection object so a server drop degrades gracefully instead of hanging/crashing.
- Fix the **`Colección modificada`** enumeration bug (snapshot or lock the collection before iterating).

### Operating system
- **Patch/upgrade Windows** off the unsupported 1909 / March-2020 level.

### Diagnostics — capture full dumps for the next occurrence
The actual crash dumps had already been purged. To capture a full, analyzable dump next time, create the following **(requires an elevated/Administrator session)**:

Registry: `HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\casparcg.exe` (and a second key for `HRC_PLAYOUT_CG.exe`)
- `DumpFolder` = `C:\APP_VIDEOREPORT\CrashDumps` (REG_EXPAND_SZ)
- `DumpType` = `2` (full dump, REG_DWORD)
- `DumpCount` = `10` (REG_DWORD)

> The dump folder `C:\APP_VIDEOREPORT\CrashDumps` has already been created. The registry keys are still pending (they require Administrator rights). With a real dump, the exact corrupting function inside the ffmpeg teardown can be resolved.

---

## 11. Reference — WER report identifiers (for traceability)

| Time | Process | Report ID |
|------|---------|-----------|
| 19:47 | casparcg | `ba9d9744-61f1-4a1e-8a24-750c69282e4d` |
| 21:04 | HRC_PLAYOUT_CG | `c8fceea5-2631-43ba-b51d-21e824522bb6` |
| 21:50 | casparcg | `bff4cfb6-e030-4f66-ab99-6aa738f78fe6` |
| 21:51 | HRC_PLAYOUT_CG | `ee6d470c-5754-445a-af3e-c659ce4e51b8` |
| 21:52 | casparcg | `3e741b33-b561-47cb-becc-19349db518c0` |
| 21:53 | HRC_PLAYOUT_CG | `8b3b11b4-5b67-4d89-9575-76e4fdf93e9d` |
| 23:18 | casparcg | `49688ffd-8de7-4f9c-b81f-a9a2f783d530` |

WER archives location: `C:\ProgramData\Microsoft\Windows\WER\ReportArchive\` (folders prefixed `AppCrash_casparcg.exe_*` and `AppCrash_HRC_PLAYOUT_CG.e_*` / `Critical_HRC_PLAYOUT_CG.e_*`).

---

*End of report.*
