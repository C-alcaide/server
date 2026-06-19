# CasparCG Crash-Dump Forensic Report

**Subject:** CasparCG server 2.4.3.0 crashes — night of **2026-06-17**
**Machine:** HRC playout workstation (`SERVIDOR_4`)
**Dumps analyzed:** `casparcg.exe.{8040, 4784, 17316, 22168}.dmp`
**Binaries:** bundled `bin\` (exact 2.4.3.0 build; `casparcg.exe` ImageBase `0x140000000`)
**Method:** Offline parse with Python `minidump`; native disassembly with `pefile` + `capstone`. No CasparCG PDBs, so application frames resolve as `module+RVA`; Windows/`ntdll` frames matched against WER bucket metadata.
**Report date:** 2026-06-19

> These are **DumpType 1** minidumps: faulting-thread stacks, registers, and the loaded-module list — **no heap pages**. They show *where* each crash tripped and *which* code path led there, but not the corrupted block's contents. Full heap forensics requires a DumpType=2 dump on the next crash (see README).

---

## 1. Executive summary

The four CasparCG dumps fall into two groups that tell one coherent story:

1. **Two heap-corruption crashes (`0xC0000374`)** — dumps **8040** (19:47) and **22168** (23:18) — are **provably the same bug**. They share a byte-identical fault signature and the **same CasparCG application culprit frame** (`casparcg.exe+0x28a044`) on the faulting thread, 3.5 hours apart. This is one reproducible teardown/free code path, not random damage.
2. **Two access-violation crashes (`0xC0000005`)** — dumps **4784** (21:50) and **17316** (21:52) — fault inside `ntdll`'s heap free/walk routine (`ntdll+0x4d5a0`). These are **downstream symptoms**: the heap was already corrupted and the next allocation/free tripped over the damage.

Two pieces of **new quantitative evidence** emerge from the stacks that the WER metadata alone could not show:

- **Thread explosion: 548–603 live threads** in the process at crash time (a 6-channel server should run a few dozen), nearly all parked in `ntdll` waits. This is hard evidence of **thread accumulation** from per-loop decoder-thread creation and detached teardown threads.
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

- **Reduce teardown/thread churn** — bound the producer-destroyer thread pool and reuse ffmpeg decoders across loop wraps instead of destroying/recreating decoder threads and filter graphs every iteration. (This is what the proposed fixes target; the 548–603 thread count is the metric to watch — a healthy build should hold a flat, low thread count over long looping playback.)
- **Capture a full dump next time.** Set `DumpType=2` for `casparcg.exe` (regkey in the README) so the corrupted heap block can be walked and the exact freed object identified.
- **Validation signal after fixes:** flat process RSS and a stable, low thread count over an extended loop-playback soak; absence of `0xC0000374` recurrence.

---

## 6. Limitations

- DumpType 1: no heap pages → the corrupted block itself cannot be inspected; the culprit is identified by faulting-stack code path, not by the damaged allocation.
- No CasparCG PDBs → `casparcg.exe+0x28a044` / `+0x250ffe` are RVAs; the destructor/unwind interpretation is structural (disassembly shape), not a resolved symbol.
- Stack frames beyond the faulting frame are recovered by a heuristic qword scan of stack memory (clamped to the containing segment), not a true unwind; candidate return addresses are validated by resolving into loaded modules.
