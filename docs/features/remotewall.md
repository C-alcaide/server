# Remotewall — CloudXR tile-wall producer

> **State:** shipped, unmeasured
> **Modules:** `src/modules/remotewall` (producer, with `vendored/cloudxr` and `vendored/recvwall`)
> **Commands:** none — a producer named `remotewall`
> **Architecture:** none — a single network receive path into a CUDA composite; the decode handoff
> it shares with the other GPU codecs is in [`../architecture/GPU_CODEC_HANDOFF.md`](../architecture/GPU_CODEC_HANDOFF.md)
> **Guide:** [`../guides/REMOTEWALL_MODULE.md`](../guides/REMOTEWALL_MODULE.md)
> **Plan it delivers:** [`../plans/REMOTEWALL_NATIVE_MODULE_PLAN.md`](../plans/REMOTEWALL_NATIVE_MODULE_PLAN.md)
> **Coverage:** **none**

Receives video from a remote source as a layer, over CloudXR. Intended for a wall driven from
elsewhere on the network rather than from local media.

---

## 1. What is implemented today

| piece | evidence |
| :--- | :--- |
| Producer named `remotewall` | `remotewall_producer.cpp:514` |
| CloudXR vendored | `vendored/cloudxr` |
| `recvwall` receiver vendored | `vendored/recvwall` |
| Config root | `configuration.remotewall` |
| Exportable Vulkan texture path | `remotewall_producer.cpp:324` |
| CUDA→VK zero-copy on the Vulkan mixer | `06c84c5b0` |
| Colour tags, timecode and camera metadata over OSC/AMCP | `e13bb4da9` |
| Config, sync groups, live `SET` | `3acdc13d6` |
| AV1, 10-bit HDR, multi-GPU device match | `c58f5f3a1` |

**This module IS the native rewrite** its plan asked for, and the plan is delivered: phases 0-4 of
its §8 landed as named commits, phase 5 as `c58f5f3a1`, and six fixes after that. It is in the build
at `src/modules/CMakeLists.txt:25`.

---

## 2. The vendored dependencies are the standing risk

**Two vendored third-party components**, and this is the fact that matters most about the module.
CloudXR is an NVIDIA SDK with its own version and platform constraints; `recvwall` is a second
vendored component. Vendored code inside a fork is the hardest kind to keep current, and **nothing
records which revision either is, or when it was taken** — so there is no way to tell whether a
published CloudXR fix is in this tree.

That is a documentation gap with a cheap fix (record the revisions), and it is worth more than any
battery this module could get first.

---

## 3. Verification

**Nothing.** No battery starts a remote source, because that needs a second endpoint.

The realistic first check is narrower than end-to-end picture: that the producer **refuses legibly
when no source is reachable**, rather than hanging or presenting black indefinitely. That needs no
sender and would cover the failure an operator actually meets.

---

## 4. Known gaps

1. **No coverage at all**, and the narrow no-source check above does not exist either.
2. **No record of the vendored CloudXR or `recvwall` revisions** — §2.
3. **The `eUndefined` layout fix** (`0f1c5fb38`) applies here too and was verified only on ProRes.
4. **The commits' phase numbers drifted from the plan's §8**, so "phase 4" in a commit message and
   "phase 4" in the plan are not the same work. Read the plan's table, not the commit titles.
