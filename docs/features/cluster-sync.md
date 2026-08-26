# Cluster sync — frame-accurate playback across servers

> **State:** shipped, unmeasured
> **Modules:** `src/modules/cluster`
> **Commands:** 4 fork-specific AMCP commands, registered by the module
> **Coverage:** **none**

Keeps playback aligned across several CasparCG servers driving one wall, so a clip started on four
machines shows the same frame on all four. A scheduled start time and a shared frame clock, with a
watchdog for the case where a node stops answering.

Operator detail is in [`../guides/CLUSTER_SYNC.md`](../guides/CLUSTER_SYNC.md). This document is the
state and what is measured.

---

## 1. What is implemented today

| command | purpose |
| :--- | :--- |
| `CLUSTER SCHEDULE` | schedule an action for a future frame, so every node performs it together |
| `CLUSTER TRACK` / `UNTRACK` | add or remove a node from the tracked set |
| `CLUSTER STATUS` | report the cluster's view of itself |

**Singleton state behind one mutex** (`g_state_mutex`), with a **watchdog** — 53 references to
`watchdog` or `frame_clock` in the module, so the failure handling is a substantial part of it
rather than an afterthought.

**One implementation detail worth surfacing, because it is a trap the module already caught
itself:** `frame_clock` is default-constructed at a **hardcoded 50 fps**, and
`sync_framerate_from_channels()` exists to correct it from the local channel's actual format. The
module's own comment says it is re-checked on every command rather than once, because it is cheap.
So the correct frame rate depends on a command having been issued — a fresh cluster that has been
scheduled but never otherwise touched is the case to think about.

---

## 2. How to drive it

```
CLUSTER TRACK 10.0.0.11
CLUSTER TRACK 10.0.0.12
CLUSTER STATUS
CLUSTER SCHEDULE ...
```

Node addressing and the schedule syntax are in the operator guide.

**No `configuration.cluster.*` keys are read by this module** despite `configuration.cluster`
appearing in it — worth knowing if you expected to configure the cluster from the config file
rather than at runtime.

---

## 3. Design decisions, and what they cost

**Scheduling by future frame, not by wall clock.** Frame-accurate alignment needs a shared frame
count; a timestamp would only be as good as NTP. The cost is that every node must agree on the
frame rate — which is exactly why the 50 fps default above matters.

**A watchdog rather than a consensus protocol.** Simpler and adequate for a wall of known nodes;
it does not attempt to keep running correctly through a partition, it detects and reports.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery starts two servers and checks they show the same frame.

The honest assessment: this is the hardest feature in the fork to test, because a real check needs
two machines and a way to compare their outputs at the same instant. But two pieces are testable on
one machine and are not:

1. **`sync_framerate_from_channels()` against a non-50 fps channel.** Start a 25p channel, issue a
   cluster command, and assert the frame clock reports 25. That is the hardcoded-default trap in §1
   and needs one server.
2. **The watchdog's timeout behaviour** with a tracked node that never answers — `CLUSTER TRACK` to
   an unroutable address, then `CLUSTER STATUS`, asserting it reports the node as lost rather than
   hanging.

Both are single-server checks of the two mechanisms most likely to be wrong.

---

## 5. Known gaps

1. **No coverage.** §4 lists two single-machine checks.
2. **The 50 fps default** is corrected by a function that runs on command dispatch. A cluster that
   is scheduled and then left alone is a case nobody has established.
3. **`configuration.cluster` is referenced but no keys are read** — either dead or unfinished.
4. **Multi-node behaviour is entirely unverified**, which is the feature's whole purpose.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Deferred rather than owed. A timing diagram would help, but it would describe intended behaviour
that nothing verifies — and per this folder's rule a diagram is a claim like any other. Worth
drawing *after* §4's two checks exist, so it illustrates something measured.
