# Cluster sync — three layers, and which one to suspect

> **State and measurements:** [`../features/cluster-sync.md`](../features/cluster-sync.md)
> **Operator guide:** [`../guides/CLUSTER_SYNC.md`](../guides/CLUSTER_SYNC.md)
> **This document is why-it-is-shaped-this-way.** Operating instructions live in `guides/`, current state and figures in `features/`.

Frame-accurate playback across several machines. The design is **three independent layers**, and the
reason to write that down is diagnostic: a cluster that drifts, a cluster that jumps, and a cluster
where one node ignores a command are three different faults in three different layers, and the
symptom does not tell you which.

---

## 1. The three layers

```
ptp/       ptp_clock          agree on TIME
             ↓
sync/      frame_clock        turn time into a frame number
           command_scheduler  execute a command AT a frame
             ↓
relay/     command_relay      get the command to every node with a frame stamp
           virtual_channel_map  decide which local channel it lands on
```

| layer | question it answers | failure looks like |
| :--- | :--- | :--- |
| **ptp** | *what time is it, everywhere?* | slow, cumulative **drift** between nodes |
| **sync** | *which frame is that, and has it arrived?* | a **jump** or a stall on one node, timing correct elsewhere |
| **relay** | *did every node get the instruction?* | one node **ignores** a command entirely, or acts on the wrong channel |

**Diagnose in that order.** A relay problem cannot cause drift, and a PTP problem cannot cause a
single node to miss a `PLAY`. Working upward from the symptom is what the layering buys.

---

## 2. Master and client are the same binary, different role

`command_relay` (`relay/command_relay.h`) states both halves:

> *"Master: connects to all members, stamps commands with target frame, forwards.
> Client: accepts connection from master, feeds received commands to scheduler."*

So the **master stamps** and the **client schedules**. That division matters more than it looks:

* A client never decides *when* — it is told a frame number and waits for it. So a client whose
  picture is early or late is a **clock** problem, not a scheduling one.
* The master's stamp is a **future** frame. If the stamp has already passed by the time a client
  receives it, the client cannot obey it and the command is late by construction — which makes the
  master→client latency budget part of the design rather than an implementation detail.

`virtual_channel_map` is the reason a cluster is not just N copies of the same rundown: a command
addressed to a virtual channel lands on whichever local channel that node maps it to, so one
instruction can drive different physical outputs per machine.

---

## 3. `command_scheduler` is a priority queue on a tight loop

From its own header: *"Priority queue that executes commands at their target frame. Runs a tight
loop checking frame_clock and dispatching commands."*

Two consequences worth knowing before changing it:

**A command whose target frame has passed is a decision, not an error.** The queue is ordered by
target frame, so a late arrival is at the head immediately — whether it fires or is dropped is
policy, and that policy is what determines whether a late node catches up or skips.

**The loop is separate from the channel tick.** Scheduling and rendering are not the same thread, so
a scheduler stall and a render stall are distinguishable in a log — and conflating them is the
easiest way to misdiagnose this module.

---

## 4. Why PTP rather than the OS clock or genlock

* **The OS clock is not good enough** — network time services correct in steps large enough to move
  a frame boundary, which is exactly the artefact this module exists to prevent.
* **Genlock alone is not sufficient** — it locks *timing* but carries no frame *numbering*, so two
  genlocked machines can be perfectly in phase and one frame apart. `frame_clock` turning agreed
  time into an agreed frame number is the part genlock cannot do.
* They compose: genlock for phase, PTP for numbering. On a rig with a sync board, expect to use
  both, and see `../guides/VULKAN_OUTPUT.md` §3 for the per-output sync group that sits underneath.

---

## 5. Unverified

**No battery drives a real cluster**, because that needs more than one machine. `../features/cluster-sync.md`
records what single-machine checks exist. Specifically untested: PTP convergence, master election
or failover, relay behaviour when a client disconnects mid-show, and what the scheduler does with a
command whose target frame has already passed — the policy question in §3.

This document is read from the source. Where it and `src/modules/cluster` disagree, the code wins
and the disagreement is a finding.
