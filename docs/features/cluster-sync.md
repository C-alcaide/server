# Cluster sync — frame-accurate playback across servers

> **State:** shipped, unmeasured
> **Modules:** `src/modules/cluster`
> **Commands:** 4 fork-specific AMCP commands, registered by the module
> **Architecture:** [`../architecture/CLUSTER_SYNC_DESIGN.md`](../architecture/CLUSTER_SYNC_DESIGN.md)
> **Guide:** [`../guides/CLUSTER_SYNC.md`](../guides/CLUSTER_SYNC.md)
> **Coverage:** `frame_clock_self_test` at boot — the frame arithmetic. No battery: multi-node behaviour is still driven by nothing

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

**And four subsystems this section used to omit entirely**, which between them are most of the
module's 2,900 lines — audited 2026-09-14 because a doc that lists four commands and a mutex
reads like a thin feature, and this is not one:

| part | what it is |
| :--- | :--- |
| `ptp/` (635 lines) | a **PTP clock** over UDP multicast, with `master`, `client` and `external` modes — `external` defers to a grandmaster on the network |
| `relay/` (718 lines) | a **TCP command relay** plus a **virtual channel map**, so a command issued on the master is executed on the member that owns that virtual channel |
| `sync/command_scheduler` | parses and executes a relayed AMCP command at a target frame |
| `sync/content_sync` | the watchdog: compares what members are playing and reports divergence |

**Singleton state behind one mutex** (`g_state_mutex`), with the watchdog above.

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

**The cluster IS configured from the config file** — this paragraph used to say the opposite,
and it was wrong from the day it was written. `parse_cluster_config(env::properties())` is
called from `cluster.cpp` and reads `mode`, `bind`, `multicast-group`, `ptp-domain`,
`sync-interval-ms`, `sync-margin`, `relay-port`, `epoch-origin`, `content-sync` (and its
threshold and max-layer), `log-ptp-status`, plus the `<channels>` and `<members>` sub-trees.
Runtime `CLUSTER TRACK` adds to what the file establishes; it is not the only way in.

**How the claim survived is worth more than the claim.** `cluster::init` is called from
`build/generated/included_modules.h` — a GENERATED header — so a grep of `src/` for the
module's entry point finds nothing and the module reads as unreachable. The same trap nearly
had the replay module written up as dead code on 2026-09-14; see `CLAUDE.md`.

---

## 3. Design decisions, and what they cost

**Scheduling by future frame, not by wall clock.** Frame-accurate alignment needs a shared frame
count; a timestamp would only be as good as NTP. The cost is that every node must agree on the
frame rate — which is exactly why the 50 fps default above matters.

**A watchdog rather than a consensus protocol.** Simpler and adequate for a wall of known nodes;
it does not attempt to keep running correctly through a partition, it detects and reports.

---

## 4. Verification — what is measured, and what is not

**`frame_clock_self_test()`, at every boot, unconditional.** The frame arithmetic is a pure
function — no cluster, no network, no channel — so it is asserted at start-up rather than left
to a two-machine battery that does not exist. It throws, which aborts the server.

It checks, for 25p, 50p, 60p, 23.976p, 29.97p and 59.94p, that `ptp_ns_to_frame` and
`frame_to_ptp_ns` are mutually consistent at five sub-second phases across 400 seconds (the
frame it reports has started, and the next one has not), and that the frame number is monotone.

**The oracle is the RELATIONSHIP between the two conversions, not a second copy of the
formula** — an expected value written as the implementation's own expression cannot fail.

**And it is gated on 1001-denominator rates because that is where the bug was.** At 25p and 50p
the denominator is 1, the defective and correct forms agree exactly, and an integer-rate check
would have passed on a broken build. Mutation-verified: with the original arithmetic compiled
back in, the server **refuses to start**, naming 29.97p.

**Measured by nothing at all:** everything multi-node. PTP convergence, master/client election,
the command relay, the virtual channel map, the scheduler firing on the right frame, and the
watchdog's divergence report. That is the feature's whole purpose and it is unverified — §5.

### 4.1 What the audit found

**`ptp_ns_to_frame` was one frame low on every fractional frame rate.** It computed
`floor(a) + floor(b)` where the answer is `floor(a + b)`: the whole-seconds term
`(elapsed_sec * num) / den` truncates, and the lost fraction was dropped rather than carried
into the sub-second term. Measured over 3000 instants per rate:

| rate | instants wrong | error |
| :--- | :--- | :--- |
| 25p, 50p | 0 of 3000 | — |
| 29.97p | 1735 of 3000 (58%) | one frame low |
| 59.94p | 2273 of 3000 (76%) | one frame low |
| 23.976p | 2349 of 3000 (78%) | one frame low |

`ptp_ns_to_frame` is how a node answers *"which frame is it"*, so on any NTSC-rate channel two
nodes sampling at slightly different sub-second phases **disagreed about the frame number** —
which is the one thing this module exists to prevent. `sync_framerate_from_channels()` passes
`video_format_desc().framerate` straight through, so 1001 denominators reach it on any such
channel.

The corrected form carries the remainder and was checked against the exact rational value over
240,000 random instants across six rates, with five orders of magnitude of `int64` headroom.

---

## 5. Known gaps

1. **No coverage.** §4 lists two single-machine checks.
2. **The 50 fps default** is corrected by a function that runs on command dispatch. A cluster that
   is scheduled and then left alone is a case nobody has established.
3. ~~**`configuration.cluster` is referenced but no keys are read.**~~ **WRONG, corrected
   2026-09-14** — thirteen keys plus two sub-trees are read; see §2. Recorded here rather than
   deleted because the reason it was believed is reusable: the module's entry point is called
   from a GENERATED header, so it looks unreachable to a grep.
4. **Multi-node behaviour is entirely unverified**, which is the feature's whole purpose — and
   it is verifiable on ONE machine, which is the useful half of this line. `create_udp_socket`
   sets `SO_REUSEADDR`, so two server processes on one host can both bind the PTP ports and
   join the multicast group, and `relay-port` is per-member configurable. **WSL is not the
   route**: there is no Linux build of this server, and WSL2 is NAT'd behind a virtual switch
   so host↔WSL2 UDP multicast does not work — which is exactly the PTP half.

5. **`relay-port` defaults to 5250, which is also the AMCP port**, so a client node with an
   otherwise-default config asks its relay listener to bind a port the AMCP server already
   holds. A bind failure is logged as an error and the thread exits — but
   `start_client_listener` logs *"Client relay listening on port N"* from the CALLING thread
   before the listener has attempted anything, so the log says both. **And the listener sets
   `SO_REUSEADDR`**, which on Windows permits binding a port another socket already holds
   rather than refusing — so the two may both bind and compete for connections, which is worse
   than a clean failure. Found by reading on 2026-09-14 and **not tested**: stated as a
   question for the one-box battery above to answer, not as a finding.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Deferred rather than owed. A timing diagram would help, but it would describe intended behaviour
that nothing verifies — and per this folder's rule a diagram is a claim like any other. Worth
drawing *after* §4's two checks exist, so it illustrates something measured.
