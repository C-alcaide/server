# Cluster sync — frame-accurate playback across servers

> **State:** shipped, unmeasured
> **Modules:** `src/modules/cluster`
> **Commands:** 4 fork-specific AMCP commands, registered by the module
> **Architecture:** [`../architecture/CLUSTER_SYNC_DESIGN.md`](../architecture/CLUSTER_SYNC_DESIGN.md)
> **Guide:** [`../guides/CLUSTER_SYNC.md`](../guides/CLUSTER_SYNC.md)
> **Coverage:** `cluster` — 8/8 both mixers, light AND 4K-loaded, two nodes on one box — plus `frame_clock_self_test` at boot

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

**The hardcoded 50 fps was a real defect, not merely a case to think about — fixed 2026-09-14.**
`frame_clock` is default-constructed at 50 fps and `sync_framerate_from_channels()` corrects it
from the local channel's format, but it was called from only **two local AMCP commands**,
`SCHEDULE` and `TRACK`. So a cluster that was configured, PTP-locked and relay-connected still
numbered frames at 50 until somebody issued one of those.

**And the client is the node that never does.** A client receives its commands over the relay;
neither of those two call sites runs on it. The master syncs when it schedules and the client did
not, so a target frame computed at 59.94 was interpreted at 50 — the two nodes acting on the same
frame number at different real times, which is exactly what this module exists to prevent.

It is now synced from `CLUSTER STATUS` as well (so the `FRAME` an operator reads is truthful) and
from the relay's own command handler, which is the path a client actually takes.

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

**`cli.py cluster`, 5/5 on both mixers, 2026-09-14 — two real nodes, on one machine.** The
multi-node half is no longer unverified.

| check | what it holds | measured |
| :--- | :--- | :--- |
| both nodes start in their configured mode | `CLUSTER STATUS` says `DISABLED` when no config was parsed, so this also proves the `<cluster>` block is read | master/client |
| the client's PTP clock LOCKS | a node that never exchanged a Sync/Delay_Req pair stays `initializing` | **locked in 1 s, offset 70 µs** |
| the relay CONNECTS | UDP multicast for the clock and TCP for commands are different transports, so PTP can be green while commands have nowhere to go | `MEMBER: 127.0.0.1:N connected` |
| **the two nodes AGREE on the frame number** | the feature's whole promise; everything else is plumbing in service of it | **0 frames** over 7 samples |
| and the clock RUNS at the channel's rate | two stopped clocks agree perfectly, so "they agree" is satisfiable by a feature that does nothing | **120 frames in 2 s** at 59.94 |
| **a command SCHEDULED on the master is executed BY THE CLIENT** | the oracle is a PICTURE on the other process — a relay that accepted the command, rewrote the channel and dropped it satisfies every check above, `MEMBER: connected` included | client renders `(32, 192, 64)`, the scheduled colour |
| **both nodes EXECUTE on the same frame, and on the one they were given** | the reason the module exists rather than an operator sending two `PLAY`s. Both halves asserted: the two ACTUALs within a frame of each other, **and** each ACTUAL within two frames of its TARGET — two nodes that both fire 120 frames early agree perfectly | **target and actual identical on both**, frame 1329784901 |
| **and it SURVIVES a 4K decode on both nodes** | every check above runs a colour producer at 1080p — the cheapest thing this server renders. A cluster is deployed where the load is | at **2160p50** with a 4K clip looping on each node: target and actual identical on both, while the channels dropped **3 and 6 late frames** |

**WHY ONE BOX IS ENOUGH.** `create_udp_socket` sets `SO_REUSEADDR`, so two processes can both
bind the PTP ports and join the multicast group; `relay-port` is per-member configurable. **WSL is
not the route**: there is no Linux build of this server, and WSL2 sits behind a NAT'd virtual
switch where host↔guest UDP multicast does not work — which is exactly the PTP half.

**The battery runs at 59.94 deliberately**, for the same reason the boot self-test covers
1001-denominator rates: at 25p and 50p the arithmetic defect below is invisible.

The last check exercises the virtual channel map too: virtual channel 2 is mapped to the client,
so `is_local(2)` is false, `route_command` sends it over the relay, and the map rewrites `2-` to
the client's physical `1-`. Mutation-verified — with `is_local` forced true the client renders
black and **only that check fails**, `MEMBER: connected` still green.

**That last check needed an observable that did not exist.** The dispatch loop logged only when a
command was **late by more than a frame**, so a *correct* execution emitted nothing and the frame
it fired on could not be asked for from outside the process at all. `CLUSTER STATUS` now reports
`EXECUTED`, `LAST-EXEC-TARGET` and `LAST-EXEC-ACTUAL`, each node reporting its own — so the
comparison is between two self-reported frame numbers and the harness's polling latency cannot
enter it. *Before recording something as untested, check whether it is observable.*

Mutation-verified, and the mutation is instructive: with the dispatch loop's `target_frame >
now_frame` guard removed, both nodes fire **120 frames early and in perfect agreement with each
other** — the "they agree" half passes and only the "on the right frame" half catches it.

**The loaded arm is the one that answers "does this work in a venue".** The light arm proves the
protocol; it runs the lightest thing the server can render, on an idle box, so a green light arm
says nothing about a busy one — and the scheduler's dispatch loop is a thread competing with
decodes and mixers for the same cores. The loaded arm puts a real 4K decode on both nodes of one
box hosting both servers, which is harsher than the two-machine deployment it stands in for, and
the late-frame counts are reported so a reader can see the load was real rather than merely
configured.

**Still measured by nothing:** the content-sync watchdog's divergence report, and any partition or
node-loss behaviour.

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

### 4.1 What the audit and the first battery run found

**The frame clock ran at 50 fps on a 59.94 channel.** Found by the battery's own logged number on
its first run — and *not* by its check, whose first gate was ±50% and admitted 100 frames where
120 were due. A tolerance wide enough to accept a different standard frame rate is not a
tolerance. The gate is ±8% now, which is comfortably tighter than the 17% gap between 50 and
59.94 and comfortably looser than the sampling jitter. Mutation-verified: with the sync removed
the check reads 100 against 120 and fails.



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
4. ~~**Multi-node behaviour is entirely unverified.**~~ **CLOSED 2026-09-14** — `cli.py cluster`
   runs two nodes on one box, §4. What remains unverified is narrower and named there: a
   scheduled command executing on both nodes at the same frame, the virtual channel map, and the
   watchdog's divergence report. The original note read — and
   it is verifiable on ONE machine, which is the useful half of this line. `create_udp_socket`
   sets `SO_REUSEADDR`, so two server processes on one host can both bind the PTP ports and
   join the multicast group, and `relay-port` is per-member configurable. **WSL is not the
   route**: there is no Linux build of this server, and WSL2 is NAT'd behind a virtual switch
   so host↔WSL2 UDP multicast does not work — which is exactly the PTP half.

5. ~~**`relay-port` defaults to 5250, which is also the AMCP port.**~~ **TESTED AND FIXED
   2026-09-14, and it was worse than the guess.** Pointed at the AMCP port, the relay's bind
   **succeeded**: `SO_REUSEADDR` on Windows permits binding a port another live socket is
   already listening on. The log read *"Client relay listening on port 5290"*, and the relay
   then accepted the harness's AMCP connections and logged each one as **"Master connected to
   client relay"**. AMCP was dead on that node — connect succeeds, every command times out —
   and nothing anywhere reported an error.

   Three fixes: the listener uses **`SO_EXCLUSIVEADDRUSE`** on Windows, which is what
   `SO_REUSEADDR` means on POSIX here (fail rather than share), so the collision is now a
   `[fatal] Failed to setup AMCP controller on port N. It is likely already in use`; the
   *"listening"* line moved to after the bind actually succeeds, having previously been logged
   from the calling thread before the listener attempted anything; and **the default is 5252**,
   so a stock config does not collide at all.

   `casparcg.config`'s reference block also showed `<member>…:5250</member>` and
   `<relay-port>5250</relay-port>`, which taught the collision to anyone who copied it.

---

## 6. Related commits

Not traced; the module predates this document.

---

## 7. Diagrams

Deferred rather than owed. A timing diagram would help, but it would describe intended behaviour
that nothing verifies — and per this folder's rule a diagram is a claim like any other. Worth
drawing *after* §4's two checks exist, so it illustrates something measured.
