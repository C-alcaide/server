# LED processor control — investigation, for future consideration

> **Status:** RESEARCH — written 2026-08-16, unchanged 2026-08-27 — not investigated against
> hardware, and that is the whole point of this file.
> **Falsifier:** `led_processor`

Nothing here is a measurement. Where a fact about a vendor protocol
would normally go, there is a question instead — deliberately, because this repo's standard
is that a claim carries the measurement that established it, and none of these could.

---

## What it would be for

`CALIBRATION` applies a `.cube` solved by OpenVPCal to the channel. That LUT is only valid
for **the wall in the state it was in when the plate was shot** — the processor's own gamut,
brightness, gamma and colour temperature settings are part of what was measured. Change any
of them afterwards and the calibration is silently describing a wall that no longer exists.

Today nothing in either repo can read or set that state. The operator is trusted to leave the
processor alone between the solve and the show, and there is no way to detect that they did
not. That is the gap: not "we should drive the LED wall", but **"the calibration has an
unverified precondition"**.

Two vendors dominate virtual production stages: **Brompton Tessera** and **Megapixel
Helios**. Both are network-controlled.

## Why it was not built

I have no processor of either kind on this machine, no way to send a command and observe the
result, and therefore no way to write a single line that could be tested rather than merely
compiled. Every other feature in this tree carries a number; this one would carry "it builds".
`CLAUDE.md` opens by pointing out that a dead code path is not a correct one — a whole
*module* of dead code aimed at hardware nobody here can exercise is that failure at scale.

It is also outside both repos as they stand: CasparVP talks to cards and GPUs, and the 360
client talks to CasparVP. Neither has a device-control layer to extend.

## What has to be established BEFORE any code

Each of these is a question because I could not answer it from here, and answering them from
documentation alone would produce exactly the plausible-and-wrong result this file exists to
avoid:

1. **What does the control API actually expose, and is it readable as well as writable?** A
   write-only API cannot support the check that motivates the work — "is the wall still in
   the state the calibration assumed?" needs a *read*.
2. **Is there a stable identifier for a processor's complete colour state?** If the state can
   be hashed or versioned, the check is cheap and exact. If it has to be reconstructed field
   by field, it becomes a second hand-maintained model — the thing the AMF study rejected.
3. **What are the failure modes on a live show?** A colour-management feature that can stall
   or drop a frame while talking to a processor over TCP is worse than not having it. The
   answer determines whether this can touch the frame path at all (it almost certainly must
   not).
4. **Who owns the state?** If an operator changes brightness from the processor's own panel
   mid-show, does the server want to fight that, report it, or ignore it? This is a product
   decision, not a technical one, and it should be made before anything is written.
5. **Licensing and access.** Both vendors' APIs may require an NDA, a licence, or a firmware
   tier. Worth confirming before designing around them.

## How it could be verified without the hardware

The part that does not need a processor is most of it, and it is the part worth doing first
if this is ever picked up:

* **Define the control surface** — what the server needs to read and write, expressed in its
  own terms rather than a vendor's.
* **Stand up a fake processor in the harness**: a small server speaking the protocol, whose
  responses the battery controls. That makes the client logic — connect, read state, detect a
  change, report it — fully testable, including the failure paths (timeout, refused
  connection, malformed reply, state changed underneath) that are the ones most likely to be
  wrong and the hardest to provoke on real hardware.
* Only the final adapter would then be unverified here, and it would be a thin one.

That is a real piece of work with a real gate, and it is the shape this should take. What it
should not be is a vendor adapter written against a PDF and merged because it compiles.

## The cheaper thing that captures most of the value

If the goal is "detect that the wall changed since calibration", a processor API is one way
and not the only one. **A calibration LUT already has a solve-time provenance problem** and
`CALIBRATION` currently stores none: no record of which OpenVPCal run produced the loaded
`.cube`, when, or against what. Recording that alongside the LUT — and surfacing it in
`CALIBRATION INFO` — costs nothing, needs no hardware, and turns "is this calibration still
current?" from unanswerable into a question an operator can at least reason about.

That is worth doing whether or not processor control ever happens, and it does not depend on
any of the five questions above.
