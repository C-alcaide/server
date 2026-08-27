# IMAGE — 16-bit capture, and the harness's own eyes

> **State:** shipped
> **Module:** `src/modules/image` — **789 lines** different from upstream across 7 files
> **Commands:** upstream's `ADD n IMAGE`, `PRINT`; no fork-specific command
> **Architecture:** none, deliberately — same -- the circularity point is state, not shape
> **Guide:** none, deliberately — Upstream owns `ADD n IMAGE` and `PRINT`; this fork adds high-bit-depth capture and byte-order handling, which is state rather than procedure. No operator guide.
> **Coverage:** indirectly enormous — this consumer is what `conformance`, `grading`, `ocio*`,
> `blend-domain`, `alpha-domain` and most 1 LSB batteries capture through

The IMAGE consumer is how nearly every number in this project is obtained. That is the reason it
has a document: a defect here does not look like a defect here.

---

## 1. What is implemented today

**High-bit-depth capture.** Upstream writes 8-bit RGBA. This fork detects a high-depth frame from
`pix_desc.planes[0].depth` and writes **PNG16 (`RGBA64`)** instead (`image_consumer.cpp:209-214`),
which is what makes a 1 LSB gate at 16 bits expressible at all.

**Byte order is per-format, not global.** A `bgra` mixer output is swizzled; a 16-bit `rgba` output
is written **packed, with no `.bgra` swizzle** (`image_consumer.cpp:276`). Both mixers reach this
code, so the rule has to be right for two different producers of the same buffer.

**A `MIXER_OUT` trace line** at `image_consumer.cpp:218-234`, logging the frame as it arrives. It
exists because everything downstream is shared by both backends, so this line is what separates
*"the mixer produced this"* from *"this consumer mangled it"*. It identified the swscale
BGRA64→RGBA64 defect written up in `../audits/UPSTREAM_SYNC_2026-08-18.md` §3.1.1 — **and without
it, that defect reads as a 16-bit precision fault in the OpenGL backend.**

Also changed: `image_loader.cpp` (EXR and DPX handling), `image_converter.{h,cpp}` ("convert to a
mixer-compatible format, preserving bit depth when possible"), `image_algorithms.h` and
`image_scroll_producer.cpp`.

---

## 2. The trap worth stating plainly

**No battery can currently distinguish a mixer fault from an IMAGE-consumer fault**, because the
batteries that would notice are the ones capturing through it. The trace line above is the only
instrument that splits them, and it is a log line rather than an assertion — so it helps a person
who already suspects the consumer, and nothing else.

This is the same circularity `screen-consumer.md` describes, and the two together cover almost every
capture route the harness has.

---

## 3. Known gaps

1. **The consumer is never the subject.** Breaking the circularity needs a second, independent
   capture route compared against it — `consumer-view`, or a DeckLink loop.
2. **EXR and DPX loading are unmeasured.** Fixtures exist for neither in a form that gates.
3. **`image_scroll_producer`** carries fork changes with no coverage and no documentation of what
   they alter.
