# Architecture - how it is built, and why

Developer notes. Includes the studies the **code itself cites** for specific numbers, which is why
they are here rather than in `plans/`: `OCIO_INTEGRATION_STUDY.md` is referenced from eight
non-documentation files at section level (`section 4.2`, `section 8.7`), so the implementation
depends on it and it is a live reference, not history.

That distinction was got wrong once during the reorganisation of 2026-08-26 - the study was filed
as deprecated before a check showed eight source comments citing it. **A document the code cites
by section is not deprecated, whatever its filename suggests.**

**Thirteen documents for twenty-five features, and that ratio is the design.** This folder exists
for implementations whose SHAPE needs explaining -- not one per module. Every `features/` document
now carries an `> **Architecture:**` line, and twelve of them say `none, deliberately` with the
reason: conventional plumbing, a thin SDK wrapper, or a structural point already recorded as state.
An absence that is stated can be argued with; one that is merely missing reads as an oversight.

`CLUSTER_SYNC_DESIGN.md` and `GPU_CODEC_HANDOFF.md` were added on 2026-08-27 -- the first because
three independent layers (ptp / sync / relay) mean drift, jumps and ignored commands are three
different faults the symptom cannot distinguish; the second because this fork has three GPU codec
producers that hand the mixer three DIFFERENT things, with no house style to copy for a fourth.

`HAP_DECODE_ROUTES.md` was added on 2026-08-27. Nine documents for twenty-three features is
correct — this folder exists for implementations whose *shape* needs explaining, not for every
module. HAP earned one because it has **three** decode routes chosen at construction and never
changed, each handing the mixer something different, plus a seek-epoch mechanism whose three
load-bearing properties were each learned by getting them wrong. All of that lived only in code
comments.

Per `CLAUDE.md`, documents here get **inline mermaid** rather than rendered PNGs: they diff as text
and change in the same commit as the code they describe. Rendered images are for the
operator-facing guides.
