# Architecture - how it is built, and why

Developer notes. Includes the studies the **code itself cites** for specific numbers, which is why
they are here rather than in `plans/`: `OCIO_INTEGRATION_STUDY.md` is referenced from eight
non-documentation files at section level (`section 4.2`, `section 8.7`), so the implementation
depends on it and it is a live reference, not history.

That distinction was got wrong once during the reorganisation of 2026-08-26 - the study was filed
as deprecated before a check showed eight source comments citing it. **A document the code cites
by section is not deprecated, whatever its filename suggests.**

`HAP_DECODE_ROUTES.md` was added on 2026-08-27. Nine documents for twenty-three features is
correct — this folder exists for implementations whose *shape* needs explaining, not for every
module. HAP earned one because it has **three** decode routes chosen at construction and never
changed, each handing the mixer something different, plus a seek-epoch mechanism whose three
load-bearing properties were each learned by getting them wrong. All of that lived only in code
comments.

Per `CLAUDE.md`, documents here get **inline mermaid** rather than rendered PNGs: they diff as text
and change in the same commit as the code they describe. Rendered images are for the
operator-facing guides.
