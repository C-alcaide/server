# Architecture - how it is built, and why

Developer notes. Includes the studies the **code itself cites** for specific numbers, which is why
they are here rather than in `plans/`: `OCIO_INTEGRATION_STUDY.md` is referenced from eight
non-documentation files at section level (`section 4.2`, `section 8.7`), so the implementation
depends on it and it is a live reference, not history.

That distinction was got wrong once during the reorganisation of 2026-08-26 - the study was filed
as deprecated before a check showed eight source comments citing it. **A document the code cites
by section is not deprecated, whatever its filename suggests.**

Per `CLAUDE.md`, documents here get **inline mermaid** rather than rendered PNGs: they diff as text
and change in the same commit as the code they describe. Rendered images are for the
operator-facing guides.
