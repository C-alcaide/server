# Plans and studies - intent, not state

Proposals for work **not yet done**, and studies of approaches considered. Some were partly
implemented and some were rejected; **none of them describes the current build**.

**Do not read a plan as a description of behaviour.** This is the most common way this tree has
misled a reader: a 59 KB integration plan reads with the same authority as a guide, and the gap
between "planned" and "shipped" is invisible from inside the document.

If a plan's feature has since shipped, the plan belongs in `../deprecated/` and the current state
belongs in `../features/`. Four plans were moved that way on 2026-08-26, after checking that
nothing - no document and no source comment - still cited them.

Also here: `FKL_INTEGRATION_ANALYSIS.md`, reviewed again on 2026-08-26 and still a *no* - with the
one item it recommended doing WITHOUT the library now done. Its conclusion held for reasons that
have changed: the growth since has been in Vulkan, which that library has no backend for.
