# Plans and studies - intent, not state

Proposals for work **not yet done**, and studies of approaches considered. Some were partly
implemented and some were rejected; **none of them describes the current build**.

**Do not read a plan as a description of behaviour.** This is the most common way this tree has
misled a reader: a 59 KB integration plan reads with the same authority as a guide, and the gap
between "planned" and "shipped" is invisible from inside the document.

If a plan's feature has since shipped, the plan belongs in `../deprecated/` and the current state
belongs in `../features/`. Four plans were moved that way on 2026-08-26, after checking that
nothing - no document and no source comment - still cited them.

**That check is what keeps three shipped plans here.** `AMF_SUPPORT_STUDY.md`,
`GPU_INTEROP_PLAN.md` and `GPU_OPTIMIZATION_PLAN.md` are cited **from source comments** -
`AMCPCommandsImpl.cpp:3403`, `d3d11_import_bridge.h:50`, `gl_export_bridge.h:34`,
`av_producer.cpp:135`, `isf_producer.cpp:301` - several by item number. Moving them would break a
citation a reader follows from the code, so they stay, with a status line that says they shipped.
Moving is the default; a source citation is the exception.

---

## Every document here declares its status, in one machine-readable form

```
> **Status:** SHIPPED | PARTIAL | UNIMPLEMENTED | RESEARCH | SUPERSEDED - <ISO date> - <why>
> **Falsifier:** `identifier`, `identifier`   OR   none - <why not>
```

The **date** is what makes staleness visible. The **falsifier** names the identifiers whose
appearance in `src/` would mean the status is wrong - a module directory, a command, a symbol. For
`SHIPPED` and `PARTIAL` write `none` with a reason: absence cannot be tested for, and the sections
carry their own outcomes.

`CasparCG-TestRunner/tests/test_plan_status.py` asserts all of it, and for `UNIMPLEMENTED` and
`RESEARCH` asserts that **every declared falsifier is absent from the tree**. Verified capable of
failing: a falsifier pointed at a real identifier and a removed status line both come back red,
naming the file.

**Why it is built that way rather than by reading the prose.** The first attempt matched phrases
like *"not implemented"* against the registered command list. It was abandoned - plans legitimately
cite `PLAY`, `INFO` and `CLS` as context, and a doc whose status had just been **corrected** still
quoted the phrase it replaced, so the two files that were now right were the two that stayed red. A
check that cannot be made green by fixing the problem gets ignored, and that is worse than no check.
The doc declares the falsifier; the test checks the declaration.

### What it found

Five of thirteen plans carried a wrong status on 2026-08-27:

| plan | said | actually |
| :--- | :--- | :--- |
| `AMF_SUPPORT_STUDY.md` | researched, not implemented | shipped `180b2fb41`, **the same day the study was written** |
| `GRADING_NODE_GRAPH_STUDY.md` | nothing is implemented | `MIXER GRADE_NODE` ships on both mixers |
| `REMOTEWALL_NATIVE_MODULE_PLAN.md` | proposal | phases 0-5 built and in the build for a month |
| `GPU_AFFINITY_PLAN.md` | *no status line at all* | partly shipped; `server.cpp:515-535` |
| `GPU_INTEROP_PLAN.md` | *no status line at all* | all three items shipped, struck through with commits |

Two of them had spread: the features doc for remotewall had copied "planned and not done"
into a **features** doc that shipped, and `features/colour-grading-and-ocio.md` and
`guides/OCIO_USER_GUIDE.md` both cite the AMF study. **A stale status does not stay in `plans/`** -
it gets read as state and quoted elsewhere, which is why this is checked rather than remembered.

Also here: `FKL_INTEGRATION_ANALYSIS.md`, reviewed again on 2026-08-26 and still a *no* - with the
one item it recommended doing WITHOUT the library now done. Its conclusion held for reasons that
have changed: the growth since has been in Vulkan, which that library has no backend for.
