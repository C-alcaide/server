# Upstream — defects we found in other projects

Reports written to be **filed with the project that owns the code**, not with this fork. Kept here
so the work is not redone and so a fork-side workaround can point at the reason it exists.

| report | project | state |
| :--- | :--- | :--- |
| `prores_ks_vulkan_qscale_corruption.md` | FFmpeg | written, reproduced on 8.1.1 and 8.1.2, **not yet filed** |
| `../swsprobe/REPORT_TO_FFMPEG.md` | FFmpeg | written, reproduced on 7.0.2 / 8.1.1 / 8.1.2, **not yet filed**; repro scripts beside it |

**These folders were invisible until 2026-08-27.** The docs reorganisation sorted 60 documents into
`features/`, `guides/`, `architecture/`, `plans/`, `audits/` and `deprecated/` and did not notice
either of these two, so `docs/README.md` — the map — did not mention them and nothing linked to
`REPORT_TO_FFMPEG.md` at all. Both are cited from four live documents, so the content was in use
while the folders holding it were undiscoverable.

Kept where they are rather than folded into `audits/`: an audit records what happened here, and
these are deliverables addressed to someone else. Moving them would also break the four inbound
citations for no gain.

**Do not reference the harness in these.** They go to maintainers who cannot run it — report the
numbers and the reproduction, not the private tool that produced them.
