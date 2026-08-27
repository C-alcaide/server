# CasparVP documentation

Sorted by **what a document is for**, because the same 60 files previously sat in one directory in
four different genres and a reader could not tell which they had opened. A plan and a guide read
alike and mean opposite things: one describes what was intended, the other what an operator should
do today.

| folder | what is in it | trust it for |
| :--- | :--- | :--- |
| **[features/](features/)** | one reference per fork feature: implementation state, the decisions behind it, and **the measurements** | **current state** - every claim carries a file, a commit, or a battery and its numbers |
| **[guides/](guides/)** | operator-facing how-to: procedure, full command syntax, worked examples | doing the thing |
| **[architecture/](architecture/)** | developer implementation notes, and the studies the CODE cites for specific numbers | why the implementation is shaped this way |
| **[plans/](plans/)** | proposals for work **not yet done**, and studies of approaches not taken | intent, never current state |
| **[audits/](audits/)** | incidents, crash reports, dumps, defect post-mortems, upstream syncs, handoffs | what happened once, on a date |
| **[deprecated/](deprecated/)** | superseded, kept for provenance | nothing - see that folder's own README |
| **[upstream/](upstream/)**, **[swsprobe/](swsprobe/)** | defect reports written for **other projects** (FFmpeg), with their reproduction scripts | filing a bug we found, or checking whether we already did |
| **[diagrams/](diagrams/)** | generator scripts | regenerating an image |
| **[images/](images/)** | committed PNGs | embedding in a document |

## features/ or guides/ — the line, and why it needs one

Both describe the same features, so the split has to be by QUESTION, not by subject:

| | features/ | guides/ |
| :--- | :--- | :--- |
| answers | *is it real, and how well does it work?* | *how do I do it?* |
| owns | state, decisions, **the numbers** | procedure, full syntax, worked examples |
| command examples | the minimum to identify the command | as many as the task needs |
| measurements | **here, once** | **cite, do not restate** |

**A number lives in exactly one place.** The original wording gave `features/` "how to drive it",
which guarantees overlap on the how-to axis, and said nothing about who owns measurements — so
guides grew Verification sections and the same figures ended up in two files. On 2026-08-27 the
seven `MIXER GRADE_NODE` figures were written into both in the same sitting.

That is the failure this tree keeps paying for: a number in two places goes stale in one, and four
such claims were found in a single sitting on 2026-08-14. A guide may state a **tolerance** an
operator acts on — "gated at 1 LSB", "expect ±1 channel" — because that changes what they do. The
measured figures behind it belong in `features/`, with a link.

**Known and not yet reconciled:** `COLOR_GRADING.md`, `HDR_GUIDE.md`,
`PIPELINE_EFFICIENCY_GUIDE.md`, `PLAYBACK_AND_RECORDING_GUIDE.md` and `LED_CALIBRATION.md` carry
Verification sections predating this rule. They are not wrong, they are duplication waiting to
rot; reconcile when next editing one, rather than in a sweep nobody reviews.

## The one rule worth reading first

`CLAUDE.md` says documents in this tree are **claims to verify against the source, not ground
truth**, and where a doc and the code disagree the code wins and the disagreement is a finding.
That is not caution for its own sake: on 2026-08-26 a single audit found four claims that had
outlived their code, including a VRAM figure understating every path by 2.5x and a picture
measurement four days stale.

`features/` is the folder that takes that seriously by construction - it is derived from code and
commits rather than from the other folders, and each claim carries its evidence. Everywhere else,
check the date and check the source.

## Reading it as one document

```
python docs/build_html.py      ->  docs/features.html
```

A single self-contained page: every `features/` document in reading order, sidebar navigation, a
light/dark toggle, no external assets and no server. Open it from the filesystem.

**It is a build output and is not committed** (`.gitignore`). The markdown is the source of truth
because it diffs, it reviews in a pull request, and it changes in the same commit as the code it
describes — the rule that stops documentation lagging. A committed HTML copy would be a second
copy of every claim, which is the duplication this folder exists to remove.

The build **warns loudly** if a document exists in `features/` but is missing from its reading
order, because a new document silently absent from the manual is exactly the quiet omission the
whole structure is meant to prevent.

## Where to start

* **"What does this fork add?"** -> [features/README.md](features/README.md) - the inventory:
  19 fork-only modules, 58 fork-specific AMCP commands, each with its state and coverage.
* **"How do I drive X?"** -> `guides/`, then the feature document for what is actually measured.
* **"Why is it built this way?"** -> `architecture/`, then the feature document's decisions section.
* **"Is this planned or done?"** -> if it is only in `plans/`, it is not done.
