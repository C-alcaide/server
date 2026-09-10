# KEYFRAMES — REMOVED

The `KEYFRAMES` command family was deleted. Every verb answers `400` with the line echoed back,
which is AMCP's reply for a command it does not know.

**What to read instead:** [`OPERATIONS_GUIDE.md`](OPERATIONS_GUIDE.md)'s `TIMELINE` section for
the operator's view, and [`../features/timeline.md`](../features/timeline.md) for the model, the
expression grammar, the ownership rules and the measurements.

**The command-by-command mapping** is in
[`../features/keyframes.md`](../features/keyframes.md), along with the five things the
replacement does that this could not.

The one thing worth repeating here, because it changes how a show is built: a timeline document
arrives over the control API (`PUT /v1/timeline/{name}`) rather than over AMCP. AMCP drives what
is loaded. A document is JSON, and the JSON interface is the one that takes JSON — AMCP's
tokenizer splits on spaces, which is why `KEYFRAMES SET` had to wrap its document in parentheses.

---

*This file is a redirect. It is kept rather than deleted because a guide that vanishes leaves a
reader with a dead link and no explanation.*
