# Keyframes — REMOVED

> **State:** **removed.** The `KEYFRAMES` command family and `src/modules/keyframes` were
> deleted, and the feature they provided is [`timeline.md`](timeline.md).
> **Commands:** none. `KEYFRAMES SET|ARM|DISARM|CLEAR|GET|PATCH|SEEK|STATUS` answer
> `400` with the line echoed back, which is AMCP's reply for a command it does not know.
> **Modules:** **not a module** — nothing. The interpolation ENGINE survived and is
> `src/core/timeline/curve.cpp`
> **Coverage:** none, deliberately: there is nothing left to cover. The temporary battery that
> measured the engine swap was deleted with the command it drove; its last green run was 8/8 on
> both mixers

## What replaced it, and what to do instead

`KEYFRAMES` animated mixer state from a keyframe list. [`timeline.md`](timeline.md) does the
same thing and more, and the mapping is direct:

| `KEYFRAMES` | the timeline |
| :--- | :--- |
| `KEYFRAMES 1-10 SET ({...})` | `PUT /v1/timeline/{name}` — a document is JSON, and the JSON interface takes it |
| `KEYFRAMES 1-10 ARM` | `TIMELINE 1 PLAY {name}` |
| `KEYFRAMES 1-10 DISARM` | `TIMELINE 1 PAUSE {name}` — which holds the parameters, or `STOP` to give them back |
| `KEYFRAMES 1-10 SEEK 12.5` | `TIMELINE 1 SEEK {name} 12.5`, and it is observable now |
| `KEYFRAMES 1-10 GET` | `GET /v1/timeline/{name}` |
| `KEYFRAMES 1-10 STATUS` | `TIMELINE 1 INFO {name}` |
| `KEYFRAMES 1-10 PATCH 2.0 ({...})` | re-`PUT` the document; a `PUT` is cheap and a partial edit is the client's job |
| the 193 frozen names, in degrees | **address-space paths in registry units** — `opacity`, `fill_translation.0`, `volume`, `producer/brightness`, `previz/screen/wall/position.0` |

Five things the replacement does that this could not, each of which is why it was replaced
rather than extended:

* **The clock is the channel's**, not the animated layer's producer. A colour fill, an empty
  layer and a paused clip all animate; `SEEK` is observable.
* **A layer is not a limit.** A document animates a producer parameter, a previz screen, a
  camera and audio volume, not only `image_transform`.
* **Releasing a parameter is lossless.** The operator's own `MIXER` value is remembered
  during an animation and comes back when the document lets go.
* **Time is exact.** `flicks` rather than `double` seconds, so a key on frame 12 of a 59.94
  channel is on frame 12 forever.
* **Objects relate to each other.** `#interview.end + 5` is written down, so moving the
  interview moves what follows it.

The `kf` names survive in the field registry's descriptors, published as `keyframe_names`, so
a client that stored them can still look up which path a name refers to.

---

*This file is a redirect. It is kept rather than deleted because a doc that vanishes leaves a
reader with a dead link and no explanation, and `KEYFRAMES` was documented for long enough that
links to it exist.*
