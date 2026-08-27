# Spout — sharing video with other Windows applications

Spout shares GPU textures between applications on the same machine, with no encode, no file and no
network. CasparVP is **both ends**: it can publish a channel as a Spout sender, and it can play
another application's Spout sender as a producer.

Typical use is a media server feeding Notch, Resolume, TouchDesigner, Unreal or a VJ tool — and the
reverse, pulling a generative scene from one of those into a CasparVP layer for compositing,
grading and projection.

Windows only.

---

## 1. Publish a channel (consumer)

```
ADD 1 SPOUT
ADD 1 SPOUT MyChannel
ADD 1 SPOUT MyChannel MAX_WIDTH 1920 MAX_HEIGHT 1080
```

| parameter | meaning |
| :--- | :--- |
| *(name)* | the **sender name** other applications will see. Optional — omit it and the server picks one |
| `MAX_WIDTH` | clamp the shared texture's width |
| `MAX_HEIGHT` | clamp the shared texture's height |

Remove it like any consumer:

```
REMOVE 1 SPOUT
```

**Name the sender explicitly in any real setup.** The receiving application picks senders by name,
so an unnamed sender means whoever is configuring the other end has to discover what it was called —
and the name is the only thing tying the two applications together.

**`MAX_WIDTH` / `MAX_HEIGHT` clamp, they do not scale to fit.** Use them when the receiver cannot
cope with the channel's full raster; leave them out otherwise, since a clamp is a downgrade of what
you are sending.

---

## 2. Receive another application's sender (producer)

Three spellings, all equivalent — use whichever your automation reads best:

```
PLAY 1-1 [SPOUT] SenderName
PLAY 1-1 spout://SenderName
PLAY 1-1 SPOUT SenderName
```

The `spout://` form is convenient in playlists and rundowns where a single string has to carry the
whole source.

Once playing it is an ordinary layer: grade it, transform it, key it, route it into projection.

```
PLAY 1-1 spout://NotchOutput
MIXER 1-1 OPACITY 0.5
MIXER 1-1 FILL 0.25 0.25 0.5 0.5
```

---

## 3. Worked example — round trip through Notch

Send the programme channel out, bring the treated result back on a second layer:

```
ADD 1 SPOUT ProgrammeOut
PLAY 1-2 spout://NotchReturn
```

**Do not point a receiver at the sender on the same channel.** The channel would be reading its own
output while producing it — a feedback loop, and the shape of it depends on frame timing rather
than on anything you configured. Use a second channel if you need a loop deliberately.

---

## 4. Things worth knowing before a show

**No configuration elements.** Everything is per-command; there is nothing in `casparcg.config` to
set up, and nothing to restart for.

**Both applications must be on the same GPU.** Spout shares a texture through a D3D11 shared
handle, and such a handle is adapter-bound. On a two-GPU machine, a sender on one adapter and a
receiver on the other will fail rather than run slowly — check which adapter each application is
using before assuming the name is wrong.

**The sender exists only while the consumer is attached.** `REMOVE 1 SPOUT` makes it disappear from
every receiver's list immediately, which looks to them like the source vanished.

**A missing sender is not an error on this side.** Playing `spout://Name` when nothing is publishing
that name gives you a layer with no picture rather than a refusal, because the sender may legitimately
appear later. If a receiving layer is black, check the *other* application is running and publishing
under exactly that name — spelling and case included.

---

## 5. What is not covered

**No battery drives Spout**, in either direction. Nothing measures the picture through a send or a
receive, so the round trip is documented and unmeasured. State and any figures live in
[`../features/spout.md`](../features/spout.md).

**Alpha behaviour through the share is not established here.** Spout can carry an alpha channel and
CasparVP layers are premultiplied; which convention the shared texture ends up in has not been
measured, so it is not written down as advice.
