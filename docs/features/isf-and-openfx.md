# ISF and OpenFX — third-party effect and generator plugins

> **State:** shipped, unmeasured
> **Modules:** `src/modules/isf`, `src/modules/ofx`
> **Commands:** none of their own — producers named `isf` and `ofx`
> **Architecture:** [`../architecture/OPENFX_IMPLEMENTATION.md`](../architecture/OPENFX_IMPLEMENTATION.md)
> **Guide:** [`../guides/ISF_USER_AND_SHADER_GUIDE.md`](../guides/ISF_USER_AND_SHADER_GUIDE.md), [`../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md`](../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md)
> **Coverage:** **none**

Two plugin hosts. **ISF** runs Interactive Shader Format shaders as producers or effects, so a
GLSL shader from the ISF ecosystem becomes a layer. **OpenFX** hosts OFX plugins — the standard
that Resolve, Nuke and Vegas use — so a commercial effect can run inside the mixer.

Detail is in [`../guides/ISF_USER_AND_SHADER_GUIDE.md`](../guides/ISF_USER_AND_SHADER_GUIDE.md),
[`../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md`](../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md) and
[`../architecture/OPENFX_IMPLEMENTATION.md`](../architecture/OPENFX_IMPLEMENTATION.md). This
document is the state and the coverage.


## ISF at 16 bits per component

`[ISF] <shader> BIT_DEPTH 16` renders and outputs at 16 bits. **Without it the producer now
follows the channel** — 16-bit on a 16-bit channel, 8-bit on an 8-bit one. It did not always:
the depth defaulted to 8 whatever the channel was, and the parameter was the only way to
reach 16. See *The depth follows the channel* below.

**Found by measuring something else.** An ISF ramp was `spout-depth`'s first fixture and
delivered **256 distinct levels** through a Spout sender correctly advertising
`rgba16-unorm` — 192 on the green channel, which is exactly 0.75 x 256 and named the cause.
Every output site in the ISF path was 8-bit, and one of them made the others irrelevant:

| site | what it is |
| :--- | :--- |
| `ensure_final` -> `make_buffer_tex(w, h, false)` | **the shader's FINAL PASS TARGET** |
| `create_texture(..., bit8, ...)` | the OGL output texture |
| `wrap_texture`'s `pixel_format_desc` | what the mixer believes it is sampling |
| `create_exportable_texture(..., bit8)` | the Vulkan shared texture |
| the CPU readback | the descriptor, the buffer arithmetic, and `glGetTexImage`'s type |

The final pass is the root: a 16-bit output texture fed by an 8-bit final pass receives an
already-quantised blit and delivers 256 levels, so fixing the visible sites alone changes
nothing.

**Measured after the fix**, both mixers, a 16-bit channel at 1080p2500, distinct levels
per channel in a received ramp:

| producer | `BIT_DEPTH 8` | `BIT_DEPTH 16` | **nothing said** |
| :--- | ---: | ---: | ---: |
| ISF shader | 256 | **1920** | **1920** |
| 16-bit clip (control) | 256 | **1920** | 1920 |

1920 is the ramp's width, so at 16 bits every column resolves and the fixture is the limit
rather than the depth. The ISF arm matches the file producer level for level.

**The third column is the one that needed a new arm**, added 2026-09-04 as `spout-depth
--producer isf-default`. The two `BIT_DEPTH` columns state the depth explicitly, so they
measure the override and would read 1920 whatever the default did — a check that cannot fail
for a change to the default. A fourth arm, `--producer isf-force8`, pins the producer to 8
while the share stays 16-bit and reads **256 with `published-depth` 16**: that is the only
cell which shows the cap is the *producer* and not the consumer truncating. 12/12 across the
three ISF arms, both mixers.

### The depth follows the channel, and the parameter overrides it

`frame_producer_dependencies` carries the channel's `channel_info` — the same `depth`,
`default_color_space` and `default_color_transfer` a consumer on that channel receives — so a
producer reads the depth from the dependencies it is already handed.

**Why it was a parameter first.** A producer genuinely could not see the depth: `frame_factory`
takes one on `create_frame` and exposed no accessor for the channel's own, and the dependencies
carried `format_desc` and nothing about colour. So the depth defaulted to 8, an operator had to
restate what the channel already knew, and **a 16-bit channel was 8-bit by default for every
generator** — truncating in silence, which is how this was found at all.

**The override is kept deliberately**, in both directions: `BIT_DEPTH 8` on a 16-bit channel is
a legitimate request for a receiver that cannot take 16 bits, and forcing 16 on an 8-bit channel
costs only memory. An absent or unparseable value follows the channel rather than forcing 8 — a
shader should not fail to load over a precision hint.

**What this does not do.** `channel_info` also carries the gamut and the transfer function, and
**no producer reads either yet**. The plumbing is there; only the depth is wired to it.

### Two traps in the implementation

* **`set_output_depth` must not free the final-pass texture.** It runs on the producer's
  thread, where deleting a GL texture is not safe. The depth is part of `ensure_final`'s
  cache key instead, so the rebuild happens on the GL thread — and without that key a depth
  change at the same raster would silently keep the old 8-bit buffer.
* **`BIT_DEPTH` has to be stripped before the source is resolved.** Everything after the
  shader name is a source producer in filter mode, so leaving the option in made
  `[ISF] shader BIT_DEPTH 16` try to open a producer named `BIT_DEPTH` and return
  `404 PLAY FAILED`.

**Coverage:** `cli.py spout-depth --producer isf` — 8/8 with the file producer as control.
The ISF arm's source gate is satisfied by construction rather than by parsing, because the
ISF producer logs no output format; the levels carry the verdict there. A log line naming
the output depth would make that gate real.

---
---

## 1. What is implemented today

| host | producer name | evidence |
| :--- | :--- | :--- |
| ISF | `isf` | `isf_producer.cpp:580` |
| OpenFX | `ofx` | `ofx_producer.cpp:904` |

Both allocate through `create_exportable_texture` (`isf_producer.cpp:401`,
`ofx_producer.cpp:251`), which puts them on the same GPU interop path as the CUDA producers, Spout
and remotewall — six consumers of one allocation function.

`configuration.ofx` is read for plugin discovery.

---

## 2. How to drive it

```
PLAY 1-1 "[ISF] shader_name"
PLAY 1-1 "[OFX] plugin_id"
```

Parameter syntax — how a shader's or plugin's own parameters are set — is in the two guides.

---

## 3. Design decisions, and what they cost

**Plugins run as producers, not as a filter stage in the mixer.** An effect is a layer, so it
composes through the existing layer machinery and needs no new stage in the shader. The cost is
that "apply this effect to that layer" is expressed as a layer arrangement rather than as a
property of the layer being affected.

**Both host third-party code in-process.** An ISF shader is GLSL that must compile, and an OFX
plugin is a DLL that runs in the server. A bad shader is a compile failure; a bad plugin is a
process-level risk. Neither host sandboxes, which is normal for OFX and worth knowing.

---

## 4. Verification — what is measured, and what is not

**Nothing.** No battery loads a shader or a plugin.

The reason this is more excusable than the other gaps in this folder — and still a gap — is that
the interesting behaviour belongs to third-party code. What is *ours* and testable:

1. **A minimal ISF shader that outputs a known flat colour** would verify the whole host path
   end to end — parse, compile, bind, render, hand to the mixer — in one check with no third-party
   dependency, since the shader would live in the harness's fixtures.
2. **The `create_exportable_texture` layout fix** (`0f1c5fb38`) applies to both hosts and was
   verified only on the ProRes path. Same assumption as Spout.
3. **Failure paths**: a shader that does not compile, and a plugin ID that does not exist, should
   produce a legible refusal rather than a silent black layer.

---

## 5. Known gaps

1. **No coverage.** §4.1 is a self-contained check needing only a fixture shader.
2. **The `eUndefined` layout fix is unverified on both paths.**
3. **No documented behaviour for a failed load** — neither guide states what happens when a shader
   fails to compile or a plugin is missing.
4. **OFX plugins are unsandboxed**, undocumented as a consideration.

---

## 6. Related commits

| commit | why it matters |
| :--- | :--- |
| `0f1c5fb38` | transitioned every exportable texture at creation — both hosts get their textures from that function; verified on ProRes, assumed here |

---

## 7. Diagrams

Not warranted. Two hosts, each a linear load-compile-render path. The interesting complexity is
inside the plugins, which no diagram of ours can usefully describe.
