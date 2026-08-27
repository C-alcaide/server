# ISF and OpenFX — third-party effect and generator plugins

> **State:** shipped, unmeasured
> **Modules:** `src/modules/isf`, `src/modules/ofx`
> **Commands:** none of their own — producers named `isf` and `ofx`
> **Guide:** [`../guides/ISF_USER_AND_SHADER_GUIDE.md`](../guides/ISF_USER_AND_SHADER_GUIDE.md), [`../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md`](../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md)
> **Coverage:** **none**

Two plugin hosts. **ISF** runs Interactive Shader Format shaders as producers or effects, so a
GLSL shader from the ISF ecosystem becomes a layer. **OpenFX** hosts OFX plugins — the standard
that Resolve, Nuke and Vegas use — so a commercial effect can run inside the mixer.

Detail is in [`../guides/ISF_USER_AND_SHADER_GUIDE.md`](../guides/ISF_USER_AND_SHADER_GUIDE.md),
[`../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md`](../guides/OPENFX_USER_AND_PLUGIN_GUIDE.md) and
[`../architecture/OPENFX_IMPLEMENTATION.md`](../architecture/OPENFX_IMPLEMENTATION.md). This
document is the state and the coverage.

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
