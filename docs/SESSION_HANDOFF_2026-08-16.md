# Session handoff — 2026-08-16

**What this file is:** the resume point for the work done on 2026-08-14 → 16, on branch
`feature/ocio-mixer`. Everything below is pushed.

**What it is not:** the OCIO measurement record. That is
[`OCIO_HANDOFF_2026-08-11.md`](OCIO_HANDOFF_2026-08-11.md), which holds the numbers and the
"what is NOT verified" ledger for the whole OCIO arc. This file does not repeat it — two
handoffs covering one thing is how the harness repo ended up with four same-dated files and
no way to tell which was current.

---

## Shipped, with the measurement that established it

Both mixers unless stated. Every one has a battery in `CasparCG-TestRunner`.

| | what | measured |
| :--- | :--- | :--- |
| `<ocio-config>` | the OCIO config is selectable; `load_config` had **no caller**, so the pin was absolute | config loads, refuses at startup on a bad path |
| `MIXER OCIO` 3D LUT | the 3D-LUT branch of both uploaders had never executed | `ocio-lut3d` 6/6 within 1 LSB, worst **0.71** |
| **`ycbcr_code_scale` fix** | the uniform was written **69 lines above `shader_->use()`**, so it landed in whatever program the previous draw left bound | conformance 100/100, `flat-decoded` 29/29, 0 GL errors |
| GL diagnostics | `SMFL_GLCheckError` discarded the failing expression; `shader::impl::set` now names the uniform and what the bound program declares there | localised the above on the first run |
| `CALIBRATION` | nothing had ever driven it — OpenVPCal's landing point | `calibration` 32/32 at 1 LSB; channel-master **identified** vs per-layer at 17.9 LSB apart; screen consumer 0.32 LSB |
| `OCIO_LOOK` | the channel's LMT, composed into the display processor | neutral **0.00**, saturated blue **37.0**, red **16.0** |
| `MIXER CDL_FILE` | ASC CDL from `.cdl`/`.ccc`/`.cc` | **0.00** against `MIXER CDL` with the same numbers |
| `AMF` | configure a channel from an ACES Metadata File | **0.00** against the three commands by hand; one-node variant **68 LSB** away |
| gamut-compress finding | the built-in operator shares ACES 1.3's **limits, not its algorithm** | mean 0.030 / max 0.350 vs OCIO's reference look |

Plus: the port-claim sweep (nine batteries were driving whichever server answered on 5250),
and an OCIO panel in `casparcg-360-client`, which had **no OCIO controls at all**.

---

## Open, in the order I would take them

> **Another session is also working in these repos**, and has its own record at
> `CasparCG-TestRunner/docs/handoff_2026-08-15.md`. Its HDR work is **SDI metadata
> signalling** — ST 352 / ST 2108-1 ancillary data, CasparVP `03e73ba62` — which is a
> different concern from item 1 below. Read both; neither supersedes the other.

1. ~~**`HDR_GUIDE.md` does not mention OCIO.**~~ **Done 2026-08-16.** New section *Two routes
   to an HDR channel (built-in vs OCIO)*, with the signal → channel → consumer diagram. The
   load-bearing finding: a display transform switches the built-in output half off entirely
   (`<auto-tone-map>` included), while consumer HDR signalling is still derived from
   `<color-depth>`/`<color-space>`/`<color-transfer>` and **nothing cross-checks the two** —
   so a PQ picture can be signalled SDR, silently. Pairing table added; not measured
   end-to-end, and the section says so.
2. ~~**`OPERATIONS_GUIDE.md` does not mention OCIO.**~~ **Done 2026-08-16.** New *Colour
   management (OCIO)* subsection in §5, plus an OCIO row in the §17 command table and five
   troubleshooting rows. It documents `ocio_panel.py`, which is still only on the client's
   local `ui-restyle` branch (see *Where things are*), and says the panel has not been run
   against a live server.
3. ~~**Diagrams where they would carry load, not decorate.**~~ **Done 2026-08-16.** Mermaid in
   the two developer docs, rendered PNGs in the operator manual, per the rule added the same
   day. Drawing them is what found the defects: `GPU_INTEROP_ARCHITECTURE.md` claimed "zero
   CPU sync points" with one at `cuda_peer_transfer.cpp:426` that cannot be removed (GL
   cannot wait on a CUDA stream), and described two device buffers where there are three
   plus a refcounted pool. `GPU_INTEROP_PLAN.md`'s mechanism table is the pre-work state, so
   the topology is now drawn as it stands — the remaining gap is one node (HTML/CEF), not one
   bridge. `PROJECTION_CALIBRATION.md` gets the closed loop and the phase→command map.
4. ~~**Not measured: a `MIXER COLORSPACE` layer carrying a tone map under an `OCIO_DISPLAY`
   channel.**~~ **Measured 2026-08-16, both mixers byte-identical.** `cli.py ws-tonemap`,
   three arms: a plain channel separates **62–71 LSB** between `tonemapping NONE` and
   `ACES_RRT`, and both `<working-space-composite>` arms separate **0.00**. So the layer's
   tone map is *inert* there — not double-applied, not undefined — and the **composite alone**
   suppresses it; the display transform is not what does it. `COLOR_GRADING.md` updated from
   "Not measured / treat as undefined" to the numbers.
5. **The client's OCIO panel has never been seen running.** Driven headless and against real
   `INFO OCIO` payloads (55 spaces / 9 displays / 1 look parse correctly, names with parens
   survive, views nest), but nobody has opened the app.
6. ~~`VIRTUAL_PRODUCTION_FEATURES.md` probably needs only a link to `COLOR_GRADING.md`.~~
   **Done 2026-08-16** — it already had that link; what it lacked were pointers to
   `OCIO_USER_GUIDE.md` and `HDR_GUIDE.md`. Not just a link, because "the commands here are
   colour-route-neutral" turned out to be false for one of them: `SHAPE`'s `COLOR1` /
   `COLOR2` / `STROKE_COLOR` are hex values injected **unconverted** between the input and
   output conversions, so on a working-space or OCIO channel they land in scene-linear ACEScg
   and `#808080` is not mid-grey. Read from the shader and the kernel, flagged as unmeasured.
7. **`d:\_ocioprobe`, 746 MB**, redundant since OCIO builds in-tree. Needs a manual
   `Remove-Item -Recurse -Force`; the sandbox blocks it.

---

## Traps that cost real time here

The measurements are recoverable from the batteries. These are not.

**A uniform set before `use()` goes into the previously bound program.** `glUniform*` writes
into whatever is *currently bound*. A steady state re-binds the same program every frame so
the write lands by accident and nothing is ever wrong — until the program changes between
draws, which is exactly what selecting an OCIO variant does. It surfaced as
`GL_INVALID_OPERATION` and an untransformed picture, in code that had run millions of times.

**"First case only" means an unwarmed shader program — but not necessarily the one you
warmed.** The AMF battery's first case failed while the rest passed. Warming both *display*
programs fixed two thirds of it and left the last third looking like a different bug; the
input transform is a **third** program. Warm every program a case will touch.

**A separation measured against a bad capture is not a separation.** The AMF variant first
read 226 LSB apart. The base frame had been captured during a display-transform compile,
when the channel emits *no frame at all*. The honest number was 1.0 — which then failed the
control, correctly, because sRGB and Gamma 2.2 render that patch almost identically.

**Pick the discriminating axis by computing it first.** Twice this session the obvious choice
could not discriminate (gamma 2.2 vs sRGB at 1 LSB; two of four "saturated" patches unmoved
by the gamut-compression look). Both were cheap to check against the CPU oracle *before*
running a server.

**OCIO silently returns the FIRST correction** from a `.ccc` when given no id. A 200-shot
collection would load shot 1's grade, correctly applied, with nothing to say so. `load_cdl`
counts them and refuses — behaviour the server adds, pinned by a test so it stays meaningful
if OCIO ever changes.

**The API surface you happen to check is not the config.** `getDescription()` and
`getAliases()` carry no AMF transform ids, and concluding from that that none exist was
wrong: serialising the config found 134. A resolution table built with `setdefault` then
reported 2 view transforms where there are 11, hiding the very pairing the AMF output mapping
depends on.

**Several sessions run against this tree at once.** Editing a doc from a stale read put
duplicate rows in the harness command table; a build wait scripted on "no `casparcg.exe`
running" over-waited because the other session was running a *different* binary
(`CasparCG-server`, not CasparVP). Check what is actually running, and never stage a file
carrying someone else's uncommitted work.

---

## Where things are

* **Server:** `feature/ocio-mixer` on `origin` (`C-alcaide/server`), pushed. `CHANGELOG.md`
  leads with the behaviour changes.
* **Harness:** `audit-reporting-fixes` on `origin` (`C-alcaide/CasparCG-TestRunner`), pushed.
  New batteries: `ocio-lut3d`, `calibration`, `ocio-look`, `cdl-file`, `amf`. Suite ~1550.
* **Client:** `casparcg-360-client`, `ocio_panel.py` + wiring in `channel_tab.py`, committed
  on branch **`ui-restyle`**, which has **no upstream** — so that work is local only. It was
  the checked-out branch and is likely another session's; confirm before pushing it anywhere.
* **Studies, deliberately not implemented:** [`AMF_SUPPORT_STUDY.md`](AMF_SUPPORT_STUDY.md)
  (built afterwards — kept for the rationale) and
  [`LED_PROCESSOR_CONTROL_STUDY.md`](LED_PROCESSOR_CONTROL_STUDY.md), which is a file of
  questions because no Brompton or Megapixel hardware is on this machine and a vendor adapter
  written against a PDF is the failure this tree's first page warns about.
