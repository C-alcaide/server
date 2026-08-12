# Four of seven ACEScg gamut matrices are wrong — found 2026-08-12

**`k_to_working` and `k_to_output` — the matrices that convert into and out of the ACEScg
working space — are not the matrices they claim to be for `bt2020`, `p3_d65`, `arri_wg3`
and `sgamut3cine`.** Identical numbers on both mixers, so both are affected equally.

Found while designing the working-space composite (`OCIO_HANDOFF_2026-08-11.md` §2 step 2),
by checking the tables the design was going to build on.

| gamut | `k_to_working` | `k_to_output` | |
| :--- | ---: | ---: | :--- |
| bt709 | 0.0002 | 0.0000 | OK |
| **bt2020** | **0.1798** | **0.2492** | **wrong** |
| **p3_d65** | **0.0867** | **0.1552** | **wrong** |
| ap0 | 0.0000 | 0.0000 | OK |
| ap1 | 0.0000 | 0.0000 | OK (identity) |
| **arri_wg3** | **0.3300** | **0.4127** | **wrong** |
| **sgamut3cine** | **0.4066** | **0.3947** | **wrong** |

Max absolute deviation per element against OCIO 2.5.2 through the pinned studio config —
the same library the server links. `ogl/image/image_kernel.cpp:643` and `:659`; the Vulkan
kernel carries the same values.

## Three independent lines of evidence, and each was needed

**1. A colorimetric derivation from primaries** (Bradford adaptation, D60→D65) reproduces
the `bt709` row to **1.5e-5** — which is what validates the method — and matches none of
the four.

**2. A round trip needing no external reference at all.** `k_to_output[i] @ k_to_working[i]`
must be the identity: a source on a channel of its own gamut converts to ACEScg and
straight back.

| gamut | `‖RT − I‖∞` |
| :--- | ---: |
| bt709, ap0, ap1 | ≤ 0.0003 |
| **bt2020** | **0.0443** |
| **p3_d65** | **0.0051** |
| arri_wg3, sgamut3cine | **0.000000** |

The last row is the important one: **arri_wg3 and sgamut3cine round-trip perfectly while
both being wrong**, because they are consistent inverses *of each other*. An internal
check alone could never have found them, which is why the third line exists.

**3. OCIO's own matrices.** For `bt2020`, OCIO gives `1.0258, −0.0201, −0.0058, …` — close
to identity, exactly as the primaries predict, since AP1's primaries were chosen near
BT.2020's. The kernel has `1.2747, −0.2692, −0.0054, …`, which is not close to identity and
is not any AP1↔X matrix.

**How two of them got there is visible in the digits.** `k_to_working[5]`
("arri wide gamut 3 → ACEScg") begins `0.6954522`; so does `k_to_output[3]`
("ACEScg → aces_ap0"). `k_to_output[5]` begins `1.4516608` against `k_to_working[3]`'s
`1.4514393`. The ARRI rows look like perturbed copies of the AP0 rows, and the S-Gamut rows
follow the same shape.

## What is affected

Anything routed through the ACEScg working space, which is **not** the default path but is
every path this fork exists for:

* **`MIXER OCIO <space>` on a BT.2020 or P3-D65 channel.** The OCIO branch ends with
  `working_to_output = k_to_output[gamut_index(params.target_color_space)]`
  (`image_kernel.cpp:750`, Vulkan `:1640`), so an HDR channel gets the wrong output matrix
  on every OCIO layer.
* **`MIXER COLORSPACE LOGC3 ARRI_WG3 …`** — the first usage example in `COLOR_GRADING.md`,
  and `k_to_working[5]`.
* **`MIXER COLORSPACE SLOG3 SGAMUT3_CINE …`** — `k_to_working[6]`.
* Any conversion with a **tone-map operator set**, which routes through ACEScg regardless
  of gamut (`if (cg.tone_mapping == 0 && ig <= 1 && og <= 1)` takes the direct route only
  when it is off).

Not affected: the default `MIXER COLORSPACE` path between BT.709 and BT.2020 with no tone
mapping, which uses `k_direct_cg`; and `auto-color-convert`, which uses `k_direct`. Those
tables were not audited here and should be.

## A second finding: no battery could have caught this

`color_math._K_TO_WORKING_AP1` / `_K_TO_OUTPUT_AP1` and `ocio_reference._K_TO_OUTPUT` in
the harness are **transcriptions of these same tables**, carrying the same numbers. So:

* `conformance --tone-maps <op>` compares the shader against a copy of the shader's own
  matrix and passes.
* `cli.py ocio` on a BT.2020 channel would do the same.

`ocio_reference`'s docstring says the comparison is *"OCIO-on-CPU against OCIO-on-our-GPU,
so a bug inside OCIO cancels and every bug in our integration does not"*. The output gamut
matrix is the exception it did not name: it is ours, transcribed identically on both sides,
so a bug in it cancels too.

`CasparCG-TestRunner/core/gamut_reference.py` closes that hole by deriving the matrices from
OCIO instead of from us, and `tests/test_gamut_matrices.py` pins the state — the four wrong
gamuts are `xfail(strict=True)`, so they **fail loudly the day the server is fixed** and the
transcriptions are not updated with it.

## The fix, and why it is not applied here

The correct rows, from OCIO 2.5.2 and the pinned config:

```
// bt2020 -> ACEScg
{0.9748950f, 0.0195991f, 0.0055059f, 0.0021796f, 0.9955355f, 0.0022850f, 0.0047972f, 0.0245320f, 0.9706708f},
// ACEScg -> bt2020
{1.0258248f, -0.0200532f, -0.0057716f, -0.0022344f, 1.0045865f, -0.0023521f, -0.0050134f, -0.0252901f, 1.0303035f},
// p3_d65 -> ACEScg
{0.7357979f, 0.2121665f, 0.0520356f, 0.0471799f, 0.9380457f, 0.0147744f, 0.0035637f, 0.0411419f, 0.9552944f},
// ACEScg -> p3_d65
{1.3792142f, -0.3088641f, -0.0703500f, -0.0693349f, 1.0822967f, -0.0129619f, -0.0021590f, -0.0454593f, 1.0476184f},
// arri_wg3 -> ACEScg
{0.9666334f, 0.1155416f, -0.0821751f, 0.0481904f, 1.1849383f, -0.2331287f, 0.0071933f, -0.0665937f, 1.0594004f},
// ACEScg -> arri_wg3
{1.0389643f, -0.0979906f, 0.0590263f, -0.0441881f, 0.8586612f, 0.1855270f, -0.0098322f, 0.0546406f, 0.9551915f},
// sgamut3cine -> ACEScg
{0.9345170f, 0.1436417f, -0.0781587f, -0.0505267f, 1.2616092f, -0.2110825f, -0.0245030f, -0.0306710f, 1.0551741f},
// ACEScg -> sgamut3cine
{1.0650269f, -0.1199250f, 0.0548981f, 0.0470203f, 0.7912176f, 0.1617622f, 0.0260986f, 0.0202137f, 0.9536877f},
```

Applying them **changes rendered output for every existing configuration that uses these
paths** — every ARRI or Sony log source through `MIXER COLORSPACE`, every BT.2020 or P3
channel under `MIXER OCIO`, every tone-mapped conversion. On saturated content the BT.2020
output matrix alone moves a pixel by roughly a quarter of full scale.

That is a correction rather than a feature, so it does not belong behind a flag the way
`<straight-alpha-grading>` does — but it is a large visible change to approved looks, and
which release it lands in is a decision rather than a detail. It also wants three things
doing together:

1. the four rows in **both** kernels;
2. the harness transcriptions, and the `xfail(strict)` marks removed;
3. `k_direct_cg` and `k_direct` audited the same way — this audit did not cover them, and
   the same copy-paste hazard applies.

Recommended before the fix lands: extend `cli.py ocio` to sweep the **channel** gamut. It
currently uses a BT.709 channel only, which is one of the three rows that happens to be
correct — so the battery that exists to prove the OCIO integration has been reporting 6/6
on the single gamut where the output matrix is right.
