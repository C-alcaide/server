#!/usr/bin/env python
"""Diagrams for the recording guides.

Two stages added in 2026-08 are order-dependent, which is this tree's test for whether a
feature earns a picture:

  * FIELD PAIRING -- an interlaced channel ticks at field rate, and a field-coded file needs
    two ticks line-interleaved into one frame. Which tick owns which lines IS the feature, and
    it has two routes (a host memcpy and a strided GPU copy) that have to agree.
  * QSCALE AUTO -- a closed loop. Prose describes a loop one step at a time and the reader has
    to hold the feedback edge in their head; a picture puts it on the page.

Run:  python docs/diagrams/generate_recording_diagrams.py
Writes PNGs into docs/images/.

Palette, helpers and `layout_check.Layout` all follow `generate_diagrams.py` and
`generate_ocio_diagrams.py` deliberately. The first draft of this file did not use Layout and
produced both defects Layout exists to catch -- a caption written across the panel above it,
and a label wider than its box -- which is the argument for it in one attempt.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches   # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_check import Layout        # noqa: E402

# ── App palette (ui_kit.COLORS), as the other generators ─────────────────────
BG = "#1e1e1e"
PANEL = "#2d2d2d"
BORDER = "#555555"
BORDER_SUBTLE = "#444444"
TEXT = "#d4d4d4"
MUTED = "#888888"
TITLE = "#9cdcfe"
ACCENT = "#2255aa"
ACCENT_HOVER = "#2f6bd0"
SUCCESS = "#1a6b1a"
WARNING = "#7a5c00"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.2, radius=0.02, z=1):
    p = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius * 100}",
        fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def _text(ax, x, y, s, *, color=TEXT, size=10, weight="normal", ha="left",
          va="center", style="normal", z=5, family=None):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha,
                   va=va, style=style, zorder=z, family=family)


def _arrow(ax, p0, p1, *, color=MUTED, lw=1.4, z=4, style="-|>", rad=0.0):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2))


def _new(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return Layout(fig, ax, panel_fn=_panel, text_fn=_text, arrow_fn=_arrow), fig, ax


def _line_stack(ax, x, y, w, rows, own_even, tint):
    """A full-height picture drawn as lines, with the ones this tick owns tinted."""
    for i in range(rows):
        ly = y + (rows - 1 - i) * (w * 0.0)  # placeholder, set below
    step = 2.6
    for i in range(rows):
        ly = y + (rows - 1 - i) * step
        owned = (i % 2 == 0) if own_even else (i % 2 == 1)
        _panel(ax, x, ly, w, 1.9, fc=tint if owned else "#232323",
               ec=BORDER_SUBTLE, lw=0.6, radius=0.0, z=3)


# ─────────────────────────────────────────────────────────────────────────────
def field_pairing():
    lay, fig, ax = _new((12.0, 6.8))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95,
             "Interlaced recording: two channel ticks become one frame",
             parent=None, color=TITLE, size=13, weight="bold", ha="center")
    lay.text("sub", 50, 90,
             "a 1080i50 channel ticks 50× a second at FULL height — the file is 25 fps",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")

    # ── the two source ticks, stacked with room for their captions ───────────
    lay.panel("tickA", 4, 58, 21, 22, fc="#0e1118", ec=BORDER)
    lay.text("labA", 14.5, 83, "tick N  ·  field A", parent=None, color=ACCENT_HOVER,
             size=9.2, weight="bold", ha="center")
    _line_stack(ax, 5.5, 60, 18, 8, True, ACCENT)

    lay.panel("tickB", 4, 22, 21, 22, fc="#0e1118", ec=BORDER)
    lay.text("labB", 14.5, 47, "tick N+1  ·  field B", parent=None, color=SUCCESS,
             size=9.2, weight="bold", ha="center")
    _line_stack(ax, 5.5, 24, 18, 8, False, SUCCESS)

    lay.text("cap", 14.5, 17.5, "each tick is full height", parent=None, color=MUTED,
             size=8, ha="center")

    # ── the interleave ───────────────────────────────────────────────────────
    lay.panel("weave", 31, 44, 21, 22, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    lay.text(None, 41.5, 61.5, "INTERLEAVE", parent="weave", color=TITLE, size=10,
             weight="bold", ha="center")
    for i, ln in enumerate(("output line y comes", "from line y of", "whichever tick",
                            "owns that parity")):
        lay.fit_text(None, 41.5, 56.5 - i * 3.6, ln, parent="weave", size=8.2,
                     color=TEXT, ha="center")

    lay.arrow((25.5, 69), (31, 58), color=ACCENT, rad=-0.15)
    lay.arrow((25.5, 33), (31, 52), color=SUCCESS, rad=0.15)

    # ── the two routes ───────────────────────────────────────────────────────
    lay.panel("routes", 29, 20, 25, 18, fc="#181818", ec=BORDER_SUBTLE)
    lay.text(None, 41.5, 34.5, "two routes, one rule", parent="routes", color=MUTED,
             size=8.6, weight="bold", ha="center")
    lay.fit_text(None, 41.5, 30, "host: per-line memcpy", parent="routes", size=8,
                 color=TEXT, ha="center")
    lay.fit_text(None, 41.5, 26.5, "GPU: strided copyImage", parent="routes", size=8,
                 color=TEXT, ha="center")
    lay.fit_text(None, 41.5, 22.8, "they must agree", parent="routes", size=7.6,
                 color=WARNING, ha="center", style="italic")

    # ── the result ───────────────────────────────────────────────────────────
    lay.panel("out", 59, 44, 21, 22, fc="#0e1118", ec=ACCENT_HOVER, lw=1.6)
    lay.text("outlab", 69.5, 69, "one field-coded frame", parent=None, color=TITLE,
             size=9.2, weight="bold", ha="center")
    for i in range(8):
        ly = 46 + (7 - i) * 2.6
        _panel(ax, 60.5, ly, 18, 1.9, fc=ACCENT if i % 2 == 0 else SUCCESS,
               ec=BORDER_SUBTLE, lw=0.6, radius=0.0, z=3)
    lay.text("outcap", 69.5, 40.5, "25 fps · field_order tt", parent=None, color=MUTED,
             size=8, ha="center")
    lay.arrow((52, 55), (59, 55), color=ACCENT_HOVER, lw=1.8)

    # ── the flags ────────────────────────────────────────────────────────────
    lay.panel("flags", 59, 18, 37, 20, fc=PANEL, ec=BORDER)
    lay.text(None, 77.5, 34.5, "signalled, not implied", parent="flags", color=TITLE,
             size=9, weight="bold", ha="center")
    lay.fit_text(None, 77.5, 30, "encoder: AV_CODEC_FLAG_INTERLACED_DCT", parent="flags",
                 size=7.6, color=TEXT, ha="center")
    lay.fit_text(None, 77.5, 26.5, "frame: AV_FRAME_FLAG_INTERLACED + TFF", parent="flags",
                 size=7.6, color=TEXT, ha="center")
    lay.fit_text(None, 77.5, 21.5, "the consumer sets these: -flags +ildct never arrives",
                 parent="flags", size=7.2, color=WARNING, ha="center")

    lay.text("opt", 50, 9, "-interlaced   auto | 0 | 1", parent=None, color=TITLE,
             size=9.5, weight="bold", ha="center")
    lay.text("optd", 50, 5,
             "auto pairs when the channel is interlaced  ·  0 writes every tick "
             "progressive  ·  1 pairs and warns if there is no second field",
             parent=None, color=MUTED, size=7.8, ha="center")

    lay.check(name="recording_field_pairing")
    _save(fig, "recording_field_pairing.png")


# ─────────────────────────────────────────────────────────────────────────────
def qscale_auto_loop():
    lay, fig, ax = _new((12.0, 5.8))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95,
             "CUDA_PRORES  QSCALE AUTO — a closed loop on the profile's data rate",
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")
    lay.text("sub", 50, 89.5,
             "a fixed quantiser has no rate control: it produces whatever the content produces",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")

    stages = [
        ("composite", 4, "COMPOSITE", ("BGRA from", "the mixer"), PANEL),
        ("encode", 24, "ENCODE", ("DCT + quantise", "at q"), ACCENT),
        ("measure", 44, "MEASURE", ("bits spent", "per macroblock"), PANEL),
        ("compare", 64, "COMPARE", ("against the", "profile target"), PANEL),
        ("step", 84, "STEP q", ("damped, toward", "the target"), SUCCESS),
    ]
    for name, x, title, body, fc in stages:
        lay.panel(name, x, 56, 15, 20, fc=fc, ec=BORDER)
        lay.text(None, x + 7.5, 71.5, title, parent=name,
                 color="#ffffff" if fc in (ACCENT, SUCCESS) else TITLE,
                 size=9, weight="bold", ha="center")
        for i, ln in enumerate(body):
            lay.fit_text(None, x + 7.5, 66 - i * 4, ln, parent=name, size=7.8,
                         color="#ffffff" if fc in (ACCENT, SUCCESS) else TEXT,
                         ha="center")
    for x in (19, 39, 59, 79):
        lay.arrow((x, 66), (x + 5, 66), color=MUTED, lw=1.5)

    # The feedback edge, which is the reason this is a picture at all -- routed BELOW the
    # stages, because a curve bowing the other way runs straight through MEASURE and COMPARE.
    #
    # LAYOUT_CHECK CANNOT SEE THAT. It tests the straight segment between an arrow's endpoints
    # (`_segment_hits_rect`), so a curved arrow whose endpoints clear every box passes while the
    # drawn path crosses two of them -- which is exactly what the first version of this diagram
    # did. The check removes the mechanical iterations; it does not remove looking at the PNG.
    # rad and the caption's y are coupled: arc3 dips about rad*span/2 below the chord, so a
    # 60-unit span at -0.18 bottoms out near y=49 and leaves the caption room at 41. At -0.30
    # it reached y=35 and the caption was drawn across it -- a text-vs-ARROW collision, which is
    # a fourth class layout_check does not cover (it checks text-in-panel, text-fit and
    # arrow-through-panel).
    lay.arrow((91.5, 54.5), (31.5, 54.5), color=SUCCESS, lw=2.0, rad=-0.18)
    lay.text("fb", 61, 41,
             "the next frame is encoded at the corrected quantiser",
             parent=None, color=SUCCESS, size=8.6, ha="center", style="italic")

    lay.panel("ceil", 4, 8, 44, 26, fc=PANEL, ec=BORDER)
    lay.text(None, 26, 30, "the target is a CEILING", parent="ceil", color=TITLE,
             size=9.2, weight="bold", ha="center")
    for i, ln in enumerate((
            "flat content floors the quantiser at 1 and still",
            "cannot spend the budget — measured 0.05× on a",
            "flat colour, which is what the reference does too")):
        lay.fit_text(None, 26, 25.5 - i * 3.8, ln, parent="ceil", size=7.9, color=TEXT,
                     ha="center")
    lay.fit_text(None, 26, 12, "it holds a rolling average, not a per-frame cap",
                 parent="ceil", size=7.7, color=WARNING, ha="center")

    lay.panel("meas", 52, 8, 44, 26, fc=PANEL, ec=BORDER)
    lay.text(None, 74, 30, "measured: every profile 1.00×", parent="meas",
             color=TITLE, size=9.2, weight="bold", ha="center")
    rows = [("Proxy", 194), ("LT", 440), ("422", 632),
            ("422 HQ", 950), ("4444", 1425), ("4444 XQ", 2137)]
    for i, (nm, bits) in enumerate(rows):
        col = 63 if i < 3 else 85
        yy = 25 - (i % 3) * 4.0
        lay.fit_text(None, col, yy, f"{nm}  {bits} bits/MB", parent="meas", size=7.6,
                     color=TEXT, ha="center", family="monospace")
    lay.fit_text(None, 74, 11, "1080p25, 1080i50 and 2160p25 alike", parent="meas",
                 size=7.7, color=MUTED, ha="center")

    lay.check(name="recording_qscale_auto")
    _save(fig, "recording_qscale_auto.png")


def main():
    field_pairing()
    qscale_auto_loop()
    print("done")


if __name__ == "__main__":
    main()
