"""Generate the OCIO colour-management diagrams.

Run from the repo root:  python docs/diagrams/generate_ocio_diagrams.py

Outputs PNGs into docs/images/. Re-run after changing a diagram; the images are committed
so the Markdown renders without a build step. Same palette and helpers as
`generate_diagrams.py`, so the guides stay visually of a piece.

Two diagrams, and each exists because prose was measurably bad at the job:

* `ocio_two_paths` — `MIXER COLORSPACE` and `MIXER OCIO` write the SAME stage and are
  mutually exclusive per layer. Saying that took a comparison table and several paragraphs
  in COLOR_GRADING.md; the branch-and-rejoin shape says it at a glance.
* `ocio_stages` — where each OCIO command acts. The input transform is per LAYER, the
  display transform is per CHANNEL and lands after the composite, and consumer views fan
  out from that same composite. Every one of those is a statement about POSITION, which is
  what prose is worst at.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from layout_check import Layout

# ── Palette, matching generate_diagrams.py ───────────────────────────────────
BG = "#1e1e1e"
PANEL = "#2d2d2d"
HOVER = "#3c3c3c"
BORDER = "#555555"
BORDER_SUBTLE = "#444444"
TEXT = "#d4d4d4"
MUTED = "#888888"
TITLE = "#9cdcfe"
ACCENT = "#2255aa"
ACCENT_HOVER = "#2f6bd0"
SUCCESS = "#1a6b1a"
WARNING = "#7a5c00"
DANGER = "#8a2a2a"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.2, radius=0.02, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def _text(ax, x, y, s, *, color=TEXT, size=10, weight="normal", ha="left", va="center",
          style="normal", z=5):
    return ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
                   fontstyle=style, zorder=z, family="DejaVu Sans")


def _layout(fig, ax):
    return Layout(fig, ax, panel_fn=_panel, text_fn=_text, arrow_fn=_arrow)


def _new(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def _arrow(ax, p0, p1, *, color=MUTED, lw=1.4, style="-|>", z=7, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, zorder=z, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def two_paths():
    """The two front ends to the working space, and their mutual exclusion."""
    fig, ax = _new((11, 5.6))
    lay = _layout(fig, ax)
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    lay.text("title", 50, 94, "Two routes into the working space \u2014 pick one per layer",
             color=TITLE, size=13, ha="center", weight="bold")

    lay.panel("src", 3, 44, 13, 14, fc=PANEL)
    lay.text(None, 9.5, 53, "Source", parent="src", color=TEXT, size=10, ha="center",
             weight="bold")
    lay.text(None, 9.5, 48.5, "clip / live", parent="src", color=MUTED, size=8, ha="center")

    lay.panel("colorspace", 24, 62, 30, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    lay.text(None, 39, 74.5, "MIXER COLORSPACE", parent="colorspace", color=TITLE,
             size=10.5, ha="center", weight="bold")
    lay.text(None, 39, 70, "built-in enums \u00b7 ACES 1.x", parent="colorspace",
             color=TEXT, size=8.5, ha="center")
    lay.text(None, 39, 66, "7 transfers \u00d7 8 gamuts", parent="colorspace",
             color=MUTED, size=8, ha="center")

    lay.panel("ocio", 24, 22, 30, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    lay.text(None, 39, 34.5, "MIXER OCIO", parent="ocio", color=TITLE, size=10.5,
             ha="center", weight="bold")
    lay.text(None, 39, 30, "config colour spaces \u00b7 ACES 2.0", parent="ocio",
             color=TEXT, size=8.5, ha="center")
    lay.text(None, 39, 26, "55 in the bundled config", parent="ocio", color=MUTED,
             size=8, ha="center")

    lay.arrow((16, 54), (24, 70))
    lay.arrow((16, 48), (24, 32))

    lay.panel("exclusion", 30.5, 45.5, 17, 9, fc=DANGER, ec="#a33", radius=0.12)
    lay.fit_text(None, 39, 51.6, "mutually exclusive", parent="exclusion", size=8.5,
                 color="#ffffff", ha="center", weight="bold")
    lay.fit_text(None, 39, 48.2, "the second returns 403", parent="exclusion", size=7.5,
                 color="#ffdddd", ha="center")
    lay.arrow((39, 62), (39, 55), color="#a33", style="<|-|>", lw=1.3)
    lay.arrow((39, 45), (39, 40), color="#a33", style="<|-|>", lw=1.3)

    lay.panel("acescg", 62, 44, 15, 14, fc=SUCCESS, ec="#2a8a2a")
    lay.text(None, 69.5, 53.5, "ACEScg", parent="acescg", color="#ffffff", size=10.5,
             ha="center", weight="bold")
    lay.text(None, 69.5, 48.8, "working space", parent="acescg", color="#dfd", size=8,
             ha="center")
    lay.arrow((54, 70), (62, 55))
    lay.arrow((54, 32), (62, 47))

    lay.panel("grade", 83, 40, 14, 22, fc=PANEL)
    lay.text(None, 90, 58, "grade", parent="grade", color=TEXT, size=10, ha="center",
             weight="bold")
    for i, t in enumerate(["EXPOSURE", "GAMUTCOMPRESS", "CDL \u00b7 LUT", "curves \u00b7 sat"]):
        lay.fit_text(None, 90, 53.5 - i * 3.6, t, parent="grade", size=7.5, color=MUTED,
                     ha="center")
    lay.arrow((77, 51), (83, 51))

    lay.text("footnote", 90, 35, "identical either way", color=SUCCESS, size=8,
             ha="center", style="italic")
    lay.text("cap1", 50, 12, "Both write pipeline steps 4\u20135. What differs is the "
             "vocabulary and the ACES generation \u2014", color=MUTED, size=8.5, ha="center")
    lay.text("cap2", 50, 8.2, "everything downstream of the working space behaves the "
             "same on either route.", color=MUTED, size=8.5, ha="center")

    lay.check(name="ocio_two_paths")
    _save(fig, "ocio_two_paths.png")


def stages():
    """Where each OCIO command acts: per layer, per channel, per consumer."""
    fig, ax = _new((11, 6.2))
    lay = _layout(fig, ax)
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    lay.text("title", 50, 94.5, "Where each OCIO command acts", color=TITLE, size=13,
             ha="center", weight="bold")

    # Containers are non-blocking: they exist to hold the boxes and labels inside them.
    lay.panel("layer_group", 3, 52, 40, 35, blocking=False, fc="#242424",
              ec=BORDER_SUBTLE, radius=0.015)
    lay.text(None, 5.5, 84, "per LAYER", color=MUTED, size=8.5, weight="bold")
    lay.text(None, 5.5, 80.5, "input transform \u2014 source encoding \u2192 ACEScg",
             color=MUTED, size=7.5)

    for i, (label, space) in enumerate([("layer 1", '"ARRI LogC3 (EI800)"'),
                                        ("layer 2", '"S-Log3 S-Gamut3.Cine"')]):
        y = 66 - i * 12
        lay.panel(label + "_box", 6, y, 12, 9, fc=PANEL)
        lay.text(None, 12, y + 4.5, label, parent=label + "_box", color=TEXT, size=9,
                 ha="center")
        lay.panel(label + "_cmd", 22, y, 18, 9, fc=PANEL, ec=ACCENT_HOVER, lw=1.5)
        lay.fit_text(None, 31, y + 6, "MIXER OCIO", parent=label + "_cmd", size=8.5,
                     color=TITLE, ha="center", weight="bold")
        lay.fit_text(None, 31, y + 2.6, space, parent=label + "_cmd", size=7,
                     color=MUTED, ha="center")
        lay.arrow((18, y + 4.5), (22, y + 4.5))

    lay.panel("composite", 47, 62, 15, 18, fc=SUCCESS, ec="#2a8a2a")
    lay.text(None, 54.5, 74, "composite", parent="composite", color="#ffffff", size=10,
             ha="center", weight="bold")
    lay.text(None, 54.5, 69.5, "in ACEScg", parent="composite", color="#dfd", size=8,
             ha="center")
    lay.text(None, 54.5, 65.5, "scene-linear", parent="composite", color="#dfd",
             size=7.5, ha="center")
    lay.arrow((40, 70.5), (47, 72))
    lay.arrow((40, 58.5), (47, 68))

    lay.panel("gate", 43, 81.5, 24, 7, fc=WARNING, ec="#a8801a", radius=0.12)
    lay.fit_text(None, 55, 86.6, "<working-space-composite>", parent="gate", size=7.5,
                 color="#ffffff", ha="center", weight="bold")
    lay.fit_text(None, 55, 83.6, "required beyond this point", parent="gate", size=7,
                 color="#ffeeba", ha="center")
    lay.arrow((55, 81.5), (55, 80.2), color="#a8801a", lw=1.2)

    lay.panel("display", 67, 62, 17, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    lay.fit_text(None, 75.5, 74, "OCIO_DISPLAY", parent="display", size=9.5, color=TITLE,
                 ha="center", weight="bold")
    lay.text(None, 75.5, 69.5, "channel display", parent="display", color=TEXT, size=8,
             ha="center")
    lay.text(None, 75.5, 65.5, "+ OCIO_LOOK", parent="display", color=MUTED, size=7.5,
             ha="center")
    lay.arrow((62, 71), (67, 71))
    lay.text(None, 70, 84, "per CHANNEL", color=MUTED, size=8.5, weight="bold")

    lay.panel("consumer_group", 3, 8, 96, 36, blocking=False, fc="#242424",
              ec=BORDER_SUBTLE, radius=0.015)
    lay.text(None, 5.5, 40, "per CONSUMER", color=MUTED, size=8.5, weight="bold")

    consumers = [("screen", "channel's view", HOVER),
                 ("decklink 1", '"Un-tone-mapped"', ACCENT),
                 ("decklink 2", '"ACES 2.0 - SDR 100 nits"', ACCENT)]
    for i, (nm, view, col) in enumerate(consumers):
        x = 8 + i * 31
        lay.panel(nm + "_box", x, 14, 26, 20, fc=PANEL)
        lay.text(None, x + 13, 29, nm, parent=nm + "_box", color=TEXT, size=9.5,
                 ha="center", weight="bold")
        lay.panel(nm + "_view", x + 2.5, 17.5, 21, 7, fc=col, ec=BORDER_SUBTLE, radius=0.1)
        lay.fit_text(None, x + 13, 21, view, parent=nm + "_view", size=7.5,
                     color="#ffffff" if col != HOVER else MUTED, ha="center")
        lay.arrow((75.5, 62), (x + 13, 34), color=BORDER, lw=1.1, ls=(0, (4, 2)))

    lay.text("note", 50, 11, "one extra post-composite pass per DISTINCT view \u2014 two "
             "consumers asking for the same view cost one", color=MUTED, size=7.5,
             ha="center")
    lay.text("cap", 50, 4.2, "The input transform is a property of each layer; the "
             "display transform is a property of the screen, so it is channel-level.",
             color=MUTED, size=8.5, ha="center")

    lay.check(name="ocio_stages")
    _save(fig, "ocio_stages.png")


if __name__ == "__main__":
    two_paths()
    stages()
