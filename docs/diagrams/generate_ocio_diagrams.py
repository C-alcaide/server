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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, family="DejaVu Sans")


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
    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    _text(ax, 50, 94, "Two routes into the working space — pick one per layer",
          color=TITLE, size=13, ha="center", weight="bold")

    # Source
    _panel(ax, 3, 44, 13, 14, fc=PANEL)
    _text(ax, 9.5, 53, "Source", color=TEXT, size=10, ha="center", weight="bold")
    _text(ax, 9.5, 48.5, "clip / live", color=MUTED, size=8, ha="center")

    # The two mutually exclusive branches
    _panel(ax, 24, 62, 30, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    _text(ax, 39, 74.5, "MIXER COLORSPACE", color=TITLE, size=10.5, ha="center",
          weight="bold")
    _text(ax, 39, 70, "built-in enums · ACES 1.x", color=TEXT, size=8.5, ha="center")
    _text(ax, 39, 66, "7 transfers × 8 gamuts", color=MUTED, size=8, ha="center")

    _panel(ax, 24, 22, 30, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    _text(ax, 39, 34.5, "MIXER OCIO", color=TITLE, size=10.5, ha="center", weight="bold")
    _text(ax, 39, 30, "config colour spaces · ACES 2.0", color=TEXT, size=8.5, ha="center")
    _text(ax, 39, 26, "55 in the bundled config", color=MUTED, size=8, ha="center")

    _arrow(ax, (16, 54), (24, 70))
    _arrow(ax, (16, 48), (24, 32))

    # Mutual exclusion
    _panel(ax, 30.5, 45.5, 17, 9, fc=DANGER, ec="#a33", radius=0.12)
    _text(ax, 39, 51.6, "mutually exclusive", color="#ffffff", size=8.5, ha="center",
          weight="bold")
    _text(ax, 39, 48.2, "the second returns 403", color="#ffdddd", size=7.5, ha="center")
    _arrow(ax, (39, 62), (39, 55), color="#a33", style="<|-|>", lw=1.3)
    _arrow(ax, (39, 45), (39, 40), color="#a33", style="<|-|>", lw=1.3)

    # Working space — where they rejoin
    _panel(ax, 62, 44, 15, 14, fc=SUCCESS, ec="#2a8a2a")
    _text(ax, 69.5, 53.5, "ACEScg", color="#ffffff", size=10.5, ha="center", weight="bold")
    _text(ax, 69.5, 48.8, "working space", color="#dfd", size=8, ha="center")
    _arrow(ax, (54, 70), (62, 55))
    _arrow(ax, (54, 32), (62, 47))

    # The shared chain
    _panel(ax, 83, 40, 14, 22, fc=PANEL)
    _text(ax, 90, 58, "grade", color=TEXT, size=10, ha="center", weight="bold")
    for i, t in enumerate(["EXPOSURE", "GAMUTCOMPRESS", "CDL · LUT", "curves · sat"]):
        _text(ax, 90, 53.5 - i * 3.6, t, color=MUTED, size=7.5, ha="center")
    _arrow(ax, (77, 51), (83, 51))

    _text(ax, 90, 35, "identical either way", color=SUCCESS, size=8, ha="center",
          style="italic")

    _text(ax, 50, 12, "Both write pipeline steps 4–5. What differs is the vocabulary and "
                      "the ACES generation —", color=MUTED, size=8.5, ha="center")
    _text(ax, 50, 8.2, "everything downstream of the working space behaves the same on "
                       "either route.", color=MUTED, size=8.5, ha="center")
    _save(fig, "ocio_two_paths.png")


def stages():
    """Where each OCIO command acts: per layer, per channel, per consumer."""
    fig, ax = _new((11, 6.2))
    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    _text(ax, 50, 94.5, "Where each OCIO command acts", color=TITLE, size=13,
          ha="center", weight="bold")

    # ── Per layer ────────────────────────────────────────────────────────────
    _panel(ax, 3, 52, 40, 35, fc="#242424", ec=BORDER_SUBTLE, radius=0.015)
    _text(ax, 5.5, 84, "per LAYER", color=MUTED, size=8.5, weight="bold")
    _text(ax, 5.5, 80.5, "input transform — source encoding → ACEScg", color=MUTED, size=7.5)

    for i, (label, sub) in enumerate([("layer 1", '"ARRI LogC3 (EI800)"'),
                                      ("layer 2", '"S-Log3 S-Gamut3.Cine"')]):
        y = 66 - i * 12  # 66..75 and 54..63, both inside the panel
        _panel(ax, 6, y, 12, 9, fc=PANEL)
        _text(ax, 12, y + 4.5, label, color=TEXT, size=9, ha="center")
        _panel(ax, 22, y, 18, 9, fc=PANEL, ec=ACCENT_HOVER, lw=1.5)
        _text(ax, 31, y + 6, "MIXER OCIO", color=TITLE, size=8.5, ha="center",
              weight="bold")
        _text(ax, 31, y + 2.6, sub, color=MUTED, size=7, ha="center")
        _arrow(ax, (18, y + 4.5), (22, y + 4.5))

    # ── Composite ────────────────────────────────────────────────────────────
    _panel(ax, 47, 62, 15, 18, fc=SUCCESS, ec="#2a8a2a")
    _text(ax, 54.5, 74, "composite", color="#ffffff", size=10, ha="center", weight="bold")
    _text(ax, 54.5, 69.5, "in ACEScg", color="#dfd", size=8, ha="center")
    _text(ax, 54.5, 65.5, "scene-linear", color="#dfd", size=7.5, ha="center")
    _arrow(ax, (40, 70.5), (47, 72))
    _arrow(ax, (40, 58.5), (47, 68))

    _panel(ax, 43, 81.5, 24, 7, fc=WARNING, ec="#a8801a", radius=0.12)
    _text(ax, 55, 86.6, "<working-space-composite>", color="#ffffff", size=7.5,
          ha="center", weight="bold")
    _text(ax, 55, 83.6, "required beyond this point", color="#ffeeba", size=7,
          ha="center")
    _arrow(ax, (55, 81.5), (55, 80.2), color="#a8801a", lw=1.2)

    # ── Per channel ──────────────────────────────────────────────────────────
    _panel(ax, 67, 62, 17, 18, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    _text(ax, 75.5, 74, "OCIO_DISPLAY", color=TITLE, size=9.5, ha="center", weight="bold")
    _text(ax, 75.5, 69.5, "channel display", color=TEXT, size=8, ha="center")
    _text(ax, 75.5, 65.5, "+ OCIO_LOOK", color=MUTED, size=7.5, ha="center")
    _arrow(ax, (62, 71), (67, 71))
    _text(ax, 70, 84, "per CHANNEL", color=MUTED, size=8.5, weight="bold")

    # ── Per consumer ─────────────────────────────────────────────────────────
    _panel(ax, 3, 8, 96, 36, fc="#242424", ec=BORDER_SUBTLE, radius=0.015)
    # Heading kept short: the fan-out arrows sweep across y=40 and clipped the longer one.
    _text(ax, 5.5, 40, "per CONSUMER", color=MUTED, size=8.5, weight="bold")
    _text(ax, 50, 11, "one extra post-composite pass per DISTINCT view — two consumers "
                      "asking for the same view cost one", color=MUTED, size=7.5,
          ha="center")

    for i, (name, view, col) in enumerate([
            ("screen", "channel's view", HOVER),
            ("decklink 1", '"Un-tone-mapped"', ACCENT),
            ("decklink 2", '"ACES 2.0 - SDR 100 nits"', ACCENT)]):
        x = 8 + i * 31
        _panel(ax, x, 14, 26, 20, fc=PANEL)
        _text(ax, x + 13, 29, name, color=TEXT, size=9.5, ha="center", weight="bold")
        _panel(ax, x + 2.5, 17.5, 21, 7, fc=col, ec=BORDER_SUBTLE, radius=0.1)
        _text(ax, x + 13, 21, view, color="#ffffff" if col != HOVER else MUTED,
              size=7.5, ha="center")
        _arrow(ax, (75.5, 62), (x + 13, 34), color=BORDER, lw=1.1, ls=(0, (4, 2)))

    _text(ax, 50, 4.2, "The input transform is a property of each layer; the display "
                       "transform is a property of the screen, so it is channel-level.",
          color=MUTED, size=8.5, ha="center")
    _save(fig, "ocio_stages.png")


if __name__ == "__main__":
    two_paths()
    stages()
