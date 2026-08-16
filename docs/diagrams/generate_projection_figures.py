"""Generate the projection-calibration diagrams.

Run from the repo root:  python docs/diagrams/generate_projection_figures.py

Outputs PNGs into docs/images/. Re-run after changing a diagram; the images are committed
so the Markdown renders without a build step. Same palette and helpers as
`generate_diagrams.py` and `generate_ocio_diagrams.py`.

`PROJECTION_CALIBRATION.md` is an operator manual — the in-depth reference behind the
Operations Guide's Projection tab — so it gets rendered images rather than inline mermaid,
per CLAUDE.md. Two diagrams, each replacing a paragraph that was doing badly:

* `projection_closed_loop` — the loop LEAVES THE COMPUTER. That is the whole method, and
  the doc's most important sentence ("the camera return is not the embedded preview") is a
  statement about a loop, which prose states and a picture shows. Everything the
  calibration corrects happens on the projector → surface edge; comparing the outgoing
  frame with itself closes nothing.
* `projection_phases` — each phase measures a different physical error and leaves it in a
  different mixer command, which is why they compose rather than supersede. Phase C applies
  nothing at all, and that is easy to miss in a list.
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


def _arrow(ax, p0, p1, *, color=MUTED, lw=1.4, style="-|>", z=7, ls="-",
           connectionstyle=None):
    kw = {}
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, zorder=z, linestyle=ls,
                                 shrinkA=0, shrinkB=0, **kw))


def closed_loop():
    """The calibration loop, and the fact that it leaves the computer."""
    fig, ax = _new((11.5, 5.8))
    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    _text(ax, 50, 94, "Camera-based projection calibration — a loop through the physical world",
          color=TITLE, size=12.5, weight="bold", ha="center")

    # The digital half, along the top.
    _panel(ax, 4, 62, 20, 16, fc=PANEL)
    _text(ax, 14, 72.5, "pattern", size=10, weight="bold", ha="center")
    _text(ax, 14, 67.5, "at the channel's\nnative resolution", size=8.2, color=MUTED,
          ha="center", va="center")

    _panel(ax, 30, 62, 20, 16, fc=ACCENT, ec=ACCENT_HOVER)
    _text(ax, 40, 72.5, "CasparVP", size=10, weight="bold", ha="center")
    _text(ax, 40, 67.5, "identity\nperspective", size=8.2, color="#cfe3ff",
          ha="center", va="center")

    _panel(ax, 56, 62, 18, 16, fc=PANEL)
    _text(ax, 65, 72.5, "projector", size=10, weight="bold", ha="center")
    _text(ax, 65, 67.5, "SDI / HDMI", size=8.2, color=MUTED, ha="center")

    _arrow(ax, (24, 70), (30, 70))
    _arrow(ax, (50, 70), (56, 70))

    # The physical edge — the only place the errors live.
    _panel(ax, 78, 62, 18, 16, fc=WARNING, ec="#a67c00")
    _text(ax, 87, 72.5, "surface", size=10, weight="bold", ha="center")
    _text(ax, 87, 67.5, "screen / cyc /\nLED volume", size=8.2, color="#f0e0b0",
          ha="center", va="center")
    _arrow(ax, (74, 70), (78, 70), color="#c9a227", lw=2.0)
    _text(ax, 76, 57, "optics + surface", size=8.6, color="#c9a227", weight="bold",
          ha="center")
    _text(ax, 76, 53, "lens distortion · keystone\ncurvature · overlap", size=7.8,
          color=MUTED, ha="center", va="center")

    # Back through the camera.
    _panel(ax, 68, 26, 20, 15, fc=PANEL)
    _text(ax, 78, 35.5, "camera", size=10, weight="bold", ha="center")
    _text(ax, 78, 30.5, "UVC or stills,\nfilming the surface", size=8.2, color=MUTED,
          ha="center", va="center")
    _arrow(ax, (87, 62), (81, 41), color="#c9a227", lw=1.6)
    _text(ax, 91, 50, "photons", size=8.4, color="#c9a227", ha="center", style="italic")

    _panel(ax, 36, 26, 24, 15, fc=PANEL)
    _text(ax, 48, 35.5, "client solve", size=10, weight="bold", ha="center")
    _text(ax, 48, 30.5, "OpenCV: detect pattern\n→ homography → warp", size=8.2,
          color=MUTED, ha="center", va="center")
    _arrow(ax, (68, 33.5), (60, 33.5))

    # Up the left side, from the solve box's edge to the server. `rad` is negative so the
    # curve bows away from the text block rather than through it.
    _arrow(ax, (36, 33.5), (38, 62), color="#3f9142", lw=1.8,
           connectionstyle="arc3,rad=-0.28")
    _text(ax, 22, 50, "AMCP", size=9, color="#5fbf62", weight="bold", ha="center")
    _text(ax, 22, 46, "MIXER PERSPECTIVE\nDISTORTION · BLEND · MESH", size=7.6,
          color=MUTED, ha="center", va="center")

    # The thing the doc most needs to deny.
    _panel(ax, 4, 6, 46, 13, fc=PANEL, ec=DANGER)
    _text(ax, 7, 15, "✗  the embedded preview is NOT the return", size=9.2,
          color="#e06c6c", weight="bold")
    _text(ax, 7, 10, "it is the signal we SENT — comparing it with itself\n"
                     "measures nothing on the projector → surface edge", size=8,
          color=MUTED, va="center")

    _panel(ax, 54, 6, 42, 13, fc=PANEL, ec=BORDER_SUBTLE)
    _text(ax, 57, 15, "everything corrected lives on the yellow edge", size=9.2,
          color="#c9a227", weight="bold")
    _text(ax, 57, 10, "which is why the loop has to leave the computer\n"
                      "and come back as photons", size=8, color=MUTED, va="center")

    _save(fig, "projection_closed_loop.png")


def phases():
    """Each phase measures a different error and leaves it in a different command."""
    fig, ax = _new((11.5, 6.2))
    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    _text(ax, 50, 94, "The phases compose — each leaves its result in a different mixer command",
          color=TITLE, size=12.5, weight="bold", ha="center")

    rows = [
        ("Phase A", "corner-pin", "where the quad lands", "MIXER PERSPECTIVE", ACCENT, False),
        ("Phase B", "distortion + blend", "lens curvature, overlap ramps",
         "MIXER PROJECTION_DISTORTION\nMIXER PROJECTION_BLEND", ACCENT, False),
        ("Phase C", "diagnostics", "uniformity · focus · contrast · straightness",
         "nothing — analysis only", HOVER, True),
        ("Phase D", "dense warp", "Gray-code structured light", "MIXER MESH  (.glb)",
         ACCENT, False),
        ("Phase E", "multi-projector", "world-UV alignment across projectors",
         "the above, per projector\n+ MIXER PROJECTION_BLEND_MASK", ACCENT, False),
    ]

    top, h, gap = 84.0, 12.5, 2.2
    for i, (phase, name, measures, command, colour, dashed) in enumerate(rows):
        y = top - (i + 1) * h - i * gap
        _panel(ax, 4, y, 15, h, fc=PANEL, ec=BORDER)
        _text(ax, 11.5, y + h / 2 + 2, phase, size=9.6, weight="bold", ha="center")
        _text(ax, 11.5, y + h / 2 - 2.6, name, size=8.2, color=MUTED, ha="center")

        _panel(ax, 21, y, 33, h, fc=PANEL, ec=BORDER_SUBTLE)
        _text(ax, 23, y + h / 2, measures, size=8.4, color=TEXT, va="center")

        _arrow(ax, (54, y + h / 2), (59, y + h / 2),
               color=MUTED if not dashed else "#6a6a6a",
               ls="-" if not dashed else (0, (3, 3)))

        _panel(ax, 59, y, 37, h, fc=colour,
               ec=ACCENT_HOVER if not dashed else BORDER_SUBTLE)
        _text(ax, 61, y + h / 2, command, size=8.4,
              color="#cfe3ff" if not dashed else MUTED,
              style="normal" if not dashed else "italic", va="center")

    _text(ax, 50, 4.5,
          "Phase C is the odd one out on purpose: it applies nothing, and exists to say "
          "whether the phase before it worked.",
          size=8.6, color=MUTED, ha="center", style="italic")

    _save(fig, "projection_phases.png")


if __name__ == "__main__":
    closed_loop()
    phases()
