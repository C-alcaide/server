"""Generate the two previz figures `previz.md` §7 records as owed.

Run from the repo root:  python docs/diagrams/generate_previz_diagrams.py

Outputs PNGs into docs/images/. Committed, so the Markdown renders with no build step.
Palette and helpers follow `generate_diagrams.py` deliberately -- a second visual language
in one docs folder is a cost with no reader on the other side of it.

WHY THESE TWO, and not a diagram per command. `CLAUDE.md` names three things that earn a
figure, and previz has exactly two of them:

  * TWO PATHS THAT REACH THE SAME PLACE -- `PREVIZ SCREEN <name> ICVFX` and
    `MIXER <ch>-<l> PROJECTION_ICVFX` both end at one layer's `icvfx_enable`, by different
    routes, and one of them LOCKS the other out. That is a sentence people re-read three times
    and a picture answers once.
  * A CHAIN WHOSE ORDER IS THE POINT -- scene, mesh, mapped channel, camera, viewport. Prose
    describes it one step at a time and the reader has to hold the order; and two of the traps
    that cost real time this year are positions in that chain rather than facts about any one
    step (a mapped channel with no consumer never ticks; auto-projection with no SOURCE applies
    to layer 0).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── App palette, same as generate_diagrams.py ────────────────────────────────
BG = "#1e1e1e"
PANEL = "#2d2d2d"
HOVER = "#3c3c3c"
BORDER = "#555555"
TEXT = "#d4d4d4"
MUTED = "#888888"
TITLE = "#9cdcfe"
ACCENT = "#2255aa"
WARNING = "#7a5c00"
DANGER = "#8a2a2a"
SUCCESS = "#1a6b1a"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.2, radius=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def _text(ax, x, y, s, *, color=TEXT, size: float = 10, weight="normal", ha="left",
          va="center", style="normal", z=5):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, family="DejaVu Sans")


def _arrow(ax, p0, p1, *, color=TEXT, lw=1.6, style="-|>", dashed=False, z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, zorder=z,
                                 linestyle="--" if dashed else "-",
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
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 1. The two routes to ICVFX state, and which one locks the other out
# ─────────────────────────────────────────────────────────────────────────────
def icvfx_two_routes():
    fig, ax = _new((10.2, 5.4))

    _text(ax, 50, 95, "Two routes to one layer's ICVFX state", color=TITLE,
          size=14, weight="bold", ha="center")
    _text(ax, 50, 89.5, "and the first thing the second one does is switch the first one off",
          color=MUTED, size=9.5, ha="center", style="italic")

    # ── Route A: the stage, recomputed per camera move ───────────────────────
    _panel(ax, 4, 56, 40, 26, fc=HOVER)
    _text(ax, 6.5, 78, "A.  THE STAGE", color=TITLE, size=10.5, weight="bold")
    _text(ax, 6.5, 73, "PREVIZ <ch> SCREEN <name> ICVFX 1", color=TEXT, size=9)
    _text(ax, 6.5, 69, "PREVIZ <ch> AUTOPROJECTION 1", color=TEXT, size=9)
    _text(ax, 6.5, 63.5, "compute_frustum() re-derives the quad", color=MUTED, size=8.5)
    _text(ax, 6.5, 60, "from the camera pose, EVERY MOVE", color=MUTED, size=8.5)

    # ── Route B: the operator, set once ──────────────────────────────────────
    _panel(ax, 56, 56, 40, 26, fc=HOVER)
    _text(ax, 58.5, 78, "B.  THE OPERATOR", color=TITLE, size=10.5, weight="bold")
    _text(ax, 58.5, 73, "MIXER <ch>-<l> PROJECTION_ICVFX ...", color=TEXT, size=9)
    _text(ax, 58.5, 69, "MIXER <ch>-<l> PROJECTION_ICVFX_COLOR ...", color=TEXT, size=8.4)
    _text(ax, 58.5, 63.5, "sets the values once and", color=MUTED, size=8.5)
    _text(ax, 58.5, 60, "sets icvfx_auto = FALSE", color="#e0b050", size=8.5, weight="bold")

    # ── The one place both land ──────────────────────────────────────────────
    _panel(ax, 26, 22, 48, 20, fc=PANEL, ec=ACCENT, lw=1.8)
    _text(ax, 50, 37.5, "the layer's image_transform.projection", color=TITLE,
          size=10.5, weight="bold", ha="center")
    _text(ax, 50, 32.5, "icvfx_enable   icvfx_q0..q3   feather   dims   gains",
          color=TEXT, size=8.8, ha="center")
    _text(ax, 50, 27, "one set of values, whichever route wrote them",
          color=MUTED, size=8.5, ha="center", style="italic")

    _arrow(ax, (24, 56), (38, 42.5), color=SUCCESS, lw=2.0)
    _arrow(ax, (76, 56), (62, 42.5), color=ACCENT, lw=2.0)

    # ── The lock, drawn as what it is: B stopping A ──────────────────────────
    _arrow(ax, (56, 66), (44, 66), color="#e0b050", lw=2.0, dashed=True, rad=-0.25)
    _text(ax, 50, 51.5, "icvfx_auto = false", color="#e0b050", size=8.6,
          ha="center", weight="bold")
    _text(ax, 50, 47.8, "B locks A out, so the operator's values survive the next camera move",
          color=MUTED, size=8.3, ha="center")

    _panel(ax, 26, 6, 48, 12, fc="#2a2118", ec=WARNING)
    _text(ax, 50, 14, "Set it by hand and auto-projection stops touching it.",
          color="#e0b050", size=9, ha="center", weight="bold")
    _text(ax, 50, 9.5, "PREVIZ <ch> SCREEN <name> ICVFX 1 restores route A.",
          color=TEXT, size=8.6, ha="center")

    _save(fig, "previz_icvfx_routes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. The chain: scene -> mesh -> mapped channel -> camera -> viewport
# ─────────────────────────────────────────────────────────────────────────────
def previz_chain():
    fig, ax = _new((11.6, 5.6))

    _text(ax, 50, 95, "What previz draws, and in what order it is assembled", color=TITLE,
          size=14, weight="bold", ha="center")
    _text(ax, 50, 89.5, "each step needs the one before it; two of them fail silently",
          color=MUTED, size=9.5, ha="center", style="italic")

    boxes = [
        (2.0, "PREVIZ <ch> SCENE", "a glTF/OBJ venue,\nor SCREEN ADD"),
        (21.0, "PREVIZ <ch> MAP", "mesh name -> channel"),
        (40.0, "that CHANNEL", "must be producing\nframes of its own"),
        (59.0, "PREVIZ <ch> CAMERA", "the PRODUCTION camera\n-- drives projections"),
        (78.0, "PREVIZ <ch> VIEW", "the VIEWPORT camera\n-- what you look from"),
    ]
    for x, head, body in boxes:
        _panel(ax, x, 50, 18, 26, fc=HOVER)
        _text(ax, x + 9, 71, head, color=TITLE, size=9.4, weight="bold", ha="center")
        for i, line in enumerate(body.split("\n")):
            _text(ax, x + 9, 64 - i * 4.4, line, color=TEXT, size=8.3, ha="center")
    for x in (20.0, 39.0, 58.0, 77.0):
        _arrow(ax, (x, 63), (x + 1.8, 63), color=TEXT, lw=1.8)

    # The two traps, pinned to the step they belong to rather than listed apart.
    _panel(ax, 33, 26, 32, 17, fc="#2a2118", ec=WARNING)
    _text(ax, 49, 39, "A mapped channel with no CONSUMER", color="#e0b050",
          size=8.8, ha="center", weight="bold")
    _text(ax, 49, 34.5, "never ticks. MAP answers 202 and", color=TEXT, size=8.3, ha="center")
    _text(ax, 49, 30.5, "the mesh stays grey. Settling does not help.",
          color=TEXT, size=8.3, ha="center")
    _arrow(ax, (49, 43), (49, 50), color=WARNING, lw=1.5, dashed=True)

    _panel(ax, 67, 26, 31, 17, fc="#2a2118", ec=WARNING)
    _text(ax, 82.5, 39, "The two cameras are NOT the same", color="#e0b050",
          size=8.8, ha="center", weight="bold")
    _text(ax, 82.5, 34.5, "compute_frustum always reads the", color=TEXT, size=8.3, ha="center")
    _text(ax, 82.5, 30.5, "production one; orbiting never moves it.",
          color=TEXT, size=8.3, ha="center")
    _arrow(ax, (82.5, 43), (75, 50), color=WARNING, lw=1.5, dashed=True)

    _panel(ax, 2, 4, 96, 17, fc=PANEL, ec=BORDER)
    _text(ax, 4.5, 16.5, "And the channel previz RENDERS ON shows the 3D view, not its own layers.",
          color=TITLE, size=9.2, weight="bold")
    _text(ax, 4.5, 11.5,
          "PREVIZ 1 ... makes channel 1 the viewport. Capturing channel 1 is then a picture of the scene -- a layer's own",
          color=TEXT, size=8.4)
    _text(ax, 4.5, 7.2,
          "grade or ICVFX on that channel is invisible in it. Measure those on a channel previz is not drawing.",
          color=TEXT, size=8.4)

    _save(fig, "previz_chain.png")


if __name__ == "__main__":
    icvfx_two_routes()
    previz_chain()
