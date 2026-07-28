"""Generate documentation diagrams for the CasparCG ISF module guide.

Run from the repo root:  python docs/diagrams/generate_isf_diagrams.py

Outputs PNGs into docs/images/. Re-run after changing a diagram; the images are
committed so the Markdown renders without a build step. Dependency-light
(matplotlib + numpy) and themed to match the CasparCG docs palette.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ── Docs palette (matches docs/diagrams/generate_diagrams.py) ─────────────────
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
GL = "#2f6bd0"
VK = "#8a5a2a"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.14)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def _new(figsize, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("auto")
    return fig, ax


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.3, radius=1.4, z=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(box)
    return box


def _text(ax, x, y, s, *, color=TEXT, size=10, weight="normal", ha="left",
          va="center", style="normal", z=6, family="DejaVu Sans"):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, family=family)


def _box(ax, x, y, w, h, title, sub=None, *, fc=PANEL, ec=BORDER, tcolor=TEXT,
         tsize=10, z=2):
    _panel(ax, x, y, w, h, fc=fc, ec=ec, z=z)
    if sub:
        _text(ax, x + w / 2, y + h * 0.63, title, color=tcolor, size=tsize,
              ha="center", weight="bold", z=z + 4)
        _text(ax, x + w / 2, y + h * 0.30, sub, color=MUTED, size=tsize - 2.2,
              ha="center", z=z + 4)
    else:
        _text(ax, x + w / 2, y + h / 2, title, color=tcolor, size=tsize,
              ha="center", weight="bold", z=z + 4)


def _arrow(ax, x1, y1, x2, y2, *, color=TITLE, lw=2.0, style="-|>", z=5,
           mut=13, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=mut, lw=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0, linestyle=ls))


def _title(ax, s, sub=None):
    _text(ax, 50, 96, s, color=TITLE, size=14, weight="bold", ha="center")
    if sub:
        _text(ax, 50, 91.5, sub, color=MUTED, size=9.5, ha="center", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 1. The three producer modes
# ─────────────────────────────────────────────────────────────────────────────
def isf_modes():
    fig, ax = _new((11, 7.4))
    _title(ax, "ISF producer modes",
           "AMCP command  →  [ISF] producer  →  channel output")

    rows = [
        (70, "Generator", "[ISF] plasma", SUCCESS,
         [("shader draws\nitself", PANEL)]),
        (44, "Filter", "[ISF] blur  <source>", ACCENT,
         [("source\nproducer", HOVER), ("inputImage", PANEL)]),
        (18, "Transition", "[ISF] dissolve TRANSITION <from> <to> [frames]", WARNING,
         [("from / to\nproducers", HOVER), ("startImage\nendImage\n+ progress", PANEL)]),
    ]

    for y, label, cmd, col, chain in rows:
        h = 15
        _box(ax, 2, y, 15, h, label, fc=col, ec=BORDER, tcolor="#ffffff", tsize=11)
        _text(ax, 2, y - 3.2, cmd, color=MUTED, size=8.2, ha="left", family="DejaVu Sans Mono")

        x = 21
        prev_cx = 17
        for (name, fc) in chain:
            _arrow(ax, prev_cx, y + h / 2, x, y + h / 2)
            _box(ax, x, y, 20, h, name, fc=fc, ec=BORDER, tsize=9.2)
            prev_cx = x + 20
            x += 26
        # output
        _arrow(ax, prev_cx, y + h / 2, x, y + h / 2)
        _box(ax, x, y, 18, h, "output\nframe", fc="#0e1118", ec=GL, tcolor=TITLE, tsize=9.2)

    _save(fig, "isf_modes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. The render pipeline (per frame)
# ─────────────────────────────────────────────────────────────────────────────
def isf_pipeline():
    fig, ax = _new((12, 6.6))
    _title(ax, "ISF render pipeline (per frame)")

    # Source acquisition (left)
    _box(ax, 2, 62, 20, 16, "Source frame", "filter / transition", fc=HOVER, tsize=9.5)
    _box(ax, 2, 40, 20, 14, "texture-backed?", "GPU frame", fc=PANEL, ec=GL, tsize=9)
    _box(ax, 2, 18, 20, 16, "generator", "no source", fc=PANEL, tsize=9.5)

    # Binding
    _box(ax, 27, 52, 20, 12, "sample directly", "same GL device", fc=PANEL, ec=GL, tsize=8.8)
    _box(ax, 27, 34, 20, 12, "convert + upload", "BGRA/RGBA → RGBA", fc=PANEL, tsize=8.8)
    _arrow(ax, 22, 47, 27, 58, color=GL)
    _arrow(ax, 22, 47, 27, 40)
    _arrow(ax, 22, 70, 27, 60, color=MUTED, ls=(0, (3, 2)))

    # Shader program
    _box(ax, 53, 34, 20, 30, "ISF shader\nprogram", "fullscreen triangle\n+ fragment", fc=ACCENT,
         ec=ACCENT_HOVER, tcolor="#ffffff", tsize=10)
    _arrow(ax, 47, 58, 53, 52)
    _arrow(ax, 47, 40, 53, 46)
    _arrow(ax, 22, 26, 53, 42, color=MUTED)

    # PASSES loop
    _box(ax, 53, 16, 20, 12, "PASSES loop", "buffers · PASSINDEX", fc=PANEL, ec=WARNING, tsize=8.8)
    _arrow(ax, 63, 34, 63, 28, color=WARNING)
    _arrow(ax, 73, 22, 78, 22, color=WARNING)
    _arrow(ax, 78, 26, 68, 34, color=WARNING, ls=(0, (2, 2)))
    _text(ax, 74.5, 30, "loop", color=WARNING, size=7.5, ha="center", style="italic")

    # Final tex + flip
    _box(ax, 79, 46, 18, 18, "final texture", "bottom-up RGBA\n→ Y-flip", fc=PANEL, tsize=9)
    _arrow(ax, 73, 50, 79, 54)

    # Two delivery paths
    _box(ax, 79, 24, 18, 14, "OpenGL mixer", "zero-copy texture", fc=GL, ec=ACCENT_HOVER,
         tcolor="#ffffff", tsize=9)
    _box(ax, 79, 4, 18, 14, "Vulkan mixer", "read back → CPU frame", fc=VK, ec=WARNING,
         tcolor="#ffffff", tsize=9)
    _arrow(ax, 88, 46, 88, 38, color=GL)
    _arrow(ax, 88, 46, 88, 18, color=VK)

    _save(fig, "isf_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-pass + persistent buffers
# ─────────────────────────────────────────────────────────────────────────────
def isf_passes():
    fig, ax = _new((11, 6.2))
    _title(ax, "Multi-pass rendering & buffers",
           "one fragment shader, run once per pass; PASSINDEX selects behaviour")

    # Single frame, multi pass (top)
    _text(ax, 4, 78, "PASSES = [ { \"TARGET\":\"bufA\" }, { } ]", color=MUTED, size=9,
          family="DejaVu Sans Mono")
    _box(ax, 4, 58, 20, 15, "Pass 0", "PASSINDEX 0", fc=ACCENT, ec=ACCENT_HOVER,
         tcolor="#ffffff", tsize=10)
    _box(ax, 34, 58, 18, 15, "bufA", "TARGET buffer", fc=PANEL, ec=WARNING, tsize=9.5)
    _box(ax, 62, 58, 20, 15, "Pass 1", "PASSINDEX 1", fc=ACCENT, ec=ACCENT_HOVER,
         tcolor="#ffffff", tsize=10)
    _box(ax, 88, 58, 10, 15, "out", None, fc="#0e1118", ec=GL, tcolor=TITLE, tsize=9)
    _arrow(ax, 24, 65.5, 34, 65.5, color=WARNING)
    _arrow(ax, 52, 65.5, 62, 65.5, color=WARNING)
    _text(ax, 57, 69, "sample", color=MUTED, size=7.5, ha="center", style="italic")
    _arrow(ax, 82, 65.5, 88, 65.5, color=GL)

    # Persistent ping-pong (bottom)
    _text(ax, 4, 44, "PERSISTENT buffer  (feedback across frames — double-buffered)",
          color=MUTED, size=9)
    _box(ax, 6, 20, 20, 16, "tex[front]", "previous frame", fc=HOVER, tsize=9)
    _box(ax, 40, 20, 20, 16, "shader pass", "read front\nwrite back", fc=ACCENT,
         ec=ACCENT_HOVER, tcolor="#ffffff", tsize=9)
    _box(ax, 74, 20, 20, 16, "tex[back]", "new frame", fc=PANEL, ec=WARNING, tsize=9)
    _arrow(ax, 26, 28, 40, 28, color=GL)
    _text(ax, 33, 31.5, "read", color=MUTED, size=7.5, ha="center", style="italic")
    _arrow(ax, 60, 28, 74, 28, color=WARNING)
    _text(ax, 67, 31.5, "write", color=MUTED, size=7.5, ha="center", style="italic")
    # swap arc
    _arrow(ax, 84, 20, 16, 20, color=SUCCESS, ls=(0, (4, 3)), lw=1.6, mut=11)
    _text(ax, 50, 14.5, "swap front ↔ back after each frame", color=SUCCESS, size=8.2,
          ha="center", style="italic")

    _save(fig, "isf_passes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mixer delivery paths
# ─────────────────────────────────────────────────────────────────────────────
def isf_mixer_paths():
    fig, ax = _new((11, 6.0))
    _title(ax, "Mixer delivery paths", "identical visual output on both mixers")

    # OpenGL column
    _panel(ax, 3, 8, 45, 74, fc="#232a36", ec=ACCENT, radius=1.6, z=1)
    _text(ax, 25.5, 76, "OpenGL mixer  ·  zero-copy", color=TITLE, size=11, ha="center",
          weight="bold")
    _box(ax, 8, 60, 35, 11, "ISF renders on the mixer's GL device", fc=PANEL, tsize=8.6)
    _box(ax, 8, 44, 35, 11, "final texture (Y-flip, BGRA-labelled)", fc=PANEL, tsize=8.6)
    _box(ax, 8, 28, 35, 11, "texture-backed frame", "no copy", fc=GL, ec=ACCENT_HOVER,
         tcolor="#ffffff", tsize=9)
    _box(ax, 8, 12, 35, 11, "mixer composites texture directly", fc="#0e1118", ec=GL,
         tcolor=TITLE, tsize=8.6)
    for y1, y2 in [(60, 55), (44, 39), (28, 23)]:
        _arrow(ax, 25.5, y1, 25.5, y2, color=GL)

    # Vulkan column
    _panel(ax, 52, 8, 45, 74, fc="#332b22", ec=WARNING, radius=1.6, z=1)
    _text(ax, 74.5, 76, "Vulkan mixer  ·  self-contained GL", color="#e0b070", size=11,
          ha="center", weight="bold")
    _box(ax, 57, 60, 35, 11, "ISF renders on its own SFML GL context", fc=PANEL, tsize=8.2)
    _box(ax, 57, 44, 35, 11, "read back  →  top-down BGRA (CPU)", fc=PANEL, tsize=8.4)
    _box(ax, 57, 28, 35, 11, "frame_factory create_frame", "CPU upload", fc=VK, ec=WARNING,
         tcolor="#ffffff", tsize=8.8)
    _box(ax, 57, 12, 35, 11, "Vulkan mixer composites the frame", fc="#0e1118", ec=WARNING,
         tcolor="#e0b070", tsize=8.4)
    for y1, y2 in [(60, 55), (44, 39), (28, 23)]:
        _arrow(ax, 74.5, y1, 74.5, y2, color=WARNING)

    _save(fig, "isf_mixer_paths.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ISF file anatomy
# ─────────────────────────────────────────────────────────────────────────────
def isf_anatomy():
    fig, ax = _new((11, 6.4))
    _title(ax, "Anatomy of an ISF file")

    # JSON header
    _panel(ax, 3, 26, 45, 56, fc="#232a2f", ec=BORDER, radius=1.6, z=1)
    _text(ax, 25.5, 77, "/*{  JSON header  }*/", color=TITLE, size=10.5, ha="center",
          weight="bold", family="DejaVu Sans Mono")
    header_lines = [
        ('"DESCRIPTION"', "human-readable label"),
        ('"CATEGORIES"', "grouping"),
        ('"INPUTS"', "float / bool / long / color / point2D / image"),
        ('"PASSES"', "TARGET · PERSISTENT · FLOAT · WIDTH/HEIGHT"),
        ('"IMPORTED"', "external image files"),
    ]
    y = 69
    for key, desc in header_lines:
        _text(ax, 6, y, key, color="#c586c0", size=9.5, family="DejaVu Sans Mono")
        _text(ax, 6, y - 3.0, desc, color=MUTED, size=7.6)
        y -= 8.2

    # GLSL body
    _panel(ax, 52, 26, 45, 56, fc="#22262a", ec=BORDER, radius=1.6, z=1)
    _text(ax, 74.5, 77, "GLSL fragment body", color=TITLE, size=10.5, ha="center",
          weight="bold")
    body = [
        "// auto-declared (do not redeclare):",
        "uniform vec2  RENDERSIZE;",
        "uniform float TIME, TIMEDELTA;",
        "uniform int   FRAMEINDEX, PASSINDEX;",
        "in      vec2  isf_FragNormCoord;",
        "",
        "void main() {",
        "  vec4 c = IMG_THIS_PIXEL(inputImage);",
        "  gl_FragColor = c * gain;",
        "}",
    ]
    y = 71
    for line in body:
        col = MUTED if line.strip().startswith("//") else TEXT
        _text(ax, 55, y, line, color=col, size=8.2, family="DejaVu Sans Mono")
        y -= 4.2

    # mapping arrow
    _arrow(ax, 48, 54, 52, 54, color=SUCCESS, lw=2.2)
    _text(ax, 50, 22, "header inputs  →  auto-declared uniforms  ·  IMG_* macros sample any image",
          color=MUTED, size=8.5, ha="center", style="italic")

    _save(fig, "isf_anatomy.png")


def main() -> None:
    isf_modes()
    isf_pipeline()
    isf_passes()
    isf_mixer_paths()
    isf_anatomy()


if __name__ == "__main__":
    main()
