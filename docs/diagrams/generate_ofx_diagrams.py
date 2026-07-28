"""Generate documentation diagrams for the CasparCG OpenFX (OFX) guides.

Run from the repo root:  python docs/diagrams/generate_ofx_diagrams.py

Outputs PNGs into docs/images/. Re-run after changing a diagram; the images are
committed so the Markdown renders without a build step. Dependency-light
(matplotlib + numpy) and themed to match the CasparCG docs palette.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
CUDA = "#3a7a2a"
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
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


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


def _arrow(ax, x1, y1, x2, y2, *, color=TITLE, lw=2.0, style="-|>", z=5, mut=13,
           ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=mut, lw=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0, linestyle=ls))


def _title(ax, s, sub=None):
    _text(ax, 50, 96, s, color=TITLE, size=14, weight="bold", ha="center")
    if sub:
        _text(ax, 50, 91.5, sub, color=MUTED, size=9.5, ha="center", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Host component map
# ─────────────────────────────────────────────────────────────────────────────
def ofx_components():
    fig, ax = _new((12, 7.0))
    _title(ax, "OFX host — component map",
           "an in-process host built on the OpenFX HostSupport library")

    # Producer (left)
    _box(ax, 2, 42, 15, 16, "[OFX]\nproducer", "wraps a source", fc=ACCENT,
         ec=ACCENT_HOVER, tcolor="#ffffff", tsize=10)

    # Host container (middle)
    _panel(ax, 21, 12, 45, 74, fc="#242a30", ec=BORDER, radius=1.8, z=1)
    _text(ax, 43.5, 81, "OFX host  (in-process)", color=TITLE, size=11, ha="center",
          weight="bold")
    _box(ax, 24, 66, 39, 11, "plug-in cache / scan", "discover .ofx bundles", fc=PANEL, tsize=9)
    _box(ax, 24, 52, 39, 11, "effect instance", "createInstance · renderAction", fc=PANEL, tsize=9)
    _box(ax, 24, 38, 18, 11, "clip instances", "Source / Output", fc=PANEL, tsize=8.5)
    _box(ax, 45, 38, 18, 11, "param instances", "double/int/bool/…", fc=PANEL, tsize=8.5)
    _box(ax, 24, 24, 39, 10, "stability layer", "try/catch (SEH) · blocklist", fc="#332b22",
         ec=WARNING, tsize=8.6)
    _text(ax, 43.5, 17.5, "openfx_host  (HostSupport lib)  +  the .ofx plug-in",
          color=MUTED, size=8.4, ha="center", style="italic")

    _arrow(ax, 17, 50, 24, 57.5)

    # Backends (right)
    _panel(ax, 70, 12, 28, 74, fc="#20262c", ec=BORDER, radius=1.8, z=1)
    _text(ax, 84, 81, "render backends", color=TITLE, size=11, ha="center", weight="bold")
    _box(ax, 73, 62, 22, 13, "CPU", "always available", fc=PANEL, tsize=9.5)
    _box(ax, 73, 44, 22, 13, "OpenGL", "SFML / zero-copy", fc=GL, ec=ACCENT_HOVER,
         tcolor="#ffffff", tsize=9.5)
    _box(ax, 73, 26, 22, 13, "CUDA", "device buffers", fc=CUDA, ec=SUCCESS,
         tcolor="#ffffff", tsize=9.5)
    _arrow(ax, 63, 57.5, 73, 68, color=MUTED)
    _arrow(ax, 63, 57.5, 73, 50, color=GL)
    _arrow(ax, 63, 57.5, 73, 32, color=CUDA)

    _save(fig, "ofx_components.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frame & data flow — the three render paths
# ─────────────────────────────────────────────────────────────────────────────
def ofx_dataflow():
    fig, ax = _new((12.5, 8.0))
    _title(ax, "OFX render paths", "the host picks the best available backend per plug-in")

    def lane(y, tag, col, boxes):
        _text(ax, 1.5, y + 7, tag, color=col, size=9.5, weight="bold")
        x = 14
        prev = None
        for (name, sub, fc, ec, tc) in boxes:
            w = 15.4
            _box(ax, x, y, w, 14, name, sub, fc=fc, ec=ec, tcolor=tc, tsize=8.2)
            if prev is not None:
                _arrow(ax, prev, y + 7, x, y + 7, color=col)
            prev = x + w
            x += 17.4

    # CPU lane
    lane(64, "CPU\n(any mixer)", MUTED, [
        ("source", "BGRA frame", HOVER, BORDER, TEXT),
        ("bridge", "swizzle + Y-flip", PANEL, BORDER, TEXT),
        ("renderAction", "RGBA (CPU)", PANEL, BORDER, TEXT),
        ("bridge back", "→ BGRA", PANEL, BORDER, TEXT),
        ("output frame", "CPU", "#0e1118", BORDER, TITLE),
    ])

    # OpenGL zero-copy lane
    lane(40, "OpenGL\nOGL mixer", GL, [
        ("source", "→ RGBA up", HOVER, GL, TEXT),
        ("upload tex", "mixer GL", PANEL, GL, TEXT),
        ("GL render", "into mixer tex", GL, ACCENT_HOVER, "#ffffff"),
        ("Y-flip", "BGRA label", PANEL, GL, TEXT),
        ("texture frame", "no read-back", "#0e1118", GL, TITLE),
    ])

    # CUDA zero-copy lane
    lane(16, "CUDA\nVulkan mixer", CUDA, [
        ("source", "→ CUDA buf", HOVER, CUDA, TEXT),
        ("render_cuda", "device buffer", CUDA, SUCCESS, "#ffffff"),
        ("cudaMemcpy2D", "→ VK array", PANEL, CUDA, TEXT),
        ("exportable VK", "CudaVkTexture", PANEL, CUDA, TEXT),
        ("texture frame", "no read-back", "#0e1118", CUDA, TITLE),
    ])

    _save(fig, "ofx_dataflow.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backend negotiation & per-plug-in gating
# ─────────────────────────────────────────────────────────────────────────────
def ofx_backends():
    fig, ax = _new((11, 6.4))
    _title(ax, "Backend selection & per-plug-in gating",
           "a backend is used only if the host enables it AND the plug-in advertises it")

    # Gating AND
    _box(ax, 4, 66, 26, 13, "host flag enabled?", "config / env opt-in", fc=PANEL, tsize=9)
    _box(ax, 4, 48, 26, 13, "plug-in advertises?", "descriptor property", fc=PANEL, tsize=9)
    _box(ax, 38, 57, 14, 13, "AND", None, fc=WARNING, ec=BORDER, tcolor="#ffffff", tsize=11)
    _arrow(ax, 30, 72.5, 38, 66, color=MUTED)
    _arrow(ax, 30, 54.5, 38, 61, color=MUTED)

    # Priority ladder
    _text(ax, 78, 82, "priority (first that qualifies)", color=MUTED, size=9, ha="center",
          style="italic")
    _box(ax, 60, 66, 36, 12, "CUDA  (Vulkan mixer · 8-bit · cuda_capable)", fc=CUDA,
         ec=SUCCESS, tcolor="#ffffff", tsize=8.6)
    _box(ax, 60, 51, 36, 12, "OpenGL  (OGL mixer · 8-bit · gl_capable)", fc=GL,
         ec=ACCENT_HOVER, tcolor="#ffffff", tsize=8.6)
    _box(ax, 60, 36, 36, 12, "CPU  (always · 8/16-bit)", fc=PANEL, tsize=8.6)
    _arrow(ax, 52, 63.5, 60, 72, color=CUDA)
    _arrow(ax, 78, 66, 78, 63, color=MUTED, ls=(0, (2, 2)), lw=1.4)
    _arrow(ax, 78, 51, 78, 48, color=MUTED, ls=(0, (2, 2)), lw=1.4)
    _text(ax, 72, 64.5, "else", color=MUTED, size=7.4, ha="right", style="italic")
    _text(ax, 72, 49.5, "else", color=MUTED, size=7.4, ha="right", style="italic")

    # Note
    _panel(ax, 6, 12, 90, 14, fc="#332b22", ec=WARNING, radius=1.4, z=1)
    _text(ax, 51, 22, "A CPU-only plug-in handed a texture/device pointer it can't use would crash "
          "or corrupt output —", color=TEXT, size=8.6, ha="center")
    _text(ax, 51, 16.5, "so the host never enables a GPU path unless the plug-in's descriptor "
          "declares support for it.", color=MUTED, size=8.4, ha="center", style="italic")

    _save(fig, "ofx_backends.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Producer modes
# ─────────────────────────────────────────────────────────────────────────────
def ofx_modes():
    fig, ax = _new((11, 7.4))
    _title(ax, "OFX producer modes",
           "AMCP command  →  [OFX] producer  →  channel output")

    rows = [
        (70, "Filter", '[OFX] "id" <source>', ACCENT,
         [("source\nproducer", HOVER), ("Source clip", PANEL)]),
        (44, "Generator", '[OFX] "id"', SUCCESS,
         [("plug-in\ngenerates", PANEL)]),
        (18, "Transition", '[OFX] "id" TRANSITION <from> <to> [frames]', WARNING,
         [("from / to\nproducers", HOVER), ("SourceFrom\nSourceTo\n+ Transition", PANEL)]),
    ]

    for y, label, cmd, col, chain in rows:
        h = 15
        _box(ax, 2, y, 15, h, label, fc=col, ec=BORDER, tcolor="#ffffff", tsize=11)
        _text(ax, 2, y - 3.2, cmd, color=MUTED, size=8.0, ha="left",
              family="DejaVu Sans Mono")

        x = 21
        prev_cx = 17
        for (name, fc) in chain:
            _arrow(ax, prev_cx, y + h / 2, x, y + h / 2)
            _box(ax, x, y, 21, h, name, fc=fc, ec=BORDER, tsize=9.0)
            prev_cx = x + 21
            x += 27
        _arrow(ax, prev_cx, y + h / 2, x, y + h / 2)
        _box(ax, x, y, 18, h, "output\nframe", fc="#0e1118", ec=GL, tcolor=TITLE, tsize=9.2)

    _save(fig, "ofx_modes.png")


def main() -> None:
    ofx_components()
    ofx_dataflow()
    ofx_backends()
    ofx_modes()


if __name__ == "__main__":
    main()
