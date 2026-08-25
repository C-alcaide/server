"""Generate the diagrams for docs/GSTREAMER_GUIDE.md.

Run from the repo root:  python docs/diagrams/generate_gstreamer_diagrams.py

Outputs PNGs into docs/images/. The images are committed so the Markdown renders with no
build step; re-run this after changing a diagram. Palette and helpers deliberately match
generate_diagrams.py so the guide looks like the rest of the manual set.

Two diagrams, and each earns its place under the rule that a diagram is for a pipeline whose
ORDER is the point or two paths reaching the same place:

  gstreamer_pipeline_anatomy.png — what a pipeline IS, for a reader who knows ffmpeg's
      flag-based command line and has never met an element graph. This is the one concept
      everything else in the guide depends on.

  gstreamer_caspar_routes.png — where CasparCG joins that graph, and the three routes a
      picture can take to the mixer. The position of the colour conversion is the whole point
      and prose keeps having to repeat it.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── App palette (ui_kit.COLORS), same as generate_diagrams.py ────────────────
BG = "#1e1e1e"
PANEL = "#2d2d2d"
HOVER = "#3c3c3c"
BORDER = "#555555"
BORDER_SUBTLE = "#444444"
TEXT = "#d4d4d4"
MUTED = "#888888"
TITLE = "#9cdcfe"
ACCENT = "#2255aa"
SUCCESS = "#1a6b1a"
WARNING = "#7a5c00"
DANGER = "#8a2a2a"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


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


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def _text(ax, x, y, s, *, color=TEXT, size=9, weight="normal", ha="center", va="center",
          style="normal", z=5, family="DejaVu Sans"):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z, family=family)


def _arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=1.4, z=3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                 linewidth=lw, mutation_scale=11, zorder=z,
                                 shrinkA=1, shrinkB=1))


def _element(ax, x, y, w, h, name, subtitle=None, *, fc=PANEL, ec=BORDER):
    """One GStreamer element: a box with pads drawn on it, which is the whole idea."""
    _panel(ax, x, y, w, h, fc=fc, ec=ec)
    _text(ax, x + w / 2, y + h / 2 + (2.2 if subtitle else 0), name,
          size=9, weight="bold", family="DejaVu Sans Mono", color=TEXT)
    if subtitle:
        _text(ax, x + w / 2, y + h / 2 - 3.0, subtitle, size=7.2, color=MUTED, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 1. What a pipeline is
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_anatomy():
    fig, ax = _new((12.4, 6.4))

    _text(ax, 50, 96, "A GStreamer pipeline is a graph of elements joined by pads",
          size=13, weight="bold", color=TITLE)
    _text(ax, 50, 90.5, "ffmpeg takes flags that describe a conversion.  GStreamer takes a "
                        "chain of parts that each do one thing.",
          size=9, color=MUTED, style="italic")

    y, h = 62, 15
    boxes = [
        (4.0, 17.0, "filesrc", "reads bytes"),
        (25.0, 17.0, "qtdemux", "splits the container"),
        (46.0, 17.0, "h264parse", "frames the stream"),
        (67.0, 17.0, "avdec_h264", "decodes"),
    ]
    for x, w, name, sub in boxes:
        _element(ax, x, y, w, h, name, sub)

    # the sink CasparCG supplies
    _panel(ax, 88.0, y, 10.0, h, fc=HOVER, ec=ACCENT, lw=1.6)
    _text(ax, 93, y + h / 2 + 2.4, "appsink", size=8.4, weight="bold",
          family="DejaVu Sans Mono")
    _text(ax, 93, y + h / 2 - 3.0, "CasparCG", size=7.0, color=TITLE)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][1]
        _arrow(ax, x1, y + h / 2, boxes[i + 1][0], y + h / 2)
        _text(ax, (x1 + boxes[i + 1][0]) / 2, y + h / 2 + 4.6, "!", size=11,
              color=ACCENT, weight="bold", family="DejaVu Sans Mono")
    _arrow(ax, boxes[-1][0] + boxes[-1][1], y + h / 2, 88.0, y + h / 2, color=ACCENT)

    # pads
    _text(ax, 21.5, y - 4.5, "src pad", size=6.8, color=MUTED)
    _text(ax, 25.5, y - 4.5, "sink pad", size=6.8, color=MUTED, ha="left")
    _arrow(ax, 22.0, y - 2.6, 22.0, y + 5.5, color=BORDER, lw=0.9, style="-")
    _arrow(ax, 27.5, y - 2.6, 27.5, y + 5.5, color=BORDER, lw=0.9, style="-")

    _text(ax, 50, 50,
          '"!"  is the link.  Read it as "then".   Every box has a sink pad in and a src pad out.',
          size=9, color=TEXT)

    # the ffmpeg comparison
    _panel(ax, 4, 8, 92, 33, fc="#252525", ec=BORDER_SUBTLE)
    _text(ax, 8, 36, "The same job, both ways", size=10, weight="bold", color=TITLE, ha="left")

    _text(ax, 8, 29.5, "ffmpeg", size=9, weight="bold", color=MUTED, ha="left")
    _text(ax, 20, 29.5, "ffmpeg -i clip.mp4 -f rawvideo -", size=9, ha="left",
          family="DejaVu Sans Mono", color=TEXT)
    _text(ax, 20, 24.5, "one command, flags describe the whole conversion", size=7.6,
          ha="left", color=MUTED, style="italic")

    _text(ax, 8, 17.5, "GStreamer", size=9, weight="bold", color=MUTED, ha="left")
    _text(ax, 20, 17.5, "filesrc location=clip.mp4 ! qtdemux ! h264parse ! avdec_h264",
          size=9, ha="left", family="DejaVu Sans Mono", color=TEXT)
    _text(ax, 20, 12.5, "a chain you assemble; each part is replaceable on its own",
          size=7.6, ha="left", color=MUTED, style="italic")

    _save(fig, "gstreamer_pipeline_anatomy.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. The three routes into the mixer
# ─────────────────────────────────────────────────────────────────────────────
def caspar_routes():
    fig, ax = _new((12.4, 8.6))

    _text(ax, 50, 96.5, "Three routes from a decoder to the CasparCG mixer",
          size=13, weight="bold", color=TITLE)
    _text(ax, 50, 91.5, "They differ only in where the picture lives on the way. "
                        "The colour conversion is the mixer's, in every one of them.",
          size=9, color=MUTED, style="italic")

    _element(ax, 3, 54, 16, 14, "decoder", "your pipeline")

    rows = [
        (80, "software decode", "avdec_h264, vpxdec …", SUCCESS,
         "system memory  →  appsink  →  host upload"),
        (61, "hardware, host", "d3d11h264dec ! d3d11download", WARNING,
         "GPU decode, downloaded, uploaded again"),
        (42, "hardware, GPU-direct", "d3d11h264dec  +  PLAY … GPU", ACCENT,
         "stays in video memory; NV12/P010 planes"),
    ]
    for y, title, how, colour, note in rows:
        _panel(ax, 26, y - 8, 40, 16, fc=PANEL, ec=colour, lw=1.6)
        _text(ax, 46, y + 3.2, title, size=9.4, weight="bold", color=TEXT)
        _text(ax, 46, y - 1.4, how, size=7.8, color=TITLE, family="DejaVu Sans Mono")
        _text(ax, 46, y - 5.4, note, size=7.0, color=MUTED, style="italic")
        _arrow(ax, 19, 61, 26, y, color=colour)
        _arrow(ax, 66, y, 76, 61, color=colour)

    _panel(ax, 76, 48, 21, 26, fc=HOVER, ec=TITLE, lw=1.6)
    _text(ax, 86.5, 68, "CasparCG mixer", size=9.6, weight="bold", color=TEXT)
    _text(ax, 86.5, 62, "ycbcra_to_rgba", size=8.2, color=TITLE, family="DejaVu Sans Mono")
    _text(ax, 86.5, 57.5, "the ONLY colour", size=7.2, color=MUTED, style="italic")
    _text(ax, 86.5, 54, "conversion", size=7.2, color=MUTED, style="italic")

    _panel(ax, 4, 3, 93, 24, fc="#252525", ec="#444444")
    _text(ax, 8, 22, "Why this matters when you write a pipeline", size=10,
          weight="bold", color=TITLE, ha="left")
    for i, line in enumerate([
        "Do NOT put videoconvert in front of the sink to \"help\" — it converts, and the mixer "
        "then has nothing left to do correctly.",
        "The GPU route is opt-in per PLAY because a source that cannot reach video memory would "
        "pay an upload for nothing.",
        "It falls back to the host route on its own, once, with the reason in the log. A silent "
        "fallback would look identical to success.",
    ]):
        _text(ax, 8, 16 - i * 4.6, "•  " + line, size=8.0, color=TEXT, ha="left")

    _save(fig, "gstreamer_caspar_routes.png")


if __name__ == "__main__":
    pipeline_anatomy()
    caspar_routes()
