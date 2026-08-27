"""Figures for the executive brief (`docs/EXECUTIVE_BRIEF.html`).

DIFFERENT AUDIENCE, SAME VISUAL LANGUAGE. The palette, helpers and `layout_check.Layout` all
follow `generate_diagrams.py`, so these sit beside the existing 42 figures rather than
introducing a second look. What differs is what they show: these answer "what does this buy
us and when", where the others answer "how does it work".

WHY ONLY SIX. Most features in the brief already have a dark-themed diagram in
`docs/images/` -- ICVFX, LED tiling, OCIO stages, the decode and encode routes, projection
geometry, tracking, previz, grading. Drawing new ones for those would be a second copy of a
picture, which is the same duplication this doc tree keeps paying for. These six are the
features that had no figure at all, plus the scope overview a brief needs and the docs do
not.

Run:  python docs/diagrams/generate_exec_diagrams.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches   # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_check import Layout        # noqa: E402

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
DANGER = "#8a2a2a"
DANGER_T = "#c85a5a"
WARNING_T = "#d0a02a"
SUCCESS_T = "#6fbf6f"

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


def _head(lay, title, sub):
    lay.text("t", 50, 94, title, parent=None, color=TITLE, size=13.5, weight="bold",
             ha="center")
    lay.text("s", 50, 88.5, sub, parent=None, color=MUTED, size=9.5, ha="center",
             style="italic")


# ─────────────────────────────────────────────────────────────────────────────
def scope():
    """What the fork adds on top of the stock server. The brief's opening picture."""
    lay, fig, ax = _new((13, 7.0))
    _head(lay, "What this fork adds on top of stock CasparCG",
          "the base server is unchanged underneath — everything above is ours")

    lay.panel("base", 6, 8, 88, 13, fc=PANEL, ec=BORDER)
    lay.text("base", 50, 17, "CasparCG Server — playout, AMCP control, consumers",
             parent="base", color=TEXT, size=10.5, weight="bold", ha="center")
    lay.text("basesub", 50, 12, "upstream, and still upstream: we track it rather than diverge from it",
             parent="base", color=MUTED, size=8.5, ha="center", style="italic")

    cols = [
        (5.0, "Picture quality", SUCCESS,
         ["ACES + OpenColorIO", "26 grading operators", "LED wall calibration", "HDR end to end"]),
        (28.6, "Virtual production", ACCENT,
         ["ICVFX inner frustum", "Curved / dome warp", "Camera tracking", "3D pre-visualisation"]),
        (52.2, "Capacity", WARNING,
         ["GPU-direct playback", "GPU-direct recording", "CUDA ProRes + NotchLC", "HAP, 5 variants"]),
        (75.8, "Reach", DANGER,
         ["Direct-to-LED output", "Multi-port SDI", "Multi-machine sync", "Lighting from content"]),
    ]
    # Width 19.2 rather than 21: at 21 the fourth column ran to x=100.5 and its right border
    # was CLIPPED by the axis. `layout_check` tests overlaps, not bounds, so nothing caught it
    # -- only looking at the PNG did.
    for x, name, colour, items in cols:
        lay.panel(f"c{x}", x, 26, 19.2, 56, fc=PANEL, ec=colour, lw=1.6)
        lay.text(f"h{x}", x + 9.6, 77, name, parent=f"c{x}", color=colour, size=10,
                 weight="bold", ha="center")
        for i, it in enumerate(items):
            lay.text(f"i{x}{i}", x + 1.4, 69 - i * 9.5, "•  " + it, parent=f"c{x}",
                     color=TEXT, size=8.6)
        lay.arrow((x + 9.6, 26), (x + 9.6, 21.4), color=BORDER, lw=1.1)

    lay.check(name="scope")
    _save(fig, "exec_scope.png")


# ─────────────────────────────────────────────────────────────────────────────
def direct_display():
    """Feeding an LED processor over HDMI/DP instead of through an SDI chain."""
    lay, fig, ax = _new((13, 6.4))
    _head(lay, "Reaching the LED processor without an SDI chain",
          "fewer boxes between the render and the wall — and fewer places to lose the picture")

    lay.text("wayA", 6, 74, "Conventional", parent=None, color=WARNING_T, size=10.5,
             weight="bold")
    boxes_a = [(6, "Render"), (26, "SDI card"), (46, "Converter"), (66, "Processor"), (86, "Wall")]
    for x, name in boxes_a:
        lay.panel(f"a{x}", x, 55, 13, 13, fc=PANEL, ec=BORDER)
        lay.text(f"al{x}", x + 6.5, 61.5, name, parent=f"a{x}", color=TEXT, size=9,
                 ha="center")
    for i in range(len(boxes_a) - 1):
        lay.arrow((boxes_a[i][0] + 13, 61.5), (boxes_a[i + 1][0], 61.5), color=WARNING_T)

    lay.text("wayB", 6, 40, "This fork", parent=None, color=SUCCESS_T, size=10.5,
             weight="bold")
    boxes_b = [(6, "Render"), (40, "Processor"), (74, "Wall")]
    for x, name in boxes_b:
        lay.panel(f"b{x}", x, 21, 20, 13, fc=PANEL, ec=SUCCESS, lw=1.6)
        lay.text(f"bl{x}", x + 10, 27.5, name, parent=f"b{x}", color=TEXT, size=9.5,
                 ha="center", weight="bold")
    for i in range(len(boxes_b) - 1):
        lay.arrow((boxes_b[i][0] + 20, 27.5), (boxes_b[i + 1][0], 27.5), color=SUCCESS_T,
                  lw=1.8)
    lay.text("dp", 32, 31.5, "HDMI / DP", parent=None, color=SUCCESS_T, size=8,
             ha="center")
    lay.text("note", 50, 10,
             "The saving is not only cost. Each conversion is a place where colour, range or "
             "HDR signalling can be silently changed.",
             parent=None, color=MUTED, size=9, ha="center", style="italic")
    lay.check(name="direct_display")
    _save(fig, "exec_direct_display.png")


# ─────────────────────────────────────────────────────────────────────────────
def cluster():
    """One rundown, several machines, frame-locked."""
    lay, fig, ax = _new((13, 6.6))
    _head(lay, "One show, several machines, one frame number",
          "a wall too wide for one server stops being a special case")

    lay.panel("master", 36, 68, 28, 15, fc=ACCENT, ec=ACCENT_HOVER, lw=1.6)
    lay.text("master", 50, 78.5, "Master", parent="master", color="#ffffff", size=10.5,
             weight="bold", ha="center")
    lay.text("master2", 50, 72.5, "stamps every command with a target frame",
             parent="master", color="#dce8fa", size=8.2, ha="center")

    for i, x in enumerate((8, 30.5, 53, 75.5)):
        lay.panel(f"n{i}", x, 34, 17, 16, fc=PANEL, ec=BORDER)
        lay.text(f"nl{i}", x + 8.5, 44, f"Node {i + 1}", parent=f"n{i}", color=TEXT,
                 size=9.5, ha="center", weight="bold")
        lay.text(f"ns{i}", x + 8.5, 38.5, "its own slice", parent=f"n{i}", color=MUTED,
                 size=8, ha="center")
        lay.arrow((50, 68), (x + 8.5, 50.4), color=ACCENT_HOVER, lw=1.2, rad=-0.12)
        lay.panel(f"w{i}", x, 15, 17, 9, fc="#123", ec=BORDER_SUBTLE)
        lay.text(f"wl{i}", x + 8.5, 19.5, "wall section", parent=f"w{i}", color=MUTED,
                 size=8, ha="center")
        lay.arrow((x + 8.5, 34), (x + 8.5, 24.4), color=BORDER, lw=1.0)

    # The caption sat at y=57.5, in the middle of the arrow fan, and `layout_check` refused it
    # -- four arrows straight through the text. Below the wall row is the only clear band.
    lay.text("ptp", 50, 9.5,
             "PTP keeps the clocks together; the frame number is agreed, not guessed",
             parent=None, color=SUCCESS_T, size=9, ha="center")
    lay.text("note", 50, 4.5,
             "Not yet exercised on real hardware — it needs two machines, which is the single "
             "cheapest thing on the readiness list.",
             parent=None, color=WARNING_T, size=8.8, ha="center", style="italic")
    lay.check(name="cluster")
    _save(fig, "exec_cluster.png")


# ─────────────────────────────────────────────────────────────────────────────
def hdr():
    """HDR that survives to the wire, not just in the render."""
    lay, fig, ax = _new((13, 6.2))
    _head(lay, "HDR that survives all the way to the wire",
          "the picture and the label it carries are decided separately — both have to be right")

    stages = [(5, "Source", "camera log\nor HDR file"),
              (26.5, "Grade", "in scene-linear\nACES"),
              (48, "Encode", "PQ or HLG\ncurve applied"),
              (69.5, "Signal", "metadata on\nthe wire"),
              (91, "Display", "shows HDR\nas graded")]
    for i, (x, name, sub) in enumerate(stages):
        w = 17.5
        colour = SUCCESS if i in (2, 3) else BORDER
        lay.panel(f"s{i}", x, 46, w, 24, fc=PANEL, ec=colour,
                  lw=1.6 if colour is SUCCESS else 1.2)
        lay.text(f"sn{i}", x + w / 2, 64, name, parent=f"s{i}", color=TEXT, size=10,
                 weight="bold", ha="center")
        lay.text(f"ss{i}", x + w / 2, 54, sub, parent=f"s{i}", color=MUTED, size=8.2,
                 ha="center")
        if i:
            lay.arrow((stages[i - 1][0] + w, 58), (x, 58), color=MUTED)

    lay.panel("warn", 12, 14, 76, 20, fc="#241f14", ec=WARNING, lw=1.4)
    lay.text("wt", 50, 28, "The failure nobody notices", parent="warn", color=WARNING_T,
             size=10, weight="bold", ha="center")
    lay.text("wb", 50, 20.5,
             "A correct HDR picture carrying an SDR label looks perfect on the bench and wrong\n"
             "on delivery. Both halves are now measured, on the card and in the file.",
             parent="warn", color=TEXT, size=8.8, ha="center")
    lay.check(name="hdr")
    _save(fig, "exec_hdr.png")


# ─────────────────────────────────────────────────────────────────────────────
def multiport():
    """A canvas wider than one SDI link."""
    lay, fig, ax = _new((13, 6.2))
    _head(lay, "A canvas wider than one cable",
          "one channel, several outputs, synchronised by the card rather than by luck")

    lay.panel("canvas", 8, 60, 84, 20, fc=PANEL, ec=ACCENT, lw=1.6)
    lay.text("cl", 50, 75, "one channel — the full wall as a single picture", parent="canvas",
             color=TITLE, size=10, weight="bold", ha="center")
    for i in range(4):
        x = 10 + i * 20.5
        lay.panel(f"sl{i}", x, 62, 19, 9, fc="#25303f", ec=BORDER_SUBTLE)
        lay.text(f"slt{i}", x + 9.5, 66.5, f"slice {i + 1}", parent=f"sl{i}", color=MUTED,
                 size=8, ha="center")

    for i in range(4):
        x = 10 + i * 20.5
        lay.panel(f"p{i}", x, 30, 19, 13, fc=PANEL, ec=BORDER)
        lay.text(f"pt{i}", x + 9.5, 39, f"SDI {i + 1}", parent=f"p{i}", color=TEXT, size=9.5,
                 ha="center", weight="bold")
        lay.text(f"ps{i}", x + 9.5, 33.5, "own crop", parent=f"p{i}", color=MUTED, size=8,
                 ha="center")
        lay.arrow((x + 9.5, 60), (x + 9.5, 43.4), color=ACCENT_HOVER, lw=1.2)

    lay.text("grp", 50, 21,
             "The driver runs them as one playback group, so the seam does not tear",
             parent=None, color=SUCCESS_T, size=9, ha="center")
    lay.text("note", 50, 10,
             "A port can carry the key channel instead of a slice, which is how fill-and-key "
             "reaches a downstream keyer without a second machine.",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")
    lay.check(name="multiport")
    _save(fig, "exec_multiport.png")


# ─────────────────────────────────────────────────────────────────────────────
def lighting():
    """House lighting driven from the picture on the wall."""
    lay, fig, ax = _new((13, 6.0))
    _head(lay, "Lighting that follows the content",
          "the wall is already the biggest light in the room — this makes the rest agree with it")

    lay.panel("content", 6, 52, 26, 24, fc=PANEL, ec=ACCENT, lw=1.6)
    lay.text("cl", 19, 70, "Content on the wall", parent="content", color=TITLE, size=9.5,
             weight="bold", ha="center")
    for i, c in enumerate(("#8a4a2a", "#2a5a8a", "#7a7a2a")):
        lay.panel(f"sw{i}", 9 + i * 7.6, 56, 6.6, 8, fc=c, ec=BORDER_SUBTLE)

    lay.panel("sample", 40, 52, 22, 24, fc=PANEL, ec=BORDER)
    lay.text("sl", 51, 70, "Sampled regions", parent="sample", color=TEXT, size=9.5,
             weight="bold", ha="center")
    lay.text("ss", 51, 61, "per fixture,\nfrom the picture itself", parent="sample",
             color=MUTED, size=8.2, ha="center")

    lay.panel("dmx", 70, 52, 24, 24, fc=PANEL, ec=SUCCESS, lw=1.6)
    lay.text("dl", 82, 70, "Art-Net / sACN", parent="dmx", color=SUCCESS_T, size=9.5,
             weight="bold", ha="center")
    lay.text("ds", 82, 61, "standard lighting\nprotocols, no bridge", parent="dmx",
             color=MUTED, size=8.2, ha="center")

    lay.arrow((32, 64), (40, 64), color=MUTED)
    lay.arrow((62, 64), (70, 64), color=SUCCESS_T)

    lay.text("note", 50, 33,
             "Practicals, cyc lights and moving heads take their colour from the same frame the\n"
             "camera sees, so a change to the content does not need a second operator to follow it.",
             parent=None, color=TEXT, size=9.2, ha="center")
    lay.text("note2", 50, 15,
             "Both transports are the ones lighting desks already speak — nothing bespoke to "
             "install at the other end.",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")
    lay.check(name="lighting")
    _save(fig, "exec_lighting.png")


if __name__ == "__main__":
    scope()
    direct_display()
    cluster()
    hdr()
    multiport()
    lighting()
