"""Figures for the executive brief (`docs/EXECUTIVE_BRIEF.html`).

DIFFERENT AUDIENCE, SAME VISUAL LANGUAGE. The palette, helpers and `layout_check.Layout` all
follow `generate_diagrams.py`, so these sit beside the existing 42 figures rather than
introducing a second look. What differs is what they show: these answer "what does this buy
us and when", where the others answer "how does it work".

WHY ONLY EIGHT FEATURE FIGURES. Most features in the brief already have a dark-themed diagram
in `docs/images/` -- ICVFX, LED tiling, OCIO stages, the decode and encode routes, projection
geometry, tracking, previz, grading. Drawing new ones for those would be a second copy of a
picture, which is the same duplication this doc tree keeps paying for. These eight are the
features that had no figure at all.

`replay()` and `audio()` were added last, and why they were missing is worth recording: the
brief's capability list was assembled from the diagram inventory rather than from
`docs/features/`, so four capabilities with no existing figure fell straight out -- replay,
PortAudio, LTC and remotewall, every one of them `Coverage: none`. Deriving a list of features
from a list of pictures selects for what was easy to illustrate.

Ten files, not eight: `exec_scope.png` is the scope overview a brief needs and the docs do not,
`exec_to_production.png` belongs to the closing argument rather than to a feature, and
`exec_cover_bg.png` is not a diagram at all -- it is a dithered wash, because the CSS gradient
it replaces banded when Chrome rasterised it for print. See `cover_bg()` for why that cannot be
fixed in CSS.

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
         ["GPU-direct playback", "GPU-direct recording", "CUDA ProRes + NotchLC",
          "HAP, 5 variants", "Instant replay"]),
        (75.8, "Reach", DANGER,
         ["Direct-to-LED output", "Multi-port SDI", "Multi-machine sync",
          "Lighting from content", "Facility audio + LTC"]),
    ]
    # Width 19.2 rather than 21: at 21 the fourth column ran to x=100.5 and its right border
    # was CLIPPED by the axis. `layout_check` tests overlaps, not bounds, so nothing caught it
    # -- only looking at the PNG did.
    for x, name, colour, items in cols:
        lay.panel(f"c{x}", x, 26, 19.2, 56, fc=PANEL, ec=colour, lw=1.6)
        lay.text(f"h{x}", x + 9.6, 77, name, parent=f"c{x}", color=colour, size=10,
                 weight="bold", ha="center")
        for i, it in enumerate(items):
            lay.text(f"i{x}{i}", x + 1.4, 69 - i * 8.6, "•  " + it, parent=f"c{x}",
                     color=TEXT, size=8.4)
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

    # Pitch 19.9 and width 16.6, so the fifth stage ends at 98.1. It used to start at x=91
    # with w=17.5 -- right edge 108.5 -- and matplotlib clipped the border while keeping the
    # labels, which is why it read as a design choice. `layout_check` now refuses it.
    stages = [(1.9, "Source", "camera log\nor HDR file"),
              (21.8, "Grade", "in scene-linear\nACES"),
              (41.7, "Encode", "PQ or HLG\ncurve applied"),
              (61.6, "Signal", "metadata on\nthe wire"),
              (81.5, "Display", "shows HDR\nas graded")]
    for i, (x, name, sub) in enumerate(stages):
        w = 16.6
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


# ─────────────────────────────────────────────────────────────────────────────
def cover_bg():
    """The cover wash, rendered as a dithered image rather than as a CSS gradient.

    WHY NOT CSS. A `radial-gradient` over a near-black background bands severely once Chrome
    rasterises it for print: the whole ramp spans about eight 8-bit steps across 300mm, so
    every step is a visible 35mm-wide stripe. Nothing about the CSS is wrong -- there is
    simply not enough bit depth to express that ramp smoothly, and no gradient syntax fixes
    an 8-bit output.

    So it is generated with per-pixel noise of about +/-1 code value, which is ordinary
    dithering: it converts the hard step edges into a fine grain the eye integrates. Costs a
    ~200 KB PNG and removes the artefact outright.
    """
    import numpy as np
    h, w = 900, 1280
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # centre near the upper-left third, matching where the headline sits
    cx, cy, rx, ry = 0.17 * w, 0.10 * h, 0.86 * w, 1.05 * h
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # smootherstep, so the ramp has no second-derivative discontinuity to band along either
    s = np.clip(d, 0.0, 1.0)
    s = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)

    near = np.array([0x22, 0x31, 0x48], dtype=np.float64)   # the tinted corner
    far = np.array([0x1e, 0x1e, 0x1e], dtype=np.float64)    # the page background, exactly
    img = near + (far - near) * s[..., None]

    rng = np.random.default_rng(20260827)                   # fixed: the PNG is committed
    img += rng.uniform(-1.1, 1.1, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(w / 130, h / 130), dpi=130)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    path = os.path.join(OUT_DIR, "exec_cover_bg.png")
    fig.savefig(path, dpi=130, facecolor=BG, pad_inches=0)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


# ─────────────────────────────────────────────────────────────────────────────
def to_production():
    """What stands between a capable server and a tool a production can be run on.

    The point of the figure is the PROPORTION: the server column is done and the four beside
    it are not, and none of the four is server work. A reader who takes only the shape away
    should take away that the remaining effort is not in the renderer.
    """
    lay, fig, ax = _new((13, 6.4))
    _head(lay, "Where the remaining effort actually sits",
          "not in the renderer — the four below are what turn built features into a service")

    lay.panel("srv", 3, 60, 30, 26, fc="#16301b", ec=SUCCESS_T, lw=1.7)
    lay.text("srvt", 18, 78, "The server", parent="srv", color=SUCCESS_T, size=11.5,
             weight="bold", ha="center")
    lay.text("srvs", 18, 71,
             "16 capabilities, 91 commands\n6 gated by measurement",
             parent="srv", color=TEXT, size=8.8, ha="center")
    lay.text("srvd", 18, 64.5, "built", parent="srv", color=SUCCESS_T, size=9,
             weight="bold", ha="center", style="italic")

    lay.panel("out", 67, 60, 30, 26, fc=PANEL, ec=ACCENT_HOVER, lw=1.7)
    lay.text("outt", 82, 78, "A production tool", parent="out", color=TITLE, size=11.5,
             weight="bold", ha="center")
    lay.text("outs", 82, 70,
             "something a crew can be\nhanded and be expected\nto deliver a show with",
             parent="out", color=TEXT, size=8.8, ha="center")

    lay.arrow((33, 73), (44, 73), color=BORDER, lw=1.3)
    lay.arrow((56, 73), (67, 73), color=BORDER, lw=1.3)
    lay.panel("gap", 44, 65, 12, 16, fc="#241f14", ec=WARNING, lw=1.4)
    lay.text("gapt", 50, 72.5, "?", parent="gap", color=WARNING_T, size=17,
             weight="bold", ha="center")

    # Pitch 23.5 against width 22.5. At width 24.1 on the same pitch the panels OVERLAPPED by
    # 0.8 and `layout_check` did not object -- it tests text against panels and arrows against
    # text, not panel against panel. Worth knowing before trusting it on a column layout.
    #
    # The state word under each is deliberately not "not started" for all four: the 360 client
    # exists and the docs exist, they are simply the wrong shape for this audience. Writing them
    # off as absent would be the same overstatement in the other direction.
    cols = [
        (3.5, "A client application", ACCENT_HOVER,
         "The 360 client proves the\ncommands work. It is lab\nwork, not an operator app.",
         "exists as lab work"),
        (27.0, "Machines that keep up", WARNING_T,
         "12K ProRes and many\nlayers set the spec, and\na show needs spares.",
         "not yet specified"),
        (50.5, "Guides and training", WARNING_T,
         "What we have is written\nfor engineers. Operators\nneed their own.",
         "engineer-facing only"),
        (74.0, "A way to run it live", DANGER_T,
         "Show files, presets, a\nbackup machine, and\nsomeone on call at 2am.",
         "not defined"),
    ]
    for x, name, colour, body, state in cols:
        lay.panel(f"c{x}", x, 10, 22.5, 40, fc=PANEL, ec=colour, lw=1.5)
        lay.text(f"ct{x}", x + 11.25, 43, name, parent=f"c{x}", color=colour, size=9.6,
                 weight="bold", ha="center")
        lay.text(f"cb{x}", x + 11.25, 28, body, parent=f"c{x}", color=TEXT, size=8.4,
                 ha="center")
        lay.text(f"cd{x}", x + 11.25, 15, state, parent=f"c{x}", color=MUTED,
                 size=8, ha="center", style="italic")

    lay.text("foot", 50, 4.5,
             "None of these four is server work — which is why none of them has been done alongside it.",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")
    lay.check(name="to_production")
    _save(fig, "exec_to_production.png")


# ─────────────────────────────────────────────────────────────────────────────
def replay():
    """Instant replay: playing back a recording that is still being written.

    `features/replay.md` §7 deliberately deferred a diagram, on the grounds that it would
    illustrate timing nothing measures. That reasoning holds for the feature document; it does
    not hold here, because the brief states readiness on every page and this one is marked
    unmeasured. The figure shows the STRUCTURE -- write head, read head, segment boundaries --
    and says on its face that the timing is not measured, so it claims no more than is known.
    """
    lay, fig, ax = _new((13, 6.4))
    _head(lay, "Reviewing a moment without stopping the record",
          "the read head follows the write head, so there is nothing to stop and nothing to reload")

    lay.panel("src", 3, 70, 20, 15, fc=PANEL, ec=BORDER)
    lay.text("srct", 13, 79.5, "Channel", parent="src", color=TEXT, size=9.5, weight="bold",
             ha="center")
    lay.text("srcs", 13, 74, "live output", parent="src", color=MUTED, size=8.2, ha="center")
    lay.arrow((23, 77.5), (31, 77.5), color=BORDER)
    lay.panel("rec", 31, 70, 22, 15, fc="#16301b", ec=SUCCESS_T, lw=1.5)
    lay.text("rect", 42, 79.5, "Recording", parent="rec", color=SUCCESS_T, size=9.5,
             weight="bold", ha="center")
    lay.text("recs", 42, 74, "continuous, never stops", parent="rec", color=TEXT, size=8.2,
             ha="center")

    # the store: segments, with the newest still open
    lay.panel("store", 3, 34, 94, 24, fc=PANEL, ec=BORDER_SUBTLE, blocking=False)
    lay.text("storet", 6, 54, "Segmented store", parent="store", color=TITLE, size=9.5,
             weight="bold")
    seg_w, seg_x0 = 11.4, 6.0
    for i in range(7):
        x = seg_x0 + i * (seg_w + 1.6)
        open_seg = i == 6
        lay.panel(f"sg{i}", x, 38, seg_w, 10,
                  fc="#22303f" if not open_seg else "#2c3a22",
                  ec=SUCCESS_T if open_seg else BORDER_SUBTLE, lw=1.5 if open_seg else 1.0)
        lay.text(f"sgl{i}", x + seg_w / 2, 43, "open" if open_seg else f"seg {i + 1}",
                 parent=f"sg{i}", color=SUCCESS_T if open_seg else MUTED, size=7.8,
                 ha="center")
    write_x = seg_x0 + 6 * (seg_w + 1.6) + seg_w / 2
    lay.arrow((write_x, 62), (write_x, 48.6), color=SUCCESS_T, lw=1.6)
    lay.text("wh", write_x, 65.5, "write head", parent=None, color=SUCCESS_T, size=8.4,
             ha="center", weight="bold")

    read_x = seg_x0 + 3 * (seg_w + 1.6) + seg_w / 2
    lay.arrow((read_x, 24), (read_x, 37.4), color=ACCENT_HOVER, lw=1.6)
    lay.panel("play", read_x - 16, 10, 32, 13, fc=PANEL, ec=ACCENT_HOVER, lw=1.5)
    lay.text("playt", read_x, 19, "Playback — any point, any speed", parent="play",
             color=TITLE, size=9, weight="bold", ha="center")
    lay.text("plays", read_x, 13.8, "LIVE mode tracks the write head", parent="play",
             color=TEXT, size=8.2, ha="center")

    lay.panel("exp", 70, 10, 27, 13, fc=PANEL, ec=WARNING, lw=1.4)
    lay.text("expt", 83.5, 19, "Export a highlight", parent="exp", color=WARNING_T, size=9,
             weight="bold", ha="center")
    lay.text("exps", 83.5, 13.8, "in and out points, to a file", parent="exp", color=TEXT,
             size=8.2, ha="center")
    lay.arrow((83.5, 37.4), (83.5, 23.6), color=WARNING_T, lw=1.4)

    lay.text("note", 50, 3.5,
             "Segment boundaries are what make reading an open recording safe. The timing of that "
             "is not yet measured.",
             parent=None, color=MUTED, size=8.6, ha="center", style="italic")
    lay.check(name="replay")
    _save(fig, "exec_replay.png")


# ─────────────────────────────────────────────────────────────────────────────
def audio():
    """Multi-channel audio out to pro interfaces, and house timecode in.

    ASIO IS DRAWN AMBER ON PURPOSE. The code is complete and correct, and `PA_USE_ASIO` is OFF
    in this build because the Steinberg SDK was not present at configure time -- verified:
    `ASIOSDK_ROOT_DIR-NOTFOUND` in CMakeCache and zero ASIO strings in casparcg.exe. Drawing it
    green would claim a route the shipped binary does not have.
    """
    lay, fig, ax = _new((13, 6.6))
    _head(lay, "The route in each direction",
          "green is available in the current build — amber needs a rebuild, not development")

    lay.panel("ch", 3, 58, 21, 26, fc=PANEL, ec=BORDER)
    lay.text("cht", 13.5, 79, "Channel audio", parent="ch", color=TEXT, size=9.5,
             weight="bold", ha="center")
    lay.text("chs", 13.5, 70,
             "as many as the device offers,\nmapped per output",
             parent="ch", color=MUTED, size=8.2, ha="center")

    lay.arrow((24, 73), (33, 73), color=BORDER)
    lay.panel("pa", 33, 58, 24, 26, fc="#22303f", ec=ACCENT_HOVER, lw=1.6)
    lay.text("pat", 45, 79, "PortAudio", parent="pa", color=TITLE, size=9.8,
             weight="bold", ha="center")
    apis = [("WASAPI", SUCCESS_T), ("DirectSound", SUCCESS_T), ("MME", SUCCESS_T),
            ("ASIO — needs a rebuild", WARNING_T)]
    for i, (nm, col) in enumerate(apis):
        lay.text(f"api{i}", 35.2, 73.6 - i * 3.6, "• " + nm, parent="pa", color=col, size=7.8)

    lay.arrow((57, 73), (66, 73), color=BORDER)
    lay.panel("dev", 66, 58, 31, 26, fc=PANEL, ec=BORDER, blocking=False)
    lay.text("devt", 81.5, 79, "Whatever the facility runs", parent="dev", color=TEXT,
             size=9.3, weight="bold", ha="center")
    for i, nm in enumerate(["Dante Virtual Soundcard", "MADI / RME, 64 channels",
                            "any USB or PCIe interface"]):
        lay.text(f"dv{i}", 68.2, 73.6 - i * 3.6, "• " + nm, parent="dev", color=MUTED,
                 size=7.9)

    # the timecode half
    lay.panel("ltc", 3, 22, 21, 24, fc=PANEL, ec=WARNING, lw=1.4)
    lay.text("ltct", 13.5, 40, "House timecode", parent="ltc", color=WARNING_T, size=9.5,
             weight="bold", ha="center")
    lay.text("ltcs", 13.5, 31,
             "LTC on an audio input,\nor the system clock",
             parent="ltc", color=MUTED, size=8.2, ha="center")
    lay.arrow((24, 34), (33, 34), color=WARNING_T)
    lay.panel("clk", 33, 22, 24, 24, fc="#241f14", ec=WARNING, lw=1.6)
    lay.text("clkt", 45, 40, "One clock", parent="clk", color=WARNING_T, size=9.8,
             weight="bold", ha="center")
    lay.text("clks", 45, 30.5,
             "process-wide, and it\nreports which source\nit is actually using",
             parent="clk", color=TEXT, size=8.2, ha="center")
    lay.arrow((57, 34), (66, 34), color=WARNING_T)
    lay.panel("use", 66, 22, 31, 24, fc=PANEL, ec=BORDER, blocking=False)
    lay.text("uset", 81.5, 40, "Three things already use it", parent="use", color=TEXT,
             size=9.3, weight="bold", ha="center")
    for i, nm in enumerate(["recordings stamped with it",
                            "tracking samples aligned to it",
                            "reported back on query"]):
        lay.text(f"us{i}", 68.2, 34.4 - i * 3.6, "• " + nm, parent="use", color=MUTED,
                 size=7.9)

    lay.text("note", 50, 9,
             "The audio consumer also becomes the channel's clock, which is how the picture stays "
             "locked to the audio device rather than drifting against it.",
             parent=None, color=TEXT, size=8.6, ha="center")
    lay.text("note2", 50, 3.5,
             "Neither half is covered by a test, and ASIO is not compiled into the current build.",
             parent=None, color=WARNING_T, size=8.4, ha="center", style="italic")
    lay.check(name="audio")
    _save(fig, "exec_audio.png")


if __name__ == "__main__":
    scope()
    direct_display()
    cluster()
    hdr()
    multiport()
    lighting()
    replay()
    audio()
    cover_bg()
    to_production()
