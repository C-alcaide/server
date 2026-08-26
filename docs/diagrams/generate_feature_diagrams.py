"""Figures for `docs/features/` — the diagrams those documents record as owed.

Palette, helpers and `layout_check.Layout` all follow `generate_diagrams.py` and
`generate_recording_diagrams.py`, so this is the same visual language rather than a new one.

WHICH FIGURES, AND WHY EACH EARNS ONE. `CLAUDE.md` sets a deliberately high bar: a feature gets a
diagram when its ORDER is the point, when TWO PATHS reach the same place, or when a stage's
POSITION is the whole feature. These three meet it and the documents say so:

  * `feature_codec_handoff.png` -- three GPU codec producers hand the mixer three DIFFERENT things,
    each for a defensible reason and none of them a house style. Two paths reaching one place, and
    the single most misleading thing to assume about this fork.

  * `feature_notchlc_pipeline.png` -- ten steps with a HOST ROUND TRIP at step 3, which is the
    structural difference from ProRes and the thing prose describes worst.

  * `feature_projection_order.png` -- the projection chain's order is the point; the document has to
    explain CURVE-versus-LENS in prose precisely because there is no picture.

Run:  python docs/diagrams/generate_feature_diagrams.py
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
DANGER_T = "#c85a5a"   # a TEXT colour; DANGER (#8a2a2a) is a fill and is unreadable at 8pt
WARNING_T = "#d0a02a"

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


# ─────────────────────────────────────────────────────────────────────────────
def codec_handoff():
    """Three producers, three handoffs. The thing not to assume about this fork.

    GEOMETRY NOTE, because `layout_check` caught both of these and a renderer would not: the
    column panels must not overlap the mixer panel (they did, 20..84 against 12..25), and the
    arrow from each handoff down to the mixer must run in the GAP below the columns rather than
    through them -- routed inside, it crossed all three lines of notes text.
    """
    lay, fig, ax = _new((13, 7.4))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    lay.text("title", 50, 95, "What each GPU codec producer hands the mixer",
             parent=None, color=TITLE, size=13, weight="bold", ha="center")
    lay.text("sub", 50, 90.5,
             "three producers, three answers — each defensible, none a house style",
             parent=None, color=MUTED, size=9.5, ha="center", style="italic")

    cols = [
        ("CUDA ProRes  4:2:2", 4.0,  SUCCESS,
         ["decode to", "Y / Cb / Cr", "10-bit planes"],
         "3 planes, bit10",
         ["4 bytes/pixel", "shader converts", "colour mgmt applies"]),
        ("HAP  /  HAP Q", 35.5, ACCENT,
         ["keep the DXT", "compressed", "all the way"],
         "compressed texture",
         ["texture unit decodes", "YCoCg resolved", "in the shader"]),
        ("CUDA NotchLC", 67.0, WARNING,
         ["decode, then", "convert YCoCg", "to BGRA16"],
         "1 packed texture",
         ["8 bytes/pixel", "converted before", "the mixer sees it"]),
    ]
    for name, x, tint, steps, handoff, notes in cols:
        cx = x + 14.5
        lay.panel(f"col{x}", x, 32, 29, 52, fc="#242424", ec=tint, lw=1.6)
        lay.text(f"nm{x}", cx, 80.5, name, parent=f"col{x}", color=tint,
                 size=10.5, weight="bold", ha="center")

        lay.panel(f"dec{x}", x + 2.5, 63, 24, 13, fc=PANEL, ec=BORDER)
        for i, t in enumerate(steps):
            lay.text(f"st{x}_{i}", cx, 72.5 - i * 3.5, t, parent=f"dec{x}",
                     color=TEXT, size=8.6, ha="center")

        lay.panel(f"ho{x}", x + 2.5, 50.5, 24, 8, fc=tint, ec=tint)
        lay.text(f"hl{x}", cx, 54.5, handoff, parent=f"ho{x}", color="#ffffff",
                 size=9.2, weight="bold", ha="center")

        for i, t in enumerate(notes):
            lay.text(f"nt{x}_{i}", cx, 44.5 - i * 3.4, t, parent=f"col{x}",
                     color=MUTED, size=8.2, ha="center")

        # the only arrow: in the clear gap between the column and the mixer
        lay.arrow((cx, 31.4), (cx, 25.6), color=tint, lw=1.7)

    lay.panel("mixer", 4, 12, 92, 13, fc=PANEL, ec=TITLE, lw=1.8)
    lay.text("mx", 50, 21, "THE MIXER", parent="mixer", color=TITLE,
             size=11, weight="bold", ha="center")
    lay.text("mx2", 50, 16,
             "samples whatever it is given — the handoff decides WHERE the colour conversion "
             "happens,\nand therefore whether the channel's colour management can reach it",
             parent="mixer", color=MUTED, size=8.2, ha="center")

    lay.text("foot", 50, 6,
             "ProRes 4444 takes the packed route too: planar 4:4:4-with-alpha is 8 bytes/pixel, "
             "exactly what BGRA16 costs — so there is nothing to win",
             parent=None, color=WARNING_T, size=8.4, ha="center", style="italic")
    lay.check(name="codec_handoff")
    _save(fig, "feature_codec_handoff.png")


# ─────────────────────────────────────────────────────────────────────────────
def notchlc_pipeline():
    """Ten steps, and the host round trip in the middle is the point."""
    lay, fig, ax = _new((14, 6.4))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    lay.text("title", 50, 94, "NotchLC decode — where the frame stops",
             parent=None, color=TITLE, size=13, weight="bold", ha="center")
    lay.text("sub", 50, 88.5,
             "step 3 comes back to the host, because the section offsets live inside the "
             "compressed payload",
             parent=None, color=MUTED, size=9.5, ha="center", style="italic")

    steps = [
        ("1", "upload", "h→d async", ACCENT),
        ("2", "LZ4", "nvcomp", ACCENT),
        ("3", "read header", "d→h  256 B", "STOP"),
        ("4", "parse", "10 × u32", ACCENT),
        ("5", "Y", "4×4 blocks", SUCCESS),
        ("6", "UV", "16×16 blocks", SUCCESS),
        ("7", "alpha", "or opaque", SUCCESS),
        ("8", "YCoCg→BGRA16", "per pixel", SUCCESS),
        ("9", "→ texture", "d→d", ACCENT),
        ("10", "sync", "stream wait", ACCENT),
    ]
    # Ten boxes, not nine: the module's own header lists a final synchronise, and a figure that
    # stops at 9 while the text says ten steps is the kind of small disagreement that makes a
    # reader distrust both.
    x = 2.8
    w = 8.8
    for n, top, bot, tint in steps:
        stop = tint == "STOP"
        col = DANGER_T if stop else tint
        fc = "#3a1f1f" if stop else "#242424"
        lay.panel(f"s{n}", x, 42, w, 26, fc=fc, ec=col, lw=2.0 if stop else 1.3)
        lay.text(f"n{n}", x + w / 2, 63.5, n, parent=f"s{n}", color=col,
                 size=11, weight="bold", ha="center")
        lay.fit_text(f"t{n}", x + w / 2, 55.5, top, parent=f"s{n}", size=8.4,
                     color=TEXT, ha="center")
        lay.fit_text(f"b{n}", x + w / 2, 48.5, bot, parent=f"s{n}", size=7.8,
                     color=MUTED, ha="center")
        if x > 3.0:
            lay.arrow((x - 0.7, 55), (x - 0.1, 55), color=MUTED, lw=1.3)
        x += w + 0.7

    lay.panel("gpu", 2, 71, 96, 8, fc="#1a2333", ec=ACCENT_HOVER)
    lay.text("gpul", 50, 75, "GPU", parent="gpu", color=ACCENT_HOVER, size=9.5,
             weight="bold", ha="center")
    lay.panel("host", 2, 30, 96, 8, fc="#33231a", ec=DANGER_T)
    lay.text("hostl", 50, 34, "HOST — one synchronisation point, mid-frame",
             parent="host", color=DANGER_T, size=9.5, weight="bold", ha="center")
    lay.arrow((24.0, 42), (24.0, 38.2), color=DANGER_T, lw=2.0)
    lay.arrow((27.2, 38.2), (27.2, 42), color=DANGER_T, lw=2.0)

    lay.text("cmp", 50, 20,
             "ProRes parses its slice table on the CPU BEFORE any upload, so it has no equivalent "
             "stop. That is the structural difference between the two routes.",
             parent=None, color=TEXT, size=9, ha="center")
    lay.text("cmp2", 50, 12,
             "unmeasured: no battery isolates the cost of this round trip, and there is no "
             "channel-count ceiling for NotchLC to compare against ProRes",
             parent=None, color=WARNING_T, size=8.4, ha="center", style="italic")
    lay.check(name="notchlc_pipeline")
    _save(fig, "feature_notchlc_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
def projection_order():
    """The chain's order is the point, and CURVE vs LENS is the confusion it removes."""
    lay, fig, ax = _new((13.5, 7.0))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    lay.text("title", 50, 95.5, "Projection and ICVFX — the order the shader actually runs",
             parent=None, color=TITLE, size=13, weight="bold", ha="center")
    lay.text("tsub", 50, 91,
             "read from shader.frag — an earlier draft of this figure guessed the order and had it wrong twice",
             parent=None, color=MUTED, size=8.6, ha="center", style="italic")

    # VERIFIED AGAINST THE SHADER, not assumed. The first version of this figure drew
    # LENS -> CURVE -> PROJECTION -> DISTORTION and was wrong twice over: the curve is applied
    # to the DESTINATION uv before anything else, and the distortion is applied BEFORE the lens
    # model rather than after the rotation. See this module's docstring for the extracted order.
    stages = [
        ("MIXER PROJECTION_CURVE", "the DESTINATION uv, warped\nfor the screen's shape", SUCCESS),
        ("MIXER PROJECTION_OFFSET", "lens shift, in NDC", ACCENT),
        ("MIXER PROJECTION_DISTORTION", "Brown–Conrady k1 k2 k3 p1 p2\n— before the lens model", WARNING),
        ("MIXER PROJECTION_LENS  +  PROJECTION\n+ _FRUSTUM", "the view vector: source lens\nAND yaw / pitch / roll / fov", ACCENT),
        ("MIXER PROJECTION_BLEND  +  _BLEND_MASK", "soft edge against\nthe next projector", ACCENT),
    ]
    y = 77
    for i, (cmd, why, tint) in enumerate(stages):
        lay.panel(f"st{i}", 4, y, 42, 10.4, fc="#242424", ec=tint, lw=1.4)
        lay.fit_text(f"c{i}", 25, y + 7.0, cmd.replace("\n", "  "), parent=f"st{i}",
                     size=8.2, color=tint, weight="bold", ha="center")
        lay.fit_text(f"w{i}", 25, y + 2.9, why.replace("\n", "  ·  "), parent=f"st{i}",
                     size=7.4, color=MUTED, ha="center")
        if i:
            lay.arrow((25, y + 12.2), (25, y + 10.6), color=MUTED, lw=1.4)
        y -= 12.4

    lay.panel("icvfx", 52, 27, 43, 45, fc="#242424", ec=WARNING, lw=1.7)
    lay.text("il", 73.5, 71, "MIXER PROJECTION_ICVFX", parent="icvfx", color=WARNING_T,
             size=9.6, weight="bold", ha="center")
    lay.panel("outer", 55, 40, 17.5, 25, fc="#2a2a2a", ec=BORDER)
    lay.text("ol", 63.75, 61.5, "OUTER", parent="outer", color=TEXT, size=8.6,
             weight="bold", ha="center")
    lay.text("ol2", 63.75, 52, "the frame\n× outer_dim\n× outer_gain", parent="outer",
             color=MUTED, size=7.6, ha="center")
    lay.panel("inner", 75.5, 40, 17.5, 25, fc="#2a2a2a", ec=WARNING)
    lay.text("il2", 84.25, 61.5, "INNER", parent="inner", color=WARNING_T, size=8.6,
             weight="bold", ha="center")
    lay.text("il3", 84.25, 52, "reprojected\n× inner_dim\n× inner_gain", parent="inner",
             color=MUTED, size=7.6, ha="center")
    lay.text("mix", 73.5, 35.5, "mixed by icvfx_mask — a signed-distance quad, no texture",
             parent="icvfx", color=TEXT, size=7.9, ha="center")
    lay.arrow((45.5, 45), (51.5, 48), color=WARNING, lw=1.6, rad=-0.15)

    lay.panel("warn", 5, 4.5, 90, 15, fc="#33231a", ec=WARNING_T, lw=1.3)
    lay.text("w1", 50, 15.6,
             "PROJECTION takes DEGREES.   PROJECTION_ICVFX takes RADIANS.",
             parent="warn", color=WARNING_T, size=9.6, weight="bold", ha="center")
    lay.text("w2", 50, 9.2,
             "The gain is per channel, so any example must use asymmetric values: equal gains are "
             "invariant under exchanging red and blue,\nwhich is how a live defect survived in the "
             "OpenGL shader until 2026-08-26.",
             parent="warn", color=TEXT, size=8.2, ha="center")
    lay.check(name="projection_order")
    _save(fig, "feature_projection_order.png")


if __name__ == "__main__":
    codec_handoff()
    notchlc_pipeline()
    projection_order()
