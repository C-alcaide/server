#!/usr/bin/env python
"""Diagrams for the recording guides.

Two stages added in 2026-08 are order-dependent, which is this tree's test for whether a
feature earns a picture:

  * FIELD PAIRING -- an interlaced channel ticks at field rate, and a field-coded file needs
    two ticks line-interleaved into one frame. Which tick owns which lines IS the feature, and
    it has two routes (a host memcpy and a strided GPU copy) that have to agree.
  * QSCALE AUTO -- a closed loop. Prose describes a loop one step at a time and the reader has
    to hold the feedback edge in their head; a picture puts it on the page.

Run:  python docs/diagrams/generate_recording_diagrams.py
Writes PNGs into docs/images/.

Palette, helpers and `layout_check.Layout` all follow `generate_diagrams.py` and
`generate_ocio_diagrams.py` deliberately. The first draft of this file did not use Layout and
produced both defects Layout exists to catch -- a caption written across the panel above it,
and a label wider than its box -- which is the argument for it in one attempt.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches   # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_check import Layout        # noqa: E402

# ── App palette (ui_kit.COLORS), as the other generators ─────────────────────
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
DANGER_T = "#c85a5a"   # readable on the dark panel; DANGER (#8a2a2a) is a FILL, not a text colour
WARNING_T = "#d0a02a"  # ditto for WARNING: a caution colour that reads as one at 8pt

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


def _line_stack(ax, x, y, w, rows, own_even, tint):
    """A full-height picture drawn as lines, with the ones this tick owns tinted."""
    for i in range(rows):
        ly = y + (rows - 1 - i) * (w * 0.0)  # placeholder, set below
    step = 2.6
    for i in range(rows):
        ly = y + (rows - 1 - i) * step
        owned = (i % 2 == 0) if own_even else (i % 2 == 1)
        _panel(ax, x, ly, w, 1.9, fc=tint if owned else "#232323",
               ec=BORDER_SUBTLE, lw=0.6, radius=0.0, z=3)


# ─────────────────────────────────────────────────────────────────────────────
def field_pairing():
    lay, fig, ax = _new((12.0, 6.8))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95,
             "Interlaced recording: two channel ticks become one frame",
             parent=None, color=TITLE, size=13, weight="bold", ha="center")
    lay.text("sub", 50, 90,
             "a 1080i50 channel ticks 50× a second at FULL height — the file is 25 fps",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")

    # ── the two source ticks, stacked with room for their captions ───────────
    lay.panel("tickA", 4, 58, 21, 22, fc="#0e1118", ec=BORDER)
    lay.text("labA", 14.5, 83, "tick N  ·  field A", parent=None, color=ACCENT_HOVER,
             size=9.2, weight="bold", ha="center")
    _line_stack(ax, 5.5, 60, 18, 8, True, ACCENT)

    lay.panel("tickB", 4, 22, 21, 22, fc="#0e1118", ec=BORDER)
    lay.text("labB", 14.5, 47, "tick N+1  ·  field B", parent=None, color=SUCCESS,
             size=9.2, weight="bold", ha="center")
    _line_stack(ax, 5.5, 24, 18, 8, False, SUCCESS)

    lay.text("cap", 14.5, 17.5, "each tick is full height", parent=None, color=MUTED,
             size=8, ha="center")

    # ── the interleave ───────────────────────────────────────────────────────
    lay.panel("weave", 31, 44, 21, 22, fc=PANEL, ec=ACCENT_HOVER, lw=1.6)
    lay.text(None, 41.5, 61.5, "INTERLEAVE", parent="weave", color=TITLE, size=10,
             weight="bold", ha="center")
    for i, ln in enumerate(("output line y comes", "from line y of", "whichever tick",
                            "owns that parity")):
        lay.fit_text(None, 41.5, 56.5 - i * 3.6, ln, parent="weave", size=8.2,
                     color=TEXT, ha="center")

    lay.arrow((25.5, 69), (31, 58), color=ACCENT, rad=-0.15)
    lay.arrow((25.5, 33), (31, 52), color=SUCCESS, rad=0.15)

    # ── the two routes ───────────────────────────────────────────────────────
    lay.panel("routes", 29, 20, 25, 18, fc="#181818", ec=BORDER_SUBTLE)
    lay.text(None, 41.5, 34.5, "two routes, one rule", parent="routes", color=MUTED,
             size=8.6, weight="bold", ha="center")
    lay.fit_text(None, 41.5, 30, "host: per-line memcpy", parent="routes", size=8,
                 color=TEXT, ha="center")
    lay.fit_text(None, 41.5, 26.5, "GPU: strided copyImage", parent="routes", size=8,
                 color=TEXT, ha="center")
    lay.fit_text(None, 41.5, 22.8, "they must agree", parent="routes", size=7.6,
                 color=WARNING, ha="center", style="italic")

    # ── the result ───────────────────────────────────────────────────────────
    lay.panel("out", 59, 44, 21, 22, fc="#0e1118", ec=ACCENT_HOVER, lw=1.6)
    lay.text("outlab", 69.5, 69, "one field-coded frame", parent=None, color=TITLE,
             size=9.2, weight="bold", ha="center")
    for i in range(8):
        ly = 46 + (7 - i) * 2.6
        _panel(ax, 60.5, ly, 18, 1.9, fc=ACCENT if i % 2 == 0 else SUCCESS,
               ec=BORDER_SUBTLE, lw=0.6, radius=0.0, z=3)
    lay.text("outcap", 69.5, 40.5, "25 fps · field_order tt", parent=None, color=MUTED,
             size=8, ha="center")
    lay.arrow((52, 55), (59, 55), color=ACCENT_HOVER, lw=1.8)

    # ── the flags ────────────────────────────────────────────────────────────
    lay.panel("flags", 59, 18, 37, 20, fc=PANEL, ec=BORDER)
    lay.text(None, 77.5, 34.5, "signalled, not implied", parent="flags", color=TITLE,
             size=9, weight="bold", ha="center")
    lay.fit_text(None, 77.5, 30, "encoder: AV_CODEC_FLAG_INTERLACED_DCT", parent="flags",
                 size=7.6, color=TEXT, ha="center")
    lay.fit_text(None, 77.5, 26.5, "frame: AV_FRAME_FLAG_INTERLACED + TFF", parent="flags",
                 size=7.6, color=TEXT, ha="center")
    lay.fit_text(None, 77.5, 21.5, "the consumer sets these: -flags +ildct never arrives",
                 parent="flags", size=7.2, color=WARNING, ha="center")

    lay.text("opt", 50, 9, "-interlaced   auto | 0 | 1", parent=None, color=TITLE,
             size=9.5, weight="bold", ha="center")
    lay.text("optd", 50, 5,
             "auto pairs when the channel is interlaced  ·  0 writes every tick "
             "progressive  ·  1 pairs and warns if there is no second field",
             parent=None, color=MUTED, size=7.8, ha="center")

    lay.check(name="recording_field_pairing")
    _save(fig, "recording_field_pairing.png")


# ─────────────────────────────────────────────────────────────────────────────
def qscale_auto_loop():
    lay, fig, ax = _new((12.0, 5.8))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95,
             "CUDA_PRORES  QSCALE AUTO — a closed loop on the profile's data rate",
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")
    lay.text("sub", 50, 89.5,
             "a fixed quantiser has no rate control: it produces whatever the content produces",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")

    stages = [
        ("composite", 4, "COMPOSITE", ("BGRA from", "the mixer"), PANEL),
        ("encode", 24, "ENCODE", ("DCT + quantise", "at q"), ACCENT),
        ("measure", 44, "MEASURE", ("bits spent", "per macroblock"), PANEL),
        ("compare", 64, "COMPARE", ("against the", "profile target"), PANEL),
        ("step", 84, "STEP q", ("damped, toward", "the target"), SUCCESS),
    ]
    for name, x, title, body, fc in stages:
        lay.panel(name, x, 56, 15, 20, fc=fc, ec=BORDER)
        lay.text(None, x + 7.5, 71.5, title, parent=name,
                 color="#ffffff" if fc in (ACCENT, SUCCESS) else TITLE,
                 size=9, weight="bold", ha="center")
        for i, ln in enumerate(body):
            lay.fit_text(None, x + 7.5, 66 - i * 4, ln, parent=name, size=7.8,
                         color="#ffffff" if fc in (ACCENT, SUCCESS) else TEXT,
                         ha="center")
    for x in (19, 39, 59, 79):
        lay.arrow((x, 66), (x + 5, 66), color=MUTED, lw=1.5)

    # The feedback edge, which is the reason this is a picture at all -- routed BELOW the
    # stages, because a curve bowing the other way runs straight through MEASURE and COMPARE.
    #
    # LAYOUT_CHECK CANNOT SEE THAT. It tests the straight segment between an arrow's endpoints
    # (`_segment_hits_rect`), so a curved arrow whose endpoints clear every box passes while the
    # drawn path crosses two of them -- which is exactly what the first version of this diagram
    # did. The check removes the mechanical iterations; it does not remove looking at the PNG.
    # rad and the caption's y are coupled: arc3 dips about rad*span/2 below the chord, so a
    # 60-unit span at -0.18 bottoms out near y=49 and leaves the caption room at 41. At -0.30
    # it reached y=35 and the caption was drawn across it -- a text-vs-ARROW collision, which is
    # a fourth class layout_check does not cover (it checks text-in-panel, text-fit and
    # arrow-through-panel).
    lay.arrow((91.5, 54.5), (31.5, 54.5), color=SUCCESS, lw=2.0, rad=-0.18)
    lay.text("fb", 61, 41,
             "the next frame is encoded at the corrected quantiser",
             parent=None, color=SUCCESS, size=8.6, ha="center", style="italic")

    lay.panel("ceil", 4, 8, 44, 26, fc=PANEL, ec=BORDER)
    lay.text(None, 26, 30, "the target is a CEILING", parent="ceil", color=TITLE,
             size=9.2, weight="bold", ha="center")
    for i, ln in enumerate((
            "flat content floors the quantiser at 1 and still",
            "cannot spend the budget — measured 0.05× on a",
            "flat colour, which is what the reference does too")):
        lay.fit_text(None, 26, 25.5 - i * 3.8, ln, parent="ceil", size=7.9, color=TEXT,
                     ha="center")
    lay.fit_text(None, 26, 12, "it holds a rolling average, not a per-frame cap",
                 parent="ceil", size=7.7, color=WARNING, ha="center")

    lay.panel("meas", 52, 8, 44, 26, fc=PANEL, ec=BORDER)
    lay.text(None, 74, 30, "measured: every profile 1.00×", parent="meas",
             color=TITLE, size=9.2, weight="bold", ha="center")
    rows = [("Proxy", 194), ("LT", 440), ("422", 632),
            ("422 HQ", 950), ("4444", 1425), ("4444 XQ", 2137)]
    for i, (nm, bits) in enumerate(rows):
        col = 63 if i < 3 else 85
        yy = 25 - (i % 3) * 4.0
        lay.fit_text(None, col, yy, f"{nm}  {bits} bits/MB", parent="meas", size=7.6,
                     color=TEXT, ha="center", family="monospace")
    lay.fit_text(None, 74, 11, "1080p25, 1080i50 and 2160p25 alike", parent="meas",
                 size=7.7, color=MUTED, ha="center")

    lay.check(name="recording_qscale_auto")
    _save(fig, "recording_qscale_auto.png")


# ─────────────────────────────────────────────────────────────────────────────
def three_questions():
    """The three "how many" shapes. The guide says these get confused constantly."""
    lay, fig, ax = _new((12.0, 6.4))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95, 'Three different "how many" questions — the numbers do not swap',
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")

    def channel(x, y, w=8.5, h=7.0, *, label=None, fc=PANEL):
        _panel(ax, x, y, w, h, fc=fc, ec=BORDER, z=2)
        if label:
            _text(ax, x + w / 2, y + h / 2, label, color=TEXT, size=7.6, ha="center", z=6)

    # ── 1. playout channels ──────────────────────────────────────────────────
    lay.panel("q1", 3, 60, 30, 26, fc="#181818", ec=BORDER_SUBTLE)
    lay.text(None, 18, 83, "how many PLAYOUT channels?", parent="q1", color=TITLE,
             size=8.8, weight="bold", ha="center")
    for i in range(3):
        y = 74 - i * 4.6
        channel(6, y, 9, 3.8, label="decode")
        _arrow(ax, (15.2, y + 1.9), (19.5, y + 1.9), color=MUTED, lw=1.1)
        channel(19.8, y, 9, 3.8, label="out", fc="#0e1118")
    lay.text(None, 18, 62.5, "N independent ticks", parent="q1", color=MUTED, size=7.6,
             ha="center", style="italic")

    # ── 2. ISO recordings ────────────────────────────────────────────────────
    lay.panel("q2", 35, 60, 30, 26, fc="#181818", ec=BORDER_SUBTLE)
    lay.text(None, 50, 83, "how many ISO RECORDINGS?", parent="q2", color=TITLE,
             size=8.8, weight="bold", ha="center")
    for i in range(3):
        y = 74 - i * 4.6
        channel(38, y, 9, 3.8, label="channel")
        _arrow(ax, (47.2, y + 1.9), (51.5, y + 1.9), color=MUTED, lw=1.1)
        channel(51.8, y, 9, 3.8, label="record", fc=SUCCESS)
    lay.text(None, 50, 62.5, "N independent ticks", parent="q2", color=MUTED, size=7.6,
             ha="center", style="italic")

    # ── 3. outputs on ONE channel ────────────────────────────────────────────
    lay.panel("q3", 67, 60, 30, 26, fc="#181818", ec=WARNING, lw=1.5)
    lay.text(None, 82, 83, "how many OUTPUTS on one channel?", parent="q3", color=TITLE,
             size=8.8, weight="bold", ha="center")
    channel(69, 68, 11, 10, label="one\nchannel")
    for i, (lbl, fc) in enumerate((("SDI out", "#0e1118"), ("record", SUCCESS),
                                   ("record", SUCCESS))):
        y = 75.5 - i * 4.6
        _arrow(ax, (80.2, 73), (85, y + 1.6), color=WARNING, lw=1.1, rad=0.08)
        channel(85.3, y, 9.5, 3.4, label=lbl, fc=fc)
    lay.text(None, 82, 62.5, "ONE frame budget, shared", parent="q3", color=WARNING,
             size=7.8, ha="center", weight="bold")

    # ── the point ────────────────────────────────────────────────────────────
    lay.panel("point", 3, 26, 94, 26, fc=PANEL, ec=WARNING, lw=1.5)
    lay.text(None, 50, 48, "The third one is the one that surprises people",
             parent="point", color=TITLE, size=10, weight="bold", ha="center")
    for i, ln in enumerate((
            "A channel walks its consumers on its own tick and cannot advance until EVERY one has taken the frame.",
            "So the slowest consumer sets the pace for all the others — that is how one slow recording makes an",
            "on-air SDI output late. Eight recordings on eight channels get eight budgets; eight on one channel share one.")):
        lay.fit_text(None, 50, 43 - i * 4.2, ln, parent="point", size=8.2, color=TEXT,
                     ha="center")
    lay.fit_text(None, 50, 29.5,
                 "And a per-unit cost cannot be multiplied into a ceiling: the mixer's work is not linear in load.",
                 parent="point", size=8, color=MUTED, ha="center", style="italic")

    lay.text("foot", 50, 18,
             "measured: 24 GPU ProRes recordings share one 1080p channel behind a screen output, "
             "zero dropped frames",
             parent=None, color=SUCCESS, size=8.4, ha="center")
    lay.text("foot2", 50, 13.5,
             "and at 24 the channel's frame budget is at 0.02 — the encoders are the limit, "
             "not the tick",
             parent=None, color=MUTED, size=8, ha="center", style="italic")
    lay.text("foot3", 50, 7,
             "the same channel with ONE CPU ProRes recording: 329 frames dropped",
             parent=None, color=DANGER_T, size=8.2, ha="center", weight="bold")

    lay.check(name="recording_three_questions")
    _save(fig, "recording_three_questions.png")


# ─────────────────────────────────────────────────────────────────────────────
def host_memory_routes():
    """Where each route touches host memory -- the column the tables lead with."""
    lay, fig, ax = _new((12.0, 6.6))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95, "Does the picture ever reach host memory?",
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")
    lay.text("sub", 50, 90,
             "every crossing of this line costs a copy and some latency — the fast paths never cross it",
             parent=None, color=MUTED, size=8.8, ha="center", style="italic")

    # the boundary
    ax.plot([4, 96], [50, 50], color=WARNING_T, lw=1.6, ls="--", zorder=2)
    # At the RIGHT end, clear of the descending arrows. On the left they sat directly under
    # the software-decode arrow and "HOST MEMORY" was drawn across it.
    lay.text("gpuside", 97, 53, "GPU side", parent=None, color=WARNING_T, size=8.4,
             weight="bold", ha="right")
    lay.text("hostside", 97, 47, "host memory", parent=None, color=WARNING_T, size=8.4,
             weight="bold", ha="right")

    # decode routes, left half
    lay.text("dec", 26, 84, "DECODE", parent=None, color=TITLE, size=10,
             weight="bold", ha="center")
    decode = [
        ("software", True, "any codec"),
        ("D3D11VA", False, "H.264 HEVC VP9 AV1"),
        ("CUDA ProRes", False, "ProRes"),
        ("FFmpeg Vulkan", False, "ProRes FFV1 DPX"),
    ]
    # THROUGH lay.fit_text, not the raw helper. The first version called `_text` directly
    # inside this loop, which bypasses the overrun check entirely -- and four of the eight
    # labels ran outside their boxes. Using the checked helper is the whole point of having it.
    for i, (name, crosses, note) in enumerate(decode):
        x = 4 + i * 11.5
        col = DANGER_T if crosses else SUCCESS
        lay.panel(f"d{i}", x, 68, 10.5, 12, fc=PANEL, ec=col, lw=1.4)
        lay.fit_text(None, x + 5.25, 76, name, parent=f"d{i}", size=7.4, color=col,
                     ha="center", weight="bold")
        lay.fit_text(None, x + 5.25, 71, note, parent=f"d{i}", size=6.3, color=MUTED,
                     ha="center")
        if crosses:
            lay.arrow((x + 5.25, 68), (x + 5.25, 43), color=DANGER_T, lw=1.6)
            lay.text(f"dc{i}", x + 5.25, 40, "a copy", parent=None, color=DANGER_T,
                     size=6.8, ha="center")
        else:
            lay.arrow((x + 5.25, 68), (x + 5.25, 56), color=SUCCESS, lw=1.6)

    # encode routes, right half
    lay.text("enc", 74, 84, "ENCODE", parent=None, color=TITLE, size=10,
             weight="bold", ha="center")
    encode = [
        ("host FFmpeg", True, "any encoder"),
        ("NVENC direct", False, "H.264 HEVC, 8-bit"),
        ("Vulkan encode", False, "ProRes FFV1, 16-bit"),
        ("CUDA_PRORES", False, "ProRes, OpenGL mixer"),
    ]
    for i, (name, crosses, note) in enumerate(encode):
        x = 52 + i * 11.5
        col = DANGER_T if crosses else SUCCESS
        lay.panel(f"e{i}", x, 68, 10.5, 12, fc=PANEL, ec=col, lw=1.4)
        lay.fit_text(None, x + 5.25, 76, name, parent=f"e{i}", size=7.4, color=col,
                     ha="center", weight="bold")
        lay.fit_text(None, x + 5.25, 71, note, parent=f"e{i}", size=6.3, color=MUTED,
                     ha="center")
        if crosses:
            lay.arrow((x + 5.25, 68), (x + 5.25, 43), color=DANGER_T, lw=1.6)
            lay.text(f"ec{i}", x + 5.25, 40, "a readback", parent=None, color=DANGER_T,
                     size=6.8, ha="center")
        else:
            lay.arrow((x + 5.25, 68), (x + 5.25, 56), color=SUCCESS, lw=1.6)

    lay.panel("why", 5, 12, 42, 22, fc=PANEL, ec=BORDER)
    lay.text(None, 26, 30, "why it matters operationally", parent="why", color=TITLE,
             size=9, weight="bold", ha="center")
    for i, ln in enumerate((
            "one consumer that needs host memory makes the",
            "channel read the composite back — for everyone.",
            "Mixing a host consumer with GPU ones pays the",
            "readback anyway.")):
        lay.fit_text(None, 26, 26 - i * 3.6, ln, parent="why", size=7.9, color=TEXT,
                     ha="center")

    lay.panel("check", 53, 12, 42, 22, fc=PANEL, ec=BORDER)
    lay.text(None, 74, 30, "how to tell which you got", parent="check", color=TITLE,
             size=9, weight="bold", ha="center")
    lay.fit_text(None, 74, 25.5, "the server says so, once, at ADD time:", parent="check",
                 size=7.9, color=TEXT, ha="center")
    for i, ln in enumerate((
            '"GPU-direct recording active"',
            '"Vulkan encode: ... never reaches host memory"',
            '"CPU readback required by consumer ..."')):
        lay.fit_text(None, 74, 21 - i * 3.4, ln, parent="check", size=7.4,
                     color=SUCCESS if i < 2 else DANGER_T, ha="center",
                     family="monospace")

    lay.check(name="recording_host_memory")
    _save(fig, "recording_host_memory.png")


# ─────────────────────────────────────────────────────────────────────────────
def ceiling_chart():
    """The recording ceilings as bars. Twenty table rows is not a shape anyone can see."""
    lay, fig, ax = _new((12.0, 7.4))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 96,
             "How many simultaneous 1080p25 recordings, one per channel",
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")
    lay.text("sub", 50, 91.5,
             "180 s per rung · every accepted recording decode-checked · this box: 12 cores, RTX A4000",
             parent=None, color=MUTED, size=8.2, ha="center", style="italic")

    # (label, channels, kind) -- kind drives the colour and the caveat
    rows = [
        ("h264_nvenc", 14, "cap"),
        ("h264_vulkan", 14, "cap"),
        ("mpeg2video (XDCAM 50)", 14, "cap"),
        ("dnxhd 120M", 12, "starved"),
        ("prores_ks_vulkan  -q:v 12", 12, "starved"),
        ("prores_ks_vulkan  -q:v 4", 11, "starved"),
        ("libx264 ultrafast", 8, "real"),
        ("CUDA_PRORES 422 / HQ", 7, "real"),
        ("hevc_vulkan", 6, "starved"),
        ("dnxhd dnxhr_hq", 4, "starved"),
        ("prores_ks_vulkan  (no -q:v)", 2, "real"),
        ("dnxhd 185M", 1, "starved"),
        ("prores_aw / prores_ks (CPU)", 0, "loss"),
        ("ffv1_vulkan · prores 4444", 0, "loss"),
    ]
    x0, wmax, cmax = 34.0, 46.0, 14.0
    for i, (name, n, kind) in enumerate(rows):
        y = 82 - i * 5.3
        # AMBER for a rig-bounded rung, not a second blue. This chart's most important
        # distinction is "the encoder stopped" against "our test rig stopped", and the first
        # version drew those as #2255aa and #2f6bd0 -- indistinguishable at reading size, so
        # the caveat was invisible in the one place it matters.
        col = {"cap": SUCCESS, "starved": WARNING_T, "real": ACCENT_HOVER,
               "loss": DANGER_T}[kind]
        _text(ax, x0 - 1.5, y + 1.7, name, color=TEXT, size=7.8, ha="right", z=6)
        _panel(ax, x0, y, wmax, 3.4, fc="#232323", ec=BORDER_SUBTLE, lw=0.5,
               radius=0.0, z=2)
        if n:
            _panel(ax, x0, y, wmax * n / cmax, 3.4, fc=col, ec=col, radius=0.0, z=3)
            _text(ax, x0 + wmax * n / cmax + 1.5, y + 1.7,
                  f"{n}{'+' if kind == 'cap' else ''}", color=col, size=8.2,
                  weight="bold", z=6)
        else:
            _text(ax, x0 + 1.5, y + 1.7, "0 — loses frames at ONE channel",
                  color=DANGER_T, size=7.6, weight="bold", z=6)

    # legend
    lay.panel("leg", 4, 3, 92, 8, fc=PANEL, ec=BORDER)
    for i, (col, txt) in enumerate((
            (SUCCESS, "hit the top of the ladder — the real limit is higher"),
            (WARNING_T, "bounded by the TEST RIG: one decoder could not feed more channels"),
            (ACCENT_HOVER, "a real encoder limit: the files stopped lining up"),
            (DANGER_T, "loses frames mid-recording at a single channel"))):
        cx = 7 + (i % 2) * 46
        cy = 8 - (i // 2) * 3.4
        _panel(ax, cx, cy - 0.9, 2.4, 1.9, fc=col, ec=col, radius=0.0, z=3)
        _text(ax, cx + 3.6, cy, txt, color=TEXT, size=7.4, z=6)

    lay.check(name="recording_ceilings")
    _save(fig, "recording_ceilings.png")


# ─────────────────────────────────────────────────────────────────────────────
def which_path():
    """The decision the guide's section 8 makes in prose."""
    lay, fig, ax = _new((12.0, 6.6))
    lay.panel("frame", 1, 1, 98, 98, blocking=False, fc=BG, ec=BORDER_SUBTLE, radius=0.01)

    lay.text("title", 50, 95, "Which recording path — start from what you have",
             parent=None, color=TITLE, size=12.5, weight="bold", ha="center")

    def node(name, x, y, w, h, head, lines, *, fc=PANEL, ec=BORDER, hc=TITLE):
        lay.panel(name, x, y, w, h, fc=fc, ec=ec, lw=1.3)
        lay.fit_text(None, x + w / 2, y + h - 4, head, parent=name, size=8.6,
                     weight="bold", color=hc, ha="center")
        for i, ln in enumerate(lines):
            lay.fit_text(None, x + w / 2, y + h - 9 - i * 3.8, ln, parent=name,
                         size=7.6, color=TEXT, ha="center")

    node("q", 33, 77, 34, 13, "what does the channel run?",
         ("the mixer and the channel depth decide",), ec=ACCENT_HOVER)

    node("vk", 4, 46, 28, 24, "Vulkan mixer, 16-bit",
         ("prores_ks_vulkan -profile:v 3", "-q:v 12  at 1080p", "-q:v 4–6 at 4K",
          "12 channels · GPU-direct"), fc="#16281a", ec=SUCCESS, hc=SUCCESS)
    node("ogl", 36, 46, 28, 24, "OpenGL mixer, 8-bit",
         ("CUDA_PRORES", "QSCALE AUTO holds the", "profile's data rate",
          "7 channels · GPU-direct"), fc="#16281a", ec=SUCCESS, hc=SUCCESS)
    node("h264", 68, 46, 28, 24, "H.264 or HEVC needed",
         ("h264_nvenc  or  h264_vulkan", "both GPU-direct", "14+ channels",
          "hevc costs more than h264"), fc="#16281a", ec=SUCCESS, hc=SUCCESS)

    lay.arrow((43, 77), (18, 70), color=MUTED, rad=0.12)
    lay.arrow((50, 77), (50, 70), color=MUTED)
    lay.arrow((57, 77), (82, 70), color=MUTED, rad=-0.12)

    # h=24, not 20: node() stacks lines every 3.8 units from y+h-9, so a fourth line
    # falls below a 20-unit box -- which layout_check caught rather than me.
    node("no", 4, 16, 44, 24, "do NOT reach for these",
         ("prores_ks / prores_aw on the CPU: they lose 240–759",
          "frames at ONE 1080p channel",
          "prores_ks_vulkan without -q:v: the quantiser search",
          "caps it at 2 channels — and -bits_per_mb is no substitute"),
         fc="#2a1616", ec=DANGER_T, hc=DANGER_T)

    node("air", 52, 16, 44, 24, "if a DeckLink output is on air",
         ("the SDI output and every recording take the SAME frame",
          "a slow recording makes the on-air output late",
          "8+ GPU ProRes recordings fit behind an SDI output; CPU ProRes fits none"),
         fc="#2a2410", ec=WARNING, hc=WARNING)

    lay.text("foot", 50, 11,
             "interlaced? the consumer pairs two ticks into one field-coded frame by default "
             "(-interlaced auto)",
             parent=None, color=MUTED, size=8, ha="center")
    lay.text("foot2", 50, 6.5,
             "50i in, 25p out? run the CHANNEL progressive and let the DeckLink producer "
             "deinterlace — nothing on the recording side",
             parent=None, color=MUTED, size=8, ha="center")

    lay.check(name="recording_which_path")
    _save(fig, "recording_which_path.png")


def main():
    field_pairing()
    qscale_auto_loop()
    three_questions()
    host_memory_routes()
    ceiling_chart()
    which_path()
    print("done")


if __name__ == "__main__":
    main()
