"""Generate documentation diagrams for the CasparCG 360 client operations guide.

Run from the repo root:  python docs/diagrams/generate_diagrams.py

Outputs PNGs into docs/images/. Re-run after changing a diagram; the images are
committed so the Markdown renders without a build step. Kept deliberately
dependency-light (matplotlib + numpy only) and themed to match the app palette.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle, Wedge)

# ── App palette (ui_kit.COLORS) ──────────────────────────────────────────────
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
LED = "#2f6bd0"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight",
                pad_inches=0.12)
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def _panel(ax, x, y, w, h, *, fc=PANEL, ec=BORDER, lw=1.2, radius=0.02, z=1):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    ax.add_patch(box)
    return box


def _text(ax, x, y, s, *, color=TEXT, size=10, weight="normal", ha="left",
          va="center", style="normal", z=5):
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
    ax.set_aspect("auto")
    return fig, ax


def _new_eq(figsize):
    """Equal-aspect canvas so circles render round. Units = figsize * 10."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, figsize[0] * 10)
    ax.set_ylim(0, figsize[1] * 10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    return fig, ax


def _hue_ring(ax, cx, cy, r_in, r_out, n=60, z=3):
    for i in range(n):
        col = mcolors.hsv_to_rgb((i / n, 0.62, 0.95))
        ax.add_patch(Wedge((cx, cy), r_out, 360 * i / n, 360 * (i + 1) / n,
                           width=r_out - r_in, facecolor=col, edgecolor="none", zorder=z))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Channel workspace UI mockup
# ─────────────────────────────────────────────────────────────────────────────
def channel_workspace():
    fig, ax = _new((11, 6.2))

    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    # File bar
    _panel(ax, 3, 90, 94, 7, fc=PANEL)
    _panel(ax, 4.5, 91.3, 9, 4.4, fc=ACCENT, ec=ACCENT_HOVER)
    _text(ax, 9, 93.5, "Select", color="#ffffff", size=9, ha="center")
    _panel(ax, 14.5, 91.3, 9, 4.4, fc=SUCCESS)
    _text(ax, 19, 93.5, "Load", color="#ffffff", size=9, ha="center")
    _text(ax, 26, 93.5, "clip.mov", color=MUTED, size=9)
    _text(ax, 95.5, 93.5, "CH 1 · L10", color=TITLE, size=9, ha="right", weight="bold")

    # Left column: viewport
    _panel(ax, 3, 30, 52, 57, fc="#0e1118", ec=BORDER)
    _text(ax, 29, 79, "EMBEDDED VIEWPORT", color=MUTED, size=10, ha="center", weight="bold")
    _text(ax, 29, 74, "live channel output", color=MUTED, size=8.5, ha="center", style="italic")
    # frustum-ish framing lines
    ax.plot([10, 48], [40, 40], color=BORDER_SUBTLE, lw=0.8, zorder=3)
    # transport
    _panel(ax, 3, 23, 52, 5.5, fc=PANEL)
    for i, gl in enumerate(["◄◄", "▶", "■", "►►"]):
        _text(ax, 8 + i * 6, 25.7, gl, color=TEXT, size=11, ha="center")
    _panel(ax, 33, 24.6, 20, 2.4, fc=HOVER, ec=BORDER_SUBTLE, radius=0.3)
    _text(ax, 43, 25.8, "speed", color=MUTED, size=7.5, ha="center")
    # playhead
    _panel(ax, 3, 16.5, 52, 4.8, fc=PANEL)
    ax.plot([5, 53], [18.9, 18.9], color=BORDER, lw=1.4, zorder=4)
    ax.add_patch(Rectangle((25, 17.6), 1.0, 2.6, facecolor=TITLE, zorder=5))
    _text(ax, 6, 18.9, "▶", color=TITLE, size=8, ha="center")
    # scopes
    for i, (lbl, c) in enumerate([("Waveform", ACCENT), ("Vector", ACCENT), ("Parade", ACCENT)]):
        _panel(ax, 3 + i * 17.5, 9, 16, 5, fc=PANEL, ec=BORDER_SUBTLE)
        _text(ax, 11 + i * 17.5, 11.5, lbl, color=TEXT, size=8.5, ha="center")

    # Right column: tool panels (scroll stack)
    _panel(ax, 58, 9, 39, 78, fc=PANEL, ec=BORDER)
    panels = [
        ("Source / media", False), ("Transform", False), ("Perspective", False),
        ("Projection (360° / ICVFX)", True), ("Colour grading", True),
        ("Hue curves · Qualifier", True), ("Colorspace (ACES)", True),
        ("Blend · Blur · Shape", False), ("Tracking", True),
        ("Keyframes · Presets", False),
    ]
    y = 83
    for label, advanced in panels:
        _panel(ax, 60, y - 4.6, 35, 4.2, fc=HOVER if advanced else "#262626",
               ec=BORDER_SUBTLE, radius=0.06)
        chev = "▾" if advanced else "▸"
        _text(ax, 62, y - 2.5, chev, color=TITLE, size=9)
        _text(ax, 65, y - 2.5, label, color=TITLE if advanced else TEXT,
              size=9, weight="bold" if advanced else "normal")
        if advanced:
            _text(ax, 93, y - 2.5, "★", color=WARNING, size=8.5, ha="right")
        y -= 7.4

    _text(ax, 50, 5.0, "Left: viewport + transport + scopes      Right: tool panels "
          "(★ = advanced)", color=MUTED, size=8.5, ha="center", style="italic")
    _text(ax, 50, 97.5, "Channel workspace", color=TITLE, size=13, ha="center", weight="bold")
    _save(fig, "channel_workspace.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage tab UI mockup
# ─────────────────────────────────────────────────────────────────────────────
def stage_layout():
    fig, ax = _new((11, 6.4))
    _panel(ax, 1, 1, 98, 98, fc=BG, ec=BORDER_SUBTLE, radius=0.01)
    _text(ax, 50, 97.5, "Stage  —  global 3D previz", color=TITLE, size=13,
          ha="center", weight="bold")

    # Toolbar
    _panel(ax, 3, 88, 94, 6.5, fc=PANEL)
    for i, t in enumerate(["Create ▾", "Add LED Screen", "Import Venue", "Export glTF"]):
        _panel(ax, 4.5 + i * 13, 89.3, 12, 4, fc=HOVER, ec=BORDER_SUBTLE, radius=0.1)
        _text(ax, 10.5 + i * 13, 91.3, t, color=TEXT, size=8, ha="center")
    # Grid/Wire/Gizmo
    for i, t in enumerate(["Grid", "Wire", "Gizmo"]):
        _panel(ax, 60 + i * 8.5, 89.3, 8, 4, fc="#262626", ec=BORDER_SUBTLE, radius=0.1)
        _text(ax, 64 + i * 8.5, 91.3, t, color=MUTED, size=7.5, ha="center")

    # View bar: previz channel + client/server toggle + live link
    _panel(ax, 3, 81, 94, 5.5, fc="#262626")
    _text(ax, 5, 83.7, "Previz channel:", color=TEXT, size=8.5)
    _panel(ax, 20, 82, 5, 3.4, fc=PANEL, ec=BORDER, radius=0.1)
    _text(ax, 22.5, 83.7, "5", color=TITLE, size=9, ha="center", weight="bold")
    _panel(ax, 28, 82, 18, 3.4, fc=ACCENT, ec=ACCENT_HOVER, radius=0.1)
    _text(ax, 37, 83.7, "Server 3D Previz", color="#ffffff", size=8, ha="center")
    _panel(ax, 78, 82, 17, 3.4, fc=SUCCESS, radius=0.1)
    _text(ax, 86.5, 83.7, "Live Link", color="#ffffff", size=8, ha="center")

    # Main viewport (client 3D)
    _panel(ax, 3, 30, 60, 49, fc="#0e1118", ec=BORDER)
    _text(ax, 33, 75, "client 3D previz  /  server output", color=MUTED, size=8,
          ha="center", style="italic")
    # draw a simple curved LED wall + camera in the viewport
    cx, cy = 30, 50
    th = np.linspace(np.deg2rad(205), np.deg2rad(335), 60)
    R = 17
    ax.plot(cx + R * np.cos(th), cy + 9 + R * np.sin(th), color=LED, lw=4, zorder=4)
    # camera
    ax.add_patch(Rectangle((28, 36), 4, 2.6, facecolor=TEXT, edgecolor=BORDER, zorder=5))
    ax.add_patch(plt.Polygon([[32, 37.3], [40, 41], [40, 33.6]], closed=True,
                              facecolor="none", edgecolor=TITLE, lw=1.2, zorder=4))
    _text(ax, 30, 33, "previz camera", color=MUTED, size=7.5, ha="center")

    # Scene tree
    _panel(ax, 65, 30, 14, 49, fc=PANEL, ec=BORDER)
    _text(ax, 66.5, 76, "Scene", color=TITLE, size=9, weight="bold")
    tree = ["▾ Stage", "  ▣ LED Wall A", "  ▣ LED Wall B", "  ◻ Set piece",
            "  ◆ Camera", "  ⬡ Reference"]
    for i, t in enumerate(tree):
        sel = i == 1
        if sel:
            _panel(ax, 65.6, 72.2 - i * 6.2, 12.8, 4.4, fc=ACCENT, ec=ACCENT_HOVER, radius=0.08)
        _text(ax, 67, 74.4 - i * 6.2, t, color="#ffffff" if sel else TEXT, size=8)

    # Properties
    _panel(ax, 81, 30, 16, 49, fc=PANEL, ec=BORDER)
    _text(ax, 82.5, 76, "Properties", color=TITLE, size=9, weight="bold")
    props = ["Panel: ROE BP2", "Pitch: 2.84 mm", "Tiles: 10 × 6", "Bezel: 0.0 mm",
             "Curve: Cylindrical", "Radius: 8.0 m", "Arc: 130°"]
    for i, p in enumerate(props):
        _text(ax, 82.5, 71 - i * 5.4, p, color=TEXT, size=8)

    # Footer
    _panel(ax, 3, 22, 94, 6, fc="#262626")
    _text(ax, 50, 25, "Live Link streams edits to the server as  PREVIZ <ch> ...  commands",
          color=MUTED, size=8.5, ha="center", style="italic")
    _save(fig, "stage_layout.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ICVFX inner/outer frustum
# ─────────────────────────────────────────────────────────────────────────────
def icvfx_frustum():
    fig, ax = _new((10, 6))
    _text(ax, 50, 96, "ICVFX — inner frustum on the LED volume", color=TITLE, size=13,
          ha="center", weight="bold")

    # Curved LED wall (top arc)
    cx, cy, R = 50, 18, 60
    th = np.linspace(np.deg2rad(58), np.deg2rad(122), 120)
    wx, wy = cx + R * np.cos(th), cy + R * np.sin(th)
    ax.plot(wx, wy, color=LED, lw=7, zorder=2, solid_capstyle="round")
    _text(ax, 50, 84, "LED volume (outer frustum content)", color=LED, size=9.5,
          ha="center", weight="bold")

    # Camera
    cam = (50, 24)
    ax.add_patch(Rectangle((46, 21), 8, 6, facecolor=PANEL, edgecolor=TEXT, lw=1.4, zorder=6))
    _text(ax, 50, 18.5, "tracked camera", color=TEXT, size=9, ha="center", weight="bold")

    # Inner frustum (camera FOV) hitting wall
    # find two points on the arc for the inner frustum edges
    a1, a2 = np.deg2rad(98), np.deg2rad(82)
    p1 = (cx + R * np.cos(a1), cy + R * np.sin(a1))
    p2 = (cx + R * np.cos(a2), cy + R * np.sin(a2))
    inner = plt.Polygon([cam, p1, p2], closed=True, facecolor=ACCENT, alpha=0.28,
                        edgecolor=ACCENT_HOVER, lw=1.6, zorder=4)
    ax.add_patch(inner)
    # the lit inner-frustum patch on the wall
    th_in = np.linspace(a2, a1, 30)
    ax.plot(cx + R * np.cos(th_in), cy + R * np.sin(th_in), color=ACCENT_HOVER,
            lw=8, zorder=5, solid_capstyle="round")
    _text(ax, 50, 70, "inner frustum", color="#cfe0ff", size=9, ha="center", weight="bold")
    _text(ax, 50, 66, "(rendered for the camera's exact POV)", color=MUTED, size=8,
          ha="center", style="italic")

    # annotation arrows
    arr = FancyArrowPatch((78, 55), (64, 64), arrowstyle="-|>", mutation_scale=12,
                          color=MUTED, lw=1.2, zorder=7)
    ax.add_patch(arr)
    _text(ax, 79, 53, "outer = set extension\n(everything off-camera)", color=MUTED,
          size=8, ha="left", va="top")

    _text(ax, 50, 7, "The wall shows the correct perspective inside the camera's "
          "frustum; the rest is set extension.", color=TEXT, size=8.5, ha="center",
          style="italic")
    _save(fig, "icvfx_frustum.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Projection calibration geometry (camera-in-the-loop)
# ─────────────────────────────────────────────────────────────────────────────
def projection_geometry():
    fig, ax = _new((10, 5.6))
    _text(ax, 50, 95, "Projection calibration — camera in the loop", color=TITLE,
          size=13, ha="center", weight="bold")

    # Projector
    ax.add_patch(Rectangle((6, 44), 14, 10, facecolor=PANEL, edgecolor=TEXT, lw=1.4, zorder=5))
    _text(ax, 13, 49, "projector", color=TEXT, size=9, ha="center", weight="bold")
    # projector cone to surface
    ax.add_patch(plt.Polygon([[20, 52], [58, 78], [58, 20], [20, 46]], closed=True,
                             facecolor=WARNING, alpha=0.18, edgecolor=WARNING, lw=1.0, zorder=2))

    # Curved/irregular surface
    ys = np.linspace(20, 78, 80)
    xs = 58 + 4 * np.sin((ys - 20) / 58 * np.pi * 1.5)
    ax.plot(xs, ys, color="#cccccc", lw=6, zorder=4, solid_capstyle="round")
    _text(ax, 63, 82, "projected surface", color="#dddddd", size=9, ha="center", weight="bold")

    # Camera filming the surface
    ax.add_patch(Rectangle((84, 44), 12, 9, facecolor=PANEL, edgecolor=SUCCESS, lw=1.6, zorder=5))
    _text(ax, 90, 48.5, "camera", color="#bfe6bf", size=9, ha="center", weight="bold")
    ax.add_patch(plt.Polygon([[84, 49], [60, 70], [60, 28]], closed=True,
                             facecolor=SUCCESS, alpha=0.16, edgecolor=SUCCESS, lw=1.0, zorder=2))

    # Loop arrows
    a1 = FancyArrowPatch((22, 40), (54, 26), arrowstyle="-|>", mutation_scale=13,
                         color=WARNING, lw=1.6, zorder=7,
                         connectionstyle="arc3,rad=-0.15")
    ax.add_patch(a1)
    _text(ax, 36, 30, "1 · project pattern", color=WARNING, size=8.5, ha="center")

    a2 = FancyArrowPatch((62, 30), (86, 42), arrowstyle="-|>", mutation_scale=13,
                         color=SUCCESS, lw=1.6, zorder=7,
                         connectionstyle="arc3,rad=-0.15")
    ax.add_patch(a2)
    _text(ax, 75, 33, "2 · capture return", color="#bfe6bf", size=8.5, ha="center")

    a3 = FancyArrowPatch((86, 56), (22, 56), arrowstyle="-|>", mutation_scale=13,
                         color=TITLE, lw=1.6, zorder=7,
                         connectionstyle="arc3,rad=-0.28")
    ax.add_patch(a3)
    _text(ax, 50, 70, "3 · solve  →  4 · apply correction  (MIXER PERSPECTIVE / MESH …)",
          color=TITLE, size=8.5, ha="center")

    _text(ax, 50, 8, "The camera return is a separate camera looking at the surface — "
          "not the output preview.", color=MUTED, size=8.5, ha="center", style="italic")
    _save(fig, "projection_geometry.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LED wall tiling / pixel pitch
# ─────────────────────────────────────────────────────────────────────────────
def led_tiling():
    fig, ax = _new((10, 5.4))
    _text(ax, 50, 95, "LED screen — tiling, pitch & bezel", color=TITLE, size=13,
          ha="center", weight="bold")

    # Tiling grid (4 x 3 tiles)
    nx, ny = 4, 3
    x0, y0, tw, th, gap = 8, 22, 14, 16, 1.2
    for j in range(ny):
        for i in range(nx):
            x = x0 + i * (tw + gap)
            y = y0 + j * (th + gap)
            ax.add_patch(Rectangle((x, y), tw, th, facecolor="#11243f",
                                   edgecolor=LED, lw=1.4, zorder=3))
    # bezel callout
    bx = x0 + tw
    a = FancyArrowPatch((bx, 70), (bx + gap, 70), arrowstyle="<|-|>", mutation_scale=9,
                        color=WARNING, lw=1.4, zorder=6)
    ax.add_patch(a)
    _text(ax, bx + gap / 2, 73.5, "bezel gap", color=WARNING, size=8, ha="center")

    # one tile zoomed with pixels
    zx, zy, zw, zh = 70, 30, 24, 30
    ax.add_patch(Rectangle((zx, zy), zw, zh, facecolor="#0b1a30", edgecolor=LED, lw=1.6, zorder=3))
    pp = 6
    for i in range(1, pp):
        ax.plot([zx + i * zw / pp] * 2, [zy, zy + zh], color="#1d3a5f", lw=0.7, zorder=4)
        ax.plot([zx, zx + zw], [zy + i * zh / pp] * 2, color="#1d3a5f", lw=0.7, zorder=4)
    ax.add_patch(Rectangle((zx + zw / pp, zy + zh / pp), zw / pp, zh / pp,
                           facecolor=ACCENT_HOVER, zorder=5))
    a2 = FancyArrowPatch((zx, zy - 3), (zx + zw / pp, zy - 3), arrowstyle="<|-|>",
                         mutation_scale=9, color=TITLE, lw=1.4, zorder=6)
    ax.add_patch(a2)
    _text(ax, zx + zw / (2 * pp) + 4, zy - 6.5, "pixel pitch (mm)", color=TITLE, size=8, ha="center")
    _text(ax, zx + zw / 2, zy + zh + 4, "one tile", color="#bcd4ef", size=8.5, ha="center")

    _text(ax, 32, 12, "tiling grid  =  H × V tiles", color="#bcd4ef", size=9, ha="center")
    _text(ax, 50, 6, "Pick a panel from the database (pitch + tile geometry) or set "
          "these by hand; the mesh rebuilds live.", color=MUTED, size=8.5, ha="center",
          style="italic")
    _save(fig, "led_tiling.png")


# ────────────────────────────────────────────────────────────────────────
# 6. Colour grading controls (the universal set)
# ────────────────────────────────────────────────────────────────────────
def grading_controls():
    fig, ax = _new_eq((11, 7))  # 110 x 70 units
    _text(ax, 55, 66.5, "Colour grading controls (the universal set)", color=TITLE,
          size=14, ha="center", weight="bold")

    # Panel A — 3-way colour wheels
    _panel(ax, 4, 34, 50, 26)
    _text(ax, 29, 57, "3-Way — Lift / Gamma / Gain", color=TITLE, size=10.5,
          ha="center", weight="bold")
    wheels = [
        (14, "Lift", "shadows", (-1.4, -1.4), "#5b8bd0"),
        (29, "Gamma", "mids", (0.0, 0.0), TEXT),
        (44, "Gain", "highlights", (1.4, 1.1), "#d8a14a"),
    ]
    for cx, name, rng, (dx, dy), dotcol in wheels:
        cy = 47
        _hue_ring(ax, cx, cy, 3.0, 4.6)
        ax.add_patch(Circle((cx, cy), 3.0, facecolor=PANEL, edgecolor=BORDER, lw=1.0, zorder=4))
        ax.plot([cx - 3, cx + 3], [cy, cy], color=BORDER, lw=0.6, zorder=5)
        ax.plot([cx, cx], [cy - 3, cy + 3], color=BORDER, lw=0.6, zorder=5)
        ax.add_patch(Circle((cx + dx, cy + dy), 0.7, facecolor=dotcol,
                            edgecolor="white", lw=0.6, zorder=6))
        _text(ax, cx, 40.4, name, color=TEXT, size=9.5, ha="center", weight="bold")
        _text(ax, cx, 38.0, rng, color=MUTED, size=8, ha="center", style="italic")
    _text(ax, 29, 35.3, "drag inside a wheel to push its colour balance", color=MUTED,
          size=7.6, ha="center", style="italic")

    # Panel B — tonal range each wheel affects
    _panel(ax, 58, 34, 48, 26)
    _text(ax, 82, 57, "Where each wheel acts", color=TITLE, size=10.5, ha="center",
          weight="bold")
    gx0, gx1, gy0, gy1 = 63, 101, 41, 54
    ax.plot([gx0, gx1], [gy0, gy0], color=BORDER, lw=1.0, zorder=3)
    t = np.linspace(0, 1, 120)
    curves = [
        ((1 - t) ** 2.2, "#5b8bd0", "Lift"),
        (np.exp(-((t - 0.5) / 0.20) ** 2), "#cccccc", "Gamma"),
        (t ** 2.2, "#d8a14a", "Gain"),
    ]
    for w, col, _lab in curves:
        xs = gx0 + t * (gx1 - gx0)
        ys = gy0 + w * (gy1 - gy0)
        ax.plot(xs, ys, color=col, lw=1.8, zorder=5)
        ax.fill_between(xs, gy0, ys, color=col, alpha=0.12, zorder=4)
    _text(ax, gx0, 38.6, "shadows", color=MUTED, size=7.4, ha="left")
    _text(ax, gx1, 38.6, "highlights", color=MUTED, size=7.4, ha="right")
    for i, (_w, col, lab) in enumerate(curves):
        lx = 67 + i * 12
        ax.add_patch(Rectangle((lx, 35.2), 1.6, 1.0, facecolor=col, edgecolor="none", zorder=6))
        _text(ax, lx + 2.2, 35.7, lab, color=TEXT, size=8, ha="left")

    # Panel C — tone curve
    _panel(ax, 4, 6, 50, 25)
    _text(ax, 29, 28, "Curves — tone response", color=TITLE, size=10.5, ha="center",
          weight="bold")
    cx0, cx1, cy0, cy1 = 14, 42, 10, 25
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, facecolor="#13233b",
                           edgecolor=BORDER_SUBTLE, lw=1.0, zorder=2))
    ax.plot([cx0, cx1], [cy0, cy1], color=MUTED, lw=1.0, ls="--", zorder=3)
    tt = np.linspace(0, 1, 120)
    sc = np.tanh((tt - 0.5) * 3.0)
    sc = (sc - sc.min()) / (sc.max() - sc.min())
    ax.plot(cx0 + tt * (cx1 - cx0), cy0 + sc * (cy1 - cy0), color=ACCENT_HOVER, lw=2.0, zorder=4)
    _text(ax, cx1, cy0 - 2.2, "input", color=MUTED, size=7.6, ha="right")
    _text(ax, cx0 - 0.5, cy1 + 1.4, "output", color=MUTED, size=7.6, ha="left")
    _text(ax, 49, 20, "dashed =\nno change\n\nsolid =\nadded\ncontrast", color=MUTED,
          size=7.0, ha="center")

    # Panel D — ASC CDL + qualifier
    _panel(ax, 58, 6, 48, 25)
    _text(ax, 82, 28, "ASC CDL + Qualifier", color=TITLE, size=10.5, ha="center", weight="bold")
    _text(ax, 82, 24.4, "out = (in × Slope + Offset) ^ Power", color=TEXT, size=9.5,
          ha="center", weight="bold")
    bars = [("Slope", 0.70, ACCENT_HOVER), ("Offset", 0.40, TITLE), ("Power", 0.55, "#d8a14a")]
    for i, (lab, val, col) in enumerate(bars):
        by = 21.0 - i * 2.4
        _text(ax, 73, by, lab, color=TEXT, size=8, ha="right")
        ax.add_patch(Rectangle((74, by - 0.7), 22, 1.4, facecolor=PANEL,
                               edgecolor=BORDER_SUBTLE, lw=0.8, zorder=3))
        ax.add_patch(Rectangle((74, by - 0.7), 22 * val, 1.4, facecolor=col,
                               edgecolor="none", zorder=4))
    _text(ax, 82, 12.4, "+ Saturation (overall)", color=MUTED, size=8, ha="center", style="italic")
    _text(ax, 82, 9.0, "Qualifier keys a hue / sat / luma region\nfor secondary, masked corrections",
          color=MUTED, size=7.6, ha="center", style="italic")
    _save(fig, "grading_controls.png")


# ────────────────────────────────────────────────────────────────────────
# 7. Keyframe interpolation / easing
# ────────────────────────────────────────────────────────────────────────
def easing_curves():
    fig, ax = _new((10, 6))
    _text(ax, 50, 95, "Keyframe interpolation & easing", color=TITLE, size=13,
          ha="center", weight="bold")

    gx0, gx1, gy0, gy1 = 16, 82, 34, 84
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="#13233b",
                           edgecolor=BORDER_SUBTLE, lw=1.0, zorder=2))
    for f in (0.25, 0.5, 0.75):
        ax.plot([gx0 + f * (gx1 - gx0)] * 2, [gy0, gy1], color="#1d3a5f", lw=0.6, zorder=3)
        ax.plot([gx0, gx1], [gy0 + f * (gy1 - gy0)] * 2, color="#1d3a5f", lw=0.6, zorder=3)
    t = np.linspace(0, 1, 120)
    eas = [
        ("Linear", t, "#9aa0a6"),
        ("Ease-in", t ** 2, "#5b8bd0"),
        ("Ease-out", 1 - (1 - t) ** 2, "#4aa564"),
        ("Ease-in-out", 3 * t ** 2 - 2 * t ** 3, ACCENT_HOVER),
    ]
    for _lab, y, col in eas:
        ax.plot(gx0 + t * (gx1 - gx0), gy0 + y * (gy1 - gy0), color=col, lw=2.0, zorder=5)
    # hold / step
    ax.plot([gx0, gx1], [gy0, gy0], color="#b0833a", lw=1.6, ls=(0, (4, 2)), zorder=4)
    ax.plot([gx1, gx1], [gy0, gy1], color="#b0833a", lw=1.6, ls=(0, (4, 2)), zorder=4)
    # legend
    legend = eas + [("Hold / step", None, "#b0833a")]
    for i, (lab, _y, col) in enumerate(legend):
        ly = 80 - i * 5
        ax.add_patch(Rectangle((86, ly - 1), 3, 2, facecolor=col, edgecolor="none", zorder=6))
        _text(ax, 90, ly, lab, color=TEXT, size=8, ha="left")
    _text(ax, 50, gy0 - 3, "time  (key A → key B)", color=MUTED, size=8.5, ha="center")
    _text(ax, gx0, gy1 + 2, "value", color=MUTED, size=8, ha="left")

    # timeline strip
    ty = 18
    ax.plot([gx0, gx1], [ty, ty], color=BORDER, lw=2.0, zorder=3)
    mid = (gx0 + gx1) / 2
    for kx, lab in [(gx0, "A"), (mid, "B"), (gx1, "C")]:
        ax.add_patch(Polygon([[kx - 2.2, ty], [kx, ty + 2.4], [kx + 2.2, ty], [kx, ty - 2.4]],
                             closed=True, facecolor=ACCENT_HOVER, edgecolor="white", lw=0.8, zorder=5))
        _text(ax, kx, ty + 5, "key " + lab, color=TEXT, size=8, ha="center")
    _text(ax, (gx0 + mid) / 2, ty - 6, "ease-in-out", color=MUTED, size=7.6, ha="center")
    _text(ax, (mid + gx1) / 2, ty - 6, "linear", color=MUTED, size=7.6, ha="center")
    _text(ax, 50, 7, "Easing is set on the destination key; the engine drives armed "
          "parameters between keys.", color=MUTED, size=8, ha="center", style="italic")
    _save(fig, "easing_curves.png")


# ────────────────────────────────────────────────────────────────────────
# 8. Corner-pin / keystone
# ────────────────────────────────────────────────────────────────────────
def corner_pin():
    fig, ax = _new_eq((9, 5))  # 90 x 50 units
    _text(ax, 45, 47, "Perspective — corner-pin / keystone", color=TITLE, size=12.5,
          ha="center", weight="bold")

    sx0, sy0, ss = 8, 12, 26
    ax.add_patch(Rectangle((sx0, sy0), ss, ss, facecolor="#13233b", edgecolor=LED, lw=1.6, zorder=3))
    _text(ax, sx0 + ss / 2, sy0 + ss + 2.5, "source (identity)", color="#bcd4ef", size=9, ha="center")
    for x, y, lab in [(sx0, sy0 + ss, "UL"), (sx0 + ss, sy0 + ss, "UR"),
                      (sx0 + ss, sy0, "LR"), (sx0, sy0, "LL")]:
        ax.add_patch(Rectangle((x - 1, y - 1), 2, 2, facecolor=ACCENT_HOVER, edgecolor="white", lw=0.6, zorder=5))
        _text(ax, x + (2 if lab[1] == "R" else -2), y + (1.5 if lab[0] == "U" else -1.5), lab,
              color=MUTED, size=7.2, ha=("left" if lab[1] == "R" else "right"))

    quad = [(52, 36), (86, 40), (82, 12), (50, 16)]  # UL, UR, LR, LL (keystone)
    ax.add_patch(Polygon(quad, closed=True, facecolor="#1c3a1c", edgecolor=SUCCESS, lw=1.6, zorder=3))
    for (x, y), lab in zip(quad, ["UL", "UR", "LR", "LL"]):
        ax.add_patch(Rectangle((x - 1, y - 1), 2, 2, facecolor=SUCCESS, edgecolor="white", lw=0.6, zorder=5))
        _text(ax, x, y + (2 if lab[0] == "U" else -2.2), lab, color="#bfe6bf", size=7.2, ha="center")

    ax.add_patch(FancyArrowPatch((37, 25), (49, 25), arrowstyle="-|>", mutation_scale=14,
                                color=MUTED, lw=1.6, zorder=6))
    _text(ax, 68, 6.5, "drag the 4 corners  →  MIXER PERSPECTIVE", color=MUTED, size=8,
          ha="center", style="italic")
    _save(fig, "corner_pin.png")


# ────────────────────────────────────────────────────────────────────────
# 9. Camera tracking → virtual scene (coordinate mapping)
# ────────────────────────────────────────────────────────────────────────
def tracking_coords():
    fig, ax = _new((10, 5.4))
    _text(ax, 50, 95, "Camera tracking → virtual scene", color=TITLE, size=13,
          ha="center", weight="bold")

    # Real world
    _panel(ax, 4, 16, 40, 64, fc="#1a2330")
    _text(ax, 24, 74, "Real world", color="#bcd4ef", size=10, ha="center", weight="bold")
    ox, oy = 12, 28
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + 12, oy), arrowstyle="-|>", mutation_scale=10,
                                color=DANGER, lw=1.6, zorder=5))
    ax.add_patch(FancyArrowPatch((ox, oy), (ox, oy + 12), arrowstyle="-|>", mutation_scale=10,
                                color=SUCCESS, lw=1.6, zorder=5))
    _text(ax, ox + 13, oy, "X", color=DANGER, size=8, ha="left")
    _text(ax, ox, oy + 13.5, "Y", color=SUCCESS, size=8, ha="center")
    _text(ax, ox - 1, oy - 3, "tracker origin", color=MUTED, size=7.6, ha="left")
    ax.add_patch(Rectangle((26, 52), 9, 6, facecolor=PANEL, edgecolor=TEXT, lw=1.4, zorder=5))
    ax.add_patch(Polygon([[35, 55], [42, 60], [42, 50]], closed=True, facecolor=ACCENT,
                         alpha=0.4, edgecolor=ACCENT_HOVER, lw=1.0, zorder=4))
    _text(ax, 30, 49, "tracked camera", color=TEXT, size=8, ha="center")

    # pose hand-off
    ax.add_patch(FancyArrowPatch((44, 48), (58, 48), arrowstyle="-|>", mutation_scale=15,
                                color=TITLE, lw=2.0, zorder=6))
    _text(ax, 51, 53, "pose", color=TITLE, size=9, ha="center", weight="bold")
    _text(ax, 51, 43.5, "x y z\nyaw pitch roll", color=MUTED, size=7.6, ha="center")

    # Virtual scene
    _panel(ax, 58, 16, 38, 64, fc="#1a2330")
    _text(ax, 77, 74, "Virtual scene", color="#bcd4ef", size=10, ha="center", weight="bold")
    cx, cy, R = 77, 18, 46
    th = np.linspace(np.deg2rad(62), np.deg2rad(118), 60)
    ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=LED, lw=5, zorder=4, solid_capstyle="round")
    _text(ax, 77, 66, "LED wall", color=LED, size=8, ha="center", weight="bold")
    ax.add_patch(Rectangle((73, 40), 8, 5, facecolor=PANEL, edgecolor=SUCCESS, lw=1.4, zorder=5))
    _text(ax, 77, 37.0, "virtual camera", color="#bfe6bf", size=7.8, ha="center")
    p1 = (cx + R * np.cos(np.deg2rad(100)), cy + R * np.sin(np.deg2rad(100)))
    p2 = (cx + R * np.cos(np.deg2rad(80)), cy + R * np.sin(np.deg2rad(80)))
    ax.add_patch(Polygon([(77, 44), p1, p2], closed=True, facecolor=ACCENT, alpha=0.25,
                         edgecolor=ACCENT_HOVER, lw=1.2, zorder=3))

    _text(ax, 50, 8, "World-alignment maps the tracker origin to the stage origin (offset + "
          "scale) so the virtual camera mirrors the real one.", color=MUTED, size=8,
          ha="center", style="italic")
    _save(fig, "tracking_coords.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Edge blend — overlapping projectors sum to uniform
# ─────────────────────────────────────────────────────────────────────────────
def edge_blend():
    fig, ax = _new((10, 5.4))
    _text(ax, 50, 95, "Edge blend — overlapping projectors sum to uniform",
          color=TITLE, size=12.5, ha="center", weight="bold")

    ax.add_patch(Rectangle((8, 80), 12, 7, facecolor=PANEL, edgecolor=TEXT, lw=1.3, zorder=5))
    _text(ax, 14, 83.5, "Proj A", color=TEXT, size=8, ha="center")
    ax.add_patch(Rectangle((80, 80), 12, 7, facecolor=PANEL, edgecolor=TEXT, lw=1.3, zorder=5))
    _text(ax, 86, 83.5, "Proj B", color=TEXT, size=8, ha="center")

    gx0, gx1, gy0, gy1 = 14, 90, 16, 66
    ax.add_patch(Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, facecolor="#13233b",
                           edgecolor=BORDER_SUBTLE, lw=1.0, zorder=2))
    t = np.linspace(0, 1, 200)
    os_, oe = 0.42, 0.64
    A = np.clip((oe - t) / (oe - os_), 0, 1)
    B = np.clip((t - os_) / (oe - os_), 0, 1)
    xs = gx0 + t * (gx1 - gx0)
    xa, xb = gx0 + os_ * (gx1 - gx0), gx0 + oe * (gx1 - gx0)
    ax.add_patch(Rectangle((xa, gy0), xb - xa, gy1 - gy0, facecolor="#ffffff", alpha=0.05, zorder=3))
    ax.fill_between(xs, gy0, gy0 + A * (gy1 - gy0), color="#d8a14a", alpha=0.16, zorder=3)
    ax.fill_between(xs, gy0, gy0 + B * (gy1 - gy0), color="#5b8bd0", alpha=0.16, zorder=3)
    ax.plot(xs, gy0 + A * (gy1 - gy0), color="#d8a14a", lw=1.8, zorder=5)
    ax.plot(xs, gy0 + B * (gy1 - gy0), color="#5b8bd0", lw=1.8, zorder=5)
    ax.plot(xs, gy0 + (A + B) * (gy1 - gy0), color="#ffffff", lw=1.6, ls=(0, (4, 2)), zorder=6)
    _text(ax, (xa + xb) / 2, gy1 + 2, "overlap (blend band)", color=MUTED, size=8, ha="center")

    leg = [("Proj A ramp", "#d8a14a"), ("Proj B ramp", "#5b8bd0"), ("Sum = uniform", "#ffffff")]
    for i, (lab, col) in enumerate(leg):
        ly = gy1 - 4 - i * 4
        ax.add_patch(Rectangle((gx0 + 2, ly - 0.8), 3, 1.6, facecolor=col, edgecolor="none", zorder=7))
        _text(ax, gx0 + 6.5, ly, lab, color=TEXT, size=7.6, ha="left")
    _text(ax, 50, gy0 - 3, "position across the seam", color=MUTED, size=8, ha="center")
    _text(ax, 50, 9, "Each projector feathers across the overlap; a pow(ramp, gamma) curve "
          "keeps the perceived sum flat.", color=MUTED, size=8, ha="center", style="italic")
    _save(fig, "edge_blend.png")


def main():
    channel_workspace()
    stage_layout()
    icvfx_frustum()
    projection_geometry()
    led_tiling()
    grading_controls()
    easing_curves()
    corner_pin()
    tracking_coords()
    edge_blend()
    print("done")


if __name__ == "__main__":
    main()
