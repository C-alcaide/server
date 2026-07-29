"""Generate figures for docs/DECKLINK_GPU_DIRECT_OUTPUT.md.

Run:  python docs/gen_decklink_figures.py
Outputs PNGs into docs/images/decklink/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "images", "decklink")
os.makedirs(OUT, exist_ok=True)

# Palette
C_BG      = "#0f1420"
C_PANEL   = "#1b2436"
C_GPU     = "#2d6cdf"
C_HOST    = "#2aa198"
C_CARD    = "#b58900"
C_ACCENT  = "#e0567a"
C_TEXT    = "#e8ecf4"
C_MUTED   = "#9aa6bd"
C_GREEN   = "#3fb950"
C_LINE    = "#3a4761"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "savefig.facecolor": C_BG,
    "text.color": C_TEXT,
    "axes.facecolor": C_BG,
    "axes.edgecolor": C_LINE,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "font.size": 11,
    "font.family": "DejaVu Sans",
})


def box(ax, x, y, w, h, text, color, tcolor=C_TEXT, fs=10, alpha=1.0, style="round,pad=0.02,rounding_size=0.06"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.4,
                       edgecolor=C_LINE, facecolor=color, alpha=alpha, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tcolor, fontsize=fs, weight="bold", zorder=5, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=C_TEXT, ls="-", lw=1.8, mut=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=mut, color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=4))


# ---------------------------------------------------------------------------
# Figure 1: end-to-end data path (packing tier + transfer tier + scheduling)
# ---------------------------------------------------------------------------
def fig_datapath():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(6.5, 6.95, "DeckLink GPU-direct output — end-to-end data path",
            ha="center", fontsize=15, weight="bold")

    # GPU domain band
    box(ax, 0.2, 4.2, 12.6, 2.35, "", C_PANEL, alpha=0.45)
    ax.text(0.45, 6.32, "GPU  (mixer device — RTX A4000)", color=C_GPU, fontsize=11, weight="bold")
    # Host domain band
    box(ax, 0.2, 2.0, 12.6, 1.7, "", C_PANEL, alpha=0.45)
    ax.text(0.45, 3.48, "HOST  (page-locked system memory)", color=C_HOST, fontsize=11, weight="bold")
    # Card domain band
    box(ax, 0.2, 0.35, 12.6, 1.25, "", C_PANEL, alpha=0.45)
    ax.text(0.45, 1.38, "DeckLink 8K Pro", color=C_CARD, fontsize=11, weight="bold")

    # GPU: mixer texture
    box(ax, 0.5, 4.55, 2.2, 1.25, "Mixer\ncomposited\ntexture\n(RGBA/f16)", C_GPU, fs=9.5)
    # Pack stage
    box(ax, 3.35, 4.55, 3.0, 1.25, "GPU pack (compute)\nv210 / BGRA\nOGL: ogl_gl_strategy\nVK: cuda_vk / vk_readback", C_GPU, fs=8.6)
    # CPU pack alt
    box(ax, 3.35, 2.15, 3.0, 1.05, "CPU pack (AVX2)\nv210 / BGRA\n<gpu-pack>cpu", C_HOST, fs=8.8)
    # SSBO
    box(ax, 6.9, 4.55, 1.7, 1.25, "packed\nGPU buffer\n(SSBO)", C_GPU, fs=9)

    # Transfer tier (two options)
    box(ax, 9.15, 5.15, 3.4, 0.72, "copy: glGetBufferSubData / cudaMemcpy", C_ACCENT, fs=8.6)
    box(ax, 9.15, 4.28, 3.4, 0.72, "dvp: GPUDirect DMA (HW sync)", C_GREEN, fs=8.8)

    # Host pinned pool
    box(ax, 6.9, 2.15, 3.0, 1.05, "page-locked pool\ngpu_output_buffer_pool\n(VirtualLock / cudaMallocHost)", C_HOST, fs=8.4)

    # decklink_frame
    box(ax, 10.4, 2.2, 2.1, 0.95, "decklink_frame\nGetBytes() ->\npinned ptr", C_HOST, fs=8.6)

    # Card DMA + SDI
    box(ax, 6.9, 0.6, 3.0, 0.8, "driver DMA from pinned frame", C_CARD, fs=9)
    box(ax, 10.4, 0.6, 2.1, 0.8, "SDI out\n(v210 / BGRA)", C_CARD, fs=9)

    # arrows GPU pack path
    arrow(ax, 2.7, 5.17, 3.35, 5.17, C_TEXT)
    arrow(ax, 6.35, 5.17, 6.9, 5.17, C_TEXT)
    arrow(ax, 8.6, 5.3, 9.15, 5.5, C_ACCENT)
    arrow(ax, 8.6, 5.05, 9.15, 4.62, C_GREEN)
    # transfer -> pinned pool
    arrow(ax, 10.85, 5.15, 9.0, 3.2, C_ACCENT, lw=1.6)
    arrow(ax, 10.85, 4.28, 9.2, 3.2, C_GREEN, lw=1.6)
    # CPU pack -> pinned pool
    arrow(ax, 6.35, 2.67, 6.9, 2.67, C_TEXT)
    # mixer texture -> CPU pack (readback) dashed
    arrow(ax, 1.6, 4.55, 1.6, 3.2, C_MUTED, ls="--", lw=1.4)
    arrow(ax, 1.6, 3.2, 3.35, 2.67, C_MUTED, ls="--", lw=1.4)
    ax.text(1.75, 3.75, "full readback\n(CPU path only)", color=C_MUTED, fontsize=7.5, ha="left")
    # pinned pool -> decklink_frame
    arrow(ax, 9.9, 2.67, 10.4, 2.67, C_TEXT)
    # decklink_frame -> card DMA
    arrow(ax, 11.45, 2.2, 11.45, 1.9, C_TEXT)
    arrow(ax, 8.4, 2.15, 8.4, 1.4, C_TEXT)
    # card DMA -> SDI
    arrow(ax, 9.9, 1.0, 10.4, 1.0, C_TEXT)

    # knobs
    ax.text(6.5, 0.06, "<gpu-pack> selects pack location (gpu/cpu)   •   <gpu-transfer> selects transfer (copy/dvp)   •   both opt-in, per-consumer",
            ha="center", color=C_MUTED, fontsize=8.5)

    legend = [
        Line2D([0], [0], color=C_ACCENT, lw=3, label="copy transfer (Tier 1, any NVIDIA/AMD)"),
        Line2D([0], [0], color=C_GREEN, lw=3, label="DVP transfer (Tier 2, GPUDirect)"),
        Line2D([0], [0], color=C_MUTED, lw=2, ls="--", label="CPU readback path"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.012, 0.008),
              framealpha=0.15, facecolor=C_PANEL, edgecolor=C_LINE, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "datapath.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: output latency by scheduling mode (measured, 1080p25)
# ---------------------------------------------------------------------------
def fig_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})

    modes = ["normal\n(default)", "low\n<latency>low", "sync\n<latency>sync"]
    app_frames = [4, 3, 1]
    colors = ["#5b6b8c", C_GPU, C_GREEN]

    bars = ax1.bar(modes, app_frames, color=colors, edgecolor=C_LINE, width=0.62, zorder=3)
    ax1.set_ylabel("app-side buffered frames (measured)")
    ax1.set_title("Output buffering depth by mode  (8K Pro, 1080p25)", fontsize=11.5)
    ax1.set_ylim(0, 5)
    ax1.grid(axis="y", color=C_LINE, alpha=0.4, zorder=0)
    for b, f in zip(bars, app_frames):
        ax1.text(b.get_x() + b.get_width() / 2, f + 0.12, f"{f} frame{'s' if f != 1 else ''}\n{f*40} ms",
                 ha="center", va="bottom", fontsize=9.5, weight="bold")

    # scheduling timeline schematic
    ax2.set_title("Frame path: scheduled vs synchronous", fontsize=11.5)
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 5)
    ax2.axis("off")

    def track(y, label, segs, col):
        ax2.text(0.15, y + 0.62, label, color=C_TEXT, fontsize=9.5, weight="bold")
        for (x, w, t) in segs:
            box(ax2, x, y - 0.32, w, 0.62, t, col, fs=7.6)

    # scheduled: composite -> pack -> preroll queue (3-4) -> card
    track(3.5, "scheduled  (normal / low)", [
        (0.2, 1.5, "composite"),
        (1.8, 1.1, "pack"),
        (3.0, 3.1, "preroll queue\n(3-4 frames)"),
        (6.2, 2.6, "card\nscan-out"),
    ], C_PANEL)
    track(1.2, "sync-display", [
        (0.2, 1.5, "composite"),
        (1.8, 1.1, "pack"),
        (3.0, 2.1, "Display\nFrameSync"),
        (5.2, 2.6, "card\nscan-out"),
    ], C_PANEL)
    ax2.annotate("", xy=(7.85, 0.55), xytext=(10.85, 0.55),
                 arrowprops=dict(arrowstyle="<->", color=C_GREEN, lw=1.6))
    ax2.text(9.35, 0.18, "~3 fewer frames", color=C_GREEN, fontsize=8.5, ha="center", weight="bold")

    fig.subplots_adjust(bottom=0.28, top=0.9, left=0.08, right=0.97, wspace=0.18)
    fig.text(0.08, 0.045,
             "40 ms / frame at 25p.  'sync' removes the preroll queue entirely.  The low-latency driver flag\n"
             "additionally removes up to ~3 frames of driver-internal latency (not visible in the app-side count).",
             fontsize=8.4, color=C_MUTED, ha="left")
    fig.savefig(os.path.join(OUT, "latency.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: sync-display threading model (primary + N secondaries)
# ---------------------------------------------------------------------------
def fig_threads():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.text(6, 6.1, "Synchronous-display threading model (multi-port + audio)",
            ha="center", fontsize=14, weight="bold")

    # channel producer
    box(ax, 0.4, 4.6, 2.3, 1.0, "channel\nframe tick\n(video rate)", C_GPU, fs=9)
    # bounded buffer
    box(ax, 3.1, 4.6, 1.9, 1.0, "consumer\nbuffer_\n(bounded)", C_HOST, fs=9)
    arrow(ax, 2.7, 5.1, 3.1, 5.1)
    ax.text(2.9, 5.32, "send()", color=C_MUTED, fontsize=7.5, ha="center")

    # primary sync loop
    box(ax, 5.4, 4.5, 3.1, 1.25, "PRIMARY sync loop  (thread)\nprec_timer.tick(1/fps)\npop -> fan-out -> pack\nDisplayVideoFrameSync\nWriteAudioSamplesSync", C_GREEN, fs=8.0)
    arrow(ax, 5.0, 5.1, 5.4, 5.1)

    # audio out
    box(ax, 9.0, 5.35, 2.6, 0.7, "SDI dev 1 + embedded audio", C_CARD, fs=8.6)
    arrow(ax, 8.5, 5.35, 9.0, 5.6)

    # fan-out to secondaries
    box(ax, 5.55, 2.7, 2.8, 0.85, "secondary[0] queue\n(bounded, drop-oldest)", C_PANEL, fs=8.2)
    box(ax, 5.55, 1.5, 2.8, 0.85, "secondary[N] queue", C_PANEL, fs=8.2)
    arrow(ax, 6.9, 4.5, 6.9, 3.55, C_ACCENT)
    ax.text(7.05, 3.95, "push_sync_frame()", color=C_ACCENT, fontsize=7.5, ha="left")
    ax.text(6.95, 2.5, "...", color=C_MUTED, fontsize=14, ha="center")

    box(ax, 8.7, 2.7, 2.9, 0.85, "SEC[0] sync loop (thread)\npack -> DisplayVideoFrameSync", C_GREEN, fs=7.8)
    box(ax, 8.7, 1.5, 2.9, 0.85, "SEC[N] sync loop (thread)", C_GREEN, fs=7.8)
    arrow(ax, 8.35, 3.12, 8.7, 3.12, C_ACCENT)
    arrow(ax, 8.35, 1.92, 8.7, 1.92, C_ACCENT)

    box(ax, 8.7, 0.35, 2.9, 0.7, "SDI dev 2..N (subregion crops)", C_CARD, fs=8.4)
    arrow(ax, 10.15, 1.5, 10.15, 1.05, C_TEXT)

    ax.text(6, 0.15, "Each device blocks on its own DisplayVideoFrameSync; genlock keeps them aligned. "
                     "Without a reference the primary prec_timer paces the channel.",
            ha="center", color=C_MUTED, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "threads.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_datapath()
    fig_latency()
    fig_threads()
    print("wrote figures to", OUT)
