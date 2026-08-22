"""Generate figures for docs/PIPELINE_EFFICIENCY_GUIDE.md.

Run:  python docs/gen_pipeline_figures.py
Outputs PNGs into docs/images/pipeline/.

Every number plotted here is measured, not illustrative. Sources are named in the
guide; re-measure with the harnesses in CasparCG-TestRunner/vkdispatch/ before
trusting them on different hardware.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "images", "pipeline")
os.makedirs(OUT, exist_ok=True)

# Palette, matching gen_decklink_figures.py so the docs look of a piece.
C_BG = "#0f1420"
C_PANEL = "#1b2436"
C_GPU = "#2d6cdf"
C_HOST = "#2aa198"
C_CARD = "#b58900"
C_ACCENT = "#e0567a"
C_TEXT = "#e8ecf4"
C_MUTED = "#9aa6bd"
C_GREEN = "#3fb950"
C_RED = "#f85149"
C_LINE = "#3a4761"

plt.rcParams.update(
    {
        "figure.facecolor": C_BG,
        "savefig.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "text.color": C_TEXT,
        "axes.labelcolor": C_TEXT,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "axes.edgecolor": C_LINE,
        "font.size": 10,
    }
)


def box(ax, x, y, w, h, label, color, sub=None, fs=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=color, facecolor=C_PANEL,
        )
    )
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), label,
            ha="center", va="center", color=C_TEXT, fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.13, sub,
                ha="center", va="center", color=C_MUTED, fontsize=fs - 2.2)


def arrow(ax, p0, p1, color, style="-", label=None, lw=1.8, rad=0.0, dy=0.12):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color, linestyle=style,
                        connectionstyle=f"arc3,rad={rad}")
    )
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, label,
                ha="center", va="bottom", color=color, fontsize=8.2)


# ---------------------------------------------------------------- figure 1
def fig_producer_path():
    """Where a decoded frame's format is decided, and what it costs."""
    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.9); ax.axis("off")

    ax.text(0.15, 5.56, "Producer → mixer: what reaches the GPU, and how big it is",
            fontsize=13, weight="bold", color=C_TEXT)

    box(ax, 0.15, 4.0, 1.9, 1.0, "decoder", C_HOST, "hardware or software")
    box(ax, 2.45, 4.0, 2.5, 1.0, "filter graph", C_ACCENT, "bwdif · fps · user filters")
    box(ax, 5.35, 4.0, 2.3, 1.0, "upload", C_GPU, "host → GPU, PCIe")
    box(ax, 8.05, 4.0, 2.0, 1.0, "mixer", C_GPU, "composition")
    box(ax, 10.45, 4.0, 1.8, 1.0, "consumers", C_CARD)

    for a, b in ((2.05, 2.45), (4.95, 5.35), (7.65, 8.05), (10.05, 10.45)):
        arrow(ax, (a, 4.5), (b, 4.5), C_LINE)

    # GPU-direct decode bypass, routed *under* the stages it skips so it reads as
    # a bypass rather than as another step. The label sits below the arc's dip.
    arrow(ax, (1.1, 4.0), (9.0, 4.0), C_GREEN, style="--", rad=0.26, lw=1.6)
    ax.text(5.05, 2.74, "GPU-direct decode (D3D11VA, OpenGL only): decoder surface → mixer, no host copy at all",
            ha="center", color=C_GREEN, fontsize=8.6)

    ax.text(0.15, 2.10, "The filter graph decides the format — and therefore the upload size",
            fontsize=10.5, weight="bold", color=C_TEXT)
    notes = [
        ("bwdif is skipped when the container declares the stream progressive.", C_GREEN),
        ("It is a pass-through on progressive frames, but its format constraints still applied:", C_MUTED),
        ("NV12 was de-interleaved to yuv420p every frame, 10-bit truncated to 8, 4:4:4 subsampled to 4:2:0.", C_MUTED),
        ("The sink is now offered only formats that cannot lose depth, chroma or alpha.", C_GREEN),
    ]
    for i, (t, c) in enumerate(notes):
        ax.text(0.3, 1.68 - i * 0.34, ("• " if i != 2 else "   ") + t, color=c, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "producer_path.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_capacity():
    """Upload bandwidth is what limits layer count."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # Ascending, so the cost ordering is visible at a glance.
    fmts = ["8-bit 4:2:0\n(NV12)", "10-bit 4:2:0", "8-bit RGBA", "16-bit RGBA\n+ alpha"]
    mb = [3.0, 5.93, 8.3, 15.0]
    colors = [C_GREEN, C_HOST, C_CARD, C_RED]
    bars = ax1.bar(fmts, mb, color=colors, edgecolor=C_LINE, linewidth=0.8)
    ax1.set_ylabel("MB per 1080p frame, per layer")
    ax1.set_title("Upload cost by source format", color=C_TEXT, fontsize=11, pad=10)
    for b, v in zip(bars, mb):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}",
                 ha="center", color=C_TEXT, fontsize=9)
    ax1.set_ylim(0, 17.5)
    ax1.grid(axis="y", color=C_LINE, alpha=0.35, linewidth=0.7)
    ax1.set_axisbelow(True)

    # Layers affordable at 9.1 GB/s
    budget_ms = [20, 40]
    labels = ["1080p50 (20 ms)", "1080p25 (40 ms)"]
    width = 0.35
    xs = range(len(fmts))
    for i, (t, lbl) in enumerate(zip(budget_ms, labels)):
        vals = [9.1 * 1024 * (t / 1000.0) / m for m in mb]
        ax2.bar([x + (i - 0.5) * width for x in xs], vals, width,
                label=lbl, color=[C_GPU, C_ACCENT][i], edgecolor=C_LINE, linewidth=0.8)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(fmts, fontsize=8.5)
    ax2.set_ylabel("layers before the bus is the limit")
    ax2.set_title("Measured ceiling: 9.1 GB/s on PCIe 3.0 x16", color=C_TEXT, fontsize=11, pad=10)
    ax2.legend(facecolor=C_PANEL, edgecolor=C_LINE, labelcolor=C_TEXT, fontsize=8.5)
    ax2.grid(axis="y", color=C_LINE, alpha=0.35, linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "capacity.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_tick():
    """Where the channel tick actually goes, and what back-pressure looks like."""
    layers = [0, 1, 4, 8, 16, 24]
    produce = [0.02, 0.16, 0.46, 0.88, 1.54, 2.44]
    mix = [0.02, 0.08, 0.50, 1.76, 27.90, 39.42]
    consume = [19.88, 19.62, 18.84, 16.74, 0.28, 0.10]

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    x = range(len(layers))
    ax.bar(x, consume, color=C_HOST, label="consume (waiting on the consumer = the clock)",
           edgecolor=C_LINE, linewidth=0.6)
    ax.bar(x, mix, bottom=consume, color=C_GPU, label="mix (composition + waiting for uploads)",
           edgecolor=C_LINE, linewidth=0.6)
    ax.bar(x, produce, bottom=[c + m for c, m in zip(consume, mix)],
           color=C_CARD, label="produce (pulling every layer)", edgecolor=C_LINE, linewidth=0.6)

    ax.axhline(20, color=C_RED, linestyle="--", linewidth=1.4)
    ax.text(5.45, 21.2, "20 ms frame budget", color=C_RED, fontsize=9, ha="right")

    ax.set_xticks(list(x)); ax.set_xticklabels([str(v) for v in layers])
    ax.set_xlabel("layers of 1080p50 (16-bit RGBA source)")
    ax.set_ylabel("ms per tick")
    ax.set_title("Channel tick decomposition — consume shrinking is back-pressure absorbing load",
                 color=C_TEXT, fontsize=11, pad=10)
    ax.legend(facecolor=C_PANEL, edgecolor=C_LINE, labelcolor=C_TEXT, fontsize=8.6, loc="upper left")
    ax.grid(axis="y", color=C_LINE, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)

    ax.annotate("slack gone → late", xy=(4, 29.7), xytext=(3.05, 36),
                color=C_RED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_RED, linewidth=1.3))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tick_decomposition.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_recording():
    """The recording round trip, and what removes it."""
    fig, ax = plt.subplots(figsize=(12.4, 4.5))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.5); ax.axis("off")

    ax.text(0.15, 4.16, "Recording: the round trip, and the path that avoids it",
            fontsize=13, weight="bold", color=C_TEXT)

    # Host path
    ax.text(0.15, 3.52, "Host path — any encoder", color=C_MUTED, fontsize=9.5)
    box(ax, 0.15, 2.42, 2.0, 0.9, "mixer", C_GPU, "composited")
    box(ax, 3.15, 2.42, 2.1, 0.9, "host memory", C_HOST, "readback")
    box(ax, 6.25, 2.42, 2.1, 0.9, "encoder", C_CARD, "uploads again")
    arrow(ax, (2.15, 2.87), (3.15, 2.87), C_RED, label="11.4 ms @ 4K")
    arrow(ax, (5.25, 2.87), (6.25, 2.87), C_RED, label="3.4 ms @ 4K")
    ax.text(8.6, 2.87, "≈ 14.8 ms of transfer\nper 4K frame", color=C_RED, fontsize=9, va="center")

    # GPU-direct
    ax.text(0.15, 1.86, "GPU-direct — NVENC, OpenGL mixer", color=C_MUTED, fontsize=9.5)
    box(ax, 0.15, 0.76, 2.0, 0.9, "mixer", C_GPU, "composited")
    box(ax, 3.15, 0.76, 2.1, 0.9, "CUDA frame", C_GPU, "device-to-device")
    box(ax, 6.25, 0.76, 2.1, 0.9, "NVENC", C_GREEN, "RGB in, no copy")
    arrow(ax, (2.15, 1.21), (3.15, 1.21), C_GREEN)
    arrow(ax, (5.25, 1.21), (6.25, 1.21), C_GREEN)
    ax.text(8.6, 1.21, "never leaves the GPU\n18 % less CPU at 4K", color=C_GREEN, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recording_paths.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_gpudirect_gate():
    """Every condition GPU-direct recording checks, and why."""
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(0.15, 6.06, "When GPU-direct recording engages", fontsize=13, weight="bold", color=C_TEXT)

    gates = [
        ("encoder is NVENC", "the only encoder here that reads device memory"),
        ("no user video filter", "lavfi filters operate on host frames"),
        ("no explicit -pix_fmt", "the frames are CUDA/RGB0; lavfi cannot reformat them"),
        ("8-bit channel", "the copy is byte-for-byte from an RGBA8 texture"),
        ("CUDA device present", "either mixer -- OpenGL or Vulkan"),
    ]
    y = 5.32
    for name, why in gates:
        box(ax, 0.15, y - 0.32, 3.0, 0.62, name, C_GPU, fs=9.5)
        ax.text(3.45, y, "→", color=C_MUTED, fontsize=12, va="center")
        ax.text(3.95, y, why, color=C_MUTED, fontsize=9, va="center")
        y -= 0.82

    box(ax, 0.15, 0.15, 3.0, 0.62, "all true → engages", C_GREEN, fs=9.5)
    ax.text(3.95, 0.46, "otherwise the host path runs, and the reason is logged",
            color=C_TEXT, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gpudirect_gate.png"), dpi=150)
    plt.close(fig)


def fig_vulkan_encode_gate():
    """The second GPU recording route, and the two things it insists on.

    A separate figure rather than more rows on the NVENC one, because the point is that these
    are TWO ROUTES to the same place with different requirements -- and the codec each can reach
    is what decides between them, not a preference.
    """
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 6.0); ax.axis("off")
    ax.text(0.15, 5.66, "When the Vulkan encoders engage", fontsize=13, weight="bold", color=C_TEXT)
    ax.text(0.15, 5.30, "NVENC cannot encode ProRes or FFV1. This route can, with no readback.",
            fontsize=9.5, color=C_MUTED)

    gates = [
        ("encoder ends _vulkan", "prores_ks / ffv1 / h264 / hevc; av1 needs an Ada GPU"),
        ("Vulkan mixer", "the exporter copies into an FFmpeg VkImage on the same device"),
        ("16-bit channel", "libplacebo exchanges red and blue on a BGRA frame; 16-bit is RGBA"),
        ("no user video filter", "this path owns the filter chain"),
    ]
    y = 4.62
    for name, why in gates:
        box(ax, 0.15, y - 0.32, 3.0, 0.62, name, C_GPU, fs=9.5)
        ax.text(3.45, y, "→", color=C_MUTED, fontsize=12, va="center")
        ax.text(3.95, y, why, color=C_MUTED, fontsize=9, va="center")
        y -= 0.90

    box(ax, 0.15, 0.85, 3.0, 0.62, "all true → engages", C_GREEN, fs=9.5)
    ax.text(3.95, 1.16, "otherwise the host path runs, and the reason is logged",
            color=C_TEXT, fontsize=9, va="center")
    ax.text(0.15, 0.34,
            "Picture agrees with the CPU encoder to a mean of 2.5 LSB. Cost is NOT measured, and "
            "rate-control defaults are untuned — set a bitrate.",
            fontsize=8.5, color=C_MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vulkan_encode_gate.png"), dpi=150)
    plt.close(fig)


def fig_recording_routes():
    """Which recording route to ask for, per codec, with what each one needs.

    A table answers "what did they cost"; this answers "which one do I type", which is the
    question an operator actually has. Every number is from `cli.py encode-matrix`, 1080p2500,
    two interleaved rounds.
    """
    # (codec, [(route, requirement, cores, note, state)]) -- state picks the colour.
    groups = [
        ("ProRes", [
            ("prores_ks_vulkan", "vulkan mixer, 16-bit", "1.46", "all frames", "best"),
            ("CUDA_PRORES", "OpenGL mixer, 8-bit", "1.64", "GPU-direct, all frames", "ok"),
            ("prores_aw", "any", "2.24", "all frames", "ok"),
            ("prores_ks", "any", "2.32", "KEEPS 138 OF 260 FRAMES", "bad"),
        ]),
        ("H.264 / HEVC", [
            ("h264_vulkan", "vulkan mixer, 16-bit", "1.42", "NVENC block 15%", "best"),
            ("hevc_vulkan", "vulkan mixer, 16-bit", "1.41", "NVENC block 39%", "best"),
            ("libx264 / libx265", "any", "2.18 / 2.83", "all frames", "ok"),
            ("h264_nvenc / hevc_nvenc", "8-bit", "--", "REFUSED by this build", "bad"),
        ]),
        ("FFV1", [
            ("ffv1_vulkan", "vulkan mixer, 16-bit", "1.50", "18x the disk", "warn"),
            ("ffv1", "any", "2.26", "all frames", "ok"),
        ]),
    ]

    # SIZED FROM THE ROW COUNT, not guessed. A fixed 7.4 clipped the last group's boxes off the
    # canvas entirely and ran the footnote through a row -- and a diagram that loses a row is
    # worse than no diagram, because the reader cannot tell it is incomplete.
    ROW, HEAD, GAP = 0.66, 0.44, 0.16
    n_rows = sum(len(r) for _, r in groups)
    height = 1.30 + len(groups) * (HEAD + GAP) + n_rows * ROW + 0.70
    fig, ax = plt.subplots(figsize=(12.4, height))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, height); ax.axis("off")
    ax.text(0.15, height - 0.34, "Which recording route, and what it needs", fontsize=13,
            weight="bold", color=C_TEXT)
    ax.text(0.15, height - 0.68, "cores are for one 1080p25 layer; the winner of each group is "
                                 "boxed in green", fontsize=9, color=C_MUTED)

    colours = {"best": C_GREEN, "ok": C_HOST, "warn": C_CARD, "bad": C_RED}
    y = height - 1.30
    for codec, routes in groups:
        ax.text(0.15, y, codec, fontsize=11, weight="bold", color=C_TEXT)
        y -= HEAD
        for name, need, cores, note, state in routes:
            box(ax, 0.35, y - 0.28, 3.5, 0.56, name, colours[state], fs=9)
            ax.text(4.05, y, need, color=C_MUTED, fontsize=8.6, va="center")
            ax.text(7.15, y, cores + " cores", color=C_TEXT, fontsize=8.6, va="center")
            ax.text(8.75, y, note, color=colours[state], fontsize=8.6, va="center")
            y -= ROW
        y -= GAP

    ax.text(0.15, max(0.22, y + 0.10),
            "NVENC is refused because the pinned FFmpeg needs driver 610+; the fix is to rebuild "
            "against nv-codec-headers n13.0, NOT to raise the\n"
            "driver, which would drop the Pascal P4000 in the other slot. No single channel serves "
            "all three fast paths: NVENC GPU-direct needs 8-bit, "
            "the Vulkan encoders need 16-bit, and CUDA_PRORES needs the OpenGL mixer.\n"
            "Cost figures are untuned defaults; the recorded sizes differ by over 10x.",
            fontsize=8.2, color=C_MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recording_routes.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_producer_path()
    fig_capacity()
    fig_tick()
    fig_recording()
    fig_gpudirect_gate()
    fig_vulkan_encode_gate()
    fig_recording_routes()
    print("wrote figures to", OUT)
