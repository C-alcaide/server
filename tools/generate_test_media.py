#!/usr/bin/env python3
"""
Generate test color-bar media for CasparVP pipeline validation.

Produces ProRes HQ files with CORRECT colorimetry for:
  - BT.709 SDR color bars (SMPTE RP 219 with black/white reference)
  - BT.2020 PQ (HDR10) — BT.2111-1 bars with proper PQ-encoded luminance
  - BT.2020 HLG — BT.2111-2 bars with HLG OETF
  - HDR reference ramp — 0 to 4000 nits in 64 PQ steps
  - BT.2020 gamut test — saturated primaries outside BT.709

All HDR patterns include:
  - Row 1 (top 67%):  75% color bars (proper OETF applied)
  - Row 2 (next 8%):  100% saturated bars
  - Row 3 (next 8%):  Grayscale ramp (black to peak)
  - Row 4 (bottom 17%): Reference patches (black, near-black, ref-white, peak)

Usage:
    python generate_test_media.py [--output-dir path] [--ffmpeg path/to/ffmpeg]
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

# Default paths
DEFAULT_FFMPEG = "ffmpeg"  # use system PATH
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1].parent / "CasparCG-TestRunner" / "media" / "sources"

# Common encoding settings
DURATION = "3"
FRAMERATE = "25"
WIDTH, HEIGHT = 1920, 1080
PRORES_PROFILE = "3"  # ProRes HQ


# =============================================================================
# Color science helpers
# =============================================================================

def pq_oetf(L):
    """ST 2084 (PQ) OETF: linear light [0,1] (normalized to 10000 nits) -> signal [0,1]."""
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    Lm1 = max(L, 0.0) ** m1
    return ((c1 + c2 * Lm1) / (1.0 + c3 * Lm1)) ** m2


def hlg_oetf(E):
    """ARIB STD-B67 (HLG) OETF: scene linear [0,1] -> signal [0,1]."""
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * math.log(4.0 * a)
    if E <= 1.0 / 12.0:
        return math.sqrt(3.0 * E)
    else:
        return a * math.log(12.0 * E - b) + c


def nits_to_pq_signal(nits, peak=10000.0):
    """Convert absolute luminance in nits to PQ signal value [0,1]."""
    return pq_oetf(nits / peak)


def signal_to_10bit_narrow(s):
    """Convert signal [0,1] to 10-bit narrow range code value (64-940)."""
    return int(round(64 + s * 876))


def rgb_to_hex_8bit(r, g, b):
    """Convert float RGB [0,1] to 8-bit hex color string for FFmpeg."""
    ri = max(0, min(255, int(round(r * 255))))
    gi = max(0, min(255, int(round(g * 255))))
    bi = max(0, min(255, int(round(b * 255))))
    return f"0x{ri:02X}{gi:02X}{bi:02X}"


# =============================================================================
# BT.2111 bar definitions (scene-linear, BT.2020 primaries)
# =============================================================================

# 75% color bars: White, Yellow, Cyan, Green, Magenta, Red, Blue, Black
BARS_75_LINEAR = [
    (0.75, 0.75, 0.75),  # White (75%)
    (0.75, 0.75, 0.00),  # Yellow
    (0.00, 0.75, 0.75),  # Cyan
    (0.00, 0.75, 0.00),  # Green
    (0.75, 0.00, 0.75),  # Magenta
    (0.75, 0.00, 0.00),  # Red
    (0.00, 0.00, 0.75),  # Blue
    (0.00, 0.00, 0.00),  # Black
]

# 100% saturated bars
BARS_100_LINEAR = [
    (1.00, 1.00, 1.00),  # White (100%)
    (1.00, 1.00, 0.00),  # Yellow
    (0.00, 1.00, 1.00),  # Cyan
    (0.00, 1.00, 0.00),  # Green
    (1.00, 0.00, 1.00),  # Magenta
    (1.00, 0.00, 0.00),  # Red
    (0.00, 0.00, 1.00),  # Blue
    (0.00, 0.00, 0.00),  # Black
]


def apply_oetf(bars_linear, oetf_func):
    """Apply an OETF to a list of linear RGB bar values."""
    result = []
    for r, g, b in bars_linear:
        result.append((oetf_func(r), oetf_func(g), oetf_func(b)))
    return result


# =============================================================================
# FFmpeg filter construction
# =============================================================================

def run_ffmpeg(ffmpeg, args, desc):
    """Run an FFmpeg command, printing status."""
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning"] + args
    print(f"  Generating: {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return False
    print(f"    OK")
    return True


def build_bars_filter(bars_encoded, width, row_height, label_prefix="bar"):
    """Build FFmpeg filter for a row of color bars using hstack."""
    n = len(bars_encoded)
    bar_w = width // n
    parts = []
    labels = []
    for i, (r, g, b) in enumerate(bars_encoded):
        color = rgb_to_hex_8bit(r, g, b)
        lbl = f"{label_prefix}{i}"
        parts.append(f"color=c={color}:s={bar_w}x{row_height}:r={FRAMERATE}:d={DURATION}[{lbl}]")
        labels.append(f"[{lbl}]")
    # Pad last bar to fill width
    remainder = width - bar_w * n
    if remainder > 0:
        lbl = f"{label_prefix}_pad"
        last_color = rgb_to_hex_8bit(*bars_encoded[-1])
        parts.append(f"color=c={last_color}:s={remainder}x{row_height}:r={FRAMERATE}:d={DURATION}[{lbl}]")
        labels.append(f"[{lbl}]")
    stack_lbl = f"{label_prefix}_row"
    parts.append(f"{''.join(labels)}hstack=inputs={len(labels)}[{stack_lbl}]")
    return parts, stack_lbl


def build_ramp_filter(steps, width, row_height, oetf_func, label_prefix="ramp"):
    """Build a grayscale ramp row with N steps from black to peak."""
    step_w = width // steps
    parts = []
    labels = []
    for i in range(steps):
        linear = i / (steps - 1)
        encoded = oetf_func(linear)
        color = rgb_to_hex_8bit(encoded, encoded, encoded)
        lbl = f"{label_prefix}{i}"
        parts.append(f"color=c={color}:s={step_w}x{row_height}:r={FRAMERATE}:d={DURATION}[{lbl}]")
        labels.append(f"[{lbl}]")
    remainder = width - step_w * steps
    if remainder > 0:
        lbl = f"{label_prefix}_pad"
        parts.append(f"color=c=0xFFFFFF:s={remainder}x{row_height}:r={FRAMERATE}:d={DURATION}[{lbl}]")
        labels.append(f"[{lbl}]")
    stack_lbl = f"{label_prefix}_row"
    parts.append(f"{''.join(labels)}hstack=inputs={len(labels)}[{stack_lbl}]")
    return parts, stack_lbl


def build_reference_patches(width, row_height, patches, label_prefix="ref"):
    """Build reference patch row: list of (label, r, g, b, width_fraction) tuples."""
    parts = []
    labels = []
    total_frac = sum(p[4] for p in patches)
    used_w = 0
    for i, (name, r, g, b, frac) in enumerate(patches):
        if i == len(patches) - 1:
            pw = width - used_w  # last patch fills remainder
        else:
            pw = int(width * frac / total_frac)
        used_w += pw
        color = rgb_to_hex_8bit(r, g, b)
        lbl = f"{label_prefix}{i}"
        parts.append(f"color=c={color}:s={pw}x{row_height}:r={FRAMERATE}:d={DURATION}[{lbl}]")
        labels.append(f"[{lbl}]")
    stack_lbl = f"{label_prefix}_row"
    parts.append(f"{''.join(labels)}hstack=inputs={len(labels)}[{stack_lbl}]")
    return parts, stack_lbl


# =============================================================================
# Generators
# =============================================================================

def generate_bt709_sdr(ffmpeg, output_dir):
    """BT.709 SDR — SMPTE HD bars (RP 219) with built-in black/white reference."""
    out = output_dir / "colorbars_bt709_sdr.mov"
    return run_ffmpeg(ffmpeg, [
        "-f", "lavfi",
        "-i", f"smptehdbars=size={WIDTH}x{HEIGHT}:rate={FRAMERATE}:duration={DURATION}",
        "-c:v", "prores_ks", "-profile:v", PRORES_PROFILE,
        "-pix_fmt", "yuv422p10le",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        str(out)
    ], f"BT.709 SDR (smptehdbars with black/white ref) -> {out.name}")


def generate_bt2020_pq(ffmpeg, output_dir):
    """BT.2020 PQ (HDR10) — BT.2111-1 color bars with correct PQ encoding.
    75% bars at 1000-nit peak + reference patches at key luminance levels."""
    out = output_dir / "colorbars_bt2020_pq.mov"

    # PQ OETF scaled to 1000-nit peak display
    def pq_1000(L):
        return pq_oetf(L * 1000.0 / 10000.0)

    bars75_pq = apply_oetf(BARS_75_LINEAR, pq_1000)
    bars100_pq = apply_oetf(BARS_100_LINEAR, pq_1000)

    # Layout heights
    h_bars = int(HEIGHT * 0.67)
    h_100 = int(HEIGHT * 0.08)
    h_ramp = int(HEIGHT * 0.08)
    h_ref = HEIGHT - h_bars - h_100 - h_ramp

    filter_parts = []

    # Row 1: 75% color bars
    p1, l1 = build_bars_filter(bars75_pq, WIDTH, h_bars, "b75")
    filter_parts.extend(p1)

    # Row 2: 100% saturated bars
    p2, l2 = build_bars_filter(bars100_pq, WIDTH, h_100, "b100")
    filter_parts.extend(p2)

    # Row 3: PQ ramp 0-1000 nits in 16 steps
    p3, l3 = build_ramp_filter(16, WIDTH, h_ramp, pq_1000, "ramp")
    filter_parts.extend(p3)

    # Row 4: Reference patches with specific nit levels
    ref_1 = nits_to_pq_signal(1)
    ref_203 = nits_to_pq_signal(203)   # BT.2408 reference white
    ref_400 = nits_to_pq_signal(400)
    ref_1000 = nits_to_pq_signal(1000)  # peak white
    ref_4000 = nits_to_pq_signal(4000)  # super-white (above 1000-nit display)
    patches = [
        ("Black 0nit", 0, 0, 0, 2),
        ("Near-black 1nit", ref_1, ref_1, ref_1, 1),
        ("Ref-white 203nit", ref_203, ref_203, ref_203, 2),
        ("Mid 400nit", ref_400, ref_400, ref_400, 2),
        ("Peak 1000nit", ref_1000, ref_1000, ref_1000, 2),
        ("Super-white 4000nit", ref_4000, ref_4000, ref_4000, 1),
    ]
    p4, l4 = build_reference_patches(WIDTH, h_ref, patches, "ref")
    filter_parts.extend(p4)

    # Stack all rows
    filter_parts.append(f"[{l1}][{l2}][{l3}][{l4}]vstack=inputs=4[out]")
    filtergraph = ";".join(filter_parts)

    return run_ffmpeg(ffmpeg, [
        "-f", "lavfi",
        "-i", filtergraph,
        "-map", "[out]",
        "-c:v", "prores_ks", "-profile:v", PRORES_PROFILE,
        "-pix_fmt", "yuv422p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        str(out)
    ], f"BT.2020 PQ HDR10 (BT.2111-1 + ref patches) -> {out.name}")


def generate_bt2020_hlg(ffmpeg, output_dir):
    """BT.2020 HLG — BT.2111-2 bars with proper HLG OETF encoding."""
    out = output_dir / "colorbars_bt2020_hlg.mov"

    bars75_hlg = apply_oetf(BARS_75_LINEAR, hlg_oetf)
    bars100_hlg = apply_oetf(BARS_100_LINEAR, hlg_oetf)

    h_bars = int(HEIGHT * 0.67)
    h_100 = int(HEIGHT * 0.08)
    h_ramp = int(HEIGHT * 0.08)
    h_ref = HEIGHT - h_bars - h_100 - h_ramp

    filter_parts = []

    p1, l1 = build_bars_filter(bars75_hlg, WIDTH, h_bars, "b75")
    filter_parts.extend(p1)

    p2, l2 = build_bars_filter(bars100_hlg, WIDTH, h_100, "b100")
    filter_parts.extend(p2)

    p3, l3 = build_ramp_filter(16, WIDTH, h_ramp, hlg_oetf, "ramp")
    filter_parts.extend(p3)

    # Reference patches at signal percentages
    # HLG: 0% = black, 75% signal ~ reference white (203 nit on 1000-nit display)
    patches = [
        ("Black 0%", 0, 0, 0, 2),
        ("25% signal", hlg_oetf(0.25), hlg_oetf(0.25), hlg_oetf(0.25), 2),
        ("50% signal", hlg_oetf(0.50), hlg_oetf(0.50), hlg_oetf(0.50), 2),
        ("75% ref-white", hlg_oetf(0.75), hlg_oetf(0.75), hlg_oetf(0.75), 2),
        ("100% peak", hlg_oetf(1.0), hlg_oetf(1.0), hlg_oetf(1.0), 2),
    ]
    p4, l4 = build_reference_patches(WIDTH, h_ref, patches, "ref")
    filter_parts.extend(p4)

    filter_parts.append(f"[{l1}][{l2}][{l3}][{l4}]vstack=inputs=4[out]")
    filtergraph = ";".join(filter_parts)

    return run_ffmpeg(ffmpeg, [
        "-f", "lavfi",
        "-i", filtergraph,
        "-map", "[out]",
        "-c:v", "prores_ks", "-profile:v", PRORES_PROFILE,
        "-pix_fmt", "yuv422p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        str(out)
    ], f"BT.2020 HLG (BT.2111-2 + ref patches) -> {out.name}")


def generate_hdr_reference_ramp(ffmpeg, output_dir):
    """Full-screen PQ luminance ramp: 0 to 4000 nits in 64 columns.
    Each column is a known nit level — verifies the entire PQ curve."""
    out = output_dir / "hdr_pq_ramp_0_to_4000nit.mov"

    steps = 64
    step_w = WIDTH // steps
    filter_parts = []
    labels = []

    for i in range(steps):
        nits = (i / (steps - 1)) * 4000.0
        sig = nits_to_pq_signal(nits)
        color = rgb_to_hex_8bit(sig, sig, sig)
        lbl = f"s{i}"
        filter_parts.append(
            f"color=c={color}:s={step_w}x{HEIGHT}:r={FRAMERATE}:d={DURATION}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    remainder = WIDTH - step_w * steps
    if remainder > 0:
        lbl = "s_pad"
        filter_parts.append(
            f"color=c=0xFFFFFF:s={remainder}x{HEIGHT}:r={FRAMERATE}:d={DURATION}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    filter_parts.append(f"{''.join(labels)}hstack=inputs={len(labels)}[out]")
    filtergraph = ";".join(filter_parts)

    return run_ffmpeg(ffmpeg, [
        "-f", "lavfi",
        "-i", filtergraph,
        "-map", "[out]",
        "-c:v", "prores_ks", "-profile:v", PRORES_PROFILE,
        "-pix_fmt", "yuv422p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        str(out)
    ], f"PQ ramp 0-4000 nit (64 steps) -> {out.name}")


def generate_bt2020_gamut_test(ffmpeg, output_dir):
    """BT.2020 saturated primaries at 1000-nit peak (PQ encoded).
    These colors are OUTSIDE the BT.709 gamut — validates wide-gamut decode."""
    out = output_dir / "gamut_bt2020_primaries_pq.mov"

    def pq_1000(L):
        return pq_oetf(L * 1000.0 / 10000.0)

    # BT.2020 primaries + secondaries at full saturation
    primaries = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.0, 1.0),  # Magenta
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 1.0, 1.0),  # White (1000 nit)
        (0.0, 0.0, 0.0),  # Black
    ]

    bar_w = WIDTH // len(primaries)
    filter_parts = []
    labels = []

    for i, (r, g, b) in enumerate(primaries):
        encoded = rgb_to_hex_8bit(pq_1000(r), pq_1000(g), pq_1000(b))
        lbl = f"gam{i}"
        filter_parts.append(
            f"color=c={encoded}:s={bar_w}x{HEIGHT}:r={FRAMERATE}:d={DURATION}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    remainder = WIDTH - bar_w * len(primaries)
    if remainder > 0:
        lbl = "gam_pad"
        filter_parts.append(
            f"color=c=0x000000:s={remainder}x{HEIGHT}:r={FRAMERATE}:d={DURATION}[{lbl}]"
        )
        labels.append(f"[{lbl}]")

    filter_parts.append(f"{''.join(labels)}hstack=inputs={len(labels)}[out]")
    filtergraph = ";".join(filter_parts)

    return run_ffmpeg(ffmpeg, [
        "-f", "lavfi",
        "-i", filtergraph,
        "-map", "[out]",
        "-c:v", "prores_ks", "-profile:v", PRORES_PROFILE,
        "-pix_fmt", "yuv422p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        str(out)
    ], f"BT.2020 gamut primaries (PQ 1000nit) -> {out.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate HDR/WCG test media for CasparVP")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG,
                        help="Path to ffmpeg executable (default: ffmpeg in PATH)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for generated .mov files")
    parser.add_argument("--only", choices=["709", "pq", "hlg", "ramp", "gamut"],
                        help="Generate only a specific variant")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify FFmpeg
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
        ver = result.stdout.split("\n")[0] if result.returncode == 0 else "unknown"
        print(f"FFmpeg: {ver}")
    except FileNotFoundError:
        print(f"ERROR: FFmpeg not found at '{ffmpeg}'")
        print("       Specify with --ffmpeg /path/to/ffmpeg")
        sys.exit(1)

    print(f"Output: {output_dir}")

    generators = {
        "709":   ("BT.709 SDR (SMPTE HD bars + black/white ref)", generate_bt709_sdr),
        "pq":    ("BT.2020 PQ HDR10 (BT.2111-1, 0/203/1000/4000 nit ref)", generate_bt2020_pq),
        "hlg":   ("BT.2020 HLG (BT.2111-2, 0%/25%/50%/75%/100% ref)", generate_bt2020_hlg),
        "ramp":  ("PQ luminance ramp 0-4000 nit (64 steps)", generate_hdr_reference_ramp),
        "gamut": ("BT.2020 saturated primaries (PQ 1000nit)", generate_bt2020_gamut_test),
    }

    if args.only:
        subset = {args.only: generators[args.only]}
    else:
        subset = generators

    # Print expected reference values
    print("\n--- PQ reference levels (10-bit narrow range Y code) ---")
    for nits in [0, 1, 100, 203, 400, 750, 1000, 4000]:
        sig = nits_to_pq_signal(nits)
        code = signal_to_10bit_narrow(sig)
        print(f"  {nits:5d} nit  ->  PQ signal {sig:.4f}  ->  Y={code}")

    print("\n--- HLG reference levels ---")
    for pct in [0, 25, 50, 75, 100]:
        sig = hlg_oetf(pct / 100.0)
        code = signal_to_10bit_narrow(sig)
        print(f"  {pct:3d}% scene  ->  HLG signal {sig:.4f}  ->  Y={code}")

    results = {}
    for key, (desc, func) in subset.items():
        print(f"\n[{key}] {desc}")
        results[key] = func(ffmpeg, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, (desc, _) in subset.items():
        status = "OK" if results.get(key) else "FAILED"
        print(f"  [{status}] {desc}")

    if all(results.values()):
        print("\nAll test media generated successfully.")
        print("\nUsage with CasparVP:")
        print('  LOADBG 1-1 CUDA_PRORES "sources/colorbars_bt709_sdr"')
        print('  LOADBG 1-1 CUDA_PRORES "sources/colorbars_bt2020_pq"')
        print('  LOADBG 1-1 CUDA_PRORES "sources/colorbars_bt2020_hlg"')
        print('  LOADBG 1-1 CUDA_PRORES "sources/hdr_pq_ramp_0_to_4000nit"')
        print('  LOADBG 1-1 CUDA_PRORES "sources/gamut_bt2020_primaries_pq"')
        print("\nPQ bar layout (bottom row reference patches):")
        print("  Black(0) | Near-black(1nit) | Ref-white(203nit) | 400nit | Peak(1000nit) | Super-white(4000nit)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
