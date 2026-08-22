#!/bin/bash
# Swap build/shell's pinned FFmpeg for a locally built one, or put the pin back.
#
#   tools/use_local_ffmpeg.sh apply     # local build -> build/shell
#   tools/use_local_ffmpeg.sh revert    # pinned build -> build/shell
#   tools/use_local_ffmpeg.sh status
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────────
# The pinned `ffmpeg-8.1.2-full_build-shared.7z` is built against nv-codec-headers
# n13.1, whose NVENC refuses any driver below 610:
#
#   [h264_nvenc] Driver does not support the required nvenc API version.
#                Required: 13.1 Found: 13.0
#   [h264_nvenc] The minimum required Nvidia driver for nvenc is 610.00 or newer
#
# so `ADD 1 FILE out.mov -vcodec h264_nvenc` returns 501 and records nothing. The check is a
# compile-time constant against the driver's runtime maximum -- `libavcodec/nvenc.c`, the line
# `(NVENCAPI_MAJOR_VERSION << 4 | NVENCAPI_MINOR_VERSION) > nvenc_max_ver` -- so there is no
# option, environment variable or runtime override. Only a rebuild changes it.
#
# AND THE DRIVER CANNOT BE RAISED ON THE REFERENCE MACHINE. Release 580 is the last branch that
# supports Quadro Maxwell, Pascal and Volta; the box's second GPU is a Pascal Quadro P4000; and
# the installed 582.53 is R580 U9, i.e. already the newest driver that serves both slots. One
# Windows package serves both, so raising it for NVENC would take the P4000's driver with it.
#
# The MSYS2 UCRT64 environment on that machine has ffnvcodec at NVENCAPI **13.0**, which needs
# only driver 570+, so a build made there has working NVENC at the same FFmpeg version.
#
# ── WHAT IT COSTS, WHICH IS WHY THIS IS A SCRIPT AND NOT THE DEFAULT ────────────
# The local build is NOT a drop-in equal of the pin:
#
#   * It is 8.1.1+7 on release/8.1, not 8.1.2. Same sonames, and the ABI holds in practice --
#     the pinned 8.1.2 `ffmpeg.exe` runs against these DLLs -- but it is not the pinned version.
#   * It carries a NARROWER codec set. Dropped against the pin, among others: libjxl, libbluray,
#     libdvdnav/libdvdread, frei0r, libvidstab, libvmaf, chromaprint, libzvbi (teletext),
#     libxvid, libfdk-aac, libtheora, libspeex, the AMR/GSM/iLBC/LC3/codec2 family, sdl2,
#     libopenmpt/libgme/libmodplug, libqrencode and whisper. Anything a user reaches for through
#     `-vcodec`/`-acodec` that is on that list stops working.
#   * `--pkg-config-flags=--static` does NOT statically link the externals into a DLL build the
#     way it does a static one, so it needs **40 mingw DLLs** copied alongside. Measured: none of
#     the 40 collides by name with the 24 the server already ships, which is the only reason this
#     is viable at all.
#   * It drops `--enable-nonfree`, so the result is GPL-v3 and redistributable. The existing
#     `configure_full.sh` in the FFmpeg tree uses nonfree for libfdk-aac and DeckLink; a nonfree
#     binary could not be shipped.
#
# ── AND WHY IT MUST BE APPLIED DELIBERATELY ────────────────────────────────────
# `casparcg_add_runtime_dependency` copies the pinned DLLs into build/shell on every build, so an
# applied swap SILENTLY REVERTS the next time anyone builds. NVENC would work until someone ran
# cmake and then stop, with nothing to explain it. Re-run `apply` after a build, or `revert` and
# leave the tree in the state its pin describes.
#
# Measured 2026-08-22 with the swap applied, 1080p2500, two interleaved rounds:
#
#   route          cores   NVENC block   picture vs CPU    MB
#   h264_nvenc     1.37    9/27 %        mean 1.66 LSB     0.1
#   h264_vulkan    1.40    15/20 %       mean 2.70 LSB     0.3
#   hevc_nvenc     1.39    11/34 %       mean 1.64 LSB     0.1
#   hevc_vulkan    1.43    36/46 %       mean 2.52 LSB     6.7
#
# NVENC is cheaper, closer to the CPU encoder and far better rate-controlled out of the box, so
# it is not redundant with the Vulkan encoders that work without this swap.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_DIR="$REPO/build/shell"
BACKUP="$SHELL_DIR/ffmpeg-pinned-backup"
LOCAL_BUILD="/d/Github/FFmpeg/wt-shared"
UCRT_BIN="/c/msys64/ucrt64/bin"

FFMPEG_DLLS="avcodec-62 avdevice-62 avfilter-11 avformat-62 avutil-60 swresample-6 swscale-9"

die() { echo "error: $*" >&2; exit 1; }

# The marker is a codec the pin has and the local build does not, so "which one is installed" is
# answered by asking the DLL rather than by trusting a filename or a timestamp.
probe() {
    local exe="$SHELL_DIR/ffmpeg.exe"
    [ -x "$exe" ] || { echo "unknown (no ffmpeg.exe)"; return; }
    if [ "$("$exe" -hide_banner -decoders 2>/dev/null | grep -ci jxl)" -gt 0 ]; then
        echo "pinned (libjxl present)"
    else
        echo "local (no libjxl)"
    fi
}

case "${1:-status}" in
  apply)
    [ -d "$LOCAL_BUILD" ] || die "no local build at $LOCAL_BUILD -- run FFmpeg's configure_shared.sh first"
    mkdir -p "$BACKUP"
    for d in $FFMPEG_DLLS; do
        # -n: never overwrite an existing backup, or applying twice would back up the local
        # build over the pin and lose the only copy of what we are meant to restore.
        cp -n "$SHELL_DIR/$d.dll" "$BACKUP/" 2>/dev/null || true
    done
    for d in $FFMPEG_DLLS; do
        src=$(find "$LOCAL_BUILD" -maxdepth 2 -name "$d.dll" | head -1)
        [ -n "$src" ] || die "local build has no $d.dll"
        cp "$src" "$SHELL_DIR/" || die "could not copy $d.dll"
    done
    n=0
    for f in "$LOCAL_BUILD"/../deps.txt; do :; done
    # Recompute the dependency list rather than trusting a stored one: the set changes with the
    # configure flags, and a stale list fails as a missing-DLL dialog at server start.
    if command -v ldd >/dev/null 2>&1; then
        for d in $FFMPEG_DLLS; do
            f=$(find "$LOCAL_BUILD" -maxdepth 2 -name "$d.dll" | head -1)
            ldd "$f" 2>/dev/null | grep "$UCRT_BIN" | awk '{print $1}'
        done | sort -u | while read -r dll; do
            cp -n "$UCRT_BIN/$dll" "$SHELL_DIR/" 2>/dev/null || true
        done
        n=$(ls "$SHELL_DIR" | wc -l)
    else
        echo "warning: no ldd; dependency DLLs not copied and the server will not start"
    fi
    echo "applied. build/shell now reports: $(probe)"
    echo "NOTE: the next cmake build overwrites this. Re-run 'apply' afterwards."
    ;;
  revert)
    [ -d "$BACKUP" ] || die "no backup at $BACKUP -- nothing to revert to"
    for d in $FFMPEG_DLLS; do
        [ -f "$BACKUP/$d.dll" ] || die "backup is missing $d.dll; refusing a partial revert"
    done
    for d in $FFMPEG_DLLS; do cp "$BACKUP/$d.dll" "$SHELL_DIR/" || die "restore of $d.dll failed"; done
    echo "reverted. build/shell now reports: $(probe)"
    ;;
  status)
    echo "build/shell FFmpeg: $(probe)"
    [ -d "$BACKUP" ] && echo "pinned backup: present ($(ls "$BACKUP" | wc -l) DLLs)" \
                     || echo "pinned backup: absent"
    ;;
  *)
    die "usage: $0 {apply|revert|status}"
    ;;
esac
