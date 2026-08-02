/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <core/fwd.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace caspar { namespace html {

/// Byte order the compositor declared for its shared surface. Mirrors
/// cef_color_type_t without dragging a CEF header into the accelerator layer.
enum class shared_surface_order
{
    bgra8,
    rgba8,
};

/**
 * Turns a compositor's shared-texture handles into mixer-ready const_frames,
 * without the pixels ever reaching host memory.
 *
 * ── Why this is two-phase ────────────────────────────────────────────────────
 * CEF hands back a handle that is pool-owned and valid only for the duration of
 * OnAcceleratedPaint, and the texture carries no keyed mutex. So the frame must
 * be copied out, and the copy must be *finished*, before the callback returns --
 * a submitted-but-unexecuted blit reading a recycled texture yields a torn frame
 * rather than a crash, which is the kind of bug that does not reproduce on
 * demand.
 *
 * That first copy therefore has to happen on the CEF UI thread. The second one --
 * into a pooled mixer texture, which has to queue behind whatever the mixer's
 * device thread is already doing -- does not, and must not: the CEF UI thread is
 * process-global and shared by every html producer in the server, so a per-frame
 * sync hop onto a single device executor would serialise the whole browser
 * message loop behind the mixer's queue depth. Phase two runs on this bridge's
 * own worker instead.
 *
 * ── Platform ─────────────────────────────────────────────────────────────────
 * Windows only for now, and no platform GPU type appears in this header: the
 * handle is a void*, the byte order an enum, the rect four ints. A Linux
 * implementation (CEF fills a dmabuf variant there, imported through
 * VK_EXT_external_memory_dma_buf) is an additive change behind the same
 * interface, not a rewrite of the producer.
 */
class html_gpu_bridge
{
  public:
    /**
     * Builds the bridge, or returns null with the reason in `reason`.
     *
     * Must be called BEFORE CefBrowserHost::CreateBrowser: CEF binds OnPaint vs
     * OnAcceleratedPaint from CefWindowInfo::shared_texture_enabled at creation
     * and will not change its mind, so the caller has to know whether the bridge
     * exists before it decides what to ask for.
     *
     * Everything that can fail without a frame in hand fails here -- adapter
     * lookup, device creation, the staging ring, the Vulkan import bridge, and a
     * self-test round trip. What cannot be tested until a frame arrives is which
     * adapter the compositor chose; see `submit`.
     *
     * `on_frame` is invoked on the worker thread, in submission order.
     */
    static std::unique_ptr<html_gpu_bridge> create(core::frame_factory&                  factory,
                                                   const void*                           stream_tag,
                                                   std::function<void(core::draw_frame)> on_frame,
                                                   std::wstring&                         reason);

    ~html_gpu_bridge(); ///< drains the worker before releasing any GPU object

    html_gpu_bridge(const html_gpu_bridge&)            = delete;
    html_gpu_bridge& operator=(const html_gpu_bridge&) = delete;

    /**
     * Phase 1. Call from OnAcceleratedPaint, on the thread where `shared_handle`
     * is valid, and do not return from that callback until this has.
     *
     * `x,y,w,h` is the visible rect: pass it, not the coded size. A coded size
     * larger than the picture is padding, and copying it shifts or rescales the
     * frame.
     *
     * Returns false when the frame was dropped -- every staging slot busy is
     * back-pressure rather than an error, and looks the same to the caller as the
     * queue-full drop the host path already does.
     */
    bool submit(void* shared_handle, shared_surface_order order, int x, int y, int w, int h);

    /// False once the bridge has given up (see kMaxConsecutiveFailures in the
    /// implementation). The producer should stop calling submit() and let the
    /// channel hold the last good frame.
    bool healthy() const;

    /// True until the first frame has been through successfully. While set, a
    /// failure in submit() is diagnosed as an adapter mismatch -- the one thing
    /// create() cannot check -- and is the caller's cue to recreate the browser
    /// without shared textures, which is only free before anything is on air.
    bool probing() const;

    int          in_flight() const;
    std::int64_t dropped() const;
    /// Microseconds spent inside submit() waiting for the GPU, accumulated since
    /// the last call. For the producer's diagnostics graph.
    std::int64_t take_stage_wait_us();

  private:
    html_gpu_bridge();
    struct impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::html
