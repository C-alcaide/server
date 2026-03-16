/*
 * Copyright (c) 2025 CasparCG Contributors
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
 *
 * This file optionally links against VRPN (https://github.com/vrpn/vrpn),
 * licensed under the Boost Software License 1.0 (BSL-1.0),
 * which is compatible with GPL-3. VRPN is only required when building
 * with -DBUILD_TRACKING_VRPN=ON.
 */

#include "vrpn_receiver.h"

#include "../camera_data.h"
#include "../tracker_registry.h"

#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// VRPN is an optional dependency. Guard everything with the feature macro.
// When VRPN is unavailable the class methods still compile to no-ops so the
// rest of the module builds cleanly.
// ---------------------------------------------------------------------------

#ifdef CASPAR_TRACKING_WITH_VRPN
#include <vrpn_Tracker.h>
#include <vrpn_Analog.h>
#endif

namespace caspar { namespace tracking {

// ---------------------------------------------------------------------------
// Quaternion → Euler (ZYX / yaw-pitch-roll, right-hand Y-up)
//   yaw   (Z): rotation about vertical axis     (+right)
//   pitch (Y): rotation about horizontal axis   (+up)
//   roll  (X): rotation about forward axis      (+clockwise)
// ---------------------------------------------------------------------------
static void quat_to_euler(double qw, double qx, double qy, double qz,
                           double& yaw, double& pitch, double& roll)
{
    // Normalise quaternion (defensive)
    double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (n < 1e-8) { yaw = pitch = roll = 0.0; return; }
    qw /= n; qx /= n; qy /= n; qz /= n;

    // ZYX Euler extraction (same convention used by most VR/tracking SDKs)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (qw * qy - qz * qx);
    pitch = (std::abs(sinp) >= 1.0)
            ? std::copysign(3.141592653589793 / 2.0, sinp)
            : std::asin(sinp);

    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

// ---------------------------------------------------------------------------

struct vrpn_receiver::impl
{
    const std::string host_url_;
    const int         camera_id_;
    const int         sensor_;
    std::atomic<bool> running_{false};
    std::thread       poll_thread_;

    // Latest partial data guarded by a mutex
    std::mutex  data_mutex_;
    camera_data partial_{};

#ifdef CASPAR_TRACKING_WITH_VRPN
    std::unique_ptr<vrpn_Tracker_Remote> tracker_;
    std::unique_ptr<vrpn_Analog_Remote>  analog_;

    static void VRPN_CALLBACK tracker_cb(void* userdata, const vrpn_TRACKERCB t)
    {
        auto* self = static_cast<impl*>(userdata);
        double yaw = 0.0, pitch = 0.0, roll = 0.0;
        quat_to_euler(t.quat[3], t.quat[0], t.quat[1], t.quat[2], yaw, pitch, roll);
        {
            std::lock_guard<std::mutex> lk(self->data_mutex_);
            self->partial_.pan       = yaw;
            self->partial_.tilt      = pitch;
            self->partial_.roll      = roll;
            self->partial_.x         = t.pos[0] * 1000.0; // VRPN position is in metres → mm
            self->partial_.y         = t.pos[1] * 1000.0;
            self->partial_.z         = t.pos[2] * 1000.0;
            self->partial_.camera_id = self->camera_id_;
            self->partial_.timestamp = std::chrono::steady_clock::now();
        }
        // Push on every pose update
        camera_data snapshot;
        {
            std::lock_guard<std::mutex> lk(self->data_mutex_);
            snapshot = self->partial_;
        }
        tracker_registry::instance().on_data(snapshot);
    }

    static void VRPN_CALLBACK analog_cb(void* userdata, const vrpn_ANALOGCB a)
    {
        auto* self = static_cast<impl*>(userdata);
        if (a.num_channel < 1)
            return;
        double raw = a.channel[0]; // typically 0.0-1.0
        if (raw < 0.0) raw = 0.0;
        if (raw > 1.0) raw = 1.0;
        std::lock_guard<std::mutex> lk(self->data_mutex_);
        self->partial_.zoom = static_cast<uint16_t>(raw * 65535.0);
    }
#endif

    explicit impl(std::string url, int cam_id, int sensor)
        : host_url_(std::move(url))
        , camera_id_(cam_id)
        , sensor_(sensor)
    {
        partial_.camera_id = camera_id_;
    }

    ~impl() { stop(); }

    void start()
    {
        if (running_.exchange(true))
            return;

#ifdef CASPAR_TRACKING_WITH_VRPN
        poll_thread_ = std::thread([this] {
            tracker_ = std::make_unique<vrpn_Tracker_Remote>(host_url_.c_str());
            tracker_->register_change_handler(this, tracker_cb, sensor_);

            analog_ = std::make_unique<vrpn_Analog_Remote>(host_url_.c_str());
            analog_->register_change_handler(this, analog_cb);

            while (running_) {
                tracker_->mainloop();
                analog_->mainloop();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            tracker_.reset();
            analog_.reset();
        });
#else
        std::cerr << "[tracking/vrpn] VRPN support was not compiled in "
                  << "(rebuild with -DBUILD_TRACKING_VRPN=ON)\n";
        running_.store(false);
#endif
    }

    void stop()
    {
        if (!running_.exchange(false))
            return;
        if (poll_thread_.joinable())
            poll_thread_.join();
    }
};

// ---------------------------------------------------------------------------

vrpn_receiver::vrpn_receiver(std::string host_url, int camera_id, int sensor)
    : impl_(std::make_unique<impl>(std::move(host_url), camera_id, sensor))
{
}

vrpn_receiver::~vrpn_receiver() = default;

void vrpn_receiver::start() { impl_->start(); }
void vrpn_receiver::stop()  { impl_->stop(); }

std::string vrpn_receiver::info() const
{
    return "VRPN " + impl_->host_url_ + " (camera " + std::to_string(impl_->camera_id_) + ")";
}

}} // namespace caspar::tracking
