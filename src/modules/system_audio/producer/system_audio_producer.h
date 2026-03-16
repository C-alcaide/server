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
 * This module uses miniaudio (https://miniaud.io), dual-licensed under
 * MIT and public domain (Unlicense), both compatible with GPL-3.
 */

#pragma once

#include <core/frame/frame_factory.h>
#include <core/producer/frame_producer.h>
#include <common/executor.h>

#include <string>
#include <vector>

namespace caspar { namespace system_audio {

class system_audio_producer : public core::frame_producer
{
public:
    explicit system_audio_producer(const core::frame_producer_dependencies& dependencies,
                                   const std::wstring&                      device_name,
                                   const std::vector<std::wstring>&         params);
    ~system_audio_producer();

    core::draw_frame receive_impl(const core::video_field, int) override;
    
    std::wstring         print() const override;
    std::wstring         name() const override;
    core::monitor::state state() const override;
    bool                 is_ready() override { return true; }

private:
    class impl;
    std::unique_ptr<impl> impl_;
};

}} // namespace caspar::system_audio