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
 * This module requires the NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit).
 * ProRes format reference: Apple Inc. "ProRes RAW White Paper" (public documentation).
 */

// timecode.h
// SMPTE timecode struct shared across the cuda_prores pipeline.
//
// Timecode flows: DeckLink capture → CaptureToken → Consumer → MOV/MXF muxer.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// SmpteTimecode
// ---------------------------------------------------------------------------
// Represents one SMPTE 12M timecode value as extracted from an SDI frame.
// When `valid == false` the timecode could not be read (e.g., the source
// sends no RP188/VITC timecode).  Consumers should fall back to the absolute
// frame counter stored in CaptureToken::timecode in that case.
struct SmpteTimecode {
    uint8_t hours      = 0;
    uint8_t minutes    = 0;
    uint8_t seconds    = 0;
    uint8_t frames     = 0;
    bool    drop_frame = false;  // true for 29.97 / 59.94 DF timecode
    bool    valid      = false;  // false = no timecode available

    // Convert to an absolute frame count from 00:00:00:00 for a given integer
    // frames-per-second value.  For drop-frame timecode, the frame count is the
    // "nominal" frame number (no correction applied — callers that need exact
    // wallclock positioning should implement DF correction separately).
    uint32_t to_frame_count(uint32_t fps) const {
        return ((uint32_t)hours * 3600u + (uint32_t)minutes * 60u + (uint32_t)seconds) * fps
               + (uint32_t)frames;
    }

    // Write a null-terminated "HH:MM:SS:FF" / "HH:MM:SS;FF" (drop frame uses ';')
    // into buf.  buf_len must be >= 12.
    void to_string(char *buf, size_t buf_len) const {
        snprintf(buf, buf_len, "%02u:%02u:%02u%c%02u",
                 (unsigned)hours, (unsigned)minutes, (unsigned)seconds,
                 drop_frame ? ';' : ':',
                 (unsigned)frames);
    }
};
