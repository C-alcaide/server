/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// One pointer-device / keyboard event, on its way from a source to whatever wants it.
//
// PLAIN DATA, DELIBERATELY. CasparCG carried an interaction API from 2013 to 2018
// (`efcd57858` removed it: 31 files, -496 lines) built as a four-level
// `spl::shared_ptr<const interaction_event>` hierarchy -- `position_event`, `mouse_move_event`,
// `mouse_button_event`, `mouse_wheel_event` -- dispatched by `dynamic_cast` through `is<T>()` /
// `as<T>()` helpers, with a virtual `clone()` per leaf so a transform could produce a new
// immutable copy. Nothing about routing a mouse position earns that: it made every consumer of
// an event pay a heap allocation and a chain of failed casts, and it could not be compared,
// copied into a queue, or built from a command string without going through a factory.
//
// A struct can be all of those. It sits in a queue on the window that produced it, is compared
// field by field to coalesce, is published to `monitor::state` if we ever want to, and is
// constructed directly by the AMCP `INPUT` command.
//
// Two conventions are load-bearing and both were wrong in the old code:
//
//   * COORDINATES ARE NORMALISED 0..1 IN THE SOURCE SURFACE, top-left origin -- and the source
//     is responsible for making that true. The 2018 screen consumer divided by the window size
//     CAPTURED AT CONSTRUCTION and ignored aspect-ratio letterboxing entirely, so coordinates
//     skewed after any resize and the pillarbox bars mapped into the visible range.
//
//   * BUTTON NUMBERING IS CEF'S: 0 left, 1 middle, 2 right. The old html producer cast
//     `sf::Mouse::Button` straight to `CefBrowserHost::MouseButtonType`, and SFML numbers
//     middle and right the other way round -- so right-click and middle-click were swapped for
//     the whole life of the feature. Every source converts to this order at the source, and the
//     sink never converts again.
#include <cstdint>

namespace caspar { namespace core {

/// Modifier and held-button state, matching CEF's `cef_event_flags_t` values so the html
/// producer can hand them over unchanged.
///
/// THE HELD-BUTTON BITS ARE THE POINT. The 2018 producer never set `CefMouseEvent::modifiers`
/// at all, so Chromium saw a move with no button down and in-page dragging never worked --
/// which is most of what an interactive page is for.
enum input_modifier : uint32_t
{
    mod_none         = 0,
    mod_shift        = 1u << 1,
    mod_control      = 1u << 2,
    mod_alt          = 1u << 3,
    mod_left_button  = 1u << 4,
    mod_middle_button = 1u << 5,
    mod_right_button = 1u << 6,
};

struct input_event
{
    enum class kind : uint8_t
    {
        move,   //< the pointer moved to (x, y)
        button, //< `button` went `pressed` at (x, y)
        wheel,  //< scrolled by (wheel_dx, wheel_dy) at (x, y)
        key,    //< `key` went `pressed`
        text,   //< `text` was typed
        leave,  //< the pointer left the surface; nothing else in the struct is meaningful
    };

    kind type = kind::move;

    /// Normalised 0..1 within the source surface, top-left origin. Outside that range means
    /// the source deliberately reported a point off its own surface, which every sink treats
    /// as a miss rather than clamping.
    double x = 0.0;
    double y = 0.0;

    /// 0 left, 1 middle, 2 right. -1 when `type` is not `button`.
    int  button  = -1;
    bool pressed = false;

    /// Wheel delta in notional lines, sign as the platform reports it. Kept as a double so a
    /// high-resolution trackpad is not quantised to integer ticks the way the old
    /// `int ticks_delta` forced.
    double wheel_dx = 0.0;
    double wheel_dy = 0.0;

    /// Platform virtual key code (`VK_*` on Windows). The old API had no keyboard event of any
    /// kind -- `interaction_event.h` defined move, button and wheel and nothing else -- so an
    /// interactive page could never be typed into.
    int key = 0;

    /// A single typed character, already translated (Win32 `WM_CHAR`). Kept separate from
    /// `key` because a keystroke and the character it produces are different events and a
    /// browser wants both.
    char32_t character = 0;

    /// Bitwise OR of `input_modifier`.
    uint32_t modifiers = mod_none;

    /// How many clicks this is part of: 1 single, 2 double, 3 triple. The old producer passed
    /// a hard-coded 1, so a page could never see a double-click.
    int click_count = 1;

    /// Which surface produced it. 0 is "the channel's own screen consumer"; a command-sourced
    /// event carries a different id so a sink can tell synthetic input from a real cursor.
    int source = 0;

    /// Is this a pointer event carrying a meaningful position?
    [[nodiscard]] bool has_position() const
    {
        return type == kind::move || type == kind::button || type == kind::wheel;
    }

    /// Is (x, y) inside the surface it was reported against?
    [[nodiscard]] bool on_surface() const { return x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0; }

    /// Same event but at a different position -- what a hit-test produces after inverting a
    /// layer's transform. A plain copy, where the old code needed a virtual `clone()` per leaf.
    [[nodiscard]] input_event at(double new_x, double new_y) const
    {
        auto out = *this;
        out.x    = new_x;
        out.y    = new_y;
        return out;
    }
};

}} // namespace caspar::core
