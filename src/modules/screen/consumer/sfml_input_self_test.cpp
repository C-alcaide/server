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

// The SFML -> raw_input translation, checked against a REAL SFML window.
//
// WHY THIS IS A STANDALONE PROGRAM AND NOT A BOOT SELF-TEST. Every other self-test in this tree
// (`compose_self_test`, `stage_math_self_test`, `binding_math_self_test`,
// `audio_analysis_self_test`) is pure arithmetic and runs inside the server at startup. This one
// cannot be: the thing under test is what a window manager and SFML actually deliver, so it needs
// a window, a display and injected input -- none of which belong in a server's boot path.
//
// It is also the only way this code is verified at all. The Windows path has `previz-interact`,
// which posts messages to a real `win32_gl_window` in a running server; the SFML path has no
// equivalent because the harness is Windows-side and the fork does not build on Linux here. So
// this exists to stop the SFML branch being written and never executed.
//
// HOW TO RUN IT, under WSL with WSLg (or any X display):
//
//     g++ -std=c++20 -I src -o /tmp/sfml_input_self_test \
//         src/modules/screen/consumer/sfml_input_self_test.cpp \
//         -lsfml-window -lsfml-system -lX11 -lXtst
//     /tmp/sfml_input_self_test
//
// WHY XTest AND NOT XSendEvent. `XSendEvent` marks its events `send_event = True` and many
// toolkits drop those; SFML does not, so it would work -- but it also bypasses the X server's own
// pointer state, which is exactly what `sfml_modifiers()` reads through
// `sf::Mouse::isButtonPressed`. A held button injected with `XSendEvent` is not held as far as
// the server is concerned, so the modifier check would fail against correct code. `XTestFakeMotionEvent`
// and friends inject at the server, so the state is real. The cost is that they move the actual
// pointer, which is why this is a standalone program a person runs and not something a battery
// fires off unattended.
//
// This is the same distinction `win32_input.py` records for `PostMessage` versus `physical_click`,
// reached from the opposite direction: on Win32 the posted message is enough because the WndProc
// reads `lParam`; here the state query forces the real thing.

#include <SFML/Window.hpp>

#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------
// The translation under test.
//
// THE SAME TEXT the consumer compiles, via `sfml_input_helpers.inl`. Not a copy: a copy would
// test the copy, and would agree with the original right up until one of them changed. The `.inl`
// exists precisely so this program and `screen_consumer.cpp` cannot diverge.
//
// What this file supplies instead is a TRIMMED `input_event` -- just the enumerators and the
// modifier bits -- because the real `core/input/input_event.h` pulls in the monitor and frame
// types and this has to link against nothing but SFML and Xlib.
// ---------------------------------------------------------------------------------------

namespace caspar { namespace core {

// A trimmed `input_event`, matching `core/input/input_event.h`'s enumerators and modifier bits.
// Trimmed rather than included because the real header pulls in the monitor and the frame types.
struct input_event
{
    enum class kind : uint8_t
    {
        move,
        button,
        wheel,
        key,
        text,
        leave
    };
};

enum input_modifier : uint32_t
{
    mod_none          = 0,
    mod_shift         = 1u << 1,
    mod_control       = 1u << 2,
    mod_alt           = 1u << 3,
    mod_left_button   = 1u << 4,
    mod_middle_button = 1u << 5,
    mod_right_button  = 1u << 6,
};

}} // namespace caspar::core

struct raw_input
{
    caspar::core::input_event::kind type    = caspar::core::input_event::kind::move;
    int                             px      = 0;
    int                             py      = 0;
    int                             button  = -1;
    bool                            pressed = false;
    double                          wheel   = 0.0;
    int                             key     = 0;
    char32_t                        ch      = 0;
    uint32_t                        mods    = 0;
    int                             clicks  = 1;
};

#include "sfml_input_helpers.inl"

// ---------------------------------------------------------------------------------------

static int failures = 0;

static void check(bool ok, const std::string& what, const std::string& detail = "")
{
    if (ok) {
        std::printf("  ok    %s\n", what.c_str());
        return;
    }
    ++failures;
    std::printf("  FAIL  %s%s%s\n", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
}

int main()
{
    std::printf("sfml_input_self_test\n");

    // ---- The pure mappings, which need no window ---------------------------------------
    //
    // First, because they are the two 2013 defects and they are checkable with no display at
    // all -- so a failure here is unambiguous rather than entangled with the windowing.
    check(sfml_button(sf::Mouse::Left) == 0, "button/left -> 0");
    check(sfml_button(sf::Mouse::Middle) == 1, "button/middle -> 1");
    check(sfml_button(sf::Mouse::Right) == 2, "button/right -> 2");
    check(sfml_button(sf::Mouse::XButton1) == -1, "button/x1 -> dropped");

    // THE EXCHANGE, stated as its own check. SFML's enum is Left=0, Right=1, Middle=2, so a
    // straight cast gives middle<->right -- and this asserts the two do NOT agree numerically,
    // which is the fact that makes the function necessary.
    check(static_cast<int>(sf::Mouse::Right) != sfml_button(sf::Mouse::Right) &&
              static_cast<int>(sf::Mouse::Middle) != sfml_button(sf::Mouse::Middle),
          "button/sfml and cef orders genuinely differ",
          "a cast would swap middle and right, which is the 2013 defect");

    check(sfml_vk(sf::Keyboard::Left) == 0x25, "vk/left");
    check(sfml_vk(sf::Keyboard::Right) == 0x27, "vk/right");
    check(sfml_vk(sf::Keyboard::Escape) == 0x1B, "vk/escape");
    check(sfml_vk(sf::Keyboard::A) == 0x41, "vk/A");
    check(sfml_vk(sf::Keyboard::Z) == 0x5A, "vk/Z");
    check(sfml_vk(sf::Keyboard::Num0) == 0x30, "vk/0");
    check(sfml_vk(sf::Keyboard::Num9) == 0x39, "vk/9");
    check(sfml_vk(sf::Keyboard::F1) == 0x70, "vk/F1");
    // UNMAPPED IS ZERO, and the caller drops it. A fallthrough that passed the raw SFML code
    // would put a wrong keystroke into a page, which is worse than no keystroke.
    check(sfml_vk(sf::Keyboard::Tilde) == 0, "vk/unmapped -> 0");

    // Coalescing: consecutive moves collapse to the last, and a button never collapses.
    {
        std::vector<raw_input> q;
        raw_input              m;
        m.type = caspar::core::input_event::kind::move;
        m.px   = 1;
        coalesce_into(q, m);
        m.px = 2;
        coalesce_into(q, m);
        m.px = 3;
        coalesce_into(q, m);
        check(q.size() == 1 && q.back().px == 3, "coalesce/moves collapse to the last",
              "size " + std::to_string(q.size()));

        raw_input b;
        b.type = caspar::core::input_event::kind::button;
        coalesce_into(q, b);
        coalesce_into(q, b);
        check(q.size() == 3, "coalesce/buttons never collapse", "size " + std::to_string(q.size()));
    }

    // ---- A REAL WINDOW, and real injected input -----------------------------------------
    if (!XOpenDisplay(nullptr)) {
        std::printf("\n  SKIPPED the window half: no X display. The pure mappings above still ran.\n");
        std::printf("%s\n", failures ? "FAILURES" : "all pure checks passed");
        return failures ? 1 : 0;
    }

    sf::Window window(sf::VideoMode(640, 360), "sfml_input_self_test", sf::Style::Default);

    auto* dpy = XOpenDisplay(nullptr);
    int   ev = 0, err = 0, major = 0, minor = 0;
    if (!XTestQueryExtension(dpy, &ev, &err, &major, &minor)) {
        std::printf("\n  SKIPPED the window half: no XTest extension, so input cannot be injected.\n");
        std::printf("%s\n", failures ? "FAILURES" : "all pure checks passed");
        return failures ? 1 : 0;
    }

    // Let the compositor map the window and give it the pointer.
    for (int i = 0; i < 60; ++i) {
        sf::Event e;
        while (window.pollEvent(e)) {
        }
        sf::sleep(sf::milliseconds(16));
    }

    const auto sz = window.getSize();

    // THE WINDOW'S TRUE ORIGIN, from the X server -- not from `sf::Window::getPosition()`.
    //
    // Under XWayland a client cannot place its own window: `setPosition` is a request the
    // compositor is free to ignore, and `getPosition` then reports somewhere the window is not.
    // The first version of this test trusted it, aimed `XTestFakeMotionEvent` at
    // `getPosition() + offset`, and got ZERO move events -- while the button and drag checks
    // below passed, because those only need focus and not a position. So the failure looked like
    // "moves do not arrive" when moves arrive perfectly and the aim was wrong.
    //
    // `XTranslateCoordinates` asks the server where the window actually is, which is the only
    // authority on a compositor that does its own placement.
    int  origin_x = 0, origin_y = 0;
    Window child = 0;
    XTranslateCoordinates(dpy, static_cast<Window>(window.getSystemHandle()),
                          DefaultRootWindow(dpy), 0, 0, &origin_x, &origin_y, &child);
    std::printf("  window origin from the X server: %d,%d  size %ux%u\n", origin_x, origin_y, sz.x, sz.y);

    auto pump = [&](std::vector<raw_input>& out) {
        for (int i = 0; i < 30; ++i) {
            sf::Event e;
            while (window.pollEvent(e)) {
                raw_input in;
                bool      have = false;
                switch (e.type) {
                    case sf::Event::MouseMoved:
                        in.type = caspar::core::input_event::kind::move;
                        in.px   = e.mouseMove.x;
                        in.py   = e.mouseMove.y;
                        have    = true;
                        break;
                    case sf::Event::MouseButtonPressed:
                    case sf::Event::MouseButtonReleased:
                        in.type    = caspar::core::input_event::kind::button;
                        in.button  = sfml_button(e.mouseButton.button);
                        in.pressed = e.type == sf::Event::MouseButtonPressed;
                        in.px      = e.mouseButton.x;
                        in.py      = e.mouseButton.y;
                        have       = in.button >= 0;
                        break;
                    case sf::Event::KeyPressed:
                    case sf::Event::KeyReleased:
                        in.type    = caspar::core::input_event::kind::key;
                        in.pressed = e.type == sf::Event::KeyPressed;
                        in.key     = sfml_vk(static_cast<int>(e.key.code));
                        have       = in.key != 0;
                        break;
                    default:
                        break;
                }
                if (have) {
                    in.mods = sfml_modifiers();
                    coalesce_into(out, in);
                }
            }
            sf::sleep(sf::milliseconds(8));
        }
    };

    // A move to a KNOWN point inside the window, so the reported client coordinates can be
    // checked against arithmetic rather than against "something arrived".
    {
        const int want_x = static_cast<int>(sz.x) / 4;
        const int want_y = static_cast<int>(sz.y) / 3;
        XTestFakeMotionEvent(dpy, -1, origin_x + want_x, origin_y + want_y, 0);
        XFlush(dpy);

        std::vector<raw_input> got;
        pump(got);

        bool found = false;
        for (const auto& in : got)
            if (in.type == caspar::core::input_event::kind::move && std::abs(in.px - want_x) <= 2 &&
                std::abs(in.py - want_y) <= 2)
                found = true;
        check(found, "window/a move arrives in CLIENT coordinates",
              "wanted ~" + std::to_string(want_x) + "," + std::to_string(want_y) + " from " +
                  std::to_string(got.size()) + " events");
    }

    // THE RIGHT BUTTON, which is the one the 2013 cast got wrong. X button 3 is the right
    // button; it must arrive as `input_event`'s 2, not as SFML's 1.
    {
        std::vector<raw_input> got;
        XTestFakeButtonEvent(dpy, 3, True, 0);
        XFlush(dpy);
        pump(got);

        bool right_pressed = false;
        uint32_t mods = 0;
        for (const auto& in : got)
            if (in.type == caspar::core::input_event::kind::button && in.pressed && in.button == 2) {
                right_pressed = true;
                mods          = in.mods;
            }
        check(right_pressed, "window/X button 3 arrives as button 2 (right), not 1");

        // AND THE MODIFIER MASK, which is the other 2013 defect: the held button must be in it.
        check((mods & caspar::core::mod_right_button) != 0,
              "window/the held right button is in the modifier mask",
              "mods=" + std::to_string(mods));

        XTestFakeButtonEvent(dpy, 3, False, 0);
        XFlush(dpy);
        std::vector<raw_input> release;
        pump(release);
    }

    // A move WHILE THE LEFT BUTTON IS HELD -- the exact case that never worked for five years,
    // because a move event carries no modifier information of its own.
    {
        XTestFakeButtonEvent(dpy, 1, True, 0);
        XFlush(dpy);
        std::vector<raw_input> down;
        pump(down);

        XTestFakeMotionEvent(dpy, -1, origin_x + static_cast<int>(sz.x) / 2,
                             origin_y + static_cast<int>(sz.y) / 2, 0);
        XFlush(dpy);
        std::vector<raw_input> moved;
        pump(moved);

        bool held_on_move = false;
        for (const auto& in : moved)
            if (in.type == caspar::core::input_event::kind::move &&
                (in.mods & caspar::core::mod_left_button))
                held_on_move = true;
        check(held_on_move, "window/a MOVE reports the held button, so an in-page drag works",
              std::to_string(moved.size()) + " move events");

        XTestFakeButtonEvent(dpy, 1, False, 0);
        XFlush(dpy);
        std::vector<raw_input> up;
        pump(up);

        bool released_clean = true;
        for (const auto& in : up)
            if (in.type == caspar::core::input_event::kind::move &&
                (in.mods & caspar::core::mod_left_button))
                released_clean = false;
        check(released_clean, "window/and stops reporting it once released");
    }

    XCloseDisplay(dpy);
    window.close();

    std::printf("\n%s\n", failures ? "FAILURES" : "all checks passed");
    return failures ? 1 : 0;
}
