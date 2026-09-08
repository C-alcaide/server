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

// The SFML -> `raw_input` translation, as an include rather than a header.
//
// AN `.inl` INCLUDED IN TWO PLACES, and the sharing is the point. `screen_consumer.cpp` pulls it
// in for the real consumer, and `sfml_input_self_test.cpp` pulls in the SAME text to check it
// against a real SFML window on a real display. A copy in the test would test the copy, and
// would agree with the original exactly until the day one of them changed.
//
// Not a normal header because it deliberately has no include guard of its own and depends on
// `raw_input` and `caspar::core::input_modifier` already being declared by its includer -- which
// is what lets the self-test supply a trimmed `input_event` instead of dragging the monitor and
// frame types into a standalone program.
//
// Windows does not compile any of this: `win32_gl_window` fills `raw_input` from `lParam`
// instead. Everything downstream of `raw_input` IS shared, which is why the letterbox rejection
// and the live-client-rect normalisation did not have to be written twice.

// ---------------------------------------------------------------------------
// The SFML -> raw_input translation helpers.
//
// Three of them, and each exists because the naive version of it is one of the five defects the
// 2013-2018 interaction API shipped with.

/// SFML's button enum -> the button number `input_event` uses, which is CEF's.
///
/// **THIS IS THE 2013 DEFECT, and it is the whole reason this is a function.** That code cast
/// `sf::Mouse::Button` straight to `CefMouseButtonType`. SFML orders Left, Right, Middle and CEF
/// orders Left, Middle, Right -- so every right-click arrived as a middle-click and every
/// middle-click as a right-click, for five years. A cast between two unrelated enumerations that
/// happen to agree on their first element is not a conversion.
///
/// -1 for a button `input_event` has no number for (SFML's XButton1/XButton2), so the caller
/// drops the event rather than inventing a button.
inline int sfml_button(int sfml)
{
    switch (sfml) {
        case sf::Mouse::Left:
            return 0;
        case sf::Mouse::Middle:
            return 1;
        case sf::Mouse::Right:
            return 2;
        default:
            return -1;
    }
}

/// The modifier mask, queried from the CURRENT keyboard and mouse state.
///
/// **THIS IS THE OTHER 2013 DEFECT.** `e.modifiers` was never set, so `SendMouseMoveEvent`
/// always told the page that no button was held and no in-page drag ever worked -- clicks were
/// fine, which is why it survived. SFML makes the same mistake easy: a `MouseMoved` event
/// carries a position and NOTHING about what is held, so the state has to be asked for.
///
/// `sf::Keyboard::isKeyPressed` and `sf::Mouse::isButtonPressed` read the real device rather
/// than the event, which is the right thing here and is worth contrasting with the Win32 path:
/// there the mask rides along in `wParam`, so it describes the moment the message was generated.
/// This describes the moment it is processed. One pump behind at worst, and a chord that is held
/// across a frame -- which is every chord an operator can actually perform -- is identical.
inline uint32_t sfml_modifiers()
{
    uint32_t m = caspar::core::mod_none;

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::RShift))
        m |= caspar::core::mod_shift;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::RControl))
        m |= caspar::core::mod_control;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LAlt) ||
        sf::Keyboard::isKeyPressed(sf::Keyboard::RAlt))
        m |= caspar::core::mod_alt;

    if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        m |= caspar::core::mod_left_button;
    if (sf::Mouse::isButtonPressed(sf::Mouse::Middle))
        m |= caspar::core::mod_middle_button;
    if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
        m |= caspar::core::mod_right_button;

    return m;
}

/// SFML's key code -> a WINDOWS virtual key.
///
/// Translated rather than passed through, and this is a deliberate cross-platform decision
/// rather than Windows chauvinism. Three things downstream already speak VK codes:
///
///   * `INPUT <ch> KEY DOWN <vk>` documents its argument as a numeric virtual key, so the same
///     command has to mean the same keystroke on either platform or the command is a lie;
///   * CEF's `CefKeyEvent::windows_key_code` is windows-style **on every platform**, including
///     Linux -- that is the field's name and its contract;
///   * `previz_renderer::input` compares against `VK_LEFT` and friends for its arrow nudges.
///
/// Passing SFML's own enum through would make `KEY DOWN 37` mean the left arrow on Windows and
/// `sf::Keyboard::B` on Linux. Silently, and only for whoever tried it.
///
/// **0 for anything unmapped, and the caller drops the event.** A partial table is honest; a
/// fallthrough that passed the raw code would put a wrong keystroke into a page, which is worse
/// than no keystroke. What is here is what a control surface and a template actually use:
/// letters, digits, the arrows, the editing keys and the function keys.
inline int sfml_vk(int code)
{
    // Letters: SFML A..Z are 0..25 in order, VK_A..VK_Z are 0x41..0x5A in order.
    if (code >= sf::Keyboard::A && code <= sf::Keyboard::Z)
        return 0x41 + (code - sf::Keyboard::A);

    // Digits: SFML Num0..Num9 in order, VK_0..VK_9 are 0x30..0x39 in order.
    if (code >= sf::Keyboard::Num0 && code <= sf::Keyboard::Num9)
        return 0x30 + (code - sf::Keyboard::Num0);

    // Function keys: SFML F1..F15 in order, VK_F1..VK_F15 are 0x70..0x7E in order.
    if (code >= sf::Keyboard::F1 && code <= sf::Keyboard::F15)
        return 0x70 + (code - sf::Keyboard::F1);

    switch (code) {
        case sf::Keyboard::Left:      return 0x25; // VK_LEFT
        case sf::Keyboard::Up:        return 0x26; // VK_UP
        case sf::Keyboard::Right:     return 0x27; // VK_RIGHT
        case sf::Keyboard::Down:      return 0x28; // VK_DOWN
        case sf::Keyboard::Escape:    return 0x1B; // VK_ESCAPE
        case sf::Keyboard::Enter:     return 0x0D; // VK_RETURN
        case sf::Keyboard::Space:     return 0x20; // VK_SPACE
        case sf::Keyboard::Tab:       return 0x09; // VK_TAB
        case sf::Keyboard::Backspace: return 0x08; // VK_BACK
        case sf::Keyboard::Delete:    return 0x2E; // VK_DELETE
        case sf::Keyboard::Insert:    return 0x2D; // VK_INSERT
        case sf::Keyboard::Home:      return 0x24; // VK_HOME
        case sf::Keyboard::End:       return 0x23; // VK_END
        case sf::Keyboard::PageUp:    return 0x21; // VK_PRIOR
        case sf::Keyboard::PageDown:  return 0x22; // VK_NEXT
        default:                      return 0;
    }
}

/// Append `in` to `out`, coalescing consecutive pointer moves.
///
/// The Windows path coalesces inside the window's own queue; SFML has no queue of ours, so it
/// happens here. Same reason either way: a fast mouse produces far more moves than there are
/// frames, and every one of them that survives is a virtual call into the stage and, for an HTML
/// layer, a post to CEF's UI thread.
inline void coalesce_into(std::vector<raw_input>& out, const raw_input& in)
{
    if (in.type == caspar::core::input_event::kind::move && !out.empty() &&
        out.back().type == caspar::core::input_event::kind::move) {
        out.back() = in;
        return;
    }

    // The same 256 cap the Win32 queue uses, and dropping MOVES rather than the oldest event:
    // a dropped move is a position that is about to be superseded, and a dropped button is a
    // gesture that never completes.
    if (out.size() > 256 && in.type == caspar::core::input_event::kind::move)
        return;

    out.push_back(in);
}
