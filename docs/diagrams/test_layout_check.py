"""Reintroduce each of the three real defects and confirm the checker catches it.

A check that cannot fail is not evidence. These are the exact three that got through
review of the OCIO diagrams on 2026-08-16.
"""
import os
import sys

sys.path.insert(0, r"d:\Github\CasparVP\docs\diagrams")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_ocio_diagrams import _arrow, _layout, _new, _panel, _text, PANEL, MUTED


def case(name, build):
    fig, ax = _new((11, 6))
    lay = _layout(fig, ax)
    build(lay)
    try:
        lay.check(name=name)
    except AssertionError as e:
        first = str(e).split("\n")[1].strip()
        print(f"  CAUGHT  {name:<22} {first}")
        plt.close(fig)
        return True
    print(f"  MISSED  {name:<22} the checker did not fire")
    plt.close(fig)
    return False


def caption_across_a_box(lay):
    lay.panel("box", 40, 50, 20, 10, fc=PANEL)
    lay.text("caption", 30, 55, "a caption written straight across the box", color=MUTED)


def label_overruns_its_box(lay):
    lay.panel("narrow", 40, 50, 8, 6, fc=PANEL)
    # No fit_text: this is the raw text() path, which is what overran.
    lay.text("label", 44, 53, "far too wide for eight units", parent="narrow",
             size=10, ha="center", color=MUTED)


def arrow_through_a_heading(lay):
    lay.text("heading", 20, 50, "per CONSUMER — one extra pass per DISTINCT view",
             color=MUTED)
    lay.arrow((10, 40), (70, 60))


def nested_label_is_not_a_collision(lay):
    """The false positive that made the first version unusable — must NOT fire."""
    lay.panel("outer", 30, 40, 30, 20, fc=PANEL)
    lay.panel("inner", 34, 44, 22, 8, fc=PANEL)
    lay.text("nested", 45, 48, "inside both", parent="inner", ha="center", color=MUTED)


if __name__ == "__main__":
    print("defects that MUST be caught:")
    caught = [case("caption-across-box", caption_across_a_box),
              case("label-overrun", label_overruns_its_box),
              case("arrow-through-text", arrow_through_a_heading)]
    print("\nlegitimate layout that must NOT be flagged:")
    clean = not case("nested-label", nested_label_is_not_a_collision)
    print(f"\n{sum(caught)}/3 defects caught; nested-label false positive: "
          f"{'absent' if clean else 'PRESENT'}")
    sys.exit(0 if all(caught) and clean else 1)
