"""Make a diagram generator able to see its own output.

A matplotlib figure exits 0 whatever it draws. Three defects got through review of
`generate_ocio_diagrams.py` on 2026-08-16 and each needed a human to look at the PNG:

* a caption written straight across an unrelated box;
* a label wider than the box it was drawn into, so it overran the edge;
* an arrow routed through a heading.

All three are geometry, so all three are checkable. This module records what a generator
draws and asserts the three at save time, which turns a look-and-nudge loop into a failing
run.

    lay = Layout(fig, ax)
    lay.panel("composite", 47, 62, 15, 18, fc=SUCCESS)
    lay.text("composite", 54.5, 74, "composite", parent="composite")   # must FIT it
    lay.text(None, 50, 4, "a caption", parent=None)                    # must clear everything
    lay.arrow((40, 70), (47, 72))
    lay.check()

WHAT IT DOES NOT DO. A diagram can pass every one of these and still explain the wrong
thing — the `<working-space-composite>` badge in its first position was perfectly legible
and implied the gate applied to the composite rather than to everything downstream. This
removes the mechanical iterations, not the review.

`blocking=False` is for panels that are meant to contain other things — the background
frame, and group containers like "per LAYER". Without it every label inside a container
would be reported as a collision with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Rect:
    name: str
    x: float
    y: float
    w: float
    h: float
    blocking: bool = True

    @property
    def bounds(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def contains(self, other, tol: float = 0.6) -> bool:
        ax0, ay0, ax1, ay1 = self.bounds
        bx0, by0, bx1, by1 = other
        return (bx0 >= ax0 - tol and by0 >= ay0 - tol
                and bx1 <= ax1 + tol and by1 <= ay1 + tol)

    def overlaps(self, other, tol: float = 0.4) -> bool:
        ax0, ay0, ax1, ay1 = self.bounds
        bx0, by0, bx1, by1 = other
        return not (bx1 <= ax0 + tol or bx0 >= ax1 - tol
                    or by1 <= ay0 + tol or by0 >= ay1 - tol)


def _segment_hits_rect(p0, p1, rect, tol: float = 0.4) -> bool:
    """Liang-Barsky against a slightly shrunk rect, so an arrow that merely grazes an
    edge is not reported — arrows legitimately pass close to the labels they point past."""
    x0, y0 = p0
    x1, y1 = p1
    rx0, ry0, rx1, ry1 = rect
    rx0, ry0, rx1, ry1 = rx0 + tol, ry0 + tol, rx1 - tol, ry1 - tol
    if rx1 <= rx0 or ry1 <= ry0:
        return False
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - rx0), (dx, rx1 - x0), (-dy, y0 - ry0), (dy, ry1 - y0)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


class Layout:
    """Records panels, texts and arrows, then checks the three geometric failure modes."""

    def __init__(self, fig, ax, *, panel_fn, text_fn, arrow_fn):
        self.fig = fig
        self.ax = ax
        self._panel_fn = panel_fn
        self._text_fn = text_fn
        self._arrow_fn = arrow_fn
        self._rects: list[_Rect] = []
        self._texts: list[tuple] = []      # (artist, parent_name, label)
        self._arrows: list[tuple] = []     # (p0, p1)

    def panel(self, name, x, y, w, h, *, blocking=True, **kw):
        self._rects.append(_Rect(name or f"panel@{x},{y}", x, y, w, h, blocking))
        return self._panel_fn(self.ax, x, y, w, h, **kw)

    def text(self, name, x, y, s, *, parent=None, **kw):
        art = self._text_fn(self.ax, x, y, s, **kw)
        if art is None:                      # helper returned nothing; re-draw to capture
            art = self.ax.texts[-1]
        self._texts.append((art, parent, name or s[:28]))
        return art

    def fit_text(self, name, x, y, s, *, parent, size=10.0, min_size=5.5, **kw):
        """Draw text and shrink it until it fits `parent`.

        Prevents the overrun class outright rather than reporting it: a label is given a
        box and cannot leave it. Returns the size actually used, so a caller that cares
        can notice it was shrunk a long way and rethink the layout instead.
        """
        p = next((r for r in self._rects if r.name == parent), None)
        if p is None:
            raise KeyError(f"fit_text: no panel named {parent!r} — declare it first")
        art = self.text(name, x, y, s, parent=parent, size=size, **kw)
        self.fig.canvas.draw()
        used = size
        while used > min_size and not p.contains(self._text_bounds(art)):
            used = round(used - 0.25, 2)
            art.set_fontsize(used)
            self.fig.canvas.draw()
        return used

    def arrow(self, p0, p1, **kw):
        self._arrows.append((p0, p1))
        return self._arrow_fn(self.ax, p0, p1, **kw)

    # -- checking ---------------------------------------------------------------
    def _text_bounds(self, art):
        bb = art.get_window_extent(renderer=self.fig.canvas.get_renderer())
        bb = bb.transformed(self.ax.transData.inverted())
        return bb.x0, bb.y0, bb.x1, bb.y1

    def _check_bounds(self, problems):
        """Flag anything laid out past the axis limits.

        THE THIRD FAILURE MODE, and it went unchecked until 2026-08-27. Patches added to an
        axes are clipped to it by default while text is not, so a panel running past the
        limit loses its border and keeps its label -- which reads as a design choice rather
        than as a mistake. `bbox_inches="tight"` does not save it: tight cropping grows the
        saved region to include artists, but the patch was already clipped when it was drawn.

        Found by eye twice in one sitting: `exec_scope.png` laid four 21-wide columns from
        x=7.5 (right edge 100.5) and `exec_hdr.png` five 17.5-wide stages from x=5 with the
        last at x=91 (right edge 108.5). Both rendered, both looked deliberate, both wrong.
        """
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        tol = 0.05

        def outside(bx0, by0, bx1, by1):
            return (bx0 < x0 - tol or bx1 > x1 + tol
                    or by0 < y0 - tol or by1 > y1 + tol)

        for r in self._rects:
            b = r.bounds
            if outside(*b):
                problems.append(
                    f"panel {r.name!r} extends past the axes "
                    f"({b[0]:.1f}..{b[2]:.1f} x {b[1]:.1f}..{b[3]:.1f}, "
                    f"axes {x0:.0f}..{x1:.0f} x {y0:.0f}..{y1:.0f}) — its border will be clipped")
        for art, _parent, label in self._texts:
            b = self._text_bounds(art)
            if outside(*b):
                problems.append(
                    f"text {label!r} extends past the axes "
                    f"({b[0]:.1f}..{b[2]:.1f} x {b[1]:.1f}..{b[3]:.1f})")
        for p0, p1 in self._arrows:
            for p in (p0, p1):
                if outside(p[0], p[1], p[0], p[1]):
                    problems.append(f"arrow endpoint {p} is past the axes")

    def _check_panel_collisions(self, problems):
        """Flag two panels that PARTIALLY overlap.

        Nesting is legitimate and common -- an inner box inside an outer one -- so full
        containment either way is fine. What is never intentional is a partial overlap: it
        means a column pitch and a column width disagree, and the result is two rounded
        borders crossing each other a few units in from the edge, which at diagram scale
        looks like a heavy divider rather than a mistake.

        Added 2026-08-27 after `exec_to_production.png` laid four 24.1-wide panels on a 23.5
        pitch. Every existing check passed: this class tests panels against TEXT and arrows,
        and nothing compared a panel with another panel.
        """
        for i, a in enumerate(self._rects):
            for b in self._rects[i + 1:]:
                if a.contains(b.bounds, tol=0.0) or b.contains(a.bounds, tol=0.0):
                    continue                      # nested, which is the intended use
                if a.overlaps(b.bounds, tol=0.0):
                    problems.append(
                        f"panels {a.name!r} and {b.name!r} partially overlap "
                        f"({a.bounds[0]:.1f}..{a.bounds[2]:.1f} against "
                        f"{b.bounds[0]:.1f}..{b.bounds[2]:.1f}) — check the pitch against the width")

    def check(self, *, name: str = "diagram"):
        self.fig.canvas.draw()
        by_name = {r.name: r for r in self._rects}
        problems: list[str] = []
        self._check_bounds(problems)
        self._check_panel_collisions(problems)

        for art, parent, label in self._texts:
            b = self._text_bounds(art)
            if parent is not None:
                p = by_name.get(parent)
                if p is None:
                    problems.append(f"text {label!r} names parent {parent!r}, which is not a panel")
                elif not p.contains(b):
                    problems.append(
                        f"text {label!r} overruns its panel {parent!r} "
                        f"(text {b[0]:.1f}..{b[2]:.1f} x {b[1]:.1f}..{b[3]:.1f}, "
                        f"panel {p.bounds[0]:.1f}..{p.bounds[2]:.1f} x {p.bounds[1]:.1f}..{p.bounds[3]:.1f})")
            # Panels nest: a label parented to an inner box legitimately overlaps the outer
            # one that holds it. Skip the parent's ancestors, or every nested label is
            # reported and the check gets switched off for crying wolf.
            ancestors = set()
            if parent in by_name:
                pr = by_name[parent]
                ancestors = {r.name for r in self._rects
                             if r.name != parent and r.contains(pr.bounds, tol=0.0)}
            for r in self._rects:
                if not r.blocking or r.name == parent or r.name in ancestors:
                    continue
                if r.overlaps(b):
                    problems.append(f"text {label!r} sits on top of panel {r.name!r}")
            for p0, p1 in self._arrows:
                if _segment_hits_rect(p0, p1, b):
                    problems.append(f"arrow {p0}->{p1} crosses text {label!r}")

        if problems:
            raise AssertionError(
                "%s: %d layout problem(s) a renderer would not report —\n  %s"
                % (name, len(problems), "\n  ".join(sorted(set(problems)))))
        return True
