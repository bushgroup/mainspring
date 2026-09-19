"""How an axis name is spelled on screen, as against what the rasteriser calls it.

`DisplayAxes.build` produces four axis names -- `"m/z"`, `"TOF bin"`, `"Arrival time
(ms)"` and `"Scan"` -- and those are the canonical strings (`mainspring/uimf/raster.py`).
They are a data-layer value: they travel with a `RasterResult`, a pipeline that never
opens a window reads them, and `mainspring.uimf` may not know what a `<span>` is. How
they are *set* in a figure is a different question and a viewer's, which is why the
answer is a small table here rather than an edit there.

Two spellings, because the two places a name appears want different things:

* **`html`** is for `AxisItem.setLabel`, which renders HTML. `m/z` is a quantity symbol
  and is italic everywhere it is printed properly, so the axis reads `<i>m</i>/<i>z</i>`.
* **`plain`** is for the status bar, which is a `QLabel` full of numbers rather than a
  figure, and where a lone italic `m` beside a value would read as an error rather than
  as typesetting.

`Arrival time (ms)` reads `Arrival Time / ms` in both. Quantity-solidus-unit is how an
axis is labelled in a paper, the parenthesis is how a variable name is written in code,
and a figure exported from this viewer goes into the first (lab record, task 24).

An axis name neither table has seen passes through as itself, escaped for the HTML
form. A writer whose frames produce an axis nobody anticipated should get an honest label
rather than nothing.

There are **two** tables, and `PANEL_NAMES` says why the chromatogram's three names are
not rows in the rasteriser's four.
"""

from __future__ import annotations

from html import escape

__all__ = ["DISPLAY_NAMES", "PANEL_NAMES", "html", "plain"]

DISPLAY_NAMES: "dict[str, tuple[str, str]]" = {
    "m/z": ("m/z", "<i>m</i>/<i>z</i>"),
    "TOF bin": ("TOF bin", "TOF bin"),
    "Arrival time (ms)": ("Arrival Time / ms", "Arrival Time / ms"),
    "Scan": ("Scan", "Scan"),
}
"""`raster.DisplayAxes.build`'s four names, each as `(plain, html)`.

Keyed by the canonical string, so the day one of them changes there this table stops
matching and the axis falls back to the new string rather than to a stale spelling of the
old one. `tests/test_labels.py` is what holds the two lists together."""

PANEL_NAMES: "dict[str, tuple[str, str]]" = {
    "Time (min)": ("Time / min", "Time / min"),
    "Frame": ("Frame", "Frame"),
    "Intensity": ("Intensity", "Intensity"),
}
"""The chromatogram panel's own axis names, in the same two spellings.

**A second table rather than three more rows in the first**, and the reason is the test
that keeps the first one honest: `tests/test_labels.py` asserts *set equality* between
the names `DisplayAxes.build` actually produces and the keys of `DISPLAY_NAMES`, which
is what makes a rename in `raster.py` loud instead of silent. A name the rasteriser
never produces, added to that table, would break the check for a name it was never about.
These three have a different owner (`viewer/chromatogram.py`) and a different source,
so they get a table of their own; `plain` and `html` read both, because a caller asking
how a name is spelled has no reason to know which vocabulary it came from."""

_SPELLINGS = DISPLAY_NAMES | PANEL_NAMES
"""Both tables as one lookup. Rebuilt here rather than merged at each call: the tables
are module constants and neither grows at runtime."""


def plain(name: str) -> str:
    """`name` as the status bar spells it: no markup, for a line of numbers."""
    return _SPELLINGS.get(name, (name, name))[0]


def html(name: str) -> str:
    """`name` as an axis label spells it, with the markup a figure deserves."""
    known = _SPELLINGS.get(name)
    return known[1] if known is not None else escape(name)
