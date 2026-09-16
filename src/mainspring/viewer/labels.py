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

An axis name this table has never seen passes through as itself, escaped for the HTML
form. A writer whose frames produce an axis nobody anticipated should get an honest label
rather than nothing.
"""

from __future__ import annotations

from html import escape

__all__ = ["DISPLAY_NAMES", "html", "plain"]

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


def plain(name: str) -> str:
    """`name` as the status bar spells it: no markup, for a line of numbers."""
    return DISPLAY_NAMES.get(name, (name, name))[0]


def html(name: str) -> str:
    """`name` as an axis label spells it, with the markup a figure deserves."""
    known = DISPLAY_NAMES.get(name)
    return known[1] if known is not None else escape(name)
