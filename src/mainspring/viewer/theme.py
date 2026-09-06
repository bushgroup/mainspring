"""Both plot palettes, the one path that applies them, and the walk that proves it.

The viewer never chose its black canvas; it inherited pyqtgraph's default, and the Qt
chrome around it has always followed the OS palette. `View > Light mode` makes that a
choice. Two states only, dark still the default, and the **plot canvas only** -- the
menu bar, toolbar, info dock and dialogs stay the operating system's (lab record,
task 14, which is also where the reasoning against a `QPalette` theme lives).

**The colour map is not part of the theme.** `COLOUR_MAPS` grows no reversed variants
and nothing here touches the user's choice: the map is a statement about the data and
the palette is a statement about the furniture around it, so a near-empty frame drawn
as a dark rectangle on a white canvas is the honest picture and not a bug.

Three mechanisms have to agree for a theme to change *live*, which is the whole
difficulty:

1. `pg.setConfigOptions(background=..., foreground=...)` is read when an item is
   **constructed**. Setting it fixes whatever is built next and nothing already on
   screen, so it is necessary and never sufficient.
2. An axis label's colour is spelled out in the style dict handed to
   `AxisItem.setLabel`, because that call *replaces* `labelStyle` wholesale rather than
   merging into it. `label_style` is therefore a function of the palette and not a
   module-level constant: a constant computed at import cannot answer a runtime toggle,
   and the failure it produces is a label that is present, bold and invisible.
3. Every colour set once at construction has to be set again. That is what
   `HeatmapView.set_palette` and `SidePlots.set_palette` are for, and `apply` is the one
   path that calls them.

`themed` is the analogue of `controls.unexplained`, with one difference that matters.
"Has a tooltip" is a property Qt answers uniformly; "the theme set this colour" is not a
bit anything records, so the strongest available test is that every pen, brush and text
pen on the items the palette owns is a colour *in the active palette*. Under the dark
palette that test passes for anything hardcoded to today's values, which is most of what
it exists to catch -- so **`themed` is run under the light palette**, the non-default
one, by `tools/check_public.py` and by `tests/test_theme.py`. Run only under dark it
would be theatre.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from .controls import name_of, numbered

__all__ = [
    "DARK",
    "LIGHT",
    "PALETTES",
    "Palette",
    "active",
    "apply",
    "label_style",
    "themed",
]


@dataclass(frozen=True)
class Palette:
    """The four colours the plot canvas is allowed to paint with.

    Four rather than one background and one foreground because the two accents are
    already in the tree and each has a job the foreground cannot do: a curve has to be
    distinguishable from an axis at a glance, and the render-time readout has to be
    legible over the image it is drawn on top of.
    """

    name: str
    background: str
    """The canvas behind the image, the axes and both projections."""
    foreground: str
    """Axis lines, tick marks, tick values and the bold axis labels."""
    curve: str
    """The mass spectrum's and the arrival-time distribution's traces."""
    debug: str
    """The render-time overlay, drawn only under `MAINSPRING_DEBUG_RENDER`."""

    def colours(self) -> "tuple[QColor, ...]":
        return tuple(
            QColor(value) for value in (self.background, self.foreground, self.curve, self.debug)
        )

    def holds(self, colour: QColor) -> bool:
        """Is `colour` one of this palette's four? What `themed` asks of every pen."""
        return colour.rgb() in {value.rgb() for value in self.colours()}


DARK = Palette(
    name="dark",
    background="#000000",
    foreground="#969696",
    curve="#bed2ff",
    debug="#dcdcdc",
)
"""Exactly what the viewer has drawn since M2: pyqtgraph's own `k` and `d` defaults,
`side_plots`' pale curve blue and the debug overlay's near-white. Spelled out rather
than read from the config options so that a pyqtgraph release that moves its defaults
cannot move this viewer's look."""

LIGHT = Palette(
    name="light",
    background="#ffffff",
    foreground="#686868",
    curve="#204080",
    debug="#202020",
)
"""The mirror of `DARK` on white, matched on contrast rather than by inverting the
channels: `#686868` on white is the 5.6:1 that `#969696` gives on black, so the axes
read as the same muted furniture at the same weight rather than as a heavier frame, and
`#204080` is the curve blue's counterpart at the ratio the pale blue has on black.
Inverting instead would put a pale blue trace on white, which is close to unreadable."""

PALETTES = {palette.name: palette for palette in (DARK, LIGHT)}
"""Keyed by name. The same two names are `settings.THEMES`, which is where they live
because `ViewerSettings.validate` has to clamp a hand-edited one and that module does
not import the plot layer; `tests/test_theme.py` is what holds the two lists together.
Not imported from there, so that nothing in the plot layer depends on the settings
module for a name it already knows."""

_ACTIVE = DARK
"""The palette `apply` last applied, and the one `themed` measures against. Module
state because the config options it sets are module state in pyqtgraph too: there is one
canvas per process and a second idea of which palette is on it would only ever disagree
with the first."""

_PAINTED = (pg.AxisItem, pg.PlotDataItem, pg.TextItem)
"""The item types `themed` inspects: everything on the canvas that paints a colour the
palette owns.

What is deliberately *not* here, and why:

* `pg.ImageItem` -- the heatmap's pixels and the colour bar's gradient strip are colour
  *map* colours, which the theme does not set and must not.
* the `ColorBarItem`'s `LinearRegionItem` and its two `InfiniteLine` handles -- drawn on
  top of that gradient, so they belong to the colour bar rather than to the canvas.
* `pg.PlotCurveItem` and `pg.ScatterPlotItem` -- the two halves of a `PlotDataItem`,
  which sets both from its own `pen` and `symbol` options. The projections are bare
  curves and never set `symbol`, so the scatter half never paints at all.
* `pg.LabelItem` -- a `PlotItem` title, and no plot here carries one.

**Hidden items are walked too**, unlike `controls.unexplained`, which skips them because
there is nothing to point at. Here there is: the render-time overlay is hidden unless
`MAINSPRING_DEBUG_RENDER` is set and the projections hide all eight of their axes, so a
walk that skipped what is not currently drawn would exempt the two places a stray colour
is least likely to be noticed by eye.
"""


def active() -> Palette:
    """The palette in force. What a widget being constructed should paint itself with."""
    return _ACTIVE


def label_style(palette: Palette) -> "dict[str, str]":
    """The style dict for a bold axis label under `palette`.

    A function and not a constant, for the reason in this module's docstring: the colour
    has to be in the dict, because `AxisItem.setLabel(**style)` replaces `labelStyle`
    rather than merging into it, and a colour computed at import time cannot answer a
    toggle. `HeatmapView` re-reads this on every `set_palette` and every `set_image`,
    so a label is repainted whichever of the two happens next.
    """
    return {"color": QColor(palette.foreground).name(), "font-weight": "bold"}


def apply(window: object, theme: str) -> Palette:
    """Repaint the whole plot canvas in `theme`, live, and return the palette used.

    The one path a theme change takes. It sets the config options so that anything
    constructed afterwards is born right, then calls each owner's `set_palette` so that
    everything already on screen is corrected -- nothing is rebuilt, so the open file,
    the frame, the view ranges and the colour levels all survive a toggle untouched.

    An unrecognised name falls back to `DARK` rather than raising: this is called with a
    restored setting, and `ViewerSettings.validate` has already clamped it, so a name
    that still gets here is a caller's bug and should cost a black canvas, not a window
    that will not open.
    """
    global _ACTIVE
    palette = PALETTES.get(theme, DARK)
    _ACTIVE = palette
    pg.setConfigOptions(background=palette.background, foreground=palette.foreground)
    window.heatmap.set_palette(palette)
    window.side_plots.set_palette(palette)
    return palette


def themed(window: object) -> "list[str]":
    """Names of everything on the plot canvas painted a colour the active palette lacks.

    Type-based like `controls.unexplained`, and for the same reason: an item added
    tomorrow is walked because of what it is. An empty list under **both** palettes is
    the rule; under the light one is the half that can fail.
    """
    palette = active()
    stray: list[str] = []
    view = window.heatmap
    _check(palette, view.backgroundBrush().color(), f"{name_of(view)} background", stray)

    for item in view.scene().items():
        if not isinstance(item, _PAINTED):
            continue
        name = name_of(item)
        if isinstance(item, pg.AxisItem):
            _check_pen(palette, item.pen(), f"{name} axis pen", stray)
            _check_pen(palette, item.tickPen(), f"{name} tick pen", stray)
            _check_pen(palette, item.textPen(), f"{name} text pen", stray)
            label = item.labelStyle.get("color")
            if label is not None:
                _check(palette, QColor(label), f"{name} label colour", stray)
        elif isinstance(item, pg.PlotDataItem):
            _check_pen(palette, item.opts.get("pen"), f"{name} curve pen", stray)
            _check_brush(palette, item.opts.get("fillBrush"), f"{name} fill brush", stray)
        else:  # pg.TextItem
            _check(palette, item.color, f"{name} text colour", stray)

    return numbered(stray)


def _check(palette: Palette, colour: "QColor | None", name: str, stray: "list[str]") -> None:
    if colour is not None and not palette.holds(QColor(colour)):
        stray.append(f"{name} ({QColor(colour).name()})")


def _check_pen(palette: Palette, pen: object, name: str, stray: "list[str]") -> None:
    """A pen that draws nothing is not a colour on the canvas.

    `Qt.PenStyle.NoPen` is how pyqtgraph says "no outline" while still carrying a colour
    in the pen -- the debug overlay's border is one -- and reporting that colour would
    send a reader looking for something that is not on screen.
    """
    if pen is None:
        return
    pen = pg.mkPen(pen)
    if pen.style() == Qt.PenStyle.NoPen:
        return
    _check(palette, pen.color(), name, stray)


def _check_brush(palette: Palette, brush: object, name: str, stray: "list[str]") -> None:
    if brush is None:
        return
    brush = pg.mkBrush(brush)
    if brush.style() == Qt.BrushStyle.NoBrush:
        return
    _check(palette, brush.color(), name, stray)
