"""Both plot palettes, the one path that applies them, and the walk that proves it.

The viewer never chose its black canvas; it inherited pyqtgraph's default, and the Qt
chrome around it has always followed the OS palette. `View > Light mode` makes that a
choice. Two states only, dark still the default, and the **plot canvas only** -- the
menu bar, toolbar, info dock and dialogs stay the operating system's (lab record,
task 14, which is also where the reasoning against a `QPalette` theme lives).

**The color map is not part of the theme.** `COLOR_MAPS` grows no reversed variants
and nothing here touches the user's choice: the map is a statement about the data and
the palette is a statement about the furniture around it, so a near-empty frame drawn
as a dark rectangle on a white canvas is the honest picture and not a bug.

Three mechanisms have to agree for a theme to change *live*, which is the whole
difficulty:

1. `pg.setConfigOptions(background=..., foreground=...)` is read when an item is
   **constructed**. Setting it fixes whatever is built next and nothing already on
   screen, so it is necessary and never sufficient.
2. An axis label's color is spelled out in the style dict handed to
   `AxisItem.setLabel`, because that call *replaces* `labelStyle` wholesale rather than
   merging into it. `label_style` is therefore a function of the palette and not a
   module-level constant: a constant computed at import cannot answer a runtime toggle,
   and the failure it produces is a label that is present, bold and invisible.
3. Every color set once at construction has to be set again. That is what
   `HeatmapView.set_palette` and `SidePlots.set_palette` are for, and `apply` is the one
   path that calls them.

`themed` is the analogue of `controls.unexplained`, with one difference that matters.
"Has a tooltip" is a property Qt answers uniformly; "the theme set this color" is not a
bit anything records, so the strongest available test is that every pen, brush and text
pen on the items the palette owns is a color *in the active palette*. Under the dark
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

from . import fonts
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
    """The four colors the plot canvas is allowed to paint with.

    Both palettes now set all three of the foreground roles to the same value -- the
    background's opposite, black or white -- because a viewer read at a glance across a
    bench wants contrast before it wants an accent (Matt, 2026-09-18), and because the
    projections carry no axis of their own for a curve to be confused with. The class
    keeps four fields anyway: they are four *roles*, the walk that proves nothing was
    painted outside a palette is written in terms of them, and an accent restored to any
    one of them is then a value and not a refactor.
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

    def role(self, name: str) -> QColor:
        """One role's color, by field name. What `themed` asks each pen about.

        There was a `holds(color)` here until both palettes became black and white: "is
        this one of the four" then accepted every stray black and every stray white,
        which is the two colors a forgotten default is most likely to be. The question
        worth asking is per role, and it is this one.
        """
        return QColor(getattr(self, name))


DARK = Palette(
    name="dark",
    background="#000000",
    foreground="#ffffff",
    curve="#ffffff",
    debug="#ffffff",
)
"""Everything the palette owns at the greatest contrast the canvas can carry: white on
black. Spelled out rather than read from the config options so that a pyqtgraph release
that moves its defaults cannot move this viewer's look."""

LIGHT = Palette(
    name="light",
    background="#ffffff",
    foreground="#000000",
    curve="#000000",
    debug="#000000",
)
"""The same rule on white: black on white. `DARK` and `LIGHT` differ only in which of
the two the background is."""

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
"""The item types `themed` inspects: everything on the canvas that paints a color the
palette owns.

What is deliberately *not* here, and why:

* `pg.ImageItem` -- the heatmap's pixels and the color bar's gradient strip are color
  *map* colors, which the theme does not set and must not.
* the `ColorBarItem`'s `LinearRegionItem` and its two `InfiniteLine` handles -- drawn on
  top of that gradient, so they belong to the color bar rather than to the canvas.
* `pg.PlotCurveItem` and `pg.ScatterPlotItem` -- the two halves of a `PlotDataItem`,
  which sets both from its own `pen` and `symbol` options. The projections are bare
  curves and never set `symbol`, so the scatter half never paints at all.
* `pg.LabelItem` -- a `PlotItem` title, and no plot here carries one.

**Hidden items are walked too**, unlike `controls.unexplained`, which skips them because
there is nothing to point at. Here there is: the render-time overlay is hidden unless
`MAINSPRING_DEBUG_RENDER` is set and the projections hide all eight of their axes, so a
walk that skipped what is not currently drawn would exempt the two places a stray color
is least likely to be noticed by eye.
"""


def active() -> Palette:
    """The palette in force. What a widget being constructed should paint itself with."""
    return _ACTIVE


def label_style(palette: Palette) -> "dict[str, str]":
    """The style dict for a bold axis label under `palette`, at the active text size.

    A function and not a constant, for the reason in this module's docstring: the color
    has to be in the dict, because `AxisItem.setLabel(**style)` replaces `labelStyle`
    rather than merging into it, and a color computed at import time cannot answer a
    toggle. `HeatmapView` re-reads this on every `set_palette` and every `set_image`,
    so a label is repainted whichever of the two happens next.

    The size is here for exactly the same reason, and being in the same dict is what
    makes the two follow each other: a palette change re-reads the scale and a scale
    change re-reads the palette, so neither can put the other's old value back. `theme`
    imports `fonts` and never the reverse -- the scale is something an axis label is
    drawn at, not something the palette decides.
    """
    return {
        "color": QColor(palette.foreground).name(),
        "font-weight": "bold",
        "font-size": fonts.css_size(),
    }


def apply(window: object, theme: str) -> Palette:
    """Repaint the whole plot canvas in `theme`, live, and return the palette used.

    The one path a theme change takes. It sets the config options so that anything
    constructed afterwards is born right, then calls each owner's `set_palette` so that
    everything already on screen is corrected -- nothing is rebuilt, so the open file,
    the frame, the view ranges and the color levels all survive a toggle untouched.

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
    """Names of everything on the plot canvas painted a color its role does not have.

    Type-based like `controls.unexplained`, and for the same reason: an item added
    tomorrow is walked because of what it is. An empty list under **both** palettes is
    the rule.

    **The test is per role, not per palette**, and it has to be. Both palettes are now
    black and white (`DARK`, `LIGHT`), so "is this color one of the palette's four"
    accepts every stray black and every stray white -- which is to say it accepts the
    two colors a forgotten default is overwhelmingly likely to be, and the walk would
    have become theatre. Asking instead whether the *axis pen* is the foreground and the
    *curve* the curve color is strictly stronger, it catches the invisible-label bug
    under either palette, and it costs one argument at each call site that already knew
    which role it was looking at.
    """
    palette = active()
    stray: list[str] = []
    view = window.heatmap
    _check(palette, "background", view.backgroundBrush().color(), f"{name_of(view)} background", stray)

    for item in view.scene().items():
        if not isinstance(item, _PAINTED):
            continue
        name = name_of(item)
        if isinstance(item, pg.AxisItem):
            _check_pen(palette, "foreground", item.pen(), f"{name} axis pen", stray)
            _check_pen(palette, "foreground", item.tickPen(), f"{name} tick pen", stray)
            _check_pen(palette, "foreground", item.textPen(), f"{name} text pen", stray)
            label = item.labelStyle.get("color")
            if label is not None:
                _check(palette, "foreground", QColor(label), f"{name} label color", stray)
        elif isinstance(item, pg.PlotDataItem):
            _check_pen(palette, "curve", item.opts.get("pen"), f"{name} curve pen", stray)
            _check_brush(palette, "curve", item.opts.get("fillBrush"), f"{name} fill brush", stray)
        else:  # pg.TextItem
            _check(palette, "debug", item.color, f"{name} text color", stray)

    return numbered(stray)


def _check(
    palette: Palette, role: str, color: "QColor | None", name: str, stray: "list[str]"
) -> None:
    if color is None:
        return
    wanted = palette.role(role)
    if QColor(color).rgb() != wanted.rgb():
        stray.append(f"{name} ({QColor(color).name()}, not the {role} {wanted.name()})")


def _check_pen(
    palette: Palette, role: str, pen: object, name: str, stray: "list[str]"
) -> None:
    """A pen that draws nothing is not a color on the canvas.

    `Qt.PenStyle.NoPen` is how pyqtgraph says "no outline" while still carrying a color
    in the pen -- the debug overlay's border is one -- and reporting that color would
    send a reader looking for something that is not on screen.
    """
    if pen is None:
        return
    pen = pg.mkPen(pen)
    if pen.style() == Qt.PenStyle.NoPen:
        return
    _check(palette, role, pen.color(), name, stray)


def _check_brush(
    palette: Palette, role: str, brush: object, name: str, stray: "list[str]"
) -> None:
    if brush is None:
        return
    brush = pg.mkBrush(brush)
    if brush.style() == Qt.BrushStyle.NoBrush:
        return
    _check(palette, role, brush.color(), name, stray)
