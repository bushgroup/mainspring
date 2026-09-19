"""The chromatogram: total signal against time for the whole file, in a dock of its own.

An operator watching a run can see the frame in front of them and nothing about how the
run is going. This is that missing view -- one point per frame, the whole file at once --
and it is also how an operator picks the part of a run worth summing: highlight a span,
read what it covers, and add it up.

**Two sources, and the instant one is the default.** The file's own per-frame `TIC`
column is one grouped query (`UimfFile.frame_totals`) and costs 79 ms on a real 45 MB
clockwork run, so it is what the panel shows from the moment a file opens. The true
sum over the heatmap's current view is a different number -- it answers "how much of
what I am looking at is in each frame", which is the question that finds where a feature
elutes -- and it costs a read and a selection per frame, 3 to 14 ms, so a whole run is
seconds to a minute. That one lives behind `Restrict to view`, runs on `LoadWorker` in
chunks, and is cancelled by whatever the user does next (lab record, task 31).

**A dock, not a cell in the heatmap's layout.** Three things follow, and all three are
why it is a dock. Docked left or right it draws vertically and sits outside the
heatmap's arrival-time axis; docked at the bottom it draws horizontally, which is how a
chromatogram is printed; floated it is a window of its own on a second screen. And
being outside the heatmap's `QGraphicsScene` is what keeps it out of an exported figure
without `export.content_rect` having to know it exists.

**The band is a marker and nothing else.** A right-drag highlights a span and says what
it covers; it does not zoom, it does not move the heatmap, and it changes nothing at all
until `Sum selection` is pressed. A gesture that both selected and acted would make the
expensive thing -- adding up a few hundred frames -- the accidental result of a slip.

**Hidden by default**, unlike the info panel. An existing user's window should not lose
width or height to a panel they have not asked for on the strength of an upgrade;
`View > Chromatogram` is where they ask.

The gestures, which are the projections' contract one plot over:

| gesture | meaning |
|---|---|
| right drag, or Shift and a left drag | highlight a span along the time axis |
| left click | show the frame under the pointer |
| double-click | clear the highlight |
| wheel, left drag | nothing |
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import fonts, labels, theme
from .controls import describe
from .heatmap import LINE_WIDTH_LIMITS, REFERENCE_VIEWPORT
from .settings import CHROMATOGRAM_SIZE

__all__ = [
    "BAND_FILL_ALPHA",
    "BAND_LINE_ALPHA",
    "CURVE_WIDTH",
    "ChromatogramPanel",
    "ChromatogramViewBox",
    "FRAME_AXIS",
    "MIN_BAND_PIXELS",
    "TIME_AXIS",
    "TOTAL_AXIS",
    "TOTAL_UNITS",
    "elapsed_minutes",
]

TIME_AXIS = "Time (min)"
FRAME_AXIS = "Frame"
TOTAL_AXIS = "Intensity"
"""The panel's three axis names, in the canonical spelling `viewer/labels.py` keys its
second table by. They are the panel's own vocabulary and deliberately not rows in
`DISPLAY_NAMES`, which is the rasteriser's four names and is held against `raster.py` by
a set-equality test: one more row there would break the check that keeps the two lists
honest, for names `raster.py` never produces.

`Intensity` rather than `Total intensity`, which is what each point is: the label is
rotated in the horizontal orientation, where the plot is about 170 pixels tall, and the
longer name plus the unit below does not fit in that and is clipped. What each point is
a total *of* is the readout's sentence, which has room for it."""

TOTAL_UNITS = "counts"
"""The unit the intensity axis is labelled with, and the reason its tick values are
readable at all.

A frame's total is hundreds of millions of ADC counts, and pyqtgraph writes that as
`1.7e+08` a tick, which two ticks apart in a panel docked on its side overlap into
`1.7e+081.8e+08`. Its own SI-prefix scaling fixes it and is on by default, but it only
engages on an axis that has a unit: with one, the ticks read 170 and 180 and the label
reads `Intensity (Mcounts)`. The unit is right as well as useful, since a stored
intensity is a count from the digitizer."""

MIN_BAND_PIXELS = 2.0
"""How far a drag travels before it is a band rather than a click that wobbled. The same
number `UimfViewBox._rect_zoom` and `ProjectionViewBox._band_zoom` use, so one hand
produces one result wherever it presses."""

CURVE_WIDTH = 1.5
"""How thick the trace is drawn at `REFERENCE_VIEWPORT`, scaled and clamped by
`LINE_WIDTH_LIMITS` the way every other line in the viewer is.

Half a pixel heavier than a projection's, because this curve is read across a bench for
its *shape* -- is the run steady, did it drop -- rather than inspected for one-bin
spikes, and because there are at most a few thousand points in it against a spectrum's
114688."""

BAND_FILL_ALPHA = 48
BAND_LINE_ALPHA = 160
"""How opaque the highlight's fill and its two edges are, over the palette's curve color.

A `LinearRegionItem` is deliberately outside `theme._PAINTED`: the color bar's own region
and its drag handles are color-*map* furniture and must not be asked whether they are in
the palette. That exemption covers this band too, so its colors are checked by a case
written for it in `tests/test_theme.py` rather than by the walk. Alpha over the curve
color rather than a fifth palette role, because the band marks the curve -- it is the
same statement at lower contrast -- and because a role would have to be invented twice,
once per palette, for something with no meaning of its own."""


def elapsed_minutes(
    frames: "tuple[int, ...]", start_times: "dict[int, float]"
) -> "tuple[float, ...] | None":
    """The frames' start times in minutes, or None when the file does not really have any.

    **Carrying `StartTimeMinutes` is not the same as meaning it.**
    `FrameSpec.start_time_minutes` defaults to 0.0 and `UimfWriter` writes it on every
    frame, so a file can hold the parameter on five thousand frames and say nothing --
    which is exactly what the synthetic scale files do. The test is therefore not "is it
    there" but "does it move": every frame has one, they never go backwards, and the
    first and last differ. A real clockwork acquisition passes it (100 distinct values
    0.638 s apart over a 64 s run) and so does a PNNL LC run through the legacy
    `StartTime` column; both synthetic files and the summed companion fail it, the first
    because the writer stored a default and the second because one frame has no spread
    (lab record, task 31).

    The values are the file's own, not shifted to start at zero. A start time is a
    number the writer recorded -- a retention time on an LC run -- and subtracting the
    first frame's would quietly answer a different question than the file was asked.
    """
    if len(frames) < 2:
        return None
    try:
        values = tuple(float(start_times[frame]) for frame in frames)
    except KeyError:
        return None  # some frame has none, so the axis would have a hole in it
    if values[-1] <= values[0]:
        return None
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        return None
    return values


class ChromatogramViewBox(pg.ViewBox):
    """The box under the trace, where a drag highlights and a click goes to a frame.

    `ProjectionViewBox`'s contract with one substitution: the band a right-drag leaves
    behind is a *marker* rather than a range to apply, so it is drawn by a
    `LinearRegionItem` that stays after the button comes up instead of by the rubber
    band that disappears with it. Everything else is the same hand -- right-drag or
    Shift and a left-drag, read on one axis and drawn across the other, a drag shorter
    than `MIN_BAND_PIXELS` treated as a click that wobbled.

    `axis` is 0 when the time axis runs horizontally and 1 when it runs vertically, so
    the two dock orientations are one number rather than two code paths. The box acts on
    nothing itself: it calls the panel back, the way a projection calls the heatmap.
    """

    def __init__(self, band, picked, cleared, parent: "object | None" = None) -> None:
        super().__init__(parent=parent, enableMenu=False, defaultPadding=0.0)
        self._axis = 0
        self._band = band
        self._picked = picked
        self._cleared = cleared
        self.setMouseEnabled(x=False, y=False)

    def set_axis(self, axis: int) -> None:
        """Say which of the box's two axes time runs along. 0 horizontal, 1 vertical."""
        self._axis = int(axis)

    # --- gestures ---------------------------------------------------------------------

    def mouseDragEvent(self, ev, axis=None) -> None:
        """Right drag, or Shift and a left drag, highlights. Nothing else drags."""
        button = ev.button()
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if button == Qt.MouseButton.RightButton or (
            button == Qt.MouseButton.LeftButton and shift
        ):
            self._highlight(ev)
            return
        ev.ignore()

    def mouseClickEvent(self, ev) -> None:
        """A double-click clears the highlight; a single left click picks a frame."""
        if ev.double():
            ev.accept()
            self._cleared()
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()
        point = self.mapToView(ev.pos())
        self._picked(point.x() if self._axis == 0 else point.y())

    def wheelEvent(self, ev, axis=None) -> None:
        """Inert, explicitly. There is one point per frame here and the whole file is
        the view worth having; a wheel that scrolled it away would only lose the run."""
        ev.ignore()

    def _highlight(self, ev) -> None:
        """The rubber band while the button is down, the marker once it comes up."""
        ev.accept()
        start = ev.buttonDownPos(ev.button())
        corners = self._corners(start, ev.pos())
        if not ev.isFinish():
            self.updateScaleBox(*corners)
            return
        self.rbScaleBox.hide()
        travelled = (
            abs(ev.pos().x() - start.x()) if self._axis == 0 else abs(ev.pos().y() - start.y())
        )
        if travelled <= MIN_BAND_PIXELS:
            return  # a click that wobbled, not a band; leave the marker where it was
        span = self.childGroup.mapRectFromParent(QRectF(*corners).normalized())
        lo, hi = (span.left(), span.right()) if self._axis == 0 else (span.top(), span.bottom())
        self._band(min(lo, hi), max(lo, hi))

    def _corners(self, start: QPointF, end: QPointF) -> "tuple[QPointF, QPointF]":
        """The drag as a band: the pointer's travel along the time axis, the box's own
        extent across the other, so what is drawn is what will be marked."""
        rect = self.boundingRect()
        if self._axis == 0:
            return QPointF(start.x(), rect.top()), QPointF(end.x(), rect.bottom())
        return QPointF(rect.left(), start.y()), QPointF(rect.right(), end.y())


class ChromatogramPanel(QDockWidget):
    """The trace, the two controls above it, and the readout that says what is marked.

    A dock like the info panel and built the same way, with one invariant taken from it
    and widened: **nothing inside has a say in how big the dock is.** The readout is a
    wrapping label of `QSizePolicy.Policy.Ignored` in both directions, because this
    panel can be docked along either edge and a label that fixed a width when it was at
    the bottom would fix a height when it was moved to the left. `CHROMATOGRAM_SIZE` is
    the floor, on whichever side the dock is currently on.
    """

    frame_picked = Signal(int)
    """A frame the user clicked on, which the window shows by the ordinary route."""
    sum_requested = Signal(object)
    """`tuple[int, ...]` -- the frames the band covers, to be added up. Emitted only
    from `Sum selection`, never from the drag that drew the band."""
    restrict_toggled = Signal(bool)
    """`Restrict to view` was ticked or unticked; the window starts or cancels the walk."""

    _curve_width = CURVE_WIDTH
    """Class-level default for the reason `SidePlots._curve_width` is one: the first
    `fit_lines` needs something to compare against."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__("Chromatogram", parent)
        self.setObjectName("chromatogram_panel")

        # Two sources, held apart rather than merged into one dictionary. They answer
        # different questions -- the whole frame's total, and the total of what is in
        # the heatmap's window -- so a chunk of one landing among the other's values
        # would put a point of a different quantity into the trace, and the trace would
        # not say so. Keeping both also makes unticking `Restrict to view` instant: the
        # stored series is still here and nothing has to be read again.
        self._stored: dict[int, float] = {}
        self._walked: dict[int, float] = {}
        self._frames: tuple[int, ...] = ()
        self._times: dict[int, float] = {}
        self._minutes: tuple[float, ...] | None = None
        self._raw_units = False
        self._span_frame: int | None = None
        self._band_range: "tuple[float, float] | None" = None
        self._vertical = False
        self._restricted = False

        container = QWidget(self)
        self._container = container
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._restrict_box = QCheckBox("Restrict to view")
        self._restrict_tip = (
            "Total only the points inside the heatmap's current view, one frame at a"
            " time, instead of the whole-frame totals the file already stores. It reads"
            " every frame, so a long run takes seconds."
        )
        describe(self._restrict_box, self._restrict_tip)
        self._restrict_box.toggled.connect(self._on_restrict_toggled)
        controls.addWidget(self._restrict_box)
        self._sum_button = QPushButton("Sum selection")
        describe(
            self._sum_button,
            "Add the frames the highlight covers into one heatmap. Drag with the right"
            " button to move the highlight.",
        )
        self._sum_button.setEnabled(False)
        self._sum_button.clicked.connect(self._on_sum_clicked)
        controls.addWidget(self._sum_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._readout = QLabel("")
        describe(
            self._readout,
            "What this trace is showing, and what the highlight covers: the frames in"
            " it, how long they span, and their total intensity.",
        )
        # Both directions `Ignored`, unlike the info panel's labels: this dock can sit
        # along any edge, so the dimension a long sentence would otherwise dictate is
        # not known when the label is built.
        self._readout.setWordWrap(True)
        self._readout.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self._readout)

        self._view_box = ChromatogramViewBox(self._set_band, self._pick, self.clear_band)
        self.plot = pg.PlotWidget(viewBox=self._view_box)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        # The same reason `HeatmapView` hides pyqtgraph's own "A": it calls
        # `enableAutoRange`, and the span here is set deliberately (`set_span`).
        self.plot.setMinimumSize(0, 0)
        # The `PlotItem` and its axes are what `controls.unexplained` walks, not the
        # `QGraphicsView` around them, and a tooltip on the widget as well would put two
        # of them under one pointer.
        describe(
            self.plot.getPlotItem(),
            "Total intensity in each frame of the file, against time. Right-drag to"
            " highlight a span, left-click to show a frame, double-click to clear.",
        )
        for name in ("left", "bottom", "top", "right"):
            describe(
                self.plot.getPlotItem().getAxis(name),
                "The run's time or frame number, and the total intensity in each frame.",
            )
        layout.addWidget(self.plot, 1)
        self.setWidget(container)

        self._curve = self.plot.plot()
        self._curve.setDownsampling(auto=True, method="peak")
        self._band: "pg.LinearRegionItem | None" = None
        self._build_band(vertical=False)

        # Qt reports the edge the dock is on whenever it moves, which is the one thing
        # the drawing direction depends on. Connected before the first layout so that a
        # dock restored to the left edge draws vertically from the start.
        self.dockLocationChanged.connect(self._on_location_changed)
        self.topLevelChanged.connect(lambda _floating: self._apply_orientation())
        self._apply_orientation()
        self.set_palette(theme.active())
        self.set_text_scale(fonts.active())

    # --- orientation -------------------------------------------------------------------

    def _on_location_changed(self, area) -> None:
        self._vertical = area in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self._apply_orientation()

    @property
    def vertical(self) -> bool:
        """Whether time runs down the plot rather than across it. Read for the tests and
        for the readout, which names the axis the band was drawn on."""
        return self._vertical

    def _apply_floor(self) -> None:
        """Fix the least the panel goes to, on whichever dimension it is drawn along.

        One dimension only: the other is the heat map's to have, and a floor on both
        would be this dock telling the window how wide to be as well as how tall. It
        follows `View > Text size` like the info panel's, because what has to fit is the
        controls and the tick values rather than a picture.
        """
        floor = round(CHROMATOGRAM_SIZE * fonts.extent())
        vertical = self._vertical and not self.isFloating()
        self._container.setMinimumWidth(floor if vertical else 0)
        self._container.setMinimumHeight(0 if vertical else floor)

    def _build_band(self, *, vertical: bool) -> None:
        """Make the highlight, in the orientation the dock is currently in.

        Rebuilt rather than turned, because a `LinearRegionItem` takes its orientation
        at construction and pyqtgraph offers no way to change it afterwards. The span it
        was covering is carried over, so moving the dock while a span is highlighted
        keeps the highlight rather than quietly losing the selection the user is about
        to sum.
        """
        region = self._band_range
        if self._band is not None:
            self.plot.removeItem(self._band)
        self._band = pg.LinearRegionItem(
            orientation="horizontal" if vertical else "vertical", movable=False
        )
        self._band.setZValue(-10)  # under the trace: a marker, not something over it
        self.plot.addItem(self._band)
        if region is None:
            self._band.hide()
        else:
            self._band.setRegion(region)

    def _apply_orientation(self) -> None:
        """Point the time axis along the dock's long side, and redraw everything on it.

        A floated panel draws horizontally whatever edge it came from: it is a window of
        its own then, and a window is wider than it is tall.
        """
        vertical = self._vertical and not self.isFloating()
        self._view_box.set_axis(1 if vertical else 0)
        self._build_band(vertical=vertical)
        self._apply_floor()
        # The band is new, so it is holding pyqtgraph's default colors until the palette
        # is put back on it; `set_palette` ends in `_redraw`, which is the rest of this.
        self.set_palette(theme.active())

    # --- the series --------------------------------------------------------------------

    def clear(self) -> None:
        """Forget the file. Called when another one is opened, before anything arrives."""
        self._stored = {}
        self._walked = {}
        self._times = {}
        self._frames = ()
        self._minutes = None
        self._span_frame = None
        self._restricted = False
        self.clear_band()
        self._restrict_box.blockSignals(True)
        self._restrict_box.setChecked(False)
        self._restrict_box.blockSignals(False)
        self._resolve()

    def set_series(
        self,
        totals: "dict[int, float]",
        start_times: "dict[int, float]",
        *,
        replace: bool = True,
    ) -> None:
        """Show the file's own per-frame totals, replacing what is there or adding to it.

        `replace=False` is the live poll's form: a poll asks only about the tail that
        can have changed (`UimfFile.frame_totals(since=...)`), so what arrives is one or
        two frames rather than the file, and merging it in is what makes a followed run
        grow a point at a time instead of re-reading itself once a second.

        The start times merge either way, because a frame's start time is written once
        and never changes: a `replace` here is a re-read of the same file rather than a
        different one, and a different file goes through `clear` first.
        """
        if replace:
            self._stored = dict(totals)
        else:
            self._stored.update(totals)
        self._times.update(start_times)
        self._resolve()

    def set_restricted(self, totals: "dict[int, float]") -> None:
        """Show the view-restricted walk's answer so far, and switch the trace to it.

        The walk's own accumulated dictionary, not a chunk of it, so the trace is
        whatever has been computed and grows as the walk does. It **replaces** rather
        than merging into the stored series: a half-finished walk drawn over the file's
        own totals would read as a complete trace with a few odd points in it, which is
        the one way this panel could lie.
        """
        self._walked = dict(totals)
        self._restricted = True
        self._resolve()

    def clear_restricted(self) -> None:
        """Go back to the file's own totals, which are still here and need no re-read."""
        self._walked = {}
        self._restricted = False
        self._resolve()

    def _resolve(self) -> None:
        """Settle which source is being drawn, and redraw from it."""
        self._totals = self._walked if self._restricted else self._stored
        self._frames = tuple(sorted(self._totals))
        self._minutes = elapsed_minutes(self._frames, self._times)
        self._redraw()

    def set_raw_units(self, raw: bool) -> None:
        """Follow `View > Raw units`: frame number instead of elapsed time.

        The highlight is **re-expressed**, not kept and not cleared: the same frames
        stay under it, at whatever numbers the new axis gives them. That is the rule
        `_rebuild_axes` already follows for the heatmap's view range, and it has to
        apply here for the same reason -- the toggle says how the axis is *labelled*,
        and a selection that silently emptied itself when a label changed would cost the
        user the span they were about to sum.

        On a file whose start times are not usable (`elapsed_minutes`) this changes
        nothing visible, because the axis is already the frame number -- and the readout
        says so, so that a toggle that appears to do nothing is explained rather than
        merely inert.
        """
        if bool(raw) == self._raw_units:
            return
        before = self._positions()
        self._raw_units = bool(raw)
        after = self._positions()
        if self._band_range is not None and len(before) == len(after) >= 2:
            source = np.asarray(before, dtype=np.float64)
            target = np.asarray(after, dtype=np.float64)
            low, high = self._band_range
            self._band_range = (float(np.interp(low, source, target)),
                                float(np.interp(high, source, target)))
            self._band.setRegion(self._band_range)
        self._redraw()

    def set_span(self, last_frame: "int | None") -> None:
        """Extend the time axis out to the frame a full method frame would end on.

        A run of 100 repetitions writes its hundredth frame 64 seconds after its first,
        and a trace that rescaled to whatever had arrived would make every moment of the
        run look the same. Given the frame the current method frame's intended
        repetition count reaches -- which every clockwork file carries and nothing else
        does -- the axis is that long from the first point on, and the trace fills it.

        `None` on a file with no grouping, where the axis is simply the frames there are.
        A parameter for the whole method's length does not exist, and inventing one
        before this has been used at the bench would be guessing (lab record, task 31).
        """
        self._span_frame = None if last_frame is None else int(last_frame)
        self._redraw()

    @property
    def restricted(self) -> bool:
        """Whether the trace on screen is the view-restricted walk rather than the
        file's own stored totals."""
        return self._restricted

    @property
    def series(self) -> "tuple[tuple[int, ...], tuple[float, ...]]":
        """`(frames, totals)` as drawn, for a caller that wants to read the trace back."""
        return self._frames, tuple(self._totals[frame] for frame in self._frames)

    # --- the band ----------------------------------------------------------------------

    def clear_band(self) -> None:
        """Take the highlight down and disable `Sum selection` with it."""
        self._band_range = None
        self._band.hide()
        self._sum_button.setEnabled(False)
        self._describe()

    def _set_band(self, low: float, high: float) -> None:
        self._band_range = (float(low), float(high))
        self._band.setRegion((float(low), float(high)))
        self._band.show()
        self._sum_button.setEnabled(bool(self.selected_frames()))
        self._describe()

    def selected_frames(self) -> "tuple[int, ...]":
        """The frames the highlight covers, ascending, or empty when there is none.

        Inclusive of both ends: a band drawn over a feature is meant to contain the
        frames whose points are under it, and half-open would silently drop the last.
        """
        if self._band_range is None or not self._frames:
            return ()
        low, high = self._band_range
        positions = self._positions()
        return tuple(
            frame for frame, position in zip(self._frames, positions)
            if low <= position <= high
        )

    def _pick(self, position: float) -> None:
        """Emit the frame nearest the click, which is what "go there" means on a trace
        of one point per frame: a click never lands exactly on a point."""
        if not self._frames:
            return
        positions = np.asarray(self._positions(), dtype=np.float64)
        index = int(np.argmin(np.abs(positions - float(position))))
        self.frame_picked.emit(int(self._frames[index]))

    def _on_sum_clicked(self) -> None:
        frames = self.selected_frames()
        if frames:
            self.sum_requested.emit(frames)

    def _on_restrict_toggled(self, checked: bool) -> None:
        self.restrict_toggled.emit(bool(checked))

    def set_restrict_enabled(self, enabled: bool, *, why: str = "") -> None:
        """Allow or refuse `Restrict to view`, and say why in the tooltip when refusing.

        Switched off while a run is being followed: the walk reads every frame of the
        file, and a poll that has to wait behind one is a poll that misses the frame it
        was watching for. Unticking it here does not emit, because turning the control
        off is not the user asking for the stored series back -- the window cancels the
        walk itself when following starts.

        The reason is **appended** to the sentence the control always carries rather
        than replacing it: a disabled control still has to say what it would do, or the
        tooltip walk's rule is met by a sentence that stops being true when it is
        enabled again.
        """
        self._restrict_box.setEnabled(bool(enabled))
        if not enabled and self._restrict_box.isChecked():
            self._restrict_box.blockSignals(True)
            self._restrict_box.setChecked(False)
            self._restrict_box.blockSignals(False)
        describe(self._restrict_box, f"{self._restrict_tip} {why}" if why
                 else self._restrict_tip)

    @property
    def restrict_checked(self) -> bool:
        """Whether `Restrict to view` is ticked, for a caller deciding what to ask for."""
        return self._restrict_box.isChecked()

    # --- drawing -----------------------------------------------------------------------

    def _positions(self) -> "tuple[float, ...]":
        """Where each frame sits on the time axis: its start time, or its own number.

        One function, so that the trace, the band's frame set and the click that picks a
        frame cannot disagree about what a coordinate on that axis means.
        """
        if self._minutes is None or self._raw_units:
            return tuple(float(frame) for frame in self._frames)
        return self._minutes

    def _axis_name(self) -> str:
        return FRAME_AXIS if (self._minutes is None or self._raw_units) else TIME_AXIS

    def _span_position(self) -> "float | None":
        """The far end of the axis, in the units it is drawn in.

        On the frame axis it is the frame `set_span` named. On the time axis that frame
        has not been written yet and so has no start time: it is projected from the
        median step between the frames that *have* arrived, which is the acquisition's
        own frame rate measured rather than assumed.
        """
        if self._span_frame is None or not self._frames:
            return None
        if self._minutes is None or self._raw_units:
            return float(self._span_frame)
        if len(self._minutes) < 2:
            return None
        steps = np.diff(np.asarray(self._minutes, dtype=np.float64))
        return float(self._minutes[-1] + float(np.median(steps))
                     * max(0, self._span_frame - self._frames[-1]))

    def _redraw(self) -> None:
        vertical = self._vertical and not self.isFloating()
        positions = np.asarray(self._positions(), dtype=np.float64)
        values = np.asarray([self._totals[f] for f in self._frames], dtype=np.float64)
        # The trace's own x is the time axis when the panel is horizontal and the
        # intensity when it is vertical, exactly as `SidePlots.set_profiles` swaps the
        # arrival-time plot: the curve is drawn in the plot's coordinates, and which of
        # them time is depends on the edge the dock is on.
        if vertical:
            self._curve.setData(values, positions)
        else:
            self._curve.setData(positions, values)
        # Downsampling and clipping both work along a curve's x axis, so they are worth
        # having in the horizontal orientation and mean nothing in the vertical one.
        self._curve.setDownsampling(auto=not vertical, method="peak")
        self._curve.setClipToView(not vertical)

        time_side, value_side = ("left", "bottom") if vertical else ("bottom", "left")
        style = theme.label_style(theme.active())
        self.plot.getPlotItem().getAxis(time_side).setLabel(
            labels.html(self._axis_name()), **style
        )
        self.plot.getPlotItem().getAxis(value_side).setLabel(
            labels.html(TOTAL_AXIS), units=TOTAL_UNITS, **style
        )
        self._apply_span(vertical, positions, values)
        self._describe()

    def _apply_span(self, vertical: bool, positions, values) -> None:
        """Set the axes: the intensity one to the data, the time one to the whole run."""
        plot = self.plot.getPlotItem()
        if positions.size == 0:
            plot.enableAutoRange(x=True, y=True)
            return
        low, high = float(positions.min()), float(positions.max())
        end = self._span_position()
        if end is not None:
            high = max(high, end)
        if high <= low:
            high = low + 1.0
        if vertical:
            plot.enableAutoRange(x=True, y=False)
            plot.setYRange(low, high, padding=0.02)
        else:
            plot.enableAutoRange(x=False, y=True)
            plot.setXRange(low, high, padding=0.02)

    def _describe(self) -> None:
        """One line under the controls: what the trace is, and what the band covers."""
        if not self._frames:
            self._readout.setText("No file open.")
            return
        source = ("summed over the heatmap's view" if self._restricted
                  else "the file's own per-frame totals")
        axis = labels.plain(self._axis_name())
        note = ""
        if self._minutes is None:
            note = ", because this file records no frame start times"
        parts = [f"{len(self._frames):,} frames, {source}, against {axis}{note}."]
        frames = self.selected_frames()
        if frames:
            total = sum(self._totals[frame] for frame in frames)
            low, high = self._band_range or (0.0, 0.0)
            covers = (f"frames {frames[0]} to {frames[-1]}" if len(frames) > 1
                      else f"frame {frames[0]}")
            width = (f", {high - low:,.3g} min" if self._axis_name() == TIME_AXIS
                     else "")
            parts.append(f"Highlighted: {covers} ({len(frames):,}{width}),"
                         f" {total:,.0f} total.")
        elif self._band_range is not None:
            parts.append("Highlighted: no frames.")
        self._readout.setText("  ".join(parts))

    # --- theme and text ------------------------------------------------------------------

    def set_palette(self, palette: "theme.Palette") -> None:
        """Repaint the trace, the axes and the band in `palette`, live.

        Called at construction and again from `theme.apply`, the one path. The band is
        the palette's curve color at `BAND_FILL_ALPHA`/`BAND_LINE_ALPHA`: a
        `LinearRegionItem` is outside the walk `theme.themed` makes, so this is the one
        color on the canvas held by a test of its own rather than by the walk.
        """
        self.plot.setBackground(palette.background)
        pen = pg.mkPen(palette.foreground)
        for name in ("left", "bottom", "top", "right"):
            axis = self.plot.getPlotItem().getAxis(name)
            axis.setPen(pen)
            axis.setTickPen(pen)
            axis.setTextPen(pen)
        self._curve.setPen(pg.mkPen(palette.curve, width=self._curve_width))
        fill = pg.mkColor(palette.curve)
        fill.setAlpha(BAND_FILL_ALPHA)
        edge = pg.mkColor(palette.curve)
        edge.setAlpha(BAND_LINE_ALPHA)
        self._band.setBrush(pg.mkBrush(fill))
        for line in self._band.lines:
            line.setPen(pg.mkPen(edge))
        # `setLabel` replaces `labelStyle` wholesale, so the labels are re-set from the
        # new palette rather than left holding the old one's color (`theme.py`).
        self._redraw()

    def set_text_scale(self, scale: float) -> None:
        """Follow `View > Text size` for the tick values and the axis labels.

        The controls and the readout are ordinary widgets and take the application font
        (`fonts.py`); the four axes are `QGraphicsItem`s and do not, which is the whole
        of what this has to do. `_redraw` carries the label size, because it goes in the
        same style dict as the color (`theme.label_style`).
        """
        font = fonts.scaled_font(scale)
        for name in ("left", "bottom", "top", "right"):
            self.plot.getPlotItem().getAxis(name).setTickFont(font)
        self._apply_floor()
        self._redraw()

    def fit_lines(self, width: int, height: int) -> bool:
        """Draw the trace at the width this viewport wants, and say whether it moved.

        `SidePlots.fit_lines` one module over, with the same reference, clamp and
        one-decimal comparison, re-applied through `set_palette` for the same reason: a
        theme toggle must not reset the thickness and a resize must not reset the color.
        """
        low, high = LINE_WIDTH_LIMITS
        fitted = min(high, max(low, CURVE_WIDTH * min(width, height) / REFERENCE_VIEWPORT))
        if round(fitted, 1) == round(self._curve_width, 1):
            return False
        self._curve_width = fitted
        self.set_palette(theme.active())
        return True

    @property
    def curve_width(self) -> float:
        """What the trace is drawn at right now, for a caller that wants to read it back."""
        return self._curve_width

    def resizeEvent(self, event) -> None:
        """Refit the trace's pen to the dock's new size. The panel is a widget of its
        own, unlike the projections, so it has a resize event to hang this on."""
        super().resizeEvent(event)
        size = self.plot.size()
        self.fit_lines(max(1, size.width()), max(1, size.height()))
