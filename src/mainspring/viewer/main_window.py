"""`MainWindow`: the one object that knows about all the others.

It owns the file, the settings, the two workers, and the widgets, and it is the only
place where they meet -- the heatmap does not know there is an info panel, the side
plots do not know there is a file. That is what keeps each of the later tasks to one or
two modules.

The flow it coordinates, once around:

    open  ->  LoadWorker decodes a frame  ->  build DisplayAxes  ->  choose the view
          ->  RenderMailbox  ->  RenderWorker  ->  image + profiles + readouts

Every arrow after the third is travelled by every gesture, every frame change, and every
toolbar toggle alike, which is what task 05 built the render path for: a wheel tick, a
frame spinner step and a swap-axes toggle all reach the render worker by the same path,
so there is one place where "what is on screen" is decided and no second, direct route
that could disagree with it.

**Results are matched to a serial, not trusted in arrival order.** The mailbox drops
superseded *requests*, but a result already in flight when the frame changes would
otherwise repaint the new frame's window with the old frame's data. Every request
carries the serial of the state that produced it, and a result whose serial is not the
current one is dropped -- which is what made frame navigation and the axis toggles below
safe to add without touching the render path.

**Choosing the view is where keep-ranges lives.** Opening a file either takes the
frame's full range or keeps whatever is already on screen, depending on the setting;
navigating between frames of the *same* file always keeps the view, regardless of the
setting -- paging through a run should never cost the zoom the user just set. Swapping
axes or switching to raw units does neither: it re-expresses the same visible bins and
scans through the new axis tables (`_rebuild_axes`), which is the third way "keep the
view" comes up here.
"""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import Qt, QByteArray, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QProgressBar,
    QProgressDialog,
    QSpinBox,
    QToolBar,
)

from ..uimf import DisplayAxes, FrameParams, GlobalParams, SparseFrame
from ..uimf.raster import AGGREGATES
from .heatmap import HeatmapView, pixel_of
from .info_panel import InfoPanel
from .settings import COLOUR_SCALES, ViewerSettings, load_settings, save_settings
from .side_plots import SidePlots
from .workers import LoadWorker, RenderMailbox, RenderRequest, RenderWorker

__all__ = ["MainWindow"]

# The sample's own frame-type convention (`notes/uimf-format.md`), extended past what
# our one sample uses: 0 and 1 both mean MS1 across the modern and legacy tables
# (`FrameParams.is_ms1`), 2 is MS/MS, 3 a calibration frame, 4 a prescan. An unrecognised
# code is shown as itself rather than dropped -- a future writer's code is still a
# choice worth filtering on, whether or not this table has a name for it yet.
_FRAME_TYPE_NAMES = {0: "MS1", 1: "MS1", 2: "MS2", 3: "Calibration", 4: "Prescan"}


def _frame_type_name(frame_type: int) -> str:
    return _FRAME_TYPE_NAMES.get(int(frame_type), f"Type {frame_type}")


class MainWindow(QMainWindow):
    """The application window, holding the heatmap, the info panel, the workers and the
    settings."""

    frame_shown = Signal(object)
    """Emitted with the `RasterResult` after each (re)paint -- what a test waits on, and
    what the self-check drives a gesture through."""

    def __init__(self, settings: "ViewerSettings | None" = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()
        self.setWindowTitle("mainspring")
        if self.settings.window_geometry:
            self.restoreGeometry(QByteArray(self.settings.window_geometry))
        else:
            self.resize(1000, 700)

        # State that a signal fired while building the toolbar below can already read --
        # a settings-restored toggle emits its own `toggled` the moment it is checked,
        # and that handler must see "no frame yet" rather than an attribute that does
        # not exist.
        self._global: GlobalParams | None = None
        self._current_frame: SparseFrame | None = None
        self._current_axes: DisplayAxes | None = None
        self._current_frame_number: int | None = None
        self._frame_params: FrameParams | None = None
        self._last_render: object | None = None
        self._serial = 0
        self._open_started = 0.0
        self._opening = False
        self._frame_message = ""
        self._frame_numbers: list[int] = []
        self._frame_types: dict[int, int] = {}
        self._active_frame_numbers: list[int] = []
        self._sum_dialog: QProgressDialog | None = None
        self._sum_count = 0

        self.heatmap = HeatmapView(colour_map=self.settings.colour_map)
        self.setCentralWidget(self.heatmap)
        self.side_plots = SidePlots(self.heatmap)
        self.heatmap.view_resized.connect(self._on_view_resized)
        self.heatmap.view_changed.connect(self._on_view_changed)
        self.heatmap.cursor_moved.connect(self._on_cursor_moved)
        self.heatmap.cursor_left.connect(self._clear_readout)

        self.info_panel = InfoPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_panel)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate: a decode's length is not known upfront
        self._busy.setMaximumWidth(120)
        self._busy.hide()
        self._readout = QLabel("")
        self.statusBar().addPermanentWidget(self._readout)
        self.statusBar().addPermanentWidget(self._busy)
        self._build_menu()
        self._build_toolbar()

        self._worker = LoadWorker(self.settings.cache_budget_mb * 1024 * 1024)
        self._worker.opened.connect(self._on_opened)
        self._worker.frame_loaded.connect(self._on_frame_loaded)
        self._worker.summing_progress.connect(self._on_summing_progress)
        self._worker.summed.connect(self._on_summed)
        self._worker.failed.connect(self._on_failed)

        self._mailbox = RenderMailbox()
        self._render_worker = RenderWorker(self._mailbox)
        self._render_worker.rendered.connect(self._on_rendered)
        self._render_worker.failed.connect(self._on_failed)
        self._render_worker.start()

    def _build_menu(self) -> None:
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._prompt_open)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)

        # A window-level action rather than a key handler on the view box: the reset must
        # work wherever the focus happens to be, which is the complaint about having to
        # find it in a context menu (lab record, task 01).
        reset_action = QAction("&Reset view", self)
        reset_action.setShortcut("Home")
        reset_action.triggered.connect(self.heatmap.reset_range)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(reset_action)

    def _build_toolbar(self) -> None:
        """Every toggle `ViewerSettings` carries, plus frame navigation and sum-all.

        Each control is fully configured -- range, items, initial value from the
        restored settings -- **before** its signal is connected, so that restoring a
        non-default setting cannot fire a handler while the rest of the window is still
        being built.
        """
        toolbar = QToolBar("View", self)
        toolbar.setObjectName("view_toolbar")  # QMainWindow.saveState() keys it by this
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel(" Aggregate: "))
        self._aggregate_box = QComboBox()
        self._aggregate_box.addItems([a.capitalize() for a in AGGREGATES])
        self._aggregate_box.setCurrentText(self.settings.aggregate.capitalize())
        self._aggregate_box.currentTextChanged.connect(self._on_aggregate_changed)
        toolbar.addWidget(self._aggregate_box)

        toolbar.addWidget(QLabel(" Colour: "))
        self._colour_box = QComboBox()
        self._colour_box.addItems([c.capitalize() for c in COLOUR_SCALES])
        self._colour_box.setCurrentText(self.settings.colour_scale.capitalize())
        self._colour_box.currentTextChanged.connect(self._on_colour_scale_changed)
        toolbar.addWidget(self._colour_box)

        self._swap_action = QAction("Swap X/Y", self)
        self._swap_action.setCheckable(True)
        self._swap_action.setChecked(self.settings.swap_axes)
        self._swap_action.toggled.connect(self._on_swap_toggled)
        toolbar.addAction(self._swap_action)

        self._raw_action = QAction("Raw units", self)
        self._raw_action.setCheckable(True)
        self._raw_action.setChecked(self.settings.raw_units)
        self._raw_action.toggled.connect(self._on_raw_units_toggled)
        toolbar.addAction(self._raw_action)

        self._keep_ranges_action = QAction("Keep ranges", self)
        self._keep_ranges_action.setCheckable(True)
        self._keep_ranges_action.setChecked(self.settings.keep_ranges)
        self._keep_ranges_action.toggled.connect(self._on_keep_ranges_toggled)
        toolbar.addAction(self._keep_ranges_action)

        self._keep_levels_action = QAction("Keep levels", self)
        self._keep_levels_action.setCheckable(True)
        self._keep_levels_action.setChecked(self.settings.keep_levels)
        self._keep_levels_action.toggled.connect(self._on_keep_levels_toggled)
        toolbar.addAction(self._keep_levels_action)

        toolbar.addWidget(QLabel(" Bits: "))
        self._bits_box = QSpinBox()
        self._bits_box.setRange(1, 32)
        self._bits_box.setValue(self.settings.detector_bits)
        self._bits_box.valueChanged.connect(self._on_detector_bits_changed)
        toolbar.addWidget(self._bits_box)

        toolbar.addWidget(QLabel(" Type: "))
        self._type_filter = QComboBox()
        self._type_filter.addItem("All frames")
        self._type_filter.currentTextChanged.connect(self._on_type_filter_changed)
        toolbar.addWidget(self._type_filter)

        toolbar.addWidget(QLabel(" Frame: "))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        toolbar.addWidget(self._frame_spin)

        sum_action = QAction("Sum all", self)
        sum_action.triggered.connect(lambda: self.sum_frames())
        toolbar.addAction(sum_action)

    def _prompt_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UIMF file", self.settings.last_directory, "UIMF files (*.uimf)"
        )
        if path:
            self.settings.last_directory = os.path.dirname(path)
            self.open_file(path)

    def open_file(self, path: str) -> None:
        """Open a UIMF file and show its first frame.

        Decoding happens on the load worker's thread, so the window and this call both
        return immediately; `_on_opened` and `_on_frame_loaded` do the rest once the
        signals arrive.
        """
        self._open_started = time.perf_counter()
        self.statusBar().showMessage(f"Opening {os.path.basename(path)}...")
        self._busy.show()
        self._opening = True
        self._worker.open(path)

    @property
    def last_render(self) -> "object | None":
        """The newest `RenderResult` drawn: the image, both profiles and what they cost.

        The window's answer to "what is on screen right now", for the self-check and
        for a test that wants the profiles the side plots were given rather than the
        pixels they became.
        """
        return self._last_render

    def show_frame(self, frame: int) -> None:
        """Switch to a frame, keeping the current view.

        Unlike opening a file, this never resets the range -- keep-ranges is about what
        happens across a *file* open, and paging through one file's frames must not cost
        the zoom the user just set regardless of that setting. `_on_frame_loaded` is
        what tells the two cases apart, by `_opening`.
        """
        if self._global is None:
            return
        self._worker.request_frame(int(frame))

    def sum_frames(self, frames: "list[int] | None" = None) -> None:
        """Sum several frames into one image, with a progress dialog and cancel.

        Defaults to the frame-type filter's active set rather than literally every
        frame: "sum all" should respect whatever the filter has already narrowed the
        frame spinner to, not silently pull in a frame type the user just excluded.
        """
        numbers = list(frames) if frames is not None else list(self._active_frame_numbers)
        if not numbers or self._global is None:
            return
        self._sum_count = len(numbers)
        dialog = QProgressDialog(
            f"Summing {len(numbers)} frames...", "Cancel", 0, len(numbers), self
        )
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self._worker.cancel_sum)
        dialog.canceled.connect(dialog.close)
        self._sum_dialog = dialog
        self._worker.sum_all(numbers)

    # --- worker callbacks -------------------------------------------------------------

    def _on_opened(
        self, global_params: GlobalParams, frame_numbers: "list[int]", frame_types: "dict[int, int]"
    ) -> None:
        self._global = global_params
        self._frame_numbers = list(frame_numbers)
        self._frame_types = dict(frame_types)
        self._populate_type_filter()
        self._active_frame_numbers = list(self._frame_numbers)
        self._frame_spin.blockSignals(True)
        if self._frame_numbers:
            self._frame_spin.setRange(min(self._frame_numbers), max(self._frame_numbers))
            self._frame_spin.setValue(self._frame_numbers[0])
        else:
            self._frame_spin.setRange(0, 0)
        self._frame_spin.blockSignals(False)
        if not frame_numbers:
            self._busy.hide()
            self._opening = False
            self.statusBar().showMessage("This file has no frames")
            return
        self._worker.request_frame(frame_numbers[0])

    def _populate_type_filter(self) -> None:
        names = sorted({_frame_type_name(t) for t in self._frame_types.values()})
        self._type_filter.blockSignals(True)
        self._type_filter.clear()
        self._type_filter.addItem("All frames")
        self._type_filter.addItems(names)
        self._type_filter.blockSignals(False)

    def _on_frame_loaded(
        self, frame_number: int, sparse_frame: SparseFrame, frame_params: FrameParams
    ) -> None:
        # `_opening` is true only for the very first frame after `open_file` -- every
        # later arrival of this same signal is a frame-navigation or filter-driven
        # request, which must keep the view regardless of keep-ranges.
        reset = (not self.settings.keep_ranges) if self._opening else False
        self._show_frame(frame_number, sparse_frame, frame_params, reset=reset)

    def _show_frame(
        self,
        frame_number: int,
        sparse_frame: SparseFrame,
        frame_params: FrameParams,
        reset: bool,
        message: "str | None" = None,
    ) -> None:
        assert self._global is not None  # a frame cannot load before opened() fires
        calibration = frame_params.calibration(self._global.bin_width_ns)
        axes = DisplayAxes.build(
            sparse_frame, calibration, frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
        )
        self._current_frame = sparse_frame
        self._current_axes = axes
        self._current_frame_number = frame_number
        self._frame_params = frame_params
        self._serial += 1
        self.info_panel.set_file(self._global, frame_params)
        self._frame_message = message or f"Frame {frame_number}: {len(sparse_frame)} points"
        self.statusBar().showMessage(self._frame_message)
        if frame_number in self._frame_numbers:
            self._frame_spin.blockSignals(True)
            self._frame_spin.setValue(frame_number)
            self._frame_spin.blockSignals(False)
        # Sets the reset target and the gesture limits, then asks for the first render;
        # every later render comes from a gesture through the same signal.
        self.heatmap.set_frame_extent(axes, reset=reset)

    def _on_failed(self, message: str) -> None:
        self._busy.hide()
        self._opening = False
        if self._sum_dialog is not None:
            self._sum_dialog.close()
            self._sum_dialog = None
        self.statusBar().showMessage(f"Error: {message}")

    def _on_view_resized(self, width: int, height: int) -> None:
        self._request_render(*self.heatmap.view_range(), width, height)

    def _on_view_changed(self, x_range: "tuple[float, float]",
                         y_range: "tuple[float, float]") -> None:
        self._request_render(x_range, y_range)

    def _request_render(
        self,
        x_range: "tuple[float, float]",
        y_range: "tuple[float, float]",
        width: "int | None" = None,
        height: "int | None" = None,
    ) -> None:
        if self._current_frame is None or self._current_axes is None:
            return
        if width is None or height is None:
            width, height = self.heatmap.pixel_size()
        self._mailbox.put(RenderRequest(
            frame=self._current_frame,
            axes=self._current_axes,
            x_range=x_range,
            y_range=y_range,
            width=width,
            height=height,
            aggregate=self.settings.aggregate,
            serial=self._serial,
        ))

    def _on_rendered(self, render: object) -> None:
        if render.serial != self._serial:
            return  # a frame or a file has moved on since this was asked for
        result = render.result
        self._last_render = render
        self.heatmap.set_image(result, colour_scale=self.settings.colour_scale)
        self.side_plots.set_profiles(render.x_profile, render.y_profile)
        self.heatmap.set_debug_text(
            f"{render.elapsed_ms:.1f} ms  {result.image.shape[1]}x{result.image.shape[0]}"
            f"  {result.points_in_view} pts"
        )
        if self._frame_params is not None:
            self.info_panel.set_view(
                result, self._frame_params.accumulations, self.settings.detector_bits
            )
        if self._opening:
            # An explicit flag and not the progress bar's visibility: a window that has
            # not been shown yet -- every pytest-qt test that does not call `show()` --
            # has no visible children to ask.
            self._opening = False
            self._busy.hide()
            elapsed_ms = (time.perf_counter() - self._open_started) * 1000.0
            self.statusBar().showMessage(
                f"{self._frame_message} -- opened in {elapsed_ms:.0f} ms"
            )
        self.frame_shown.emit(result)

    def _on_summing_progress(self, done: int, total: int) -> None:
        if self._sum_dialog is not None:
            self._sum_dialog.setValue(done)

    def _on_summed(self, sparse_frame: "SparseFrame | None", frame_params: "FrameParams | None") -> None:
        if self._sum_dialog is not None:
            self._sum_dialog.close()
            self._sum_dialog = None
        if frame_params is None:
            return  # cancelled, or an empty frame list -- nothing to show
        self._show_frame(
            0, sparse_frame, frame_params, reset=False,
            message=f"Sum of {self._sum_count} frames: {len(sparse_frame)} points",
        )

    # --- toolbar callbacks --------------------------------------------------------------

    def _on_aggregate_changed(self, text: str) -> None:
        self.settings.aggregate = text.lower()
        self._request_render(*self.heatmap.view_range())

    def _on_colour_scale_changed(self, text: str) -> None:
        self.settings.colour_scale = text.lower()
        if self._last_render is not None:
            self.heatmap.set_image(self._last_render.result, colour_scale=self.settings.colour_scale)

    def _on_swap_toggled(self, checked: bool) -> None:
        self.settings.swap_axes = checked
        self._rebuild_axes()

    def _on_raw_units_toggled(self, checked: bool) -> None:
        self.settings.raw_units = checked
        self._rebuild_axes()

    def _on_keep_ranges_toggled(self, checked: bool) -> None:
        self.settings.keep_ranges = checked

    def _on_keep_levels_toggled(self, checked: bool) -> None:
        self.settings.keep_levels = checked
        if checked:
            self.heatmap.set_levels(*self.heatmap.levels())
        else:
            self.heatmap.release_levels()

    def _on_detector_bits_changed(self, value: int) -> None:
        self.settings.detector_bits = int(value)
        if self._last_render is not None and self._frame_params is not None:
            self.info_panel.set_view(
                self._last_render.result, self._frame_params.accumulations, self.settings.detector_bits
            )

    def _on_type_filter_changed(self, text: str) -> None:
        if not text or text == "All frames":
            self._active_frame_numbers = list(self._frame_numbers)
        else:
            self._active_frame_numbers = [
                n for n in self._frame_numbers if _frame_type_name(self._frame_types.get(n, 0)) == text
            ]
        if not self._active_frame_numbers:
            return
        self._frame_spin.blockSignals(True)
        self._frame_spin.setRange(min(self._active_frame_numbers), max(self._active_frame_numbers))
        self._frame_spin.blockSignals(False)
        if self._current_frame_number not in self._active_frame_numbers:
            nearest = min(
                self._active_frame_numbers,
                key=lambda n: abs(n - (self._current_frame_number or 0)),
            )
            self.show_frame(nearest)

    def _on_frame_spin_changed(self, value: int) -> None:
        if not self._active_frame_numbers:
            return
        nearest = min(self._active_frame_numbers, key=lambda n: abs(n - value))
        if nearest != value:
            self._frame_spin.blockSignals(True)
            self._frame_spin.setValue(nearest)
            self._frame_spin.blockSignals(False)
        if nearest != self._current_frame_number:
            self.show_frame(nearest)

    def _rebuild_axes(self) -> None:
        """Rebuild `DisplayAxes` for the swap or raw-units toggle, translating the
        visible region through the old and new axis tables rather than resetting it.

        A display coordinate's fractional source-element index is recovered by
        interpolating into the *old* edge table and re-expressed through the *new*
        one -- both are monotonic by construction, the calibration clamped so that it
        never decreases and a raw index table trivially increasing -- so the same bins
        and scans stay in view across the toggle, only relabelled or swapped.
        """
        if self._current_frame is None or self._current_axes is None or self._frame_params is None:
            return
        assert self._global is not None
        old_axes = self._current_axes
        (x0, x1), (y0, y1) = self.heatmap.view_range()
        bin_lo, bin_hi = (y0, y1) if old_axes.swapped else (x0, x1)
        scan_lo, scan_hi = (x0, x1) if old_axes.swapped else (y0, y1)
        bin_index_lo = _index_of(old_axes.bin_edges, bin_lo)
        bin_index_hi = _index_of(old_axes.bin_edges, bin_hi)
        scan_index_lo = _index_of(old_axes.scan_edges, scan_lo)
        scan_index_hi = _index_of(old_axes.scan_edges, scan_hi)

        calibration = self._frame_params.calibration(self._global.bin_width_ns)
        new_axes = DisplayAxes.build(
            self._current_frame, calibration, self._frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
        )
        new_bin_lo = _value_at(new_axes.bin_edges, bin_index_lo)
        new_bin_hi = _value_at(new_axes.bin_edges, bin_index_hi)
        new_scan_lo = _value_at(new_axes.scan_edges, scan_index_lo)
        new_scan_hi = _value_at(new_axes.scan_edges, scan_index_hi)
        if new_axes.swapped:
            new_x_range = tuple(sorted((new_scan_lo, new_scan_hi)))
            new_y_range = tuple(sorted((new_bin_lo, new_bin_hi)))
        else:
            new_x_range = tuple(sorted((new_bin_lo, new_bin_hi)))
            new_y_range = tuple(sorted((new_scan_lo, new_scan_hi)))

        self._current_axes = new_axes
        self._serial += 1
        # The box's own `set_extent`, not `HeatmapView.set_frame_extent` -- that also
        # asks for an immediate render at whatever range is still on screen, in the old
        # axes' numbers, which would be a wasted request a moment before `setRange`
        # below asks for the translated one.
        self.heatmap.view_box.set_extent(new_axes, reset=False)
        self.heatmap.view_box.setRange(xRange=new_x_range, yRange=new_y_range, padding=0.0)

    # --- cursor readout ---------------------------------------------------------------

    def _on_cursor_moved(self, x: float, y: float) -> None:
        """The status-bar and info-panel readout: where the pointer is, in every unit
        the frame has.

        Both the display units and the raw bin and scan are shown, because they answer
        different questions -- an m/z identifies a species, a bin identifies the sample
        of the digitizer trace it came from -- and neither can be recovered from the
        other by eye. The intensity is the one **drawn at that pixel**, read back out of
        the image on screen rather than recomputed, and it is labelled with the
        aggregation that produced it: a `sum` pixel is a total over however many bins
        and scans that pixel covers, and quoting it as if it were a stored intensity is
        the mistake this label exists to prevent.
        """
        axes = self._current_axes
        if axes is None or self._last_render is None:
            return
        result = self._last_render.result
        parts = [f"{axes.x_label} {x:,.4g}", f"{axes.y_label} {y:,.4g}"]
        bin_value, scan_value = (y, x) if axes.swapped else (x, y)
        bin_index = _element_of(axes.bin_edges, bin_value)
        scan_index = _element_of(axes.scan_edges, scan_value)
        if bin_index is not None and scan_index is not None:
            parts.append(f"bin {bin_index}  scan {scan_index}")
        cell = pixel_of(result, x, y)
        if cell is not None:
            row, column = cell
            parts.append(f"{result.aggregate} {float(result.image[row, column]):,.0f}")
        text = "   |   ".join(parts)
        self._readout.setText(text)
        self.info_panel.set_cursor(text)

    def _clear_readout(self) -> None:
        self._readout.setText("")
        self.info_panel.set_cursor("")

    def closeEvent(self, event) -> None:
        self.settings.window_geometry = bytes(self.saveGeometry())
        save_settings(self.settings)
        self._mailbox.close()
        self._render_worker.wait(2000)
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(event)


def _element_of(edges: "np.ndarray", value: float) -> "int | None":
    """The index of the source element containing `value`, or None if it is outside.

    `searchsorted` and not a division: the edge tables are monotonic but not uniform --
    m/z is quadratic in bin -- so this is the one place in the viewer where the axis
    really does have to be searched rather than divided through (`uimf/raster.py` says
    why the rasteriser gets to divide).
    """
    index = int(np.searchsorted(edges, value, side="right")) - 1
    if 0 <= index < len(edges) - 1:
        return index
    return None


def _index_of(edges: "np.ndarray", value: float) -> float:
    """The fractional source-element index a display coordinate names, by interpolating
    into a monotonic edge table -- the inverse of building that table in the first
    place, and how `_rebuild_axes` recovers "which bins and scans" from "which pixels"."""
    return float(np.interp(value, edges, np.arange(len(edges), dtype=np.float64)))


def _value_at(edges: "np.ndarray", index: float) -> float:
    """The inverse of `_index_of`: the display coordinate a fractional index maps to."""
    return float(np.interp(index, np.arange(len(edges), dtype=np.float64), edges))
