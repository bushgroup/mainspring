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
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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
from .controls import add_labelled, describe, make_action
from .export import ExportDialog, content_rect, export_display
from .heatmap import HeatmapView, pixel_of
from .info_panel import InfoPanel
from .settings import COLOUR_MAPS, COLOUR_SCALES, ViewerSettings, load_settings, save_settings
from .side_plots import SidePlots
from .workers import LoadWorker, RenderMailbox, RenderRequest, RenderWorker

__all__ = ["APP_TITLE", "MainWindow"]

APP_TITLE = "mainspring"
"""The window title with no file open. With one open it is `"<file name> -- mainspring"`,
file first, the way Windows names a document window, so that two viewers on the taskbar
can be told apart by what they show rather than by what they are."""

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
        self.setWindowTitle(APP_TITLE)
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
        self._path: str | None = None
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
        self.info_panel.setVisible(self.settings.show_info_panel)
        # The dock's own action rather than a hand-rolled one: Qt keeps it in step with
        # the dock's visibility in both directions, so the X button on the dock, the
        # toolbar button and the menu entry are one switch with three handles.
        self._info_action = self.info_panel.toggleViewAction()
        self._info_action.setText("Info")
        self._info_action.setShortcut("Ctrl+I")
        describe(
            self._info_action,
            "Show the panel of frame parameters and in-view totals.",
            shortcut="Ctrl+I",
        )
        self._info_action.toggled.connect(self._on_info_toggled)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate: a decode's length is not known upfront
        self._busy.setMaximumWidth(120)
        self._busy.hide()
        describe(self._busy, "A frame is being decoded.")
        self._readout = QLabel("")
        describe(
            self._readout,
            "The axis values under the pointer, and the intensity of the pixel it is over.",
        )
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
        self.open_action = make_action(
            self,
            "&Open...",
            tip="Open a UIMF file.",
            shortcut="Ctrl+O",
            triggered=self._prompt_open,
        )
        # One entry per format rather than one "Export..." with a format chooser: the
        # File menu is where a user goes looking for the words PNG and PDF, and the
        # dialog behind both then has one thing in it (`export.ExportDialog`).
        self.export_png_action = make_action(
            self,
            "Export &PNG...",
            tip="Save the heatmap and its projections as a PNG image, without the colour bar.",
            triggered=lambda: self._prompt_export("png"),
        )
        self.export_pdf_action = make_action(
            self,
            "Export P&DF...",
            tip="Save the heatmap and its projections as a PDF, without the colour bar.",
            triggered=lambda: self._prompt_export("pdf"),
        )
        # Nothing to export until something has been rendered, and an entry that opens a
        # file dialog and then reports that there is no image is worse than a grey one.
        for action in (self.export_png_action, self.export_pdf_action):
            action.setEnabled(False)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_png_action)
        file_menu.addAction(self.export_pdf_action)

        # A window-level action rather than a key handler on the view box: the reset must
        # work wherever the focus happens to be, which is the complaint about having to
        # find it in a context menu (lab record, task 01).
        self.reset_action = make_action(
            self,
            "&Reset view",
            tip="Return the heatmap to the frame's full range.",
            shortcut="Home",
            triggered=self.heatmap.reset_range,
        )
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.reset_action)
        view_menu.addAction(self._info_action)
        self._build_colour_map_menu(view_menu)

    def _build_colour_map_menu(self, view_menu: "object") -> None:
        """A curated four-map choice, radio-style, in its own `View` submenu.

        pyqtgraph's `ColorBarItem` ships a right-click menu of its own listing every
        registered colormap; it is turned off at construction (`heatmap.py`) because a
        choice made through it would neither tick an entry here nor reach
        `ViewerSettings` -- this menu is the one place the colour map lives.
        """
        colour_map_menu = view_menu.addMenu("Colour map")
        group = QActionGroup(self)
        group.setExclusive(True)
        for name in COLOUR_MAPS:
            action = make_action(
                self,
                name.capitalize(),
                tip=f"Colour the heatmap with the {name.capitalize()} colormap.",
                checkable=True,
                checked=(name == self.settings.colour_map),
                toggled=lambda checked, n=name: self._on_colour_map_changed(n, checked),
            )
            group.addAction(action)
            colour_map_menu.addAction(action)

    def _build_toolbar(self) -> None:
        """Every toggle `ViewerSettings` carries, plus frame navigation and sum-all.

        Each control is fully configured -- range, items, initial value from the
        restored settings -- **before** its signal is connected, so that restoring a
        non-default setting cannot fire a handler while the rest of the window is still
        being built. `make_action` holds that order for the actions; the widgets below
        keep it by hand, because their value has to be set before `add_labelled` can
        hand them over.
        """
        toolbar = QToolBar("View", self)
        toolbar.setObjectName("view_toolbar")  # QMainWindow.saveState() keys it by this
        self.addToolBar(toolbar)
        # Qt makes this one itself and offers it in the window's right-click menu, which
        # is the only place it appears -- so it is the control most easily left mute.
        describe(toolbar.toggleViewAction(), "Show the toolbar.")

        self._aggregate_box = QComboBox()
        self._aggregate_box.addItems([a.capitalize() for a in AGGREGATES])
        self._aggregate_box.setCurrentText(self.settings.aggregate.capitalize())
        self._aggregate_box.currentTextChanged.connect(self._on_aggregate_changed)
        self._aggregate_label = add_labelled(
            toolbar,
            " Aggregate: ",
            self._aggregate_box,
            tip="Choose how the intensities inside one screen pixel are combined.",
        )

        self._colour_box = QComboBox()
        self._colour_box.addItems([c.capitalize() for c in COLOUR_SCALES])
        self._colour_box.setCurrentText(self.settings.colour_scale.capitalize())
        self._colour_box.currentTextChanged.connect(self._on_colour_scale_changed)
        self._colour_label = add_labelled(
            toolbar,
            " Colour: ",
            self._colour_box,
            tip="Choose how intensity maps onto colour: linear, log or square root.",
        )

        self._swap_action = make_action(
            self,
            "Swap X/Y",
            tip="Put arrival time on the horizontal axis and m/z on the vertical.",
            checkable=True,
            checked=self.settings.swap_axes,
            toggled=self._on_swap_toggled,
        )
        toolbar.addAction(self._swap_action)

        self._raw_action = make_action(
            self,
            "Raw units",
            tip="Show TOF bin and scan number instead of calibrated m/z and arrival time.",
            checkable=True,
            checked=self.settings.raw_units,
            toggled=self._on_raw_units_toggled,
        )
        toolbar.addAction(self._raw_action)

        self._arrival_offset_box = QDoubleSpinBox()
        self._arrival_offset_box.setRange(-100_000.0, 100_000.0)
        self._arrival_offset_box.setDecimals(3)
        self._arrival_offset_box.setSingleStep(1.0)
        self._arrival_offset_box.setValue(self.settings.arrival_offset_ms)
        self._arrival_offset_box.valueChanged.connect(self._on_arrival_offset_changed)
        self._arrival_offset_label = add_labelled(
            toolbar,
            " Arrival offset (ms): ",
            self._arrival_offset_box,
            tip="Shift the displayed arrival-time axis by this many milliseconds.",
        )

        self._keep_ranges_action = make_action(
            self,
            "Keep ranges",
            tip="Keep the current zoom when the next file is opened.",
            checkable=True,
            checked=self.settings.keep_ranges,
            toggled=self._on_keep_ranges_toggled,
        )
        toolbar.addAction(self._keep_ranges_action)

        self._keep_levels_action = make_action(
            self,
            "Keep levels",
            tip="Keep the current colour limits instead of rescaling to each frame.",
            checkable=True,
            checked=self.settings.keep_levels,
            toggled=self._on_keep_levels_toggled,
        )
        toolbar.addAction(self._keep_levels_action)

        toolbar.addAction(self._info_action)  # built with the dock, above

        self._bits_box = QSpinBox()
        self._bits_box.setRange(1, 32)
        self._bits_box.setValue(self.settings.detector_bits)
        self._bits_box.valueChanged.connect(self._on_detector_bits_changed)
        self._bits_label = add_labelled(
            toolbar,
            " Bits: ",
            self._bits_box,
            tip="Detector bit depth, 1 to 32, that the per-push readout assumes.",
        )

        self._type_filter = QComboBox()
        self._type_filter.addItem("All frames")
        self._type_filter.currentTextChanged.connect(self._on_type_filter_changed)
        self._type_label = add_labelled(
            toolbar,
            " Type: ",
            self._type_filter,
            tip="Show only frames of one type, or all of them.",
        )

        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        self._frame_label = add_labelled(
            toolbar,
            " Frame: ",
            self._frame_spin,
            tip="Go to a frame by number, within the frames the type filter allows.",
        )

        self.sum_action = make_action(
            self,
            "Sum all",
            tip="Add every frame passing the type filter into one heatmap.",
            triggered=lambda: self.sum_frames(),
        )
        toolbar.addAction(self.sum_action)

    def _prompt_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UIMF file", self.settings.last_directory, "UIMF files (*.uimf)"
        )
        if path:
            self.settings.last_directory = os.path.dirname(path)
            self.open_file(path)

    def _prompt_export(self, fmt: str) -> None:
        """Ask where, then at what resolution, then write it.

        Path first and options second, the order Windows puts them in: a cancelled file
        dialog costs nothing, and the resolution dialog can then name the size the file
        it is about to write will be.
        """
        if self._current_frame is None or self._last_render is None:
            return
        filters = {"png": "PNG image (*.png)", "pdf": "PDF document (*.pdf)"}
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", self._export_default_path(fmt), filters[fmt]
        )
        if not path:
            return
        dialog = ExportDialog(
            self, fmt=fmt, dpi=self.settings.export_dpi,
            rect=content_rect(self.heatmap, self.side_plots),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.export_dpi = dialog.dpi()
        self.statusBar().showMessage(f"Exporting {os.path.basename(path)}...")
        # A re-rasterise at 600 dpi is seconds of a frozen window on a dense frame, and
        # it is not put on the render worker: that thread's mailbox drops whatever it is
        # holding when a newer request arrives, which is right for a gesture and would
        # silently lose an export. A wait cursor is the honest way to say so.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            width, height = export_display(
                self.heatmap, self.side_plots, self._current_frame,
                self._last_render.result, self.settings.colour_scale,
                path, fmt, dialog.dpi(),
            )
        except Exception as exc:  # noqa: BLE001 -- shown in the status bar, as a failed render is
            self._on_failed(f"could not export {os.path.basename(path)}: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Exported {os.path.basename(path)}, {width} x {height} pixels"
            f" at {dialog.dpi()} dpi"
        )

    def _export_default_path(self, suffix: str) -> str:
        """The file the save dialog opens on: the frame being looked at, beside the file
        it came from. A figure is nearly always named after both."""
        stem = os.path.splitext(os.path.basename(self._path))[0] if self._path else APP_TITLE
        frame = self._current_frame_number
        name = f"{stem}-frame{frame}.{suffix}" if frame else f"{stem}.{suffix}"
        return os.path.join(self.settings.last_directory, name)

    def open_file(self, path: str) -> None:
        """Open a UIMF file and show its first frame.

        Decoding happens on the load worker's thread, so the window and this call both
        return immediately; `_on_opened` and `_on_frame_loaded` do the rest once the
        signals arrive.
        """
        self._open_started = time.perf_counter()
        self._path = path
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
        describe(dialog, "Summing the frames the type filter allows. Cancel keeps the "
                 "frame already on screen.")
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
        # Named as soon as the file has opened, frames or no frames: an empty file is
        # still the file on screen.
        self.setWindowTitle(f"{os.path.basename(self._path or '')} — {APP_TITLE}")
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
            t0_offset_ms=self.settings.arrival_offset_ms,
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
        if self._opening:
            # The open itself failed: nothing is on screen for the title to name.
            self._path = None
            self.setWindowTitle(APP_TITLE)
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
        self.export_png_action.setEnabled(True)
        self.export_pdf_action.setEnabled(True)
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

    def _on_arrival_offset_changed(self, value: float) -> None:
        self.settings.arrival_offset_ms = float(value)
        self._rebuild_axes()

    def _on_keep_ranges_toggled(self, checked: bool) -> None:
        self.settings.keep_ranges = checked

    def _on_keep_levels_toggled(self, checked: bool) -> None:
        self.settings.keep_levels = checked
        if checked:
            self.heatmap.set_levels(*self.heatmap.levels())
        else:
            self.heatmap.release_levels()

    def _on_colour_map_changed(self, name: str, checked: bool) -> None:
        # The exclusive `QActionGroup` also toggles the previously-checked entry off,
        # which fires this same handler with `checked=False` -- only the newly-checked
        # one is the change to act on.
        if not checked:
            return
        self.settings.colour_map = name
        self.heatmap.set_colour_map(name)

    def _on_info_toggled(self, checked: bool) -> None:
        self.settings.show_info_panel = checked

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
            t0_offset_ms=self.settings.arrival_offset_ms,
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
        """The status-bar readout: where the pointer is, in every unit the frame has.

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
        self._readout.setText("   |   ".join(parts))

    def _clear_readout(self) -> None:
        self._readout.setText("")

    def closeEvent(self, event) -> None:
        self.settings.window_geometry = bytes(self.saveGeometry())
        # Read off the dock itself, in case a visibility change reached it by a route
        # the action's `toggled` did not report.
        self.settings.show_info_panel = not self.info_panel.isHidden()
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
