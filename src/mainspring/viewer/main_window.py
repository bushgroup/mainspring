"""`MainWindow`: the one object that knows about all the others.

It owns the file, the settings, the two workers, and the four widgets, and it is the
only place where they meet -- the heatmap does not know there is an info panel, the
side plots do not know there is a file. That is what keeps each of the later tasks to
one or two modules.

The flow it coordinates, once around:

    open  ->  LoadWorker decodes a frame  ->  build DisplayAxes  ->  choose the view
          ->  RenderMailbox  ->  RenderWorker  ->  image + profiles + readouts

Every arrow after the third is now travelled by every gesture as well as by every open,
which is the whole of task 05: a wheel tick and a newly opened file reach the render
worker by the same path, so there is one place where "what is on screen" is decided and
no second, direct-to-the-rasteriser route that could disagree with it.

**Results are matched to a serial, not trusted in arrival order.** The mailbox drops
superseded *requests*, but a result already in flight when the frame changes would
otherwise repaint the new frame's window with the old frame's data. Every request
carries the serial of the state that produced it, and a result whose serial is not the
current one is dropped -- which is also what makes frame navigation and the axis
toggles (task 06) safe to add without touching the render path.

**Choosing the view is where keep-ranges lives.** On opening a file the window either
takes the frame's full range or keeps the ranges already on screen, depending on the
setting, and that decision belongs here rather than in the heatmap because it is a
property of the session and not of the widget. Task 05 always takes the full range;
keep-ranges arrives with the settings it depends on (task 06), and needs only the
`reset=False` that `set_frame_extent` already takes.

Frame navigation, the toolbar toggles, and "sum all frames" with its progress and
cancel are also here, since each of them changes what is asked of the workers rather
than how a widget draws.
"""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QProgressBar

from ..uimf import DisplayAxes, FrameParams, GlobalParams, SparseFrame
from .heatmap import HeatmapView, pixel_of
from .settings import ViewerSettings
from .side_plots import SidePlots
from .workers import LoadWorker, RenderMailbox, RenderRequest, RenderWorker

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """The application window, holding the heatmap, the workers, and the settings."""

    frame_shown = Signal(object)
    """Emitted with the `RasterResult` after each (re)paint -- what a test waits on, and
    what the info panel will read from once it exists (task 06)."""

    def __init__(self, settings: "ViewerSettings | None" = None) -> None:
        super().__init__()
        self.settings = settings or ViewerSettings()
        self.setWindowTitle("mainspring")
        self.resize(1000, 700)

        self.heatmap = HeatmapView(colour_map=self.settings.colour_map)
        self.setCentralWidget(self.heatmap)
        self.side_plots = SidePlots(self.heatmap)
        self.heatmap.view_resized.connect(self._on_view_resized)
        self.heatmap.view_changed.connect(self._on_view_changed)
        self.heatmap.cursor_moved.connect(self._on_cursor_moved)
        self.heatmap.cursor_left.connect(self._clear_readout)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)  # indeterminate: a decode's length is not known upfront
        self._busy.setMaximumWidth(120)
        self._busy.hide()
        self._readout = QLabel("")
        self.statusBar().addPermanentWidget(self._readout)
        self.statusBar().addPermanentWidget(self._busy)
        self._build_menu()

        self._worker = LoadWorker(self.settings.cache_budget_mb * 1024 * 1024)
        self._worker.opened.connect(self._on_opened)
        self._worker.frame_loaded.connect(self._on_frame_loaded)
        self._worker.failed.connect(self._on_failed)

        self._mailbox = RenderMailbox()
        self._render_worker = RenderWorker(self._mailbox)
        self._render_worker.rendered.connect(self._on_rendered)
        self._render_worker.failed.connect(self._on_failed)
        self._render_worker.start()

        self._global: GlobalParams | None = None
        self._current_frame: SparseFrame | None = None
        self._current_axes: DisplayAxes | None = None
        self._last_render: object | None = None
        self._serial = 0
        self._open_started = 0.0
        self._opening = False
        self._frame_message = ""

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
        return immediately; `_on_frame_loaded` does the rest once the signal arrives.
        """
        self._open_started = time.perf_counter()
        self.statusBar().showMessage(f"Opening {os.path.basename(path)}...")
        self._busy.show()
        self._opening = True
        self._worker.open(path)

    @property
    def last_render(self) -> "object | None":
        """The newest `RenderResult` drawn: the image, both profiles and what they cost.

        The window's answer to "what is on screen right now", for the info panel (task
        06), for the self-check, and for a test that wants the profiles the side plots
        were given rather than the pixels they became.
        """
        return self._last_render

    def show_frame(self, frame: int) -> None:
        """Switch to a frame, keeping the current view. Arrives with the lab record's task 06."""
        raise NotImplementedError("frame navigation arrives with the lab record's task 06")

    def sum_frames(self, frames: "list[int] | None" = None) -> None:
        """Sum several frames into one image, with progress and cancel. Arrives with task 06."""
        raise NotImplementedError("summing frames arrives with the lab record's task 06")

    # --- worker callbacks -------------------------------------------------------------

    def _on_opened(self, global_params: GlobalParams, frame_numbers: "list[int]") -> None:
        self._global = global_params
        if not frame_numbers:
            self._busy.hide()
            self._opening = False
            self.statusBar().showMessage("This file has no frames")
            return
        self._worker.request_frame(frame_numbers[0])

    def _on_frame_loaded(
        self, frame_number: int, sparse_frame: SparseFrame, frame_params: FrameParams
    ) -> None:
        assert self._global is not None  # a frame cannot load before opened() fires
        calibration = frame_params.calibration(self._global.bin_width_ns)
        axes = DisplayAxes.build(
            sparse_frame, calibration, frame_params.average_tof_length_ns,
            raw_units=self.settings.raw_units, swapped=self.settings.swap_axes,
        )
        self._current_frame = sparse_frame
        self._current_axes = axes
        self._serial += 1
        self.side_plots.set_axes(axes)
        self._frame_message = f"Frame {frame_number}: {len(sparse_frame)} points"
        self.statusBar().showMessage(self._frame_message)
        # Sets the reset target and the gesture limits, then asks for the first render;
        # every later render comes from a gesture through the same signal.
        self.heatmap.set_frame_extent(axes)

    def _on_failed(self, message: str) -> None:
        self._busy.hide()
        self._opening = False
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
        self.heatmap.set_image(result)
        self.side_plots.set_profiles(render.x_profile, render.y_profile)
        self.heatmap.set_debug_text(
            f"{render.elapsed_ms:.1f} ms  {result.image.shape[1]}x{result.image.shape[0]}"
            f"  {result.points_in_view} pts"
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
