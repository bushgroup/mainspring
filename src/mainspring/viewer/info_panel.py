"""The info panel: the file's parameters, and the readout the instrument is judged by.

Two halves. The upper one lists the global and frame parameters a file happens to carry
-- whatever keys are in the tables, not a fixed list, since the 2016 files carry fifteen
optics voltages the sample does not and a future writer will carry something else again.
It reads `GlobalParams.extra`/`FrameParams.extra` for exactly that reason: those are the
raw `ParamName -> ParamValue` pairs the reader kept alongside the handful of fields it
parses, so a key nobody anticipated is still shown rather than dropped.

The lower half is the number PNNL's viewer never showed and the reason a scientist
opens the viewer during an acquisition: **intensity per TOF push**. The stored value is
a sum over the frame's `Accumulations` pulses, so the largest single-push value in view
is `max(intensity) / Accumulations` in ADC counts, and the panel reports it as a
percentage of full scale as well, from the detector-bits setting.

**Bit depth is not in the file.** It is 8 today on SLIMPHONY and will be 14 under
clockwork; UIMF-Library's own saturation helpers guess it from the acquisition date,
which we do not, because a setting the user can see beats a rule they cannot. So this
panel never shows a per-push number without also showing the `Accumulations` and the
assumed bit depth that produced it (lab record, task 01) -- a count quoted without both
is not reproducible, and someone will quote it.

Also here: max intensity in view, total ion current in view and the points in view, all
taken from the `RasterResult` so that they describe the image on screen rather than a
newer view the render has not caught up with. The cursor readout is *not* here -- it is
the status bar's -- because a label whose text changes on every mouse move was what made
the dock, and the heatmap beside it, change width under the pointer (lab record, task
11).

**The panel has a fixed width.** A dock's width follows its content's minimum size hint,
and a `QLabel`'s follows its text, so any live readout could otherwise resize the whole
window. The one long readout, per push, wraps inside that width instead.
"""

from __future__ import annotations

from typing import Mapping

from PySide6.QtWidgets import QDockWidget, QFormLayout, QLabel, QTreeWidget, QTreeWidgetItem, QWidget

__all__ = ["INFO_PANEL_WIDTH", "InfoPanel", "per_push"]

INFO_PANEL_WIDTH = 320
"""Pixels across the dock's content, docked or floating. Wide enough for the parameter
tree's two columns and the per-push line on two rows."""


def per_push(max_intensity: float, accumulations: int, detector_bits: int) -> tuple[float, float]:
    """`(counts_per_push, percent_of_full_scale)` from an in-view maximum.

    `accumulations` is clamped to at least 1 -- `FrameParams.accumulations` already
    guarantees that from a real file, but a caller handing this function a bare number
    should not be able to divide by zero over a value it typed by hand.
    """
    accumulations = max(1, int(accumulations))
    counts_per_push = float(max_intensity) / accumulations
    full_scale = float(2 ** max(1, int(detector_bits)) - 1)
    percent = 100.0 * counts_per_push / full_scale
    return counts_per_push, percent


def _rows(params: object) -> list[tuple[str, str]]:
    """A parameter object's raw `(name, value)` pairs, sorted for a stable display order."""
    extra: Mapping[str, str] = getattr(params, "extra", None) or {}
    return sorted(extra.items())


class InfoPanel(QDockWidget):
    """The parameter tree and the per-push readout, docked beside the heatmap.

    A dock and not a fixed side panel, so a researcher who wants the whole screen for
    the heatmap can float or close it. Whether it is shown is
    `ViewerSettings.show_info_panel`, toggled from the window's toolbar and View menu
    through the dock's own `toggleViewAction()`, so closing it with its X button and
    unticking the action are the same thing.
    """

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__("Info", parent)
        self.setObjectName("info_panel")

        container = QWidget(self)
        container.setFixedWidth(INFO_PANEL_WIDTH)
        self._tree = QTreeWidget(container)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Parameter", "Value"])
        self._global_root = QTreeWidgetItem(self._tree, ["Global", ""])
        self._frame_root = QTreeWidgetItem(self._tree, ["Frame", ""])

        self._max_label = QLabel("-", container)
        self._per_push_label = QLabel("-", container)
        self._per_push_label.setWordWrap(True)  # the one readout longer than the panel
        self._tic_label = QLabel("-", container)
        self._points_label = QLabel("-", container)
        live = QFormLayout()
        # A field too wide for the space beside its label drops to the next line rather
        # than squeezing into a sliver -- the per-push line, at the fixed panel width.
        live.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        live.addRow("Max intensity in view:", self._max_label)
        live.addRow("Per push:", self._per_push_label)
        live.addRow("TIC in view:", self._tic_label)
        live.addRow("Points in view:", self._points_label)

        layout = QFormLayout(container)
        layout.addRow(self._tree)
        layout.addRow(live)
        self.setWidget(container)

    def set_file(self, global_params: object, frame_params: object) -> None:
        """Replace the parameter tree with one file's global and frame parameters."""
        self._global_root.takeChildren()
        for name, value in _rows(global_params):
            QTreeWidgetItem(self._global_root, [name, value])
        self._frame_root.takeChildren()
        for name, value in _rows(frame_params):
            QTreeWidgetItem(self._frame_root, [name, value])
        self._tree.expandAll()
        for column in range(2):
            self._tree.resizeColumnToContents(column)

    def set_view(self, result: object, accumulations: int, detector_bits: int) -> None:
        """Update the in-view readouts from a rendered `RasterResult`.

        `accumulations` and `detector_bits` travel with every call rather than being
        remembered from `set_file`, because the detector-bits setting can change without
        a new frame arriving and the readout must move with it immediately.
        """
        self._max_label.setText(f"{result.max_intensity:,.0f}")
        counts, percent = per_push(result.max_intensity, accumulations, detector_bits)
        self._per_push_label.setText(
            f"{counts:,.1f} ADC/push  ({accumulations} accum., {detector_bits}-bit"
            f" -- {percent:.2f}% full scale)"
        )
        self._tic_label.setText(f"{result.tic_in_view:,.0f}")
        self._points_label.setText(f"{result.points_in_view:,}")
