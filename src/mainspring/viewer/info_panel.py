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

**Bit depth is in the file only if mainspring wrote it.** PNNL's parameter set has no
name for it, so on their files it is 8 today on SLIMPHONY and a setting the user can
see; UIMF-Library's own saturation helpers guess it from the acquisition date, which we
do not, because a setting beats a rule the user cannot see. A clockwork acquisition
stores its digitizer's own depth (`MainspringDetectorBits`, lab record, task 16) and
the readout then uses that. Either way this panel never shows a per-push number without
also showing the `Accumulations`, the bit depth, **and which of the two places the bit
depth came from** (lab record, tasks 01 and 17) -- a count quoted without all three is
not reproducible, and someone will quote it.

**Clipping is the other half of that readout**, and the half that covers a whole file
rather than one view: the file's raw maximum, and how many points reached full scale in
the frame on screen and in the whole file. The full scale they are judged against is one
row of its own, said once, because it is the number all three per-push claims share and
the one that is wrong when they are -- on a clockwork file it is the stored 16-bit
ceiling, not the 14 bits the ADC resolves (`mainspring.uimf.full_scale`, lab record,
task 36).

Also here: max intensity in view, total ion current in view and the points in view, all
taken from the `RasterResult` so that they describe the image on screen rather than a
newer view the render has not caught up with. The cursor readout is *not* here -- it is
the status bar's -- because a label whose text changes on every mouse move was what made
the dock, and the heatmap beside it, change width under the pointer (lab record, task
11).

**The panel has a floor, not a fixed width.** A dock's width follows its content's
minimum size hint and a `QLabel`'s follows its text, so a live readout that grew by a
digit could otherwise resize the whole window -- which is the bug task 11 fixed by
fixing the width outright. What holds that invariant now is narrower and lets the panel
be dragged wider: every readout label is `QSizePolicy.Policy.Ignored` horizontally and
wraps, so its text has no say in how wide anything is, and `INFO_PANEL_WIDTH` is a
minimum rather than a fixed size. The width the user drags to is a persisted setting
(`ViewerSettings.info_panel_width`), because the file whose parameters do not fit in 320
pixels is every file, and re-widening the panel on every launch is exactly the kind of
thing this viewer exists not to make people do (lab record, task 24).
"""

from __future__ import annotations

from typing import Mapping

from PySide6.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from . import fonts
from .controls import describe
from .settings import INFO_PANEL_WIDTH

__all__ = ["InfoPanel", "full_scale_words", "per_push"]


def per_push(
    max_intensity: float,
    accumulations: int,
    detector_bits: int,
    *,
    full_scale: "int | None" = None,
) -> tuple[float, float]:
    """`(counts_per_push, percent_of_full_scale)` from an in-view maximum.

    `full_scale` is the per-push ceiling (`mainspring.uimf.full_scale`), and the caller
    that knows where the bit depth came from passes it; without it the depth is taken as
    the stored width, `2^bits - 1`, which is right for a depth set by hand and four
    times too small for the one a clockwork file declares (lab record, task 36).

    `accumulations` is clamped to at least 1 -- `FrameParams.accumulations` already
    guarantees that from a real file, but a caller handing this function a bare number
    should not be able to divide by zero over a value it typed by hand.
    """
    accumulations = max(1, int(accumulations))
    counts_per_push = float(max_intensity) / accumulations
    if full_scale is None:
        full_scale = 2 ** max(1, int(detector_bits)) - 1
    percent = 100.0 * counts_per_push / float(max(1, int(full_scale)))
    return counts_per_push, percent


def _depth_words(detector_bits: int, full_scale: int, from_file: bool) -> str:
    """`14-bit from file, stored as 16-bit`, or `8-bit from setting`: where the depth
    came from, and the stored width when that is not the depth itself."""
    words = f"{detector_bits}-bit {'from file' if from_file else 'from setting'}"
    stored = int(full_scale).bit_length()
    if stored != int(detector_bits):
        words += f", stored as {stored}-bit"
    return words


def full_scale_words(full_scale: int, detector_bits: int, from_file: bool) -> str:
    """The `Full scale:` row: the ceiling one push can store, and where it came from."""
    return f"{full_scale:,} per push ({_depth_words(detector_bits, full_scale, from_file)})"


def _over_words(over: int) -> str:
    """What to add when points sit *above* full scale, which no push can read."""
    if not over:
        return ""
    return (f"; {over:,} above full scale, so the bit depth is too low for this file"
            " and these counts are not clipping")


def run_words(global_params: object) -> str:
    """How the file's run ended, as a sentence fragment for the panel.

    `unknown` is spelled out, because a file from before the outcome was recorded is
    the commonest case there is and "unknown" alone reads as a fault. The counts go
    beside `stopped` and `failed` in particular, where they are what decides whether the
    file is usable, and beside the others where they are there to be read.
    """
    outcome = getattr(global_params, "run_outcome", "unknown") or "unknown"
    if outcome == "unknown":
        return "not recorded in this file"
    words = {"incomplete": "never closed, or still running"}.get(outcome, outcome)
    planned = getattr(global_params, "repetitions_planned", None)
    acquired = getattr(global_params, "repetitions_acquired", None)
    if planned is not None and acquired is not None:
        words += f", {acquired} of {planned} repetitions"
    reason = getattr(global_params, "run_reason", "")
    if reason:
        words += f": {reason}"
    return words


def _tree_row(parent: QTreeWidgetItem, name: str, value: str) -> QTreeWidgetItem:
    """One parameter row, with its name and its full value on both columns' tooltips."""
    item = QTreeWidgetItem(parent, [name, value])
    item.setToolTip(0, f"{name}: {value}")
    item.setToolTip(1, value)
    return item


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

        self._container = QWidget(self)
        container = self._container
        container.setMinimumWidth(INFO_PANEL_WIDTH)
        self._tree = QTreeWidget(container)
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Parameter", "Value"])
        # The value column takes whatever width the panel is dragged to, so widening the
        # dock is what shows a long parameter value rather than a wider panel with the
        # same elided text in it. `set_file` still sizes the name column to its contents.
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        describe(
            self._tree,
            "The file's global parameters and the open frame's, as the file stores them.",
        )
        self._global_root = QTreeWidgetItem(self._tree, ["Global", ""])
        self._frame_root = QTreeWidgetItem(self._tree, ["Frame", ""])

        self._run_label = QLabel("-", container)
        self._state_label = QLabel("-", container)
        self._max_label = QLabel("-", container)
        self._per_push_label = QLabel("-", container)
        self._tic_label = QLabel("-", container)
        self._points_label = QLabel("-", container)
        self._full_scale_label = QLabel("-", container)
        self._raw_max_label = QLabel("-", container)
        self._frame_clip_label = QLabel("-", container)
        self._file_clip_label = QLabel("-", container)
        # What keeps a live readout from resizing the window, now that the container's
        # width is no longer fixed: a label whose horizontal policy is `Ignored`
        # contributes nothing to the layout's width, whatever its text says, and wraps
        # into whatever width it is given. Both halves are needed -- `Ignored` alone
        # would elide, and word wrap alone would still report a wide size hint.
        for label in (
            self._run_label,
            self._state_label,
            self._max_label,
            self._per_push_label,
            self._tic_label,
            self._points_label,
            self._full_scale_label,
            self._raw_max_label,
            self._frame_clip_label,
            self._file_clip_label,
        ):
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        live = QFormLayout()
        # A field too wide for the space beside its label drops to the next line rather
        # than squeezing into a sliver -- the per-push line, at the narrowest width.
        live.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        live.addRow("Run:", self._run_label)
        live.addRow("Frame:", self._state_label)
        live.addRow("Max intensity in view:", self._max_label)
        live.addRow("Per push:", self._per_push_label)
        live.addRow("TIC in view:", self._tic_label)
        live.addRow("Points in view:", self._points_label)
        live.addRow("Full scale:", self._full_scale_label)
        live.addRow("Raw maximum:", self._raw_max_label)
        live.addRow("Clipped in frame:", self._frame_clip_label)
        live.addRow("Clipped in file:", self._file_clip_label)
        # Both halves of each row, because the field name is what a reader questions.
        # The per-push sentence carries the caveat the number is meaningless without:
        # it is a quotient, over an accumulation count and a bit depth the file may not
        # have told us (lab record, task 01).
        for widget, tip in (
            (
                self._run_label,
                "How the acquisition that wrote this file ended, as the file records it:"
                " completed, stopped, failed, or never closed.",
            ),
            (
                self._state_label,
                "Whether the instrument may still be adding scans to the frame on"
                " screen, which is what the viewer knows rather than a guess at it.",
            ),
            (self._max_label, "The largest single stored intensity inside the view."),
            (
                self._per_push_label,
                "The largest stored intensity divided by Accumulations, as a fraction of"
                " full scale at the detector bit depth, which is either stored in the"
                " file or set in the Data settings menu.",
            ),
            (self._tic_label, "The total stored intensity inside the view."),
            (self._points_label, "How many stored, non-zero points are inside the view."),
            (
                self._full_scale_label,
                "The largest value one push can store, from the detector bit depth in"
                " the file or in the Data settings menu, which the clipping counts and"
                " the per-push line are judged against.",
            ),
            (
                self._raw_max_label,
                "The largest stored intensity anywhere in the file, not divided by"
                " Accumulations.",
            ),
            (
                self._frame_clip_label,
                "How many points in the whole frame on screen reached full scale, which"
                " a point can only do by saturating on every push added into it.",
            ),
            (
                self._file_clip_label,
                "How many points in every frame of the file reached full scale, counted"
                " in the background when Count clipping in file is on in the Data"
                " settings menu.",
            ),
        ):
            describe(widget, tip)
            describe(live.labelForField(widget), tip)

        layout = QFormLayout(container)
        layout.addRow(self._tree)
        layout.addRow(live)
        self.setWidget(container)
        self.set_text_scale(fonts.active())

    def set_text_scale(self, scale: float) -> None:
        """Follow `View > Text size`: grow the floor the panel cannot go below.

        Every widget in here takes its font from `QApplication.setFont`, so there is no
        text to set (`fonts.py`). What does not follow on its own is the width: 320
        pixels holds two columns and a wrapped per-push line at 100 per cent and holds
        neither at 200, so the minimum moves with the type. A panel the user has already
        dragged wider than the new floor keeps the width they chose.
        """
        self._container.setMinimumWidth(round(INFO_PANEL_WIDTH * fonts.extent()))

    def set_file(self, global_params: object, frame_params: object) -> None:
        """Replace the parameter tree with one file's global and frame parameters.

        Every row carries its own value as a tooltip. The value column stretches to the
        panel's width and elides what does not fit, and a calibration coefficient or a
        method path read to fourteen characters and an ellipsis is worse than not shown
        at all -- so the whole value is one hover away at any panel width.
        """
        self._run_label.setText(run_words(global_params))
        self._global_root.takeChildren()
        for name, value in _rows(global_params):
            _tree_row(self._global_root, name, value)
        self._frame_root.takeChildren()
        for name, value in _rows(frame_params):
            _tree_row(self._frame_root, name, value)
        self._tree.expandAll()
        self._tree.resizeColumnToContents(0)

    def set_frame_state(self, provisional: "bool | None") -> None:
        """Say whether the frame on screen is finished, in the panel's own words.

        The status bar says it too, in the line under the plot; this is where a reader
        checking what they are about to quote looks, beside the numbers they would
        quote. `None` for a frame that is not a frame of the file at all -- a sum -- for
        which the question is about its inputs, and the status line names those.

        The words are "still being written" rather than "provisional", which is the
        code's word for it and says nothing to an operator watching a run.
        """
        if provisional is None:
            self._state_label.setText("-")
        else:
            self._state_label.setText("still being written" if provisional else "complete")

    def set_view(
        self,
        result: object,
        accumulations: int,
        detector_bits: int,
        *,
        from_file: bool = False,
        full_scale: "int | None" = None,
    ) -> None:
        """Update the in-view readouts from a rendered `RasterResult`.

        `accumulations` and `detector_bits` travel with every call rather than being
        remembered from `set_file`, because the detector-bits setting can change without
        a new frame arriving and the readout must move with it immediately.

        `from_file` says where the bit depth came from, and the readout says so too. A
        clockwork acquisition stores its digitizer's depth and a PNNL-written file has
        nowhere to store one, so the same sentence covers two different claims -- "the
        file says 14 bits" and "you told us 8" -- and the rule that a per-push number is
        never quoted without what produced it is not met by quoting a number that could
        be either (lab record, tasks 01 and 17). `full_scale` is the per-push ceiling that
        follows from both (`mainspring.uimf.full_scale`); omitted, it is `2^bits - 1`.
        """
        self._max_label.setText(f"{result.max_intensity:,.0f}")
        if full_scale is None:
            full_scale = 2 ** max(1, int(detector_bits)) - 1
        counts, percent = per_push(result.max_intensity, accumulations, detector_bits,
                                   full_scale=full_scale)
        self._per_push_label.setText(
            f"{counts:,.1f} ADC/push  ({accumulations} accum.,"
            f" {_depth_words(detector_bits, full_scale, from_file)}"
            f" -- {percent:.2f}% full scale)"
        )
        self._tic_label.setText(f"{result.tic_in_view:,.0f}")
        self._points_label.setText(f"{result.points_in_view:,}")

    # --- clipping -----------------------------------------------------------------------

    def set_full_scale(self, full_scale: int, detector_bits: int, from_file: bool) -> None:
        """Say once what every clipping number below is judged against, and why.

        The bit depth's source is the panel's standing rule (`set_view`): a number
        derived from bits is never shown without them and without where they came from.
        """
        self._full_scale_label.setText(full_scale_words(full_scale, detector_bits, from_file))

    def set_raw_max(self, value: "float | None", *, so_far: bool = False) -> None:
        """The file's largest stored intensity, raw. `None` while it is being read.

        `so_far` while the run is still being written, since a larger value can still
        arrive; the words go away when following stops.
        """
        if value is None:
            self._raw_max_label.setText("reading...")
            return
        self._raw_max_label.setText(f"{value:,.0f}" + (" so far" if so_far else ""))

    def set_frame_clipping(self, counts: "tuple[int, int] | None") -> None:
        """Clipped points in the whole frame on screen, or why there is no count.

        `None` for a sum: the heat map is then several frames added together, and a
        bin of a sum reaching one frame's full scale says nothing about whether any of
        its frames did. The count is of single frames, and the label says so rather than
        adding the member frames' counts behind the operator's back.
        """
        if counts is None:
            self._frame_clip_label.setText("single frames only, and a sum is shown")
            return
        clipped, over = counts
        self._frame_clip_label.setText(
            f"{clipped:,} point{'' if clipped == 1 else 's'}" + _over_words(over)
        )

    def set_file_clipping(
        self,
        clipping: "object | None",
        *,
        enabled: bool,
        done: int = 0,
        total: int = 0,
        so_far: bool = False,
    ) -> None:
        """The file-wide count: off, counting, provisional, or the answer.

        Off says where to turn it on, because a row that reads only "off" is a row
        nobody can act on. Counting gives the frames reached and what they held so
        far. `so_far` is a run still being written, whose count can only grow.
        """
        if not enabled:
            self._file_clip_label.setText("off (Data settings > Count clipping in file)")
            return
        if clipping is None:
            self._file_clip_label.setText("counting...")
            return
        clipped = int(clipping.total)
        frames = int(clipping.frames)
        text = (f"{clipped:,} point{'' if clipped == 1 else 's'}"
                f" in {frames:,} frame{'' if frames == 1 else 's'}")
        if total and done < total:
            text = f"counting, {done:,} of {total:,} frames: " + text + " so far"
        elif so_far:
            text += " so far, still being written"
        self._file_clip_label.setText(text + _over_words(int(clipping.over_total)))

    def readouts(self) -> "dict[str, str]":
        """Every readout row's text by its field name, for a check that reads the panel."""
        form = {
            "Full scale": self._full_scale_label,
            "Raw maximum": self._raw_max_label,
            "Clipped in frame": self._frame_clip_label,
            "Clipped in file": self._file_clip_label,
            "Per push": self._per_push_label,
        }
        return {name: label.text() for name, label in form.items()}
