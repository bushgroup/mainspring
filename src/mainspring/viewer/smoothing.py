"""The `Data settings > Arrival-time smoothing width...` dialog.

One number, the Gaussian's σ in milliseconds, which the arrival-time projection is
smoothed by while `Data settings > Smooth arrival time` is ticked. A text box rather than
a spin box, because the rule is the smoother's own and nothing narrower: **any finite
number above zero** (`mainspring.analysis.gaussian_smooth`). A spin box would need a
lowest and a highest value and a number of decimals, three bounds nobody chose (lab
record, task 44).

The width is in milliseconds whatever the arrival-time axis shows: under `Raw units` the
axis counts scans, and the smoother still converts σ through the frame's own scan time.
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from .controls import describe

__all__ = ["SmoothingWidthDialog", "parse_width"]


def parse_width(text: str) -> "float | None":
    """`text` as a width in ms if it is a finite number above zero, else None."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


class SmoothingWidthDialog(QDialog):
    """Ask for σ in ms; OK is offered only while the text is a usable width."""

    def __init__(self, parent: object, sigma_ms: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("Arrival-time smoothing width")
        self.setModal(True)

        self._edit = QLineEdit(f"{sigma_ms:g}")
        self._edit.selectAll()
        describe(
            self._edit,
            "The standard deviation of the Gaussian that smooths the arrival-time plot,"
            " in milliseconds; any number above zero.",
        )
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Width, σ (ms):", self._edit)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        note = QLabel("In milliseconds even when Raw units is on.")
        note.setEnabled(False)  # a footnote, not a control
        layout.addWidget(note)
        layout.addWidget(self._buttons)

        self._edit.textChanged.connect(lambda _: self._check())
        self._check()

    def value(self) -> "float | None":
        """The width typed, or None if it is not one."""
        return parse_width(self._edit.text())

    def _check(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(self.value() is not None)
