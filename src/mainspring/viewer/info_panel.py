"""The info panel: the file's parameters, and the readout the instrument is judged by.

Two halves. The upper one lists the global and frame parameters a file happens to carry
-- whatever keys are in the tables, not a fixed list, since the 2016 files carry fifteen
optics voltages the sample does not and a future writer will carry something else again.

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

Also here: max intensity in view, total ion current in view, and the point count, all
taken from the `RasterResult` so that they describe the image on screen rather than a
newer view the render has not caught up with.
"""

from __future__ import annotations

__all__ = ["InfoPanel", "per_push"]


def per_push(max_intensity: float, accumulations: int, detector_bits: int) -> tuple[float, float]:
    """`(counts_per_push, percent_of_full_scale)` from an in-view maximum.

    Arrives with the lab record's task 06.
    """
    raise NotImplementedError("the per-push readout arrives with the lab record's task 06")


class InfoPanel:
    """The parameter list and the per-push readout. Arrives with the lab record's task 06."""

    def __init__(self) -> None:
        raise NotImplementedError("the info panel arrives with the lab record's task 06")

    def set_file(self, global_params: object, frame_params: object) -> None:
        """Show a file's parameters. Arrives with the lab record's task 06."""
        raise NotImplementedError("the info panel arrives with the lab record's task 06")

    def set_view(self, result: object, accumulations: int, detector_bits: int) -> None:
        """Update the in-view readouts from a rendered image. Arrives with task 06."""
        raise NotImplementedError("the info panel arrives with the lab record's task 06")
