"""The two side plots: the mass spectrum and the arrival-time distribution in view.

Both are projections of **what is on screen**, not of the whole frame: the mass
spectrum sums the points inside the current arrival-time range and the arrival-time
plot sums those inside the current m/z range, so narrowing one axis sharpens the other
plot rather than leaving it describing a window the user has left. That linkage is the
reason the two plots earn their space; a static full-frame spectrum beside a zoomed
heatmap tells you nothing you did not already know.

Axes are shared with the heatmap in the obvious way -- the mass spectrum's x is the
heatmap's x, the arrival-time plot's y is the heatmap's y -- and they stay shared when
the axes are swapped, which is why the plots take their orientation from `DisplayAxes`
rather than from their own idea of which is which.

Their projections come from `mainspring.uimf.raster.profile`, on the render worker with
the image, so a gesture produces one consistent set of three pictures rather than three
that arrive at different times.
"""

from __future__ import annotations

__all__ = ["SidePlots"]


class SidePlots:
    """The mass spectrum above and the arrival-time distribution beside the heatmap.

    Arrives with the lab record's task 05.
    """

    def __init__(self) -> None:
        raise NotImplementedError("the side plots arrive with the lab record's task 05")

    def set_profiles(self, spectrum: object, arrival: object) -> None:
        """Show a matched pair of projections. Arrives with the lab record's task 05."""
        raise NotImplementedError("the side plots arrive with the lab record's task 05")

    def clear(self) -> None:
        """Empty both plots, for a file being closed. Arrives with task 05."""
        raise NotImplementedError("the side plots arrive with the lab record's task 05")
