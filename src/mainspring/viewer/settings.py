"""`ViewerSettings`: what the viewer remembers between sessions, in one place.

Every toggle the toolbar carries and every preference the info panel needs is a field
here, and `QSettings` is the only thing that knows where they are stored. Two reasons
that matters more than it looks:

* **Keep ranges is a persisted setting, not a session flag.** The complaint that
  started this project is that PNNL's viewer resets the m/z and arrival-time ranges
  every time a file is opened; a researcher comparing twenty acquisitions sets the
  window once. Persisting it across restarts, and optionally the colour levels with it,
  is the whole feature.
* **`detector_bits` has no defensible default from the file.** Bit depth is not
  recorded in UIMF. Today's SLIMPHONY digitizer is 8-bit and clockwork will be 14, so
  the per-push readout is only as right as this setting -- which is why the info panel
  states it alongside the number rather than quoting counts on their own (lab record,
  task 01).

Plain dataclass, no Qt in the type itself, so a test can exercise the defaults and the
round trip without a `QApplication`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ORGANISATION", "APPLICATION", "ViewerSettings", "load_settings", "save_settings"]

ORGANISATION = "University of Washington"
APPLICATION = "mainspring"
"""The `QSettings` scope. On Windows this is `HKCU\\Software\\...`; changing either
string orphans every user's saved preferences, so do not."""


@dataclass
class ViewerSettings:
    """The persisted viewer state. Defaults are what a first run gets.

    Mutable by design: the toolbar edits one of these and hands it back to be saved.
    """

    aggregate: str = "sum"
    swap_axes: bool = False
    raw_units: bool = False
    log_colour: bool = False
    keep_ranges: bool = False
    keep_levels: bool = False
    detector_bits: int = 8
    colour_map: str = "viridis"
    cache_budget_mb: int = 512
    last_directory: str = ""
    window_geometry: bytes = b""

    def validate(self) -> "ViewerSettings":
        """Clamp anything a hand-edited registry could have put out of range.

        Settings read from disk are input like any other: a `detector_bits` of 0 would
        divide by zero in the per-push readout. Arrives with the lab record's task 06.
        """
        raise NotImplementedError("settings validation arrives with the lab record's task 06")


def load_settings() -> ViewerSettings:
    """Read the persisted settings, falling back to the defaults field by field.

    Arrives with the lab record's task 06.
    """
    raise NotImplementedError("settings persistence arrives with the lab record's task 06")


def save_settings(settings: ViewerSettings) -> None:
    """Write the settings back. Arrives with the lab record's task 06."""
    raise NotImplementedError("settings persistence arrives with the lab record's task 06")
