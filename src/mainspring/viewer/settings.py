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
round trip without a `QApplication`. `load_settings`/`save_settings` are the only two
functions that touch `QSettings`, and they are the only two functions in this module
that need one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from PySide6.QtCore import QByteArray, QSettings

from ..uimf.raster import AGGREGATES

__all__ = [
    "APPLICATION",
    "COLOUR_MAPS",
    "COLOUR_SCALES",
    "EXPORT_DPIS",
    "ORGANISATION",
    "ViewerSettings",
    "load_settings",
    "save_settings",
]

ORGANISATION = "University of Washington"
APPLICATION = "mainspring"
"""The `QSettings` scope. On Windows this is `HKCU\\Software\\...`; changing either
string orphans every user's saved preferences, so do not."""

COLOUR_SCALES = ("linear", "log", "sqrt")
"""How the image maps intensity to colour. Linear is the honest default; log and sqrt
compress a spectrum's dynamic range for a view dominated by one bright peak. Purely a
display transform -- the readouts always quote the untransformed `RasterResult`."""

COLOUR_MAPS = ("viridis", "plasma", "inferno", "magma")
"""The View > Colour map choices, all perceptually uniform so a gradient never implies
a step in intensity that is not there -- unlike jet or a stock rainbow map, which the
viewer deliberately does not offer."""


EXPORT_DPIS = (96, 150, 300, 600)
"""The File > Export resolutions, in dots per inch. 96 is the window's own pixels one
for one (`export.BASE_DPI`), 150 is a slide, and 300 and 600 are what a journal asks for.
Here rather than in `export.py` so that `validate` can clamp a hand-edited value without
the settings module having to import a module that pulls in Qt's widgets."""


@dataclass
class ViewerSettings:
    """The persisted viewer state. Defaults are what a first run gets.

    Mutable by design: the toolbar edits one of these and hands it back to be saved.
    """

    aggregate: str = "sum"
    swap_axes: bool = False
    raw_units: bool = False
    colour_scale: str = "linear"
    keep_ranges: bool = False
    keep_levels: bool = False
    show_info_panel: bool = True
    detector_bits: int = 8
    colour_map: str = "viridis"
    export_dpi: int = 300
    cache_budget_mb: int = 512
    arrival_offset_ms: float = 0.0
    last_directory: str = ""
    window_geometry: bytes = b""

    def validate(self) -> "ViewerSettings":
        """Clamp anything a hand-edited registry could have put out of range.

        Settings read from disk are input like any other: a `detector_bits` of 0 would
        divide by zero in the per-push readout, and an unrecognised `aggregate` or
        `colour_scale` would raise deep inside the render path rather than here, where
        the bad value can be seen and swapped for the default.
        """
        detector_bits = self.detector_bits if 1 <= self.detector_bits <= 32 else 8
        cache_budget_mb = self.cache_budget_mb if self.cache_budget_mb > 0 else 512
        return replace(
            self,
            aggregate=self.aggregate if self.aggregate in AGGREGATES else "sum",
            colour_scale=self.colour_scale if self.colour_scale in COLOUR_SCALES else "linear",
            detector_bits=detector_bits,
            colour_map=self.colour_map if self.colour_map in COLOUR_MAPS else "viridis",
            export_dpi=self.export_dpi if self.export_dpi in EXPORT_DPIS else 300,
            cache_budget_mb=cache_budget_mb,
            arrival_offset_ms=self.arrival_offset_ms if math.isfinite(self.arrival_offset_ms) else 0.0,
            last_directory=self.last_directory or "",
        )


def _store() -> QSettings:
    """The one `QSettings` every read and write goes through.

    **Not `QSettings(ORGANISATION, APPLICATION)`** -- that two-argument constructor is
    documented to mean `QSettings(NativeFormat, UserScope, organization, application)`
    unconditionally, ignoring `QSettings.setDefaultFormat()` no matter what it was set
    to. A test that redirects the default format to a per-test ini file (`conftest.py`)
    would silently keep hitting the real Windows registry through that form, which is
    exactly the pollution the redirect exists to prevent. The four-argument form below
    asks for the format explicitly and reads `defaultFormat()` to get it, so a test's
    redirect is honoured and a real session still gets the registry it always has.
    """
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope, ORGANISATION, APPLICATION)


def load_settings() -> ViewerSettings:
    """Read the persisted settings, falling back to the defaults field by field.

    A key `QSettings` has never seen -- a first run, or one written by an older version
    missing a field this one added -- comes back as that field's default rather than
    failing the whole load, which is why each is read with the default already in hand
    rather than by reconstructing a dict and hoping every key is present.
    """
    defaults = ViewerSettings()
    store = _store()
    settings = ViewerSettings(
        aggregate=str(store.value("aggregate", defaults.aggregate)),
        swap_axes=_as_bool(store.value("swap_axes", defaults.swap_axes)),
        raw_units=_as_bool(store.value("raw_units", defaults.raw_units)),
        colour_scale=str(store.value("colour_scale", defaults.colour_scale)),
        keep_ranges=_as_bool(store.value("keep_ranges", defaults.keep_ranges)),
        keep_levels=_as_bool(store.value("keep_levels", defaults.keep_levels)),
        show_info_panel=_as_bool(store.value("show_info_panel", defaults.show_info_panel)),
        detector_bits=_as_int(store.value("detector_bits", defaults.detector_bits),
                               defaults.detector_bits),
        colour_map=str(store.value("colour_map", defaults.colour_map)),
        export_dpi=_as_int(store.value("export_dpi", defaults.export_dpi), defaults.export_dpi),
        cache_budget_mb=_as_int(store.value("cache_budget_mb", defaults.cache_budget_mb),
                                 defaults.cache_budget_mb),
        arrival_offset_ms=_as_float(store.value("arrival_offset_ms", defaults.arrival_offset_ms),
                                     defaults.arrival_offset_ms),
        last_directory=str(store.value("last_directory", defaults.last_directory)),
        window_geometry=_as_bytes(store.value("window_geometry", defaults.window_geometry)),
    )
    return settings.validate()


def save_settings(settings: ViewerSettings) -> None:
    """Write the settings back, one value per key, and flush them to disk.

    `sync()` rather than leaving it to process exit: the viewer's own close saves once,
    and a crash between a toggle and the next clean shutdown should not cost the user
    every preference they set this session.
    """
    store = _store()
    store.setValue("aggregate", settings.aggregate)
    store.setValue("swap_axes", settings.swap_axes)
    store.setValue("raw_units", settings.raw_units)
    store.setValue("colour_scale", settings.colour_scale)
    store.setValue("keep_ranges", settings.keep_ranges)
    store.setValue("keep_levels", settings.keep_levels)
    store.setValue("show_info_panel", settings.show_info_panel)
    store.setValue("detector_bits", settings.detector_bits)
    store.setValue("colour_map", settings.colour_map)
    store.setValue("export_dpi", settings.export_dpi)
    store.setValue("cache_budget_mb", settings.cache_budget_mb)
    store.setValue("arrival_offset_ms", settings.arrival_offset_ms)
    store.setValue("last_directory", settings.last_directory)
    store.setValue("window_geometry", QByteArray(settings.window_geometry))
    store.sync()


def _as_bool(value: object) -> bool:
    """A `QSettings` value as a bool. The ini backend round-trips real Python bools; the
    registry backend does not, and hands one back as `"true"`/`"false"` or `0`/`1`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_bytes(value: object) -> bytes:
    if isinstance(value, QByteArray):
        return bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return b""
