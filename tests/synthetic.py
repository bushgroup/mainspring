"""Write a real UIMF file with no acquisition in it.

A fresh clone has no `.uimf` file and never will -- samples live in the private lab
repository and PNNL's excerpts are fetched, not committed -- so the only way the decode
path can be exercised end to end in a bare clone is to write a file through the real
SQLite schema and read it back. That is what this module does, and since task 16 it
does it through `mainspring.uimf.writer`: the schema, the parameter keys and the four
stored summary columns are the package's, so there is one definition of what a UIMF
file is rather than one for the product and one for the tests.

What is left here is the *content* -- a few dozen points whose every stored column we
know because we computed it -- and the deliberate awkwardness that makes this a fixture
for real files rather than for tidy ones (lab record, task 01):

* **both parameter table forms**, modern and legacy, as our sample carries them --
  and with the frame type disagreeing between them, 0 modern against 1 legacy, exactly
  as the sample does, so that a reader which picks the wrong table is caught. The
  writer emits one value in both tables by default; this is the one caller that asks
  it not to;
* **only the scans that have signal**, starting well past zero, because the SLIMPHONY
  writer stores 1656 of 5000 and a reader must not assume a row per scan;
* **explicit zeros in the intensity stream**, which is why `NonZeroCount` comes out
  above the number of points and is an upper bound rather than a count;
* the three intensity element types, including the two no real file we have uses.

`BPI_MZ` here is exact -- the calibration formula applied to the argmax bin -- unlike
every real writer, which is off by up to three bins or, in one 2011 file, never computed
it at all. A test wanting that behaviour should use the PNNL excerpts, and one wanting
what PNNL's console does with the column should use `console_stub.py`.

    from synthetic import write_synthetic_uimf
    spec = write_synthetic_uimf(tmp_path / "synth.uimf", frames=2, scans=16)
    spec.scan(1, 4).intensity      # what the file must decode to
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from mainspring.uimf.decode import dtype_for, encode_intensities
from mainspring.uimf.writer import FrameSpec, GlobalSpec, UimfWriter

__all__ = ["SyntheticFile", "SyntheticScan", "write_synthetic_uimf"]

# The sample's own calibration, so that a synthetic m/z axis lands where a real one does.
SLOPE = 0.738123
INTERCEPT = 0.07690495
AVERAGE_TOF_LENGTH_NS = 129003.607843137
ACCUMULATIONS = 100

# The US locale string real files carry, frozen rather than taken from the clock so that
# two runs of the suite write the same bytes.
DATE_STARTED = "8/25/2026 3:23:14 PM"


@dataclass(frozen=True)
class SyntheticScan:
    """One written scan and every column the file records for it.

    `bin_index` and `intensity` are what a decoder must return: the explicit zeros this
    fixture writes into the stream are *not* here, which is exactly why `non_zero_count`
    is larger than `bin_index.size`.
    """

    frame: int
    scan: int
    bin_index: np.ndarray
    intensity: np.ndarray
    non_zero_count: int
    bpi: float
    bpi_mz: float
    tic: float
    blob: bytes


@dataclass(frozen=True)
class SyntheticFile:
    """The file that was written, and the answers a test compares against."""

    path: str
    frames: tuple[int, ...]
    scans: int
    bins: int
    dtype: np.dtype
    tof_intensity_type: str
    bin_width_ns: float
    average_tof_length_ns: float
    accumulations: int
    slope: float
    intercept: float
    legacy_only: bool
    modern_only: bool
    scan_rows: dict[tuple[int, int], SyntheticScan] = field(default_factory=dict)

    def scan(self, frame: int, scan: int) -> SyntheticScan:
        """The written row for one `(frame, scan)`; KeyError if that scan was not stored."""
        return self.scan_rows[(frame, scan)]

    def stored_scans(self, frame: int) -> list[int]:
        """The scan numbers actually written for a frame, ascending. Not `range(scans)`."""
        return sorted(s for (f, s) in self.scan_rows if f == frame)

    def points(self, frame: int) -> int:
        """The number of non-zero points in a frame, over every stored scan."""
        return sum(row.bin_index.size for (f, _), row in self.scan_rows.items() if f == frame)

    def tic(self, frame: int) -> float:
        """The frame's total ion current: the sum of its rows' `TIC` columns."""
        return float(sum(row.tic for (f, _), row in self.scan_rows.items() if f == frame))


def _mz_of_bin(bin_index: float, bin_width_ns: float, slope: float, intercept: float) -> float:
    """The calibration of `notes/uimf-format.md`, written out once so this fixture is
    independent of the implementation it exists to test."""
    t_us = bin_index * bin_width_ns / 1000.0
    return float((slope * (t_us - intercept)) ** 2)


def _peaks(frame: int, scan: int, bins: int, amplitude: int) -> tuple[list[int], list[int]]:
    """A deterministic little spectrum: two peaks that drift with scan, plus a fixed ridge.

    Deterministic and cheap rather than realistic. It only has to give every scan a
    different argmax, put points at both ends of the bin axis, and produce a heatmap with
    visible structure for the viewer tasks to look at.
    """
    centre_a = 12 + (scan * 37 + frame * 11) % max(1, bins - 40)
    centre_b = 20 + (scan * 211 + frame * 97) % max(1, bins - 40)
    out: dict[int, int] = {}
    for centre, height in ((centre_a, amplitude), (centre_b, amplitude // 3 + 1)):
        for offset in (-2, -1, 0, 1, 2):
            b = centre + offset
            if 0 <= b < bins:
                out[b] = out.get(b, 0) + max(1, height >> abs(offset))
    for b in (1, bins // 2, bins - 2):  # a ridge at both edges and the middle
        out[b] = out.get(b, 0) + 3
    ordered = sorted(out)
    return ordered, [out[b] for b in ordered]


def write_synthetic_uimf(
    path: str | os.PathLike[str],
    frames: int = 2,
    scans: int = 16,
    bins: int = 4096,
    tof_intensity_type: str = "ADC",
    bin_width_ns: float = 1.0,
    legacy_only: bool = False,
    modern_only: bool = False,
    store_all_scans: bool = False,
    explicit_zeros: bool = True,
    first_scan: int = 3,
    journal_mode: str = "wal",
    finalise: bool = True,
    grouped: bool = False,
    detector_bits: int | None = None,
) -> SyntheticFile:
    """Write a UIMF file at `path` and return what is in it.

    `frames` and `scans` are counts; frame numbers are 1-based, as writers number them,
    and scan numbers run from `first_scan` -- non-zero by default, because our sample's
    first stored scan is 12 and a reader that assumes 0 must fail here.

    `legacy_only` writes a 2011-style file with only `Global_Parameters` and
    `Frame_Parameters`; `modern_only` writes only the EAV tables; the default writes
    both, as the sample does, with the frame type disagreeing between them on purpose.

    `store_all_scans` writes a row for every scan including empty ones, as the 2011
    writers do; the default stores only scans with signal. `explicit_zeros` puts zeros
    into the intensity stream and counts them in `NonZeroCount`, as real writers do.

    `journal_mode` is WAL, which is what a clockwork acquisition is and what a live
    reader needs; pass `"delete"` for the mode every file we have from PNNL's writers is
    in. `finalise` leaves the last frame without its completion marker when false, which
    is what a run cut short by a power failure looks like. `grouped` writes each frame as
    a repetition of one method frame; `detector_bits` stores a bit depth.
    """
    if legacy_only and modern_only:
        raise ValueError("legacy_only and modern_only are mutually exclusive")
    dtype = dtype_for(tof_intensity_type)
    amplitude = 900 if dtype == np.dtype("<i2") else 9000
    path = os.fspath(path)

    spec = SyntheticFile(
        path=os.path.abspath(path),
        frames=tuple(range(1, frames + 1)),
        scans=scans,
        bins=bins,
        dtype=dtype,
        tof_intensity_type=tof_intensity_type,
        bin_width_ns=bin_width_ns,
        average_tof_length_ns=AVERAGE_TOF_LENGTH_NS,
        accumulations=ACCUMULATIONS,
        slope=SLOPE,
        intercept=INTERCEPT,
        legacy_only=legacy_only,
        modern_only=modern_only,
    )

    tables = "legacy" if legacy_only else ("modern" if modern_only else "both")
    writer = UimfWriter(
        spec.path,
        GlobalSpec(
            bins=bins,
            bin_width_ns=bin_width_ns,
            instrument_name="SLIM3",
            date_started=DATE_STARTED,
            tof_intensity_type=tof_intensity_type,
            time_offset_ns=20000,  # present, and deliberately not applied to the calibration
            prescan_tof_pulses=5000,
            prescan_accumulations=ACCUMULATIONS,
            detector_bits=detector_bits,
        ),
        tables=tables,
        journal_mode=journal_mode,
        overwrite=True,
    )
    with writer:
        for frame in spec.frames:
            writer.add_frame(
                FrameSpec(
                    scans=scans,
                    accumulations=ACCUMULATIONS,
                    # 0 in the modern table and 1 in the legacy one, as our sample
                    # disagrees with itself. Everything else this writer produces says 1
                    # in both; see `FrameSpec.legacy_frame_type`.
                    frame_type=0,
                    legacy_frame_type=1,
                    calibration_slope=SLOPE,
                    calibration_intercept=INTERCEPT,
                    average_tof_length_ns=AVERAGE_TOF_LENGTH_NS,
                    method_frame=1 if grouped else None,
                    repetition=frame if grouped else None,
                    repetitions=len(spec.frames) if grouped else None,
                ),
                frame=frame,
            )
            rows = _frame_rows(spec, frame, amplitude, store_all_scans, explicit_zeros,
                               first_scan)
            writer.write_scans(frame, rows)
            if finalise or frame != spec.frames[-1]:
                writer.finalise_frame(frame, duration_s=_duration_s(spec))
    return spec


def _duration_s(spec: SyntheticFile) -> float:
    """What a frame of this shape would have taken: scans x pusher period x repeats."""
    return spec.scans * spec.average_tof_length_ns * 1e-9 * spec.accumulations


def _frame_rows(
    spec: SyntheticFile,
    frame: int,
    amplitude: int,
    store_all_scans: bool,
    explicit_zeros: bool,
    first_scan: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The scans of one frame, and the record of them in `spec.scan_rows`.

    What goes to the writer is the *stream* -- the points plus this fixture's explicit
    zeros -- and what is recorded is the points, because the points are what a decoder
    must give back.
    """
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    # Clamp so that a small `scans` still produces signal: the point of the offset is
    # that scan 0 is not stored, not that a particular scan is the first.
    start = min(first_scan, max(0, spec.scans - 2))
    for scan in range(spec.scans):
        has_signal = start <= scan < spec.scans - 1
        if not has_signal and not store_all_scans:
            continue
        if has_signal:
            bins_list, values_list = _peaks(frame, scan, spec.bins, amplitude)
        else:
            bins_list, values_list = [], []

        # Explicit zeros go into the stream and are counted, but are not points: this is
        # what makes NonZeroCount an upper bound (lab record, task 01). They sit in the
        # quiet stretch after the first ridge bin, where no peak can reach.
        stream = dict(zip(bins_list, values_list))
        zeros = 0
        if explicit_zeros and bins_list:
            for candidate in (bins_list[0] + 6, bins_list[0] + 7):
                if candidate < spec.bins and candidate not in stream:
                    stream[candidate] = 0
                    zeros += 1
        stream_bins = np.array(sorted(stream), dtype=np.int64)
        stream_values = np.array([stream[b] for b in stream_bins], dtype=spec.dtype)

        bin_index = np.array(bins_list, dtype=np.int64)
        intensity = np.array(values_list, dtype=spec.dtype)
        if intensity.size:
            argmax = int(np.argmax(intensity))
            bpi = float(intensity[argmax])
            bpi_mz = _mz_of_bin(float(bin_index[argmax]), spec.bin_width_ns, spec.slope,
                                spec.intercept)
            tic = float(intensity.sum())
        else:
            bpi, bpi_mz, tic = 0.0, 0.0, 0.0

        spec.scan_rows[(frame, scan)] = SyntheticScan(
            frame=frame, scan=scan, bin_index=bin_index, intensity=intensity,
            non_zero_count=int(intensity.size + zeros), bpi=bpi, bpi_mz=bpi_mz, tic=tic,
            blob=encode_intensities(stream_bins, stream_values, spec.dtype),
        )
        rows.append((scan, stream_bins, stream_values))
    return rows
