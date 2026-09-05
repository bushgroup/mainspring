"""Write a real UIMF file with no acquisition in it.

A fresh clone has no `.uimf` file and never will -- samples live in the private lab
repository and PNNL's excerpts are fetched, not committed -- so the only way the decode
path can be exercised end to end in a bare clone is to write a file through the real
SQLite schema and read it back. That is what this module does: the schema below is the
one a 2026 SLIMPHONY acquisition carries, table for table and index for index, filled
with a few dozen points whose every stored column we know because we computed it.

It is a fixture, not a writer. mainspring reads UIMF files; nothing outside `tests/`
imports this, and the encoder it leans on (`mainspring.uimf.decode.encode_intensities`)
is pure Python for the same reason.

What it deliberately reproduces from real files (lab record, task 01):

* **both parameter table forms**, modern and legacy, as our sample carries them --
  and with the frame type disagreeing between them, 0 modern against 1 legacy, exactly
  as the sample does, so that a reader which picks the wrong table is caught;
* **only the scans that have signal**, starting well past zero, because the SLIMPHONY
  writer stores 1656 of 5000 and a reader must not assume a row per scan;
* **explicit zeros in the intensity stream**, which is why `NonZeroCount` comes out
  above the number of points and is an upper bound rather than a count;
* the three intensity element types, including the two no real file we have uses.

`BPI_MZ` here is exact -- the calibration formula applied to the argmax bin -- unlike
every real writer, which is off by up to three bins or, in one 2011 file, never computed
it at all. A test wanting that behaviour should use the PNNL excerpts.

    from synthetic import write_synthetic_uimf
    spec = write_synthetic_uimf(tmp_path / "synth.uimf", frames=2, scans=16)
    spec.scan(1, 4).intensity      # what the file must decode to
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from mainspring.uimf.decode import dtype_for, encode_intensities

__all__ = ["SyntheticFile", "SyntheticScan", "write_synthetic_uimf"]

# --- the schema, as a 2026 SLIMPHONY file carries it -------------------------------

_MODERN_SCHEMA = (
    "CREATE TABLE Global_Params ( ParamID INTEGER NOT NULL, ParamName TEXT NOT NULL,"
    " ParamValue TEXT, ParamDataType TEXT NOT NULL, ParamDescription TEXT NULL)",
    "CREATE TABLE Frame_Param_Keys ( ParamID INTEGER NOT NULL, ParamName TEXT NOT NULL,"
    " ParamDataType TEXT NOT NULL, ParamDescription TEXT NULL)",
    "CREATE TABLE Frame_Params ( FrameNum INTEGER NOT NULL, ParamID INTEGER NOT NULL,"
    " ParamValue TEXT)",
    "CREATE UNIQUE INDEX pk_index_GlobalParams on Global_Params(ParamID)",
    "CREATE UNIQUE INDEX pk_index_FrameParamKeys on Frame_Param_Keys(ParamID)",
    "CREATE UNIQUE INDEX pk_index_FrameParams on Frame_Params(FrameNum, ParamID)",
    "CREATE INDEX ix_index_FrameParams_By_ParamID on Frame_Params(ParamID, FrameNum)",
    "CREATE VIEW V_Frame_Params AS SELECT FP.FrameNum, FPK.ParamName, FP.ParamID,"
    " FP.ParamValue, FPK.ParamDescription, FPK.ParamDataType FROM Frame_Params FP"
    " INNER JOIN Frame_Param_Keys FPK ON FP.ParamID = FPK.ParamID",
)

_LEGACY_SCHEMA = (
    "CREATE TABLE Global_Parameters ( DateStarted TEXT, NumFrames INTEGER NOT NULL,"
    " TimeOffset INTEGER NOT NULL, BinWidth DOUBLE NOT NULL, Bins INTEGER NOT NULL,"
    " TOFCorrectionTime FLOAT NOT NULL, FrameDataBlobVersion FLOAT NOT NULL,"
    " ScanDataBlobVersion FLOAT NOT NULL, TOFIntensityType TEXT NOT NULL,"
    " DatasetType TEXT, Prescan_TOFPulses INTEGER, Prescan_Accumulations INTEGER,"
    " Prescan_TICThreshold INTEGER, Prescan_Continuous BOOLEAN, Prescan_Profile TEXT,"
    " Instrument_Name TEXT)",
    "CREATE TABLE Frame_Parameters ( FrameNum INTEGER PRIMARY KEY, StartTime DOUBLE,"
    " Duration DOUBLE, Accumulations SMALLINT, FrameType SMALLINT, Scans INTEGER,"
    " IMFProfile TEXT, TOFLosses DOUBLE, AverageTOFLength DOUBLE NOT NULL,"
    " CalibrationSlope DOUBLE, CalibrationIntercept DOUBLE, a2 DOUBLE, b2 DOUBLE,"
    " c2 DOUBLE, d2 DOUBLE, e2 DOUBLE, f2 DOUBLE, Temperature DOUBLE,"
    " PressureFront DOUBLE, PressureBack DOUBLE, MPBitOrder TINYINT,"
    " FragmentationProfile BLOB, HighPressureFunnelPressure DOUBLE,"
    " IonFunnelTrapPressure DOUBLE, RearIonFunnelPressure DOUBLE,"
    " QuadrupolePressure DOUBLE, ESIVoltage DOUBLE, FloatVoltage DOUBLE,"
    " CalibrationDone INTEGER, Decoded INTEGER)",
)

_SCANS_SCHEMA = (
    "CREATE TABLE Frame_Scans ( FrameNum INTEGER NOT NULL, ScanNum SMALLINT NOT NULL,"
    " NonZeroCount INTEGER NOT NULL, BPI INTEGER NOT NULL, BPI_MZ DOUBLE NOT NULL,"
    " TIC INTEGER NOT NULL, Intensities BLOB)",
    "CREATE UNIQUE INDEX pk_index_FrameScans on Frame_Scans(FrameNum, ScanNum)",
)

# (ParamID, ParamName, ParamDataType, ParamDescription) as the sample carries them.
_GLOBAL_KEYS = (
    (1, "InstrumentName", "System.String", "Instrument name"),
    (2, "DateStarted", "System.String", "Time that the data acquisition started"),
    (3, "NumFrames", "System.Int32", "Number of frames in the dataset"),
    (4, "TimeOffset", "System.Int32",
     "Time offset from 0 (in nanoseconds). All bin numbers must be offset by this amount"),
    (5, "BinWidth", "System.Double", "Width of TOF bins (in ns)"),
    (6, "Bins", "System.Int32", "Total number of TOF bins in frame"),
    (8, "TOFIntensityType", "System.String",
     "Data type of intensity in each TOF record (ADC is int, TDC is short, FOLDED is float)"),
    (9, "DatasetType", "System.String", "Type of dataset (HMS, HMSn, or HMS-HMSn)"),
    (10, "PrescanTOFPulses", "System.String", "Prescan TOF pulses"),
    (11, "PrescanAccumulations", "System.String", "Number of prescan accumulations"),
)

_FRAME_KEYS = (
    (3, "Accumulations", "System.Int32", "Number of collected and summed acquisitions in a frame"),
    (4, "FrameType", "System.Int32",
     "Frame Type: 0=MS (Legacy); 1=MS (Regular); 2=MS/MS (Frag); 3=Calibration; 4=Prescan"),
    (6, "CalibrationDone", "System.Int32",
     "Tracks whether frame has been calibrated: 1 if calibrated"),
    (7, "Scans", "System.Int32", "Number of TOF scans in a frame"),
    (11, "AverageTOFLength", "System.Double",
     "Average time between TOF trigger pulses, in nanoseconds"),
    (12, "CalibrationSlope", "System.Double", "Calibration slope, k0"),
    (13, "CalibrationIntercept", "System.Double", "Calibration intercept, t0"),
    (14, "MassCalibrationCoefficienta2", "System.Double",
     "a2 parameter for residual mass error correction"),
    (15, "MassCalibrationCoefficientb2", "System.Double",
     "b2 parameter for residual mass error correction"),
    (16, "MassCalibrationCoefficientc2", "System.Double",
     "c2 parameter for residual mass error correction"),
    (17, "MassCalibrationCoefficientd2", "System.Double",
     "d2 parameter for residual mass error correction"),
    (18, "MassCalibrationCoefficiente2", "System.Double",
     "e2 parameter for residual mass error correction"),
    (19, "MassCalibrationCoefficientf2", "System.Double",
     "f2 parameter for residual mass error correction"),
    (20, "AmbientTemperature", "System.Single", "Ambient temperature, in Celcius"),
)

# The sample's own calibration, so that a synthetic m/z axis lands where a real one does.
SLOPE = 0.738123
INTERCEPT = 0.07690495
AVERAGE_TOF_LENGTH_NS = 129003.607843137
ACCUMULATIONS = 100


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
    """
    if legacy_only and modern_only:
        raise ValueError("legacy_only and modern_only are mutually exclusive")
    dtype = dtype_for(tof_intensity_type)
    amplitude = 900 if dtype == np.dtype("<i2") else 9000
    path = os.fspath(path)
    if os.path.exists(path):
        os.remove(path)

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

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode = delete")  # what every writer we have seen uses
        for statement in _SCANS_SCHEMA:
            conn.execute(statement)
        if not legacy_only:
            for statement in _MODERN_SCHEMA:
                conn.execute(statement)
        if not modern_only:
            for statement in _LEGACY_SCHEMA:
                conn.execute(statement)

        _write_global(conn, spec)
        if not spec.legacy_only:
            conn.executemany(
                "INSERT INTO Frame_Param_Keys (ParamID, ParamName, ParamDataType,"
                " ParamDescription) VALUES (?, ?, ?, ?)",
                list(_FRAME_KEYS),
            )
        for frame in spec.frames:
            _write_frame_params(conn, spec, frame)
            _write_frame_scans(conn, spec, frame, amplitude, store_all_scans, explicit_zeros,
                               first_scan)
        conn.commit()
    finally:
        conn.close()
    return spec


def _write_global(conn: sqlite3.Connection, spec: SyntheticFile) -> None:
    values = {
        "InstrumentName": "SLIM3",
        "DateStarted": "8/25/2026 3:23:14 PM",  # the US locale string real files carry
        "NumFrames": str(len(spec.frames)),
        "TimeOffset": "20000",  # present, and deliberately not applied to the calibration
        "BinWidth": repr(spec.bin_width_ns),
        "Bins": str(spec.bins),
        "TOFIntensityType": spec.tof_intensity_type,
        "DatasetType": "",
        "PrescanTOFPulses": "5000",
        "PrescanAccumulations": str(spec.accumulations),
    }
    if not spec.legacy_only:
        conn.executemany(
            "INSERT INTO Global_Params (ParamID, ParamName, ParamValue, ParamDataType,"
            " ParamDescription) VALUES (?, ?, ?, ?, ?)",
            [(pid, name, values[name], dtype, desc) for pid, name, dtype, desc in _GLOBAL_KEYS],
        )
    if not spec.modern_only:
        conn.execute(
            "INSERT INTO Global_Parameters (DateStarted, NumFrames, TimeOffset, BinWidth,"
            " Bins, TOFCorrectionTime, FrameDataBlobVersion, ScanDataBlobVersion,"
            " TOFIntensityType, DatasetType, Prescan_TOFPulses, Prescan_Accumulations,"
            " Prescan_TICThreshold, Prescan_Continuous, Prescan_Profile, Instrument_Name)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (values["DateStarted"], len(spec.frames), 20000, spec.bin_width_ns, spec.bins,
             0.0, 0.1, 0.1, spec.tof_intensity_type, "", 5000, spec.accumulations, 0, 0, "",
             "SLIM3"),
        )


def _write_frame_params(conn: sqlite3.Connection, spec: SyntheticFile, frame: int) -> None:
    values = {
        3: str(spec.accumulations),
        4: "0",  # MS (Legacy) in the modern table ...
        6: "1",
        7: str(spec.scans),
        11: repr(spec.average_tof_length_ns),
        12: repr(spec.slope),
        13: repr(spec.intercept),
        14: "0", 15: "0", 16: "0", 17: "0", 18: "0", 19: "0",
        20: "0",
    }
    if not spec.legacy_only:
        conn.executemany(
            "INSERT INTO Frame_Params (FrameNum, ParamID, ParamValue) VALUES (?, ?, ?)",
            [(frame, pid, value) for pid, value in sorted(values.items())],
        )
    if not spec.modern_only:
        conn.execute(
            "INSERT INTO Frame_Parameters (FrameNum, StartTime, Duration, Accumulations,"
            " FrameType, Scans, IMFProfile, TOFLosses, AverageTOFLength, CalibrationSlope,"
            " CalibrationIntercept, a2, b2, c2, d2, e2, f2, Temperature, PressureFront,"
            " PressureBack, MPBitOrder, FragmentationProfile, HighPressureFunnelPressure,"
            " IonFunnelTrapPressure, RearIonFunnelPressure, QuadrupolePressure, ESIVoltage,"
            " FloatVoltage, CalibrationDone, Decoded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame, 0.0, 0.0, spec.accumulations,
             1,  # ... and MS (Regular) in the legacy one, as the sample disagrees
             spec.scans, "", 0.0, spec.average_tof_length_ns, spec.slope, spec.intercept,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, None, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1, 0),
        )


def _write_frame_scans(
    conn: sqlite3.Connection,
    spec: SyntheticFile,
    frame: int,
    amplitude: int,
    store_all_scans: bool,
    explicit_zeros: bool,
    first_scan: int,
) -> None:
    rows = []
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
        stream_bins = sorted(stream)
        stream_values = [stream[b] for b in stream_bins]

        bin_index = np.array(bins_list, dtype=np.int64)
        intensity = np.array(values_list, dtype=spec.dtype)
        blob = encode_intensities(
            np.array(stream_bins, dtype=np.int64),
            np.array(stream_values, dtype=spec.dtype),
            spec.dtype,
        )

        if intensity.size:
            argmax = int(np.argmax(intensity))
            bpi = float(intensity[argmax])
            bpi_mz = _mz_of_bin(float(bin_index[argmax]), spec.bin_width_ns, spec.slope,
                                spec.intercept)
            tic = float(intensity.sum())
        else:
            bpi, bpi_mz, tic = 0.0, 0.0, 0.0
        non_zero_count = int(intensity.size + zeros)

        spec.scan_rows[(frame, scan)] = SyntheticScan(
            frame=frame, scan=scan, bin_index=bin_index, intensity=intensity,
            non_zero_count=non_zero_count, bpi=bpi, bpi_mz=bpi_mz, tic=tic, blob=blob,
        )
        rows.append((frame, scan, non_zero_count, int(bpi), bpi_mz, int(tic), blob))

    conn.executemany(
        "INSERT INTO Frame_Scans (FrameNum, ScanNum, NonZeroCount, BPI, BPI_MZ, TIC,"
        " Intensities) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
