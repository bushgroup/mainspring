"""`UimfWriter`: creating a UIMF file, in two phases per frame.

mainspring reads UIMF files; this module is the one place it writes them, and it exists
because the lab's acquisition software has to hand PNNL's acquisition console a file
that already has its schema and its parameters in it. The console opens that file
read-write **without creating it** and inserts `Frame_Scans` rows only, one transaction
per batch, never touching the journal mode. So the division is: this writer owns the
schema, `Global_Params`, `Frame_Param_Keys`, `Frame_Params` and the legacy twins; the
console owns the scans; and whatever journal mode the creator sets is the mode the
console writes under.

**Two phases per frame.** `add_frame` writes what is known before an acquisition starts
-- `Scans`, `Accumulations`, `FrameType`, the calibration, the start time, and which
method frame and repetition the frame belongs to. `finalise_frame` writes what only the
client knows afterwards: the duration, and a **completion marker**. The marker is needed
because nothing else in the file says a frame is finished -- the console publishes
`finished` before its own writer has drained, and a frame stores only the scans that had
signal, so neither the row count nor the last scan number can be compared against
anything (lab record, task 16).

**A frame that was never finalised is provisional, not corrupt.** The acquisition PC has
no UPS, so a run can end mid-write. A file created here is in WAL mode and the console
commits with `synchronous=0`; a power loss therefore loses recent commits rather than
the database. If the lost commit is a completion marker, that frame reads as provisional
for ever, which is the honest answer: it may indeed be short. `UimfFile.is_provisional`
reads the marker on files this writer created and falls back to its file-age heuristic
on everything else, which is every file PNNL's own writers produce.

**Two kinds of file, one writer.** The acquisition client writes a raw file with one
frame per ion mobility experiment (`Accumulations` = 1, the console filling the scans)
and folds those into a summed companion in today's shape (`Accumulations` = A). The
fold is the client's; `write_scans` and `write_sparse_frame` are what it writes the
companion with.

    from mainspring.uimf import FrameSpec, GlobalSpec, UimfWriter

    with UimfWriter(path, GlobalSpec(bins=114688, detector_bits=14)) as writer:
        writer.add_frame(FrameSpec(scans=5000, method_frame=1, repetition=1))
        ...                                    # the console appends Frame_Scans
        writer.finalise_frame(1, duration_s=0.645)
"""

from __future__ import annotations

import contextlib
import datetime
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

import numpy as np

from .calib import Calibration
from .decode import dtype_for, encode_intensities
from .frame import SparseFrame

__all__ = [
    "CUSTOM_PARAM_ID_BASE",
    "DETECTOR_BITS",
    "FRAME_COMPLETE",
    "FRAME_KEYS",
    "GLOBAL_KEYS",
    "METHOD_FRAME",
    "REPETITION",
    "REPETITIONS",
    "WRITER_BUSY_TIMEOUT_MS",
    "WRITER_STAMP",
    "FrameSpec",
    "GlobalSpec",
    "ParamDef",
    "UimfWriter",
]

WRITER_BUSY_TIMEOUT_MS = 5000
"""How long a parameter write waits for the console's lock. Twenty times the reader's,
and for the opposite reason: a stale view costs nothing and a lost completion marker
costs a frame its provenance, so this side waits rather than gives up. The console's own
transactions are one batch of scans long, so the wait is never anywhere near this."""

CUSTOM_PARAM_ID_BASE = 1000
"""Where mainspring's own parameter IDs start.

PNNL's `FrameParamKeyType` reaches 52 and `GlobalParamKeyType` reaches 22, and both
grow. UIMF-Library maps an ID it does not know to `Unknown` and *skips the row with a
warning* rather than failing, so a custom key costs a downstream PNNL tool nothing --
but an ID PNNL later assigns to something else would be read as that something else.
A thousand is far enough away to be safe and small enough to read."""

# The four things the viewer and the acquisition client's fold need that PNNL's
# parameter set has no name for. They are prefixed because `Frame_Param_Keys` is keyed
# by name as well as by ID: UIMF-Library resolves an unrecognised *name* by parsing it
# against its enum, so a bare `Repetition` would silently become a standard parameter
# the day PNNL adds one.
METHOD_FRAME = "MainspringMethodFrame"
"""Frame parameter: which method frame this frame is a repetition of, 1-based."""

REPETITION = "MainspringRepetition"
"""Frame parameter: which repetition of that method frame this is, 1-based."""

REPETITIONS = "MainspringRepetitions"
"""Frame parameter: how many repetitions the method asked for. A method frame with
fewer than this many frames in the file was cut short."""

FRAME_COMPLETE = "MainspringFrameComplete"
"""Frame parameter: the completion marker, written by `finalise_frame` and by nothing
else. Absent or zero means the frame may still be growing."""

WRITER_STAMP = "MainspringWriter"
"""Global parameter: what wrote this file. Its presence is what makes the absence of a
completion marker meaningful -- a file without this stamp is one of PNNL's, where every
frame is finished and none of them says so."""

DETECTOR_BITS = "MainspringDetectorBits"
"""Global parameter: the digitizer's bit depth. Not in PNNL's parameter set at all, and
not derivable from anything that is; the viewer's "% of full scale" readout falls back
to a user setting when a file does not carry it (lab record, task 06)."""


@dataclass(frozen=True)
class ParamDef:
    """One row of `Global_Params` or `Frame_Param_Keys`: its ID, name, type and text."""

    param_id: int
    name: str
    data_type: str
    description: str | None = None


def _defs(rows: Iterable[tuple]) -> dict[str, ParamDef]:
    return {row[1]: ParamDef(*row) for row in rows}


# PNNL's `GlobalParamKeyType`, read off UIMF-Library (2026-09-10), plus mainspring's
# own two. The `System.*` type names and the descriptions are the library's where it
# states one and the files' where it does not.
GLOBAL_KEYS = _defs((
    (1, "InstrumentName", "System.String", "Instrument name"),
    (2, "DateStarted", "System.String", "Time that the data acquisition started"),
    (3, "NumFrames", "System.Int32", "Number of frames in the dataset"),
    (4, "TimeOffset", "System.Int32",
     "Time offset from 0 (in nanoseconds). All bin numbers must be offset by this amount"),
    (5, "BinWidth", "System.Double", "Width of TOF bins (in ns)"),
    (6, "Bins", "System.Int32", "Total number of TOF bins in frame"),
    (7, "TOFCorrectionTime", "System.Single", "TOF correction time"),
    (8, "TOFIntensityType", "System.String",
     "Data type of intensity in each TOF record (ADC is int, TDC is short, FOLDED is float)"),
    (9, "DatasetType", "System.String", "Type of dataset (HMS, HMSn, or HMS-HMSn)"),
    (10, "PrescanTOFPulses", "System.Int32", "Prescan TOF pulses"),
    (11, "PrescanAccumulations", "System.Int32", "Number of prescan accumulations"),
    (12, "PrescanTICThreshold", "System.Int32", "Prescan TIC threshold"),
    (13, "PrescanContinuous", "System.Int32", "Prescan continuous mode"),
    (14, "PrescanProfile", "System.String", "Prescan profile"),
    (15, "InstrumentClass", "System.Int32", "Instrument class: 0 for TOF; 1 for ppm bin-based"),
    (16, "PpmBinBasedStartMz", "System.Double",
     "Starting m/z; used only when InstrumentClass is 1"),
    (17, "PpmBinBasedEndMz", "System.Double", "Ending m/z; used only when InstrumentClass is 1"),
    (18, "DriftTubeLength", "System.Double", "Drift tube length"),
    (19, "DriftGas", "System.String", "Drift gas"),
    (20, "ADCName", "System.String", "Name of the analog to digital converter"),
    (21, "OneBasedDriftScans", "System.Int32", "1 if the first drift scan is scan 1, not scan 0"),
    (22, "AcquisitionMethod", "System.String", "Acquisition method"),
    (CUSTOM_PARAM_ID_BASE + 1, WRITER_STAMP, "System.String",
     "Software that created this file and wrote its parameters"),
    (CUSTOM_PARAM_ID_BASE + 2, DETECTOR_BITS, "System.Int32",
     "Digitizer bit depth, for reporting intensity as a fraction of full scale"),
))

# PNNL's `FrameParamKeyType`, same source and same rule. One deliberate divergence:
# `CalibrationSlope` is described the way the files describe it rather than the way
# UIMF-Library's current source does. The library says "k is slope / 10000"; every file
# we have stores K itself, which is what its own `BPI_MZ` column confirms to eleven
# decimal places (lab record, task 01), so the library's text would be a false statement
# about the number in the row. `MassCalibrationCoefficientd2` keeps the library's own
# "db2" typo, which our sample carries too.
FRAME_KEYS = _defs((
    (1, "StartTimeMinutes", "System.Double", "Start time of frame, in minutes"),
    (2, "DurationSeconds", "System.Double", "Frame duration, in seconds"),
    (3, "Accumulations", "System.Int32",
     "Number of collected and summed acquisitions in a frame"),
    (4, "FrameType", "System.Int32",
     "Frame Type: 0=MS (Legacy); 1=MS (Regular); 2=MS/MS (Frag); 3=Calibration; 4=Prescan"),
    (5, "Decoded", "System.Int32",
     "Tracks whether frame has been decoded: 0 for non-multiplexed or encoded; 1 if decoded"),
    (6, "CalibrationDone", "System.Int32",
     "Tracks whether frame has been calibrated: 1 if calibrated"),
    (7, "Scans", "System.Int32", "Number of TOF scans in a frame"),
    (8, "MultiplexingEncodingSequence", "System.String",
     "The name of the sequence used to encode the data when acquiring multiplexed data"),
    (9, "MPBitOrder", "System.Int32",
     "Multiplexing bit order; Determines size of the bit sequence"),
    (10, "TOFLosses", "System.Int32",
     "Number of TOF Losses (lost/skipped scans due to I/O problems)"),
    (11, "AverageTOFLength", "System.Double",
     "Average time between TOF trigger pulses, in nanoseconds"),
    (12, "CalibrationSlope", "System.Double", "Calibration slope, k0"),
    (13, "CalibrationIntercept", "System.Double", "Calibration intercept, t0"),
    (14, "MassCalibrationCoefficienta2", "System.Double",
     "a2 parameter for residual mass error correction; ResidualMassError ="
     " a2*t + b2*t^3 + c2*t^5 + d2*t^7 + e2*t^9 + f2*t^11"),
    (15, "MassCalibrationCoefficientb2", "System.Double",
     "b2 parameter for residual mass error correction"),
    (16, "MassCalibrationCoefficientc2", "System.Double",
     "c2 parameter for residual mass error correction"),
    (17, "MassCalibrationCoefficientd2", "System.Double",
     "db2 parameter for residual mass error correction"),
    (18, "MassCalibrationCoefficiente2", "System.Double",
     "e2 parameter for residual mass error correction"),
    (19, "MassCalibrationCoefficientf2", "System.Double",
     "f2 parameter for residual mass error correction"),
    (20, "AmbientTemperature", "System.Single", "Ambient temperature, in Celsius"),
    (21, "VoltHVRack1", "System.Single", "Volt hv rack 1"),
    (22, "VoltHVRack2", "System.Single", "Volt hv rack 2"),
    (23, "VoltHVRack3", "System.Single", "Volt hv rack 3"),
    (24, "VoltHVRack4", "System.Single", "Volt hv rack 4"),
    (25, "VoltCapInlet", "System.Single", "Capillary Inlet Voltage"),
    (26, "VoltEntranceHPFIn", "System.Single", "HPF In Voltage"),
    (27, "VoltEntranceHPFOut", "System.Single", "HPF Out Voltage"),
    (28, "VoltEntranceCondLmt", "System.Single", "Entrance Cond Limit Voltage"),
    (29, "VoltTrapOut", "System.Single", "Trap Out Voltage"),
    (30, "VoltTrapIn", "System.Single", "Trap In Voltage"),
    (31, "VoltJetDist", "System.Single", "Jet Disruptor Voltage"),
    (32, "VoltQuad1", "System.Single", "Fragmentation Quadrupole Voltage 1"),
    (33, "VoltCond1", "System.Single", "Fragmentation Conductance Voltage 1"),
    (34, "VoltQuad2", "System.Single", "Fragmentation Quadrupole Voltage 2"),
    (35, "VoltCond2", "System.Single", "Fragmentation Conductance Voltage 2"),
    (36, "VoltIMSOut", "System.Single", "IMS Out Voltage"),
    (37, "VoltExitHPFIn", "System.Single", "HPF In Voltage"),
    (38, "VoltExitHPFOut", "System.Single", "HPF Out Voltage"),
    (39, "VoltExitCondLmt", "System.Single", "Exit Cond Limit Voltage"),
    (40, "PressureFront", "System.Single", "Pressure at front of Drift Tube"),
    (41, "PressureBack", "System.Single", "Pressure at back of Drift Tube"),
    (42, "HighPressureFunnelPressure", "System.Single", "High pressure funnel pressure"),
    (43, "IonFunnelTrapPressure", "System.Single", "Ion funnel trap pressure"),
    (44, "RearIonFunnelPressure", "System.Single", "Rear ion funnel pressure"),
    (45, "QuadrupolePressure", "System.Single", "Quadrupole pressure"),
    (46, "ESIVoltage", "System.Single", "ESI voltage"),
    (47, "FloatVoltage", "System.Single", "Float voltage"),
    (48, "FragmentationProfile", "System.String",
     "Voltage profile used in fragmentation (array of doubles, converted to an array of"
     " bytes, then stored as a Base 64 encoded string)"),
    (49, "ScanNumFirst", "System.Int32", "First scan"),
    (50, "ScanNumLast", "System.Int32", "Last scan"),
    (51, "PressureUnits", "System.String", "Units for pressure"),
    (52, "DriftTubeTemperature", "System.Single", "Drift tube temperature, in Celsius"),
    (CUSTOM_PARAM_ID_BASE + 1, METHOD_FRAME, "System.Int32",
     "Which method frame this frame is a repetition of, 1-based"),
    (CUSTOM_PARAM_ID_BASE + 2, REPETITION, "System.Int32",
     "Which repetition of that method frame this frame is, 1-based"),
    (CUSTOM_PARAM_ID_BASE + 3, REPETITIONS, "System.Int32",
     "How many repetitions of that method frame were asked for"),
    (CUSTOM_PARAM_ID_BASE + 4, FRAME_COMPLETE, "System.Int32",
     "1 once the client has finished writing this frame; absent while it may still grow"),
))

# The schema, statement for statement as a 2026 SLIMPHONY acquisition carries it. Every
# index real files have is here: the console inserts under them, and the unique index on
# `Frame_Scans` is what stops a re-sent batch becoming duplicate rows.
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

# The legacy fixed-column name for each modern key that has one.
_LEGACY_FRAME_COLUMNS = {
    "StartTimeMinutes": "StartTime",
    "DurationSeconds": "Duration",
    "AmbientTemperature": "Temperature",
    "MassCalibrationCoefficienta2": "a2",
    "MassCalibrationCoefficientb2": "b2",
    "MassCalibrationCoefficientc2": "c2",
    "MassCalibrationCoefficientd2": "d2",
    "MassCalibrationCoefficiente2": "e2",
    "MassCalibrationCoefficientf2": "f2",
}

# The keys every frame gets, whatever the client asks for. Declared in
# `Frame_Param_Keys` when the file is created so that a mid-acquisition reader sees a
# complete dictionary; `extra` adds to it as it is used.
_DEFAULT_FRAME_KEYS = (
    "StartTimeMinutes", "DurationSeconds", "Accumulations", "FrameType",
    "CalibrationDone", "Scans", "AverageTOFLength", "CalibrationSlope",
    "CalibrationIntercept", "MassCalibrationCoefficienta2",
    "MassCalibrationCoefficientb2", "MassCalibrationCoefficientc2",
    "MassCalibrationCoefficientd2", "MassCalibrationCoefficiente2",
    "MassCalibrationCoefficientf2", METHOD_FRAME, REPETITION, REPETITIONS,
    FRAME_COMPLETE,
)

# `Frame_Parameters` columns `finalise_frame` may update, after that mapping. Anything
# else it is given -- mainspring's own keys, and most of PNNL's later ones -- has no
# column in the legacy table and lives only in the modern one.
_LEGACY_FRAME_FIELDS = frozenset({
    "StartTime", "Duration", "Accumulations", "FrameType", "Scans", "TOFLosses",
    "AverageTOFLength", "CalibrationSlope", "CalibrationIntercept", "a2", "b2", "c2",
    "d2", "e2", "f2", "Temperature", "PressureFront", "PressureBack", "MPBitOrder",
    "HighPressureFunnelPressure", "IonFunnelTrapPressure", "RearIonFunnelPressure",
    "QuadrupolePressure", "ESIVoltage", "FloatVoltage", "CalibrationDone", "Decoded",
})


@dataclass(frozen=True)
class GlobalSpec:
    """The dataset-wide parameters, as a client states them before the first frame.

    `bins` and `bin_width_ns` set the TOF axis; `tof_intensity_type` fixes the element
    type of every blob in the file and cannot change afterwards. `time_offset_ns` is
    recorded and, as everywhere else in mainspring, is **not** part of the calibration.
    `detector_bits` is mainspring's own: see `DETECTOR_BITS`.

    **Set `prescan_tof_pulses` to the frame's `Scans`.** It reads like a prescan
    setting and nothing in mainspring uses it, but FALKOR writes the method's scan count
    there and PNNL's `uimfpy` takes its `num_scans` from it and from nowhere else, so a
    file that leaves it at zero tells that tool it has no scans (lab record, task 16).

    `extra` carries any other parameter PNNL's `GlobalParamKeyType` names, by that name;
    an unrecognised name is refused rather than given an invented ID, because an ID this
    repository made up is one a future UIMF-Library may assign to something else.
    """

    bins: int
    bin_width_ns: float = 1.0
    instrument_name: str = ""
    date_started: str = ""
    tof_intensity_type: str = "ADC"
    time_offset_ns: int = 0
    dataset_type: str = ""
    prescan_tof_pulses: int = 0
    prescan_accumulations: int = 0
    detector_bits: int | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.bins) <= 0:
            raise ValueError(f"bins must be positive, not {self.bins}")
        if float(self.bin_width_ns) <= 0:
            raise ValueError(f"bin_width_ns must be positive, not {self.bin_width_ns}")
        dtype_for(self.tof_intensity_type)  # raises on an element type nothing can decode
        if self.detector_bits is not None and not 1 <= int(self.detector_bits) <= 64:
            raise ValueError(f"detector_bits outside 1..64: {self.detector_bits}")
        _check_extra(self.extra, GLOBAL_KEYS, "global")


@dataclass(frozen=True)
class FrameSpec:
    """One frame's parameters, as they are known *before* it is acquired.

    Everything here is a property of the experiment the client is about to run, which is
    why the grouping is here and not in `finalise_frame`: a frame cut short by a power
    loss still says which method frame and repetition it was, and that is exactly the
    frame whose provenance is worth having.

    `frame_type` defaults to 1, "MS (Regular)". Real files disagree with themselves --
    our sample says 0 in the modern table and 1 in the legacy one -- and 0 is documented
    as the legacy value, so a downstream tool filtering for MS1 frames looks for 1. This
    writer emits one unambiguous value in both tables; `legacy_frame_type` overrides the
    legacy one and exists so that the test fixture can reproduce the disagreement real
    files have, which is a reader test and not a thing to write on purpose.

    `extra` carries any other parameter PNNL's `FrameParamKeyType` names; see
    `GlobalSpec.extra` for why an unknown name is refused.
    """

    scans: int
    accumulations: int = 1
    frame_type: int = 1
    calibration_slope: float = 0.0
    calibration_intercept: float = 0.0
    average_tof_length_ns: float = 0.0
    start_time_minutes: float = 0.0
    method_frame: int | None = None
    repetition: int | None = None
    repetitions: int | None = None
    legacy_frame_type: int | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.scans) <= 0:
            raise ValueError(f"scans must be positive, not {self.scans}")
        if int(self.accumulations) < 1:
            raise ValueError(f"accumulations must be at least 1, not {self.accumulations}")
        grouped = (self.method_frame is not None, self.repetition is not None)
        if any(grouped) and not all(grouped):
            raise ValueError(
                "method_frame and repetition go together: a repetition with no method"
                " frame cannot be grouped, and a method frame with no repetition cannot"
                " be ordered"
            )
        for name in ("method_frame", "repetition", "repetitions"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} is 1-based, not {value}")
        if self.repetitions is not None and self.repetition is not None:
            if int(self.repetition) > int(self.repetitions):
                raise ValueError(
                    f"repetition {self.repetition} of {self.repetitions} asked for"
                )
        _check_extra(self.extra, FRAME_KEYS, "frame")

    @property
    def calibration_done(self) -> bool:
        """Whether this frame carries a calibration that can produce an m/z axis."""
        return float(self.calibration_slope) > 0.0


class UimfWriter:
    """A UIMF file being created: the schema, the parameters, and optionally the scans.

    The connection is opened once and held for the life of the writer, which is the one
    place mainspring does that. The reader's rule -- open, one query, close -- protects a
    writer from a reader; here we *are* the writer, and the client holds the file open
    for the length of an acquisition anyway. What the rule becomes on this side is that
    **no transaction outlives a call**: every method below opens one, commits it and
    returns, so the writer is never inside a transaction while the client waits on
    hardware and the console tries to insert a batch.

    Creating only. A file that already exists is refused unless `overwrite` says so:
    silently truncating a previous acquisition is not a thing to do by accident.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        globals_: GlobalSpec,
        *,
        tables: str = "both",
        journal_mode: str = "wal",
        overwrite: bool = False,
        busy_timeout_ms: int = WRITER_BUSY_TIMEOUT_MS,
    ) -> None:
        if tables not in ("both", "modern", "legacy"):
            raise ValueError(f"tables must be both, modern or legacy, not {tables!r}")
        self.path = os.path.abspath(os.fspath(path))
        self.globals = globals_
        self.modern = tables in ("both", "modern")
        self.legacy = tables in ("both", "legacy")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._frames: list[int] = []
        self._declared: set[str] = set()
        self._closed = False

        if os.path.exists(self.path):
            if not overwrite:
                raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
            _remove_database(self.path)

        self._conn = sqlite3.connect(
            self.path, timeout=self.busy_timeout_ms / 1000.0, isolation_level=None
        )
        try:
            self._conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            mode = self._conn.execute(f"PRAGMA journal_mode = {journal_mode}").fetchone()[0]
            if str(mode).lower() != journal_mode.lower():
                raise OSError(
                    f"{self.path}: asked for journal_mode={journal_mode}, got {mode}."
                    " A network or read-only filesystem will do this, and the console"
                    " inherits whatever mode the file ends up in."
                )
            # Parameters are a few dozen short rows per frame, so a full fsync per commit
            # would cost nothing measurable -- but the console writes the same file with
            # `synchronous=0`, and `synchronous` is a property of a connection rather
            # than of the file, so ours binds only our own writes. NORMAL is WAL's
            # recommended setting: a power loss can lose the most recent commits, and the
            # most recent commit here is a completion marker, whose loss reads as
            # "provisional" rather than as damage.
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._create_schema()
        except BaseException:
            self._conn.close()
            raise

    # --- lifecycle --------------------------------------------------------------------

    def __enter__(self) -> "UimfWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r}, frames={len(self._frames)})"

    def close(self) -> None:
        """Checkpoint the write-ahead log and let go. Idempotent; `__exit__` calls it.

        The checkpoint is what leaves a single file behind rather than a file and a
        `-wal` beside it, which matters because the finished file is the thing a user
        copies off the instrument.
        """
        if self._closed:
            return
        self._closed = True
        try:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self._conn.close()

    @property
    def frames(self) -> tuple[int, ...]:
        """The frame numbers added so far, in the order they were added."""
        return tuple(self._frames)

    # --- the two phases ---------------------------------------------------------------

    def add_frame(self, spec: FrameSpec, frame: int | None = None) -> int:
        """Write a frame's parameters before it is acquired; return its frame number.

        `frame` defaults to one past the highest so far, which is how a client that
        acquires in order never has to count. `NumFrames` is brought up to date in the
        same transaction, so a reader that trusts that number and a reader that counts
        `Frame_Params` rows agree at every commit boundary.
        """
        self._require_open()
        if frame is None:
            frame = (max(self._frames) + 1) if self._frames else 1
        frame = int(frame)
        if frame in self._frames:
            raise ValueError(f"frame {frame} has already been added")

        values = self._frame_values(spec)
        with self._transaction():
            if self.modern:
                self._declare_frame_keys(values)
                self._conn.executemany(
                    "INSERT INTO Frame_Params (FrameNum, ParamID, ParamValue) VALUES (?, ?, ?)",
                    _by_param_id(frame, values, FRAME_KEYS),
                )
            if self.legacy:
                self._insert_legacy_frame(frame, spec, values)
            self._set_frame_count(len(self._frames) + 1)
        self._frames.append(frame)
        return frame

    def finalise_frame(
        self,
        frame: int,
        duration_s: float | None = None,
        *,
        complete: bool = True,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Write what is only known once the frame has been acquired.

        `complete` writes the completion marker; pass `False` to record a duration on a
        frame known to have been cut short, which leaves it provisional. Calling this
        twice is allowed and overwrites -- a client that finalises, folds, and then has
        something to add should not have to remember which call it is on.
        """
        self._require_open()
        frame = int(frame)
        values: dict[str, str] = {}
        if duration_s is not None:
            values["DurationSeconds"] = _as_text(duration_s, FRAME_KEYS["DurationSeconds"])
        if complete:
            values[FRAME_COMPLETE] = "1"
        for name, value in (extra or {}).items():
            definition = FRAME_KEYS.get(name)
            if definition is None:
                raise ValueError(_unknown_key_message(name, "frame"))
            values[name] = _as_text(value, definition)
        if not values:
            return

        with self._transaction():
            if self.modern:
                if not self._conn.execute(
                    "SELECT 1 FROM Frame_Params WHERE FrameNum = ? LIMIT 1", (frame,)
                ).fetchone():
                    raise KeyError(f"frame {frame} has no parameters to finalise")
                self._declare_frame_keys(values)
                self._conn.executemany(
                    "INSERT INTO Frame_Params (FrameNum, ParamID, ParamValue) VALUES (?, ?, ?)"
                    " ON CONFLICT(FrameNum, ParamID)"
                    " DO UPDATE SET ParamValue = excluded.ParamValue",
                    _by_param_id(frame, values, FRAME_KEYS),
                )
            if self.legacy:
                self._update_legacy_frame(frame, values)

    # --- scans, for the fold and for the fixture ---------------------------------------

    def write_scans(
        self,
        frame: int,
        scans: Iterable[tuple[int, np.ndarray, np.ndarray]],
    ) -> int:
        """Insert `Frame_Scans` rows for a frame; return how many were written.

        `scans` yields `(scan number, bin index, intensity)`, ascending in bin within a
        scan. The four summary columns are computed here rather than asked for, because
        they are the file's own ground truth and a client that could get them wrong would
        turn `uimf-info --verify` into a test of the client instead of the decoder. A
        zero passed in `intensity` is written into the stream and counted in
        `NonZeroCount`, exactly as real writers do -- which is why that column is an
        upper bound on the number of points rather than a count of them.

        `BPI_MZ` is the m/z UIMF-Library's calibration puts the base peak's bin at, which
        is what the column is for and what our own files hold. PNNL's SA220P console
        stores the base peak's **bin index** in it instead -- read off its source, not
        yet seen on a file -- so a raw acquisition and the summed companion this writes
        will disagree about what that column means. Neither the reader nor the viewer
        uses it; `uimf-info --verify` does (lab record, task 16).

        Nothing here is on the acquisition path: during a real run it is the console that
        fills this table. This is what the fold writes the summed companion with, and
        what the test fixture writes everything with.
        """
        self._require_open()
        frame = int(frame)
        dtype = dtype_for(self.globals.tof_intensity_type)
        calibration = self._calibration_for(frame)
        rows = [self._scan_row(frame, scan, bin_index, intensity, dtype, calibration)
                for scan, bin_index, intensity in scans]
        if not rows:
            return 0
        with self._transaction():
            self._conn.executemany(
                "INSERT INTO Frame_Scans (FrameNum, ScanNum, NonZeroCount, BPI, BPI_MZ,"
                " TIC, Intensities) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def write_sparse_frame(
        self,
        frame: int,
        data: SparseFrame,
        *,
        store_empty_scans: bool = False,
    ) -> int:
        """Write a `SparseFrame` as this frame's scans; return how many rows were written.

        The other half of the round trip the reader makes, and what the fold writes a
        summed frame with once `sum_frames` has produced it. Empty scans are left out by
        default, which is what the SLIMPHONY writer does and what makes a 5000-scan frame
        1656 rows.
        """
        def stored() -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
            for scan in range(data.scans):
                bin_index, intensity = data.scan(scan)
                if bin_index.size or store_empty_scans:
                    yield scan, bin_index, intensity

        return self.write_scans(frame, stored())

    # --- internals ---------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError(f"{self.path}: writer is closed")

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One immediate transaction, committed on the way out, and never longer.

        `BEGIN IMMEDIATE` takes the write lock up front rather than half way through, so
        a clash with the console's batch is a wait here instead of a rollback later.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    def _create_schema(self) -> None:
        spec = self.globals
        with self._transaction():
            for statement in _SCANS_SCHEMA:
                self._conn.execute(statement)
            if self.modern:
                for statement in _MODERN_SCHEMA:
                    self._conn.execute(statement)
            if self.legacy:
                for statement in _LEGACY_SCHEMA:
                    self._conn.execute(statement)

            values = self._global_values()
            if self.modern:
                self._conn.executemany(
                    "INSERT INTO Global_Params (ParamID, ParamName, ParamValue,"
                    " ParamDataType, ParamDescription) VALUES (?, ?, ?, ?, ?)",
                    [(GLOBAL_KEYS[name].param_id, name, text,
                      GLOBAL_KEYS[name].data_type, GLOBAL_KEYS[name].description)
                     for name, text in sorted(values.items(),
                                              key=lambda kv: GLOBAL_KEYS[kv[0]].param_id)],
                )
                self._declare_frame_keys(_DEFAULT_FRAME_KEYS)
            if self.legacy:
                self._conn.execute(
                    "INSERT INTO Global_Parameters (DateStarted, NumFrames, TimeOffset,"
                    " BinWidth, Bins, TOFCorrectionTime, FrameDataBlobVersion,"
                    " ScanDataBlobVersion, TOFIntensityType, DatasetType,"
                    " Prescan_TOFPulses, Prescan_Accumulations, Prescan_TICThreshold,"
                    " Prescan_Continuous, Prescan_Profile, Instrument_Name)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (values["DateStarted"], 0, int(spec.time_offset_ns),
                     float(spec.bin_width_ns), int(spec.bins), 0.0, 0.1, 0.1,
                     spec.tof_intensity_type, spec.dataset_type,
                     int(spec.prescan_tof_pulses), int(spec.prescan_accumulations),
                     0, 0, "", spec.instrument_name),
                )

    def _declare_frame_keys(self, names: Iterable[str]) -> None:
        """Make sure `Frame_Param_Keys` has a row for each of these, and remember it.

        `Frame_Param_Keys` is the file's own dictionary, and real files carry only the
        keys they use -- 14 rows in our sample, 33 in a PNNL one. So the dictionary grows
        with use rather than being written out whole at creation, and the writer keeps a
        set of what it has declared so that adding the five-thousandth frame is not
        fourteen no-op inserts.
        """
        missing = [name for name in names if name not in self._declared]
        if not missing:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO Frame_Param_Keys (ParamID, ParamName, ParamDataType,"
            " ParamDescription) VALUES (?, ?, ?, ?)",
            [(FRAME_KEYS[name].param_id, name, FRAME_KEYS[name].data_type,
              FRAME_KEYS[name].description)
             for name in sorted(missing, key=lambda n: FRAME_KEYS[n].param_id)],
        )
        self._declared.update(missing)

    def _global_values(self) -> dict[str, str]:
        spec = self.globals
        values: dict[str, object] = {
            "InstrumentName": spec.instrument_name,
            "DateStarted": spec.date_started or _now_as_files_write_it(),
            "NumFrames": 0,
            "TimeOffset": int(spec.time_offset_ns),
            "BinWidth": float(spec.bin_width_ns),
            "Bins": int(spec.bins),
            "TOFIntensityType": spec.tof_intensity_type,
            "DatasetType": spec.dataset_type,
            "PrescanTOFPulses": int(spec.prescan_tof_pulses),
            "PrescanAccumulations": int(spec.prescan_accumulations),
            WRITER_STAMP: _writer_stamp(),
        }
        if spec.detector_bits is not None:
            values[DETECTOR_BITS] = int(spec.detector_bits)
        values.update(spec.extra)
        return {name: _as_text(value, GLOBAL_KEYS[name]) for name, value in values.items()}

    def _frame_values(self, spec: FrameSpec) -> dict[str, str]:
        values: dict[str, object] = {
            "StartTimeMinutes": float(spec.start_time_minutes),
            # Written now, as a zero, so that the console's own timing survives. Its
            # `update_timing_information` is an `UPDATE Frame_Params ... WHERE FrameNum
            # = ? AND ParamID = ?` wrapped in a `catch (...) {}`, so against a row that
            # does not exist yet it changes nothing and says nothing. The duration it
            # would write is counted off the card's own sample clock, which is a better
            # number than the client's wall clock, so the row is here to receive it and
            # `finalise_frame` leaves it alone unless it is given one.
            "DurationSeconds": 0.0,
            "Accumulations": int(spec.accumulations),
            "FrameType": int(spec.frame_type),
            "CalibrationDone": 1 if spec.calibration_done else 0,
            "Scans": int(spec.scans),
            "AverageTOFLength": float(spec.average_tof_length_ns),
            "CalibrationSlope": float(spec.calibration_slope),
            "CalibrationIntercept": float(spec.calibration_intercept),
        }
        for suffix in ("a2", "b2", "c2", "d2", "e2", "f2"):
            values[f"MassCalibrationCoefficient{suffix}"] = 0.0
        if spec.method_frame is not None:
            values[METHOD_FRAME] = int(spec.method_frame)
        if spec.repetition is not None:
            values[REPETITION] = int(spec.repetition)
        if spec.repetitions is not None:
            values[REPETITIONS] = int(spec.repetitions)
        values.update(spec.extra)
        return {name: _as_text(value, FRAME_KEYS[name]) for name, value in values.items()}

    def _insert_legacy_frame(
        self, frame: int, spec: FrameSpec, values: Mapping[str, str]
    ) -> None:
        """The legacy row, from the same values, with the columns it has and no more.

        The legacy table is fixed-column, so mainspring's own parameters and most of
        PNNL's later ones have nowhere to go in it. That is not a loss: a tool that reads
        only this table is a 2011-era tool, and the grouping and the completion marker
        exist for tools that are not.
        """
        legacy_type = (
            spec.frame_type if spec.legacy_frame_type is None else spec.legacy_frame_type
        )
        self._conn.execute(
            "INSERT INTO Frame_Parameters (FrameNum, StartTime, Duration, Accumulations,"
            " FrameType, Scans, IMFProfile, TOFLosses, AverageTOFLength, CalibrationSlope,"
            " CalibrationIntercept, a2, b2, c2, d2, e2, f2, Temperature, PressureFront,"
            " PressureBack, MPBitOrder, FragmentationProfile, HighPressureFunnelPressure,"
            " IonFunnelTrapPressure, RearIonFunnelPressure, QuadrupolePressure, ESIVoltage,"
            " FloatVoltage, CalibrationDone, Decoded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (frame, float(spec.start_time_minutes), 0.0, int(spec.accumulations),
             int(legacy_type), int(spec.scans), "", 0.0,
             float(spec.average_tof_length_ns), float(spec.calibration_slope),
             float(spec.calibration_intercept), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             _legacy_float(values, "AmbientTemperature"),
             _legacy_float(values, "PressureFront"), _legacy_float(values, "PressureBack"),
             0, None,
             _legacy_float(values, "HighPressureFunnelPressure"),
             _legacy_float(values, "IonFunnelTrapPressure"),
             _legacy_float(values, "RearIonFunnelPressure"),
             _legacy_float(values, "QuadrupolePressure"),
             _legacy_float(values, "ESIVoltage"), _legacy_float(values, "FloatVoltage"),
             1 if spec.calibration_done else 0, 0),
        )

    def _update_legacy_frame(self, frame: int, values: Mapping[str, str]) -> None:
        columns = {
            _LEGACY_FRAME_COLUMNS.get(name, name): value
            for name, value in values.items()
            if _LEGACY_FRAME_COLUMNS.get(name, name) in _LEGACY_FRAME_FIELDS
        }
        if not columns:
            return
        assignments = ", ".join(f"{column} = ?" for column in columns)
        self._conn.execute(
            f"UPDATE Frame_Parameters SET {assignments} WHERE FrameNum = ?",
            (*(float(value) for value in columns.values()), frame),
        )

    def _set_frame_count(self, count: int) -> None:
        if self.modern:
            self._conn.execute(
                "UPDATE Global_Params SET ParamValue = ? WHERE ParamName = 'NumFrames'",
                (str(int(count)),),
            )
        if self.legacy:
            self._conn.execute("UPDATE Global_Parameters SET NumFrames = ?", (int(count),))

    def _calibration_for(self, frame: int) -> Calibration:
        """The calibration this frame's own parameters state, read back from the file.

        Read rather than remembered: `BPI_MZ` has to be the calibration the file carries
        rather than the one the caller happens to be holding, and on a legacy-only file
        the two tables are not even written by the same code path here.
        """
        if self.modern:
            found = {
                int(param_id): float(value)
                for param_id, value in self._conn.execute(
                    "SELECT ParamID, ParamValue FROM Frame_Params"
                    " WHERE FrameNum = ? AND ParamID IN (?, ?)",
                    (frame, FRAME_KEYS["CalibrationSlope"].param_id,
                     FRAME_KEYS["CalibrationIntercept"].param_id),
                )
            }
            slope = found.get(FRAME_KEYS["CalibrationSlope"].param_id, 0.0)
            intercept = found.get(FRAME_KEYS["CalibrationIntercept"].param_id, 0.0)
        else:
            row = self._conn.execute(
                "SELECT CalibrationSlope, CalibrationIntercept FROM Frame_Parameters"
                " WHERE FrameNum = ?", (frame,),
            ).fetchone()
            slope, intercept = (float(row[0]), float(row[1])) if row else (0.0, 0.0)
        return Calibration(
            slope=slope, intercept=intercept,
            bin_width_ns=float(self.globals.bin_width_ns), done=slope > 0.0,
        )

    def _scan_row(
        self,
        frame: int,
        scan: int,
        bin_index: np.ndarray,
        intensity: np.ndarray,
        dtype: np.dtype,
        calibration: Calibration,
    ) -> tuple:
        bin_index = np.asarray(bin_index, dtype=np.int64)
        intensity = np.asarray(intensity, dtype=dtype)
        if bin_index.size != intensity.size:
            raise ValueError(
                f"frame {frame} scan {scan}: {bin_index.size} bins against"
                f" {intensity.size} intensities"
            )
        if bin_index.size and (bin_index[0] < 0 or bin_index[-1] >= int(self.globals.bins)):
            raise ValueError(
                f"frame {frame} scan {scan}: bins outside 0..{int(self.globals.bins) - 1}"
            )
        if bin_index.size > 1 and np.any(np.diff(bin_index) <= 0):
            raise ValueError(f"frame {frame} scan {scan}: bins are not ascending")

        blob = encode_intensities(bin_index, intensity, dtype)
        if intensity.size:
            argmax = int(np.argmax(intensity))
            bpi = intensity[argmax].item()
            bpi_mz = float(calibration.mz(float(bin_index[argmax])))
            total = intensity.sum().item()
        else:
            bpi, bpi_mz, total = 0, 0.0, 0
        if dtype.kind in "iu":
            bpi, total = int(bpi), int(total)
        return (frame, int(scan), int(bin_index.size), bpi, bpi_mz, total, blob)


def _by_param_id(
    frame: int, values: Mapping[str, str], keys: Mapping[str, ParamDef]
) -> list[tuple[int, int, str]]:
    """`(frame, id, text)` rows in ID order, which is the order files store them in."""
    return [(frame, keys[name].param_id, text)
            for name, text in sorted(values.items(), key=lambda kv: keys[kv[0]].param_id)]


def _check_extra(extra: Mapping[str, object], keys: Mapping[str, ParamDef], what: str) -> None:
    for name in extra:
        if name not in keys:
            raise ValueError(_unknown_key_message(name, what))


def _unknown_key_message(name: str, what: str) -> str:
    return (
        f"{name!r} is not a UIMF {what} parameter. Only the names PNNL's UIMF-Library"
        " knows can be written, because an ID invented here is one the library may later"
        " assign to something else; add it to the key table in mainspring.uimf.writer if"
        " a real file needs it."
    )


def _as_text(value: object, definition: ParamDef) -> str:
    """A parameter value as the TEXT the modern tables store, typed by its own key.

    `repr` on a float rather than `str`: both round trip in Python 3, and `repr` is what
    the values in our sample look like when read back (`129003.607843137`).
    """
    if definition.data_type == "System.Int32":
        return str(int(value))
    if definition.data_type in ("System.Double", "System.Single"):
        return repr(float(value))
    return "" if value is None else str(value)


def _legacy_float(values: Mapping[str, str], name: str) -> float:
    try:
        return float(values[name])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _now_as_files_write_it() -> str:
    """Now, in the US-locale string every UIMF writer we have seen puts in `DateStarted`.

    Built by hand rather than by `strftime`, whose no-padding flag is `%#m` on Windows
    and `%-m` everywhere else, and whose `%p` follows the C locale.
    """
    now = datetime.datetime.now()
    hour = now.hour % 12 or 12
    meridiem = "AM" if now.hour < 12 else "PM"
    return f"{now.month}/{now.day}/{now.year} {hour}:{now.minute:02d}:{now.second:02d} {meridiem}"


def _writer_stamp() -> str:
    from .. import __version__

    return f"mainspring {__version__}"


def _remove_database(path: str) -> None:
    """Remove a database and the sidecars a journal or a write-ahead log leaves behind.

    Deleting the database alone would leave a `-wal` belonging to it, which SQLite would
    then read back into the new file.
    """
    for suffix in ("", "-wal", "-shm", "-journal"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path + suffix)
