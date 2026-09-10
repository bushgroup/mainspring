"""A stand-in for PNNL's acquisition console, so the two-phase protocol has two parties.

The real console is a C++ service that talks to an Acqiris digitizer, and it is the
half of the writing that mainspring does *not* do: `mainspring.uimf.writer` creates the
file and owns every parameter, the console opens that file and inserts `Frame_Scans`
rows into it. There is no way to exercise the protocol against the real thing without
the instrument, so this appends rows the way the console's `UimfWriter` does, from a
reading of its source in the lab record (`UIMFWriter.cpp`, `uimfacquisitionrecord.cpp`;
lab record, task 16). Everything below that is a claim about the console is a claim
about that source and not about a file anyone has seen yet.

What it copies, and why each one matters:

* **Read-write, never create.** The console opens with `OPEN_READWRITE` and no
  `OPEN_CREATE`, so a client that has not written the schema first gets an error rather
  than an empty database. `mode=rw` in a URI is the same thing.
* **`synchronous = 0` around each batch, and back to 1 after.** A per-connection pragma,
  set and unset inside `write_scan_data`, so it binds the console's writes and not the
  creator's. It is also why a power cut can lose recent scans.
* **One transaction per batch**, `NotifyOnScansCount` scans long (500 by default).
* **`ScanNum` counts from `start_trigger`**, not from the trigger index.
* **A record with one stream element or none is dropped**, unless it is scan 0. That is
  the console's `encoded_spectra.size() > 1 || er.scan == 0`, and it is the reason a
  frame stores only the scans that had signal.
* **`BPI_MZ` gets the base peak's *bin index*, not its m/z** (`bpi_mz =
  index_max_intensity`). Every other writer we have puts an m/z there. Setting
  `bpi_mz_as_bin` false writes a zero instead -- the other thing a real writer does
  with this column, on the 2011 file that never computed it -- so that a test can show
  what each of the two does to `uimf-info --verify`.
* **The journal mode is never touched**, so the file stays in whatever mode its creator
  put it in.

    stub = ConsoleStub(path)
    stub.acquire_frame(1, scans)      # scans: (scan, bin_index, intensity) triples
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from typing import Iterable, Iterator

import numpy as np

from mainspring.uimf.decode import dtype_for, lzf_compress, rlz_encode

__all__ = ["ConsoleStub", "NOTIFY_ON_SCANS_COUNT"]

NOTIFY_ON_SCANS_COUNT = 500
"""Scans per batch and per write. The console's `config.txt` default, and the size of
one of its transactions."""


class ConsoleStub:
    """Appends `Frame_Scans` to a file someone else created. See the module docstring."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        tof_intensity_type: str = "ADC",
        batch_size: int = NOTIFY_ON_SCANS_COUNT,
        bpi_mz_as_bin: bool = True,
    ) -> None:
        self.path = os.path.abspath(os.fspath(path))
        self.dtype = dtype_for(tof_intensity_type)
        self.batch_size = int(batch_size)
        self.bpi_mz_as_bin = bool(bpi_mz_as_bin)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"{self.path}: the console does not create the file")

    @contextlib.contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        uri = "file:" + self.path.replace("?", "%3f").replace("#", "%23") + "?mode=rw"
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def acquire_frame(
        self,
        frame: int,
        scans: Iterable[tuple[int, np.ndarray, np.ndarray]],
        start_trigger: int = 0,
    ) -> int:
        """Insert one frame's scans in batches; return how many rows were written.

        `scans` yields `(trigger index, bin index, intensity)`. The stored `ScanNum` is
        the trigger index less `start_trigger`, which is what makes a frame's scans
        start at zero however far into the run its first trigger was.
        """
        with self._connection() as conn:
            written = 0
            for batch in _batched(
                (self._row(frame, scan - start_trigger, bin_index, intensity)
                 for scan, bin_index, intensity in scans
                 if scan >= start_trigger),
                self.batch_size,
            ):
                rows = [row for row in batch if row is not None]
                # The pragmas sit inside the loop because they sit inside the console's
                # `write_scan_data`, which is called once per batch.
                conn.execute("PRAGMA synchronous = 0")
                conn.execute("BEGIN TRANSACTION")
                try:
                    conn.executemany(
                        "INSERT INTO Frame_Scans (FrameNum, ScanNum, NonZeroCount, BPI,"
                        " BPI_MZ, TIC, Intensities) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                except BaseException:
                    conn.rollback()
                    raise
                conn.commit()
                conn.execute("PRAGMA synchronous = 1")
                written += len(rows)
        return written

    def _row(
        self, frame: int, scan: int, bin_index: np.ndarray, intensity: np.ndarray
    ) -> tuple | None:
        bin_index = np.asarray(bin_index, dtype=np.int64)
        intensity = np.asarray(intensity, dtype=self.dtype)
        stream = rlz_encode(bin_index, intensity, self.dtype)
        if stream.size <= 1 and scan != 0:
            return None
        if intensity.size:
            argmax = int(np.argmax(intensity))
            bpi = int(intensity[argmax])
            peak_bin = int(bin_index[argmax])
            total = int(intensity.sum())
        else:
            bpi, peak_bin, total = 0, 0, 0
        bpi_mz = float(peak_bin) if self.bpi_mz_as_bin else 0.0
        return (frame, int(scan), int(intensity.size), bpi, bpi_mz, total,
                lzf_compress(stream.tobytes()))


def _batched(items: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
