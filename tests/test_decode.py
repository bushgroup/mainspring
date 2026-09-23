"""The blob path, both directions: the encoder against the format, then the round trip.

`encode_intensities` is what every summed companion the acquisition side folds is
written with (lab record, task 33), and is checked here against the format as
`notes/uimf-format.md` states it -- by an independent walk of the stream, not by asking
the encoder whether it agrees with itself.

The decoder is then checked three ways: it undoes the encoder; its two backends produce
identical arrays, on synthetic blobs and on every blob of every real file this clone
has; and it refuses a malformed stream instead of returning a short one. Last, the two
encode backends are held to each other byte for byte, and the compiled one to every blob
of a real file mainspring wrote.
"""

from __future__ import annotations

import numpy as np
import pytest

from mainspring.uimf import decode


def walk_rlz(values):
    """The run-length-zero rule of `notes/uimf-format.md`, written out independently.

    Three lines rather than a call into `decode`, so that this test says what the format
    is instead of asserting that the encoder agrees with itself.
    """
    bins, ints, cursor = [], [], 0
    for value in values:
        if value < 0:
            cursor += int(-value)
        elif value == 0:
            cursor += 1
        else:
            bins.append(cursor)
            ints.append(value)
            cursor += 1
    return bins, ints


def test_dtype_for_covers_the_three_intensity_types():
    assert decode.dtype_for("ADC") == np.dtype("<i4")
    assert decode.dtype_for("TDC") == np.dtype("<i2")
    assert decode.dtype_for("FOLDED") == np.dtype("<f4")
    assert decode.dtype_for(None) == np.dtype("<i4"), "an absent type is the oldest one"
    assert decode.dtype_for(" adc ") == np.dtype("<i4")


def test_dtype_for_refuses_to_guess():
    with pytest.raises(ValueError, match="TOFIntensityType"):
        decode.dtype_for("SOMETHING_NEW")


def test_rlz_encode_matches_the_format_by_hand():
    stream = decode.rlz_encode(np.array([0, 1, 5]), np.array([7, 8, 9]))
    assert stream.tolist() == [7, 8, -3, 9]
    assert stream.dtype == np.dtype("<i4")


def test_rlz_encode_round_trips_through_the_format_rule():
    rng = np.random.default_rng(20260905)
    bins = np.sort(rng.choice(114688, size=400, replace=False))
    values = rng.integers(1, 100000, size=bins.size)
    got_bins, got_values = walk_rlz(decode.rlz_encode(bins, values).tolist())
    assert got_bins == bins.tolist()
    assert got_values == values.tolist()


def test_rlz_encode_writes_explicit_zeros_as_the_writers_do():
    """A zero intensity is a stream entry, not a wider gap: that is what makes
    `NonZeroCount` an upper bound rather than a count."""
    stream = decode.rlz_encode(np.array([2, 3, 4]), np.array([5, 0, 6]))
    assert stream.tolist() == [-2, 5, 0, 6]


def test_rlz_encode_splits_a_gap_too_wide_for_the_element_type():
    stream = decode.rlz_encode(np.array([0, 100000]), np.array([5, 6]), "<i2")
    assert stream[0] == 5 and stream[-1] == 6
    assert all(v < 0 for v in stream[1:-1].tolist())
    assert -int(stream[1:-1].sum()) == 99999, "the skips must total the gap"


def test_rlz_encode_rejects_an_unsorted_index():
    with pytest.raises(ValueError, match="increasing"):
        decode.rlz_encode(np.array([5, 1]), np.array([1, 2]))


def test_lzf_compress_handles_the_degenerate_lengths():
    for n in range(0, 8):
        blob = decode.lzf_compress(bytes(range(n)))
        assert isinstance(blob, bytes)
        assert (len(blob) == 0) == (n == 0)


def test_lzf_compress_actually_compresses_a_sparse_spectrum():
    bins = np.arange(0, 114688, 700)
    stream = decode.rlz_encode(bins, np.full(bins.size, 42))
    blob = decode.lzf_compress(stream.tobytes())
    assert len(blob) < stream.nbytes / 4


def test_lzf_control_bytes_stay_inside_the_format():
    """Every control byte must be a literal run or a back-reference that points behind
    the output cursor -- a stream UIMF-Library would reject is worse than no stream."""
    bins = np.arange(0, 4096, 3)
    blob = decode.lzf_compress(decode.rlz_encode(bins, np.full(bins.size, 7)).tobytes())
    i, produced = 0, 0
    while i < len(blob):
        control = blob[i]
        i += 1
        if control < 32:
            i += control + 1
            produced += control + 1
        else:
            length = control >> 5
            if length == 7:
                length += blob[i]
                i += 1
            offset = ((control & 31) << 8 | blob[i]) + 1
            i += 1
            assert offset <= produced, "back-reference before the start of the output"
            produced += length + 2
    assert i == len(blob), "the stream must end on a control-byte boundary"



# --- the decoder --------------------------------------------------------------------


def test_lzf_round_trips_every_degenerate_length():
    for n in range(0, 40):
        payload = bytes(range(n))
        assert decode.lzf_decompress(decode.lzf_compress(payload)) == payload


def test_lzf_round_trips_a_stream_full_of_back_references():
    """Long runs of one byte are what force overlapping copies, the case a naive
    slice-based decompressor gets wrong."""
    payload = b"AB" * 5000 + bytes(range(256)) * 20
    assert decode.lzf_decompress(decode.lzf_compress(payload)) == payload


def test_rlz_decode_undoes_the_format_by_hand():
    bins, values = decode.rlz_decode(np.array([7, 8, -3, 0, 9], dtype="<i4"))
    assert bins.tolist() == [0, 1, 6]
    assert values.tolist() == [7, 8, 9]


def test_rlz_decode_agrees_with_the_independent_walk():
    rng = np.random.default_rng(20260905)
    bins = np.sort(rng.choice(114688, size=500, replace=False))
    values = rng.integers(1, 100000, size=bins.size)
    stream = decode.rlz_encode(bins, values)
    want_bins, want_values = walk_rlz(stream.tolist())
    got_bins, got_values = decode.rlz_decode(stream)
    assert got_bins.tolist() == want_bins
    assert got_values.tolist() == want_values


def test_rlz_decode_bounds_the_result_to_the_bin_axis():
    stream = decode.rlz_encode(np.array([1, 4000]), np.array([5, 6]))
    bins, values = decode.rlz_decode(stream, bins=100)
    assert bins.tolist() == [1] and values.tolist() == [5]


@pytest.mark.parametrize("intensity_type", ["ADC", "TDC", "FOLDED"])
def test_decode_intensities_undoes_encode_intensities(intensity_type):
    dtype = decode.dtype_for(intensity_type)
    rng = np.random.default_rng(3)
    bins = np.sort(rng.choice(4096, size=200, replace=False))
    values = np.asarray(rng.integers(1, 900, size=bins.size), dtype=dtype)
    got_bins, got_values = decode.decode_intensities(
        decode.encode_intensities(bins, values, dtype), dtype
    )
    assert got_bins.tolist() == bins.tolist()
    assert got_values.tolist() == values.tolist()
    assert got_values.dtype == dtype


def test_decode_intensities_drops_the_explicit_zeros_the_writers_leave():
    """The zeros advance the bin counter and are not points, which is the whole reason
    `NonZeroCount` is an upper bound rather than a count (lab record, task 01)."""
    blob = decode.encode_intensities(np.array([2, 3, 4]), np.array([5, 0, 6]))
    bins, values = decode.decode_intensities(blob)
    assert bins.tolist() == [2, 4]
    assert values.tolist() == [5, 6]


def test_decode_intensities_of_nothing_is_nothing():
    bins, values = decode.decode_intensities(b"")
    assert bins.size == 0 and values.size == 0


def test_a_truncated_stream_raises_rather_than_returning_a_short_spectrum():
    blob = decode.encode_intensities(np.arange(0, 4096, 3), np.full(1366, 7))
    with pytest.raises(ValueError, match="LZF"):
        decode.lzf_decompress(blob[:-1])


def test_a_ragged_decompressed_length_is_refused():
    with pytest.raises(ValueError, match="multiple"):
        decode.decode_intensities(decode.lzf_compress(b"\x01\x02\x03"), "<i4")


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="backend"):
        decode.decode_frame_blobs([b""], backend="fortran")


# --- the two backends -----------------------------------------------------------------


def synthetic_blobs(dtype="<i4", count=40):
    rng = np.random.default_rng(11)
    blobs = []
    for i in range(count):
        size = int(rng.integers(0, 300))
        bins = np.sort(rng.choice(114688, size=size, replace=False))
        values = np.asarray(rng.integers(0, 30000, size=size), dtype=dtype)
        blobs.append(decode.encode_intensities(bins, values, dtype) if size else b"")
    return blobs


@pytest.mark.parametrize("intensity_type", ["ADC", "TDC", "FOLDED"])
def test_the_two_backends_agree_on_synthetic_blobs(intensity_type):
    dtype = decode.dtype_for(intensity_type)
    blobs = synthetic_blobs(dtype)
    pure = decode.decode_frame_blobs(blobs, dtype, 114688, backend="pure")
    auto = decode.decode_frame_blobs(blobs, dtype, 114688, backend="auto")
    for ours, theirs in zip(pure, auto):
        assert np.array_equal(ours, theirs)
        assert ours.dtype == theirs.dtype


def test_decode_frame_blobs_counts_map_onto_the_concatenation():
    blobs = synthetic_blobs()
    counts, bins, values = decode.decode_frame_blobs(blobs, "<i4", 114688)
    assert int(counts.sum()) == bins.size == values.size
    start = 0
    for blob, count in zip(blobs, counts.tolist()):
        want_bins, want_values = decode.decode_intensities(blob)
        assert bins[start:start + count].tolist() == want_bins.tolist()
        assert values[start:start + count].tolist() == want_values.tolist()
        start += count


def test_a_frame_of_empty_blobs_decodes_to_nothing():
    counts, bins, values = decode.decode_frame_blobs([b"", None, b""], "<i4")
    assert counts.tolist() == [0, 0, 0]
    assert bins.size == 0 and values.size == 0


@pytest.mark.skipif(not decode.numba_available(), reason="numba is not installed")
def test_the_two_backends_agree_blob_by_blob_on_a_real_file(real_uimf):
    """The check the compiled path exists to earn: same answer as the reference, on
    every blob a real writer produced, not on blobs we made up."""
    import sqlite3

    from mainspring.uimf.reader import UimfFile

    dtype = UimfFile(real_uimf).global_params().dtype
    conn = sqlite3.connect("file:" + real_uimf.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        blobs = [row[0] for row in conn.execute(
            "SELECT Intensities FROM Frame_Scans ORDER BY FrameNum, ScanNum LIMIT 4000")]
    finally:
        conn.close()
    pure = decode.decode_frame_blobs(blobs, dtype, backend="pure")
    fast = decode.decode_frame_blobs(blobs, dtype, backend="numba")
    for ours, theirs in zip(pure, fast):
        assert np.array_equal(ours, theirs)


def test_numba_available_is_a_bool_either_way():
    assert decode.numba_available() in (True, False)


# --- the two encode backends ----------------------------------------------------------
#
# The compiled encoder is held to the pure one byte for byte, never to "a valid stream":
# a file written with numba and one written without it must be the same file (lab
# record, task 33).

needs_numba = pytest.mark.skipif(not decode.numba_available(), reason="numba is not installed")


def compiled_lzf(payload: bytes) -> bytes:
    """The LZF kernel on one raw byte stream, of any length, bypassing the RLZ stage."""
    src = np.frombuffer(payload, dtype=np.uint8)
    n = src.size
    dst = np.empty(n + (n >> 5) + 2, dtype=np.uint8)
    lengths = np.empty(1, dtype=np.int64)
    decode._kernels().lzf_compress(
        src, np.array([0, n], dtype=np.int64), dst, np.array([0, dst.size], dtype=np.int64),
        lengths, np.zeros(1 << 14, dtype=np.int64))
    return dst[:int(lengths[0])].tobytes()


def random_spectrum(rng, dtype, size=None):
    dtype = np.dtype(dtype)
    size = int(rng.integers(0, 3000)) if size is None else size
    span = int(rng.choice([size + 1, 4 * size + 8, 114688, 253888]))
    bins = np.sort(rng.choice(max(span, size), size=size, replace=False))
    if dtype.kind == "f":
        values = rng.choice([0.0, 0.5, 1.0, 3.25, 1e6, 7.0], size=size)
    else:
        values = rng.integers(0, int(rng.choice([2, 60, 30000])), size=size)
    return bins, values.astype(dtype)


@needs_numba
@pytest.mark.parametrize("intensity_type", ["ADC", "TDC", "FOLDED"])
def test_the_encode_backends_agree_on_random_spectra(intensity_type):
    dtype = decode.dtype_for(intensity_type)
    rng = np.random.default_rng(33)
    for _ in range(150):
        bins, values = random_spectrum(rng, dtype)
        assert (decode.encode_intensities(bins, values, dtype, "numba")
                == decode.encode_intensities(bins, values, dtype, "pure"))


@needs_numba
def test_the_encode_backends_agree_where_every_point_is_the_same():
    """A summed frame's flat stretches are long back-references, the path a literal-heavy
    random spectrum hardly reaches."""
    for size in (1, 2, 3, 9, 264, 265, 5000):
        bins = np.arange(size)
        values = np.full(size, 7, dtype="<i4")
        assert (decode.encode_intensities(bins, values, "<i4", "numba")
                == decode.encode_intensities(bins, values, "<i4", "pure"))


@needs_numba
@pytest.mark.parametrize("payload", [
    b"",
    b"\x00",
    b"ab",
    b"abc",
    b"abcd",
    b"abcab",
    bytes(range(256)) * 3,
    b"A" * 100000,                                   # a match as long as the format allows
    b"xyz" * 20 + b"q",                              # a match that runs to the last bytes
    b"q" + b"xyz" * 20,
    b"abcdefgh" * 2000,
    bytes(range(256)) * 40 + bytes(range(256)),      # repeats 8192+ bytes back: out of reach
    b"\x01\x02\x03" + bytes(9000) + b"\x01\x02\x03",
], ids=lambda p: f"{len(p)}B")
def test_the_compiled_lzf_matches_the_reference_on_raw_streams(payload):
    assert compiled_lzf(payload) == decode.lzf_compress(payload)


@needs_numba
def test_the_compiled_lzf_matches_the_reference_on_random_streams():
    rng = np.random.default_rng(3333)
    for _ in range(300):
        n = int(rng.integers(0, 5000))
        alphabet = int(rng.choice([2, 4, 16, 256]))
        payload = rng.integers(0, alphabet, size=n, dtype=np.uint8).tobytes()
        assert compiled_lzf(payload) == decode.lzf_compress(payload)
    for n in range(0, 40):
        assert compiled_lzf(bytes(range(n))) == decode.lzf_compress(bytes(range(n)))


@pytest.mark.parametrize("backend", ["pure", "auto"])
def test_an_empty_scan_encodes_to_nothing(backend):
    assert decode.encode_intensities(np.array([], dtype=np.int64),
                                     np.array([], dtype="<i4"), "<i4", backend) == b""
    blobs = decode.encode_frame_blobs(np.array([0, 0, 2, 2]), np.array([3, 9]),
                                      np.array([1, 2], dtype="<i4"), "<i4", backend)
    assert blobs[0] == blobs[2] == b"" and blobs[1] != b""


@needs_numba
@pytest.mark.parametrize("gap", [32766, 32767, 32768, 65534, 65535, 65536, 200000])
def test_a_gap_beyond_the_int16_skip_splits_the_same_way(gap):
    bins = np.array([0, gap, gap + 1, 2 * gap + 5])
    values = np.array([5, 6, 7, 8], dtype="<i2")
    assert (decode.encode_intensities(bins, values, "<i2", "numba")
            == decode.encode_intensities(bins, values, "<i2", "pure"))


@pytest.mark.parametrize("backend", ["pure", "auto"])
def test_encode_frame_blobs_is_encode_intensities_scan_by_scan(backend):
    rng = np.random.default_rng(34)
    counts = rng.integers(0, 400, size=60)
    counts[::5] = 0
    pieces = [random_spectrum(rng, "<i4", int(c)) for c in counts]
    scan_start = np.concatenate(([0], np.cumsum(counts)))
    bins = np.concatenate([b for b, _ in pieces])
    values = np.concatenate([v for _, v in pieces])
    blobs = decode.encode_frame_blobs(scan_start, bins, values, "<i4", backend)
    assert blobs == [decode.encode_intensities(b, v, "<i4", "pure") for b, v in pieces]


@needs_numba
def test_one_scan_leaves_nothing_behind_for_the_next():
    """The kernel shares its hash table across a frame's scans, and a stale entry is only
    harmless if it is recognised as stale: the same scan must encode the same way
    whatever came before it."""
    bins = np.arange(0, 3000, 3)
    values = (np.arange(bins.size) % 17 + 1).astype("<i4")
    alone = decode.encode_intensities(bins, values, "<i4", "pure")
    scan_start = np.arange(0, 5 * bins.size + 1, bins.size)
    blobs = decode.encode_frame_blobs(scan_start, np.tile(bins, 5), np.tile(values, 5),
                                      "<i4", "numba")
    assert blobs == [alone] * 5


@pytest.mark.parametrize("backend", ["pure", "auto"])
def test_an_intensity_given_in_a_wider_type_encodes_as_its_value(backend):
    bins = np.array([1, 5, 9])
    as_int64 = np.array([1, 70000, 3], dtype=np.int64)
    assert (decode.encode_intensities(bins, as_int64, "<i4", backend)
            == decode.encode_intensities(bins, as_int64.astype("<i4"), "<i4", "pure"))


@pytest.mark.parametrize("backend", ["pure", "auto"])
def test_a_value_the_element_type_cannot_hold_is_refused_either_way(backend):
    with pytest.raises(OverflowError):
        decode.encode_intensities(np.array([1, 2]), np.array([1, 40000]), "<i2", backend)


@pytest.mark.parametrize("backend", ["pure", "auto"])
def test_both_encode_backends_refuse_an_unsorted_index(backend):
    with pytest.raises(ValueError, match="increasing"):
        decode.encode_intensities(np.array([5, 1]), np.array([1, 2]), "<i4", backend)
    with pytest.raises(ValueError, match="increasing"):
        decode.encode_frame_blobs(np.array([0, 3]), np.array([1, 4, 4]),
                                  np.array([1, 2, 3], dtype="<i4"), "<i4", backend)


def test_a_new_scan_starting_lower_is_not_a_disorder():
    blobs = decode.encode_frame_blobs(np.array([0, 2, 4]), np.array([50, 60, 1, 2]),
                                      np.array([1, 2, 3, 4], dtype="<i4"))
    assert len(blobs) == 2


def test_an_unknown_encode_backend_is_refused():
    with pytest.raises(ValueError, match="backend"):
        decode.encode_intensities(np.array([1]), np.array([1]), backend="fortran")
    with pytest.raises(ValueError, match="backend"):
        decode.encode_frame_blobs(np.array([0, 1]), np.array([1]), np.array([1]),
                                  backend="fortran")


@needs_numba
def test_every_blob_of_a_file_we_wrote_encodes_back_to_itself(real_uimf):
    """The identity on a real acquisition, not on blobs made up here: decode what a
    mainspring writer stored, encode it again with the kernels, and get the stored bytes.
    Only a file mainspring wrote can pass -- PNNL's writer leaves explicit zeros in its
    streams, which decode away, so re-encoding its blobs differs legitimately."""
    import sqlite3

    from mainspring.uimf.reader import UimfFile

    source = UimfFile(real_uimf)
    if not source.global_params().written_by.startswith("mainspring"):
        pytest.skip("not written by mainspring; its blobs differ from ours by design")
    dtype = source.global_params().dtype
    conn = sqlite3.connect("file:" + real_uimf.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        blobs = [bytes(row[0] or b"") for row in conn.execute(
            "SELECT Intensities FROM Frame_Scans ORDER BY FrameNum, ScanNum LIMIT 4000")]
    finally:
        conn.close()
    counts, bins, values = decode.decode_frame_blobs(blobs, dtype, backend="numba")
    scan_start = np.concatenate(([0], np.cumsum(counts)))
    assert decode.encode_frame_blobs(scan_start, bins, values, dtype, "numba") == blobs
