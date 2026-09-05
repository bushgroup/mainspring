"""The encoder: the half of the blob path that exists before the decoder does.

Only tests write UIMF files, so `encode_intensities` exists to make the synthetic
fixture possible (lab record, task 02) and is checked here against the format as
`notes/uimf-format.md` states it. The round trip against our own decoder is not here
because our own decoder is not written yet; it arrives with the lab record's task 03,
and `tools/check_public.py` is where it will be checked.
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


def test_the_decoder_says_it_is_not_written_yet():
    for call in (
        lambda: decode.decode_intensities(b"\x00"),
        lambda: decode.lzf_decompress(b"\x00"),
        lambda: decode.rlz_decode(np.array([1])),
    ):
        with pytest.raises(NotImplementedError, match="task 03"):
            call()
