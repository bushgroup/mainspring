"""`mainspring.analysis`'s shape statistics, against closed forms and small cases.

A planted Gaussian gives every position at its centre and the widths as multiples of σ
(FWHM 2.355σ, IQR 1.349σ, a flank's 10–90 % rise 1.687σ); a step through a Gaussian
gives the 2.563σ rise; two peaks and a plateau pin the local-maxima rule; and a profile
with nothing in it gives `None` throughout (lab record, task 42).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr

from mainspring.analysis import (
    argmax, centroid, fwhm, iqr, local_maxima, median, rise_10_90,
)

SIGMA = 20.0
CENTRE = 1000
FWHM_SIGMAS = 2 * math.sqrt(2 * math.log(2))
IQR_SIGMAS = 2 * 0.6744897501960817
FLANK_SIGMAS = math.sqrt(2 * math.log(10)) - math.sqrt(2 * math.log(1 / 0.9))
STEP_SIGMAS = 2 * 1.2815515655446004
STATISTICS = (argmax, centroid, median, iqr, fwhm, rise_10_90)

x = np.arange(2001, dtype=np.float64)


def gaussian(centre=CENTRE, sigma=SIGMA, height=1.0):
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def test_a_gaussian_is_read_at_its_centre():
    g = gaussian()
    assert argmax(g) == CENTRE
    assert centroid(g) == pytest.approx(CENTRE, abs=1e-9)
    assert median(g) == CENTRE


def test_a_gaussian_has_its_widths_in_sigma():
    g = gaussian()
    assert fwhm(g) == pytest.approx(FWHM_SIGMAS * SIGMA, rel=1e-3)
    assert iqr(g) == pytest.approx(IQR_SIGMAS * SIGMA, rel=1e-3)
    assert rise_10_90(g) == pytest.approx(FLANK_SIGMAS * SIGMA, rel=1e-3)


def test_a_step_through_a_gaussian_rises_in_2_563_sigma():
    step = ndtr((x - CENTRE) / SIGMA)
    assert rise_10_90(step) == pytest.approx(STEP_SIGMAS * SIGMA, rel=1e-3)


def test_every_result_is_a_float():
    for statistic in STATISTICS:
        assert type(statistic(gaussian())) is float, statistic.__name__
    assert local_maxima(gaussian(), fraction=0.5).dtype == np.float64


@pytest.mark.parametrize("empty", [np.zeros(50), np.zeros(0), [0.0, -1.0, 0.0]],
                         ids=["zeros", "no scans", "nothing above zero"])
def test_a_profile_with_nothing_in_it_has_no_statistic(empty):
    for statistic in (argmax, centroid, fwhm, rise_10_90):
        assert statistic(empty) is None, statistic.__name__
    assert local_maxima(empty, fraction=0.15) is None
    if np.min(empty, initial=0.0) >= 0.0:
        assert median(empty) is None and iqr(empty) is None


def test_a_tie_reads_its_first_scan():
    assert argmax([0.0, 2.0, 1.0, 2.0]) == 1.0


def test_the_centroid_is_taken_over_the_half_maximum_run_only():
    # The second peak is below half the first and separated from it: it is left out.
    profile = np.zeros(40)
    profile[10:13] = [4.0, 8.0, 6.0]
    profile[30] = 3.0
    assert centroid(profile) == pytest.approx((10 * 4 + 11 * 8 + 12 * 6) / 18)
    spike = np.zeros(9)
    spike[4] = 5.0
    assert centroid(spike) == 4.0


def test_the_median_is_a_whole_scan_and_the_iqr_interpolates():
    profile = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    assert median(profile) == 2.0   # the cumulative reaches 2 of 4 at scan 2
    # 1 of 4 is reached a whole scan into scan 1 and 3 of 4 a whole scan into scan 3,
    # each scan's intensity spread over the interval ending at it.
    assert iqr(profile) == pytest.approx(2.0)


def test_a_peak_cut_by_the_end_crosses_just_past_it():
    # Zero beyond the end, as the smoother takes it: half of 4 is crossed half a scan
    # before scan 0.
    assert fwhm([4.0, 2.0, 0.0]) == pytest.approx(1.5)
    assert rise_10_90([4.0, 0.0]) == pytest.approx((-1 + 0.9) - (-1 + 0.1))


def test_local_maxima_count_peaks_above_the_fraction_without_merging():
    two = gaussian(950, 10.0) + gaussian(1050, 10.0, 0.5) + gaussian(1150, 10.0, 0.1)
    np.testing.assert_array_equal(local_maxima(two, fraction=0.15), [950.0, 1050.0])
    np.testing.assert_array_equal(local_maxima(two, fraction=0.05), [950.0, 1050.0, 1150.0])
    np.testing.assert_array_equal(local_maxima([0.0, 1.0, 0.5, 1.0, 0.0], fraction=0.0),
                                  [1.0, 3.0])


def test_a_plateau_is_one_maximum_at_its_first_scan_and_an_end_is_none():
    np.testing.assert_array_equal(local_maxima([0, 2, 2, 2, 0], fraction=0.0), [1.0])
    np.testing.assert_array_equal(local_maxima([0, 2, 2, 3, 0], fraction=0.0), [3.0])
    assert local_maxima([5.0, 1.0, 0.0], fraction=0.0).size == 0
    assert local_maxima([0.0, 3.0, 3.0], fraction=0.0).size == 0


def test_no_statistic_smooths():
    # A spike on a broad peak: argmax and FWHM read the spike, as the docstrings say.
    profile = gaussian()
    profile[1100] = 5.0
    assert argmax(profile) == 1100.0
    assert fwhm(profile) < 2.0


@pytest.mark.parametrize("statistic", STATISTICS)
def test_a_profile_that_is_not_one_dimensional_or_finite_is_refused(statistic):
    with pytest.raises(ValueError, match="one-dimensional"):
        statistic(np.ones((3, 3)))
    with pytest.raises(ValueError, match="NaN"):
        statistic([0.0, math.nan, 1.0])


@pytest.mark.parametrize("statistic", [median, iqr])
def test_a_percentile_refuses_a_negative_value(statistic):
    with pytest.raises(ValueError, match="negative"):
        statistic([1.0, -0.5, 2.0])


@pytest.mark.parametrize("fraction", [-0.1, 1.5, math.nan])
def test_local_maxima_refuse_a_fraction_outside_0_to_1(fraction):
    with pytest.raises(ValueError, match="fraction"):
        local_maxima(gaussian(), fraction=fraction)


def test_local_maxima_have_no_default_floor():
    with pytest.raises(TypeError):
        local_maxima(gaussian())  # type: ignore[call-arg]
