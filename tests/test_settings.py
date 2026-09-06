"""`ViewerSettings`: defaults, validation, and the round trip through `QSettings`.

`conftest._isolated_qsettings` redirects every test's `QSettings` to a per-test ini file
(autouse), so the round trip here is real persistence and never touches the registry a
real viewer session would use.
"""

from __future__ import annotations

from mainspring.viewer.settings import ViewerSettings, load_settings, save_settings


def test_defaults_are_already_valid():
    """The defaults must pass their own `validate()` unchanged -- a first run should
    never see its own settings "corrected"."""
    assert ViewerSettings().validate() == ViewerSettings()


def test_validate_clamps_an_unrecognised_aggregate_or_colour_scale():
    settings = ViewerSettings(aggregate="bogus", colour_scale="bogus")
    fixed = settings.validate()
    assert fixed.aggregate == "sum"
    assert fixed.colour_scale == "linear"


def test_validate_clamps_an_unrecognised_colour_map():
    assert ViewerSettings(colour_map="bogus").validate().colour_map == "viridis"
    assert ViewerSettings(colour_map="plasma").validate().colour_map == "plasma"


def test_validate_clamps_a_non_finite_arrival_offset():
    assert ViewerSettings(arrival_offset_ms=float("nan")).validate().arrival_offset_ms == 0.0
    assert ViewerSettings(arrival_offset_ms=float("inf")).validate().arrival_offset_ms == 0.0
    assert ViewerSettings(arrival_offset_ms=-150.0).validate().arrival_offset_ms == -150.0


def test_validate_clamps_detector_bits_to_a_usable_range():
    assert ViewerSettings(detector_bits=0).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=-3).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=33).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=14).validate().detector_bits == 14  # clockwork's


def test_validate_clamps_a_non_positive_cache_budget():
    assert ViewerSettings(cache_budget_mb=0).validate().cache_budget_mb == 512
    assert ViewerSettings(cache_budget_mb=-100).validate().cache_budget_mb == 512
    assert ViewerSettings(cache_budget_mb=256).validate().cache_budget_mb == 256


def test_load_settings_is_the_defaults_when_nothing_is_stored():
    assert load_settings() == ViewerSettings()


def test_settings_round_trip_through_qsettings():
    """Every field survives a save and a fresh load, keep-ranges and the detector-bits
    setting included -- the two the whole feature exists for."""
    written = ViewerSettings(
        aggregate="max",
        swap_axes=True,
        raw_units=True,
        colour_scale="log",
        keep_ranges=True,
        keep_levels=True,
        show_info_panel=False,
        detector_bits=14,
        colour_map="plasma",
        cache_budget_mb=256,
        arrival_offset_ms=-42.5,
        last_directory="F:/data/acquisitions",
        window_geometry=b"\x00binary-ish\x01geometry\x02bytes\x03",
    )

    save_settings(written)
    read_back = load_settings()

    assert read_back == written


def test_an_unrecognised_stored_value_falls_back_to_the_default_rather_than_failing():
    """A value `QSettings` cannot coerce to the right type -- or one an older or hand-
    edited store never wrote at all -- must not fail the whole load; `load_settings`
    reads each field with its default already in hand for exactly this."""
    save_settings(ViewerSettings(detector_bits=14, aggregate="max"))

    import mainspring.viewer.settings as settings_module

    # `settings_module._store()`, not `QSettings(ORGANISATION, APPLICATION)` -- the
    # two-argument constructor always means `NativeFormat`, ignoring the redirect
    # `conftest._isolated_qsettings` put in place, and would silently corrupt this
    # value in the real registry rather than in the isolated store this test means to
    # exercise (`_store`'s own docstring says why).
    store = settings_module._store()
    store.setValue("detector_bits", "not-a-number")
    store.sync()

    settings = load_settings()
    assert settings.detector_bits == 8  # the default, not a crash
    assert settings.aggregate == "max"  # everything else still round-trips
