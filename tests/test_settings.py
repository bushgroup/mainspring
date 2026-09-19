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


def test_validate_clamps_an_unrecognised_aggregate_or_color_scale():
    settings = ViewerSettings(aggregate="bogus", color_scale="bogus")
    fixed = settings.validate()
    assert fixed.aggregate == "sum"
    assert fixed.color_scale == "linear"


def test_validate_clamps_an_unrecognised_text_scale():
    """Four steps, and a hand-edited store cannot ask for a fifth."""
    assert ViewerSettings(text_scale=3.0).validate().text_scale == 1.0
    assert ViewerSettings(text_scale=1.25).validate().text_scale == 1.25


def test_validate_keeps_the_info_panel_at_least_its_minimum_width():
    """A hand-edited width of nothing would hide the panel behind its own splitter."""
    from mainspring.viewer.settings import INFO_PANEL_WIDTH

    assert ViewerSettings(info_panel_width=0).validate().info_panel_width == INFO_PANEL_WIDTH
    assert ViewerSettings(info_panel_width=900).validate().info_panel_width == 900


def test_validate_clamps_an_unrecognised_color_map():
    assert ViewerSettings(color_map="bogus").validate().color_map == "viridis"
    assert ViewerSettings(color_map="plasma").validate().color_map == "plasma"


def test_validate_clamps_a_non_finite_arrival_offset():
    assert ViewerSettings(arrival_offset_ms=float("nan")).validate().arrival_offset_ms == 0.0
    assert ViewerSettings(arrival_offset_ms=float("inf")).validate().arrival_offset_ms == 0.0
    assert ViewerSettings(arrival_offset_ms=-150.0).validate().arrival_offset_ms == -150.0


def test_validate_clamps_detector_bits_to_a_usable_range():
    assert ViewerSettings(detector_bits=0).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=-3).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=33).validate().detector_bits == 8
    assert ViewerSettings(detector_bits=14).validate().detector_bits == 14  # clockwork's


def test_validate_clamps_an_unrecognised_export_dpi():
    """A resolution the dialog cannot offer would leave its combo on no choice at all."""
    assert ViewerSettings(export_dpi=137).validate().export_dpi == 300
    assert ViewerSettings(export_dpi=600).validate().export_dpi == 600


def test_validate_clamps_the_rolling_sum_window_to_a_length_it_can_keep_up_with():
    """A total recomputed every time a frame finishes has about a second to do it in, so
    the top of the range is what fits in one (`ROLLING_SUM_MAX`); a whole run added up is
    what `Sum all` is for."""
    from mainspring.viewer.settings import ROLLING_SUM_MAX

    assert ViewerSettings(rolling_sum_frames=0).validate().rolling_sum_frames == 5
    assert ViewerSettings(rolling_sum_frames=-4).validate().rolling_sum_frames == 5
    assert (ViewerSettings(rolling_sum_frames=ROLLING_SUM_MAX + 1)
            .validate().rolling_sum_frames == 5)
    assert ViewerSettings(rolling_sum_frames=25).validate().rolling_sum_frames == 25


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
        color_scale="log",
        keep_ranges=True,
        keep_levels=True,
        show_info_panel=False,
        info_panel_width=480,
        text_scale=1.5,
        detector_bits=14,
        color_map="plasma",
        export_dpi=600,
        cache_budget_mb=256,
        arrival_offset_ms=-42.5,
        rolling_sum_frames=25,
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


def test_the_old_british_spelling_of_the_two_color_keys_is_read_once_and_then_removed():
    """An upgrade keeps the map and the scale the user chose.

    The keys were `colour_map` and `colour_scale` up to 1.3.0. A store written by that
    version has to keep answering, or every existing installation silently loses two
    settings on upgrade; and the old key has to go once the new one is written, or a
    store carries two keys for one setting and they drift.
    """
    from mainspring.viewer.settings import _store

    store = _store()
    store.setValue("colour_map", "magma")
    store.setValue("colour_scale", "log")
    store.sync()

    loaded = load_settings()
    assert loaded.color_map == "magma"
    assert loaded.color_scale == "log"

    save_settings(loaded)
    store = _store()
    assert not store.contains("colour_map")
    assert not store.contains("colour_scale")
    assert store.value("color_map") == "magma"
    assert load_settings().color_map == "magma"


def test_the_new_spelling_wins_when_a_store_somehow_carries_both():
    """A store hand-edited, or written by two versions in turn, is not ambiguous: the
    key this version writes is the one it reads."""
    from mainspring.viewer.settings import _store

    store = _store()
    store.setValue("colour_map", "magma")
    store.setValue("color_map", "plasma")
    store.sync()

    assert load_settings().color_map == "plasma"
