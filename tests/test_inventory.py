import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import inventory
from tests import _reference as ref


# --------------------------- preprocessing ---------------------------

def test_add_consumption_matches_original(inventory_wide):
    new = inventory.add_consumption(inventory.melt_inventory(inventory_wide))
    assert_frame_equal(new, ref.app_preprocess(inventory_wide), check_exact=True)


def test_consumption_is_positive_drops_only():
    wide = pd.DataFrame({
        "material": ["A", "B"],
        "2026-01-01": [10.0, 5.0],
        "2026-01-02": [7.0, 5.0],
        "2026-01-03": [9.0, 1.0],   # A sube (recepción) -> consumo 0
        "2026-01-04": [4.5, 0.0],
    })
    df = inventory.add_consumption(inventory.melt_inventory(wide))

    a = df[df["material"] == "A"]["consumo"].tolist()
    b = df[df["material"] == "B"]["consumo"].tolist()
    assert a == [0.0, 3.0, 0.0, 4.5]
    assert b == [0.0, 0.0, 4.0, 1.0]


# --------------------------- filtering ---------------------------

def test_filter_matches_original_on_real_data(inventory_wide):
    df = inventory.add_consumption(inventory.melt_inventory(inventory_wide))
    first, last = df["fecha"].min().date(), df["fecha"].max().date()
    ranges = [(first, last), (dt.date(2026, 6, 1), dt.date(2026, 7, 15)), (last, last),
              (dt.date(2030, 1, 1), dt.date(2030, 2, 1))]
    for material in df["material"].unique():
        for start, end in ranges:
            assert_frame_equal(
                inventory.filter_inventory(df, material, start, end),
                ref.app_filter(df, material, start, end),
            )


def test_filter_boundaries_include_whole_end_day():
    df = pd.DataFrame({
        "material": pd.Categorical(["A"] * 5),
        "fecha": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 23:59", "2026-01-02 12:00",
                                 "2026-01-03 00:00", "2025-12-31 23:59"]),
        "inventario": range(5),
    })
    for start, end in [(dt.date(2026, 1, 1), dt.date(2026, 1, 2)), (dt.date(2026, 1, 2), dt.date(2026, 1, 2))]:
        assert_frame_equal(inventory.filter_inventory(df, "A", start, end),
                           ref.app_filter(df, "A", start, end))


# --------------------------- business rules ---------------------------

@pytest.mark.parametrize("material, expected", [
    ("ARMEX DE 15 x 15", 7),
    ("  armex 15x20 ", 7),
    ("ARMEX 15 x 20", 7),
    ("ARMEX DE 20 x 20", 14),
    ("VARILLA CORRUGADA 3/8\"", 14),
])
def test_target_days_for(material, expected):
    assert inventory.target_days_for(material) == expected


def _metrics_inputs(inventory_wide, material):
    df = inventory.add_consumption(inventory.melt_inventory(inventory_wide))
    filtered = inventory.filter_inventory(df, material, df["fecha"].min().date(), df["fecha"].max().date())
    return filtered


@pytest.mark.parametrize("lead_time_days", [1, 7, 15])
def test_replenishment_metrics_match_original(inventory_wide, lead_time_days):
    rng = np.random.default_rng(0)
    for material in inventory_wide["material"]:
        filtered = _metrics_inputs(inventory_wide, material)
        target_days = inventory.target_days_for(material)
        forecast = pd.Series(rng.uniform(0, 50, lead_time_days + target_days))

        new = inventory.replenishment_metrics(filtered, forecast, lead_time_days, target_days)
        old = ref.app_metrics(filtered, forecast, lead_time_days, material)
        old.pop("target_days")
        assert new == old, material


def test_replenishment_metrics_hand_computed():
    filtered = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "inventario": [100.0, 90.0, 80.0, 60.0],
        "consumo": [0.0, 10.0, 10.0, 20.0],
    })
    forecast = pd.Series([10.0] * 16)
    m = inventory.replenishment_metrics(filtered, forecast, lead_time_days=2, target_days=14)

    std = filtered["consumo"].std()
    safety = 1.5 * std * np.sqrt(2)
    target_stock = 10.0 * 16 + safety
    assert m["avg_consumption"] == 10.0
    assert m["safety_stock"] == pytest.approx(safety)
    assert m["reorder_point"] == pytest.approx(160.0 + safety)
    assert m["recommended_qty"] == pytest.approx(target_stock - (60.0 - 20.0))
    assert m["maximum_inventory"] == pytest.approx(target_stock * (1 + np.clip(std / 10.0, 0.1, 0.5)))
    assert m["minimum_stock"] == 140.0
    assert m["buy_now"]
    assert m["estimated_stockout_date"] == pd.Timestamp("2026-01-10")


def test_replenishment_metrics_without_consumption():
    filtered = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "inventario": [50.0, 50.0],
        "consumo": [0.0, 0.0],
    })
    m = inventory.replenishment_metrics(filtered, pd.Series([0.0] * 21), 7, 14)
    assert m["estimated_stockout_date"] is None
    assert m["days_until_stockout"] == np.inf
    assert m["recommended_qty"] == 0
    assert not m["buy_now"]


# --------------------------- ARIMA ---------------------------

def test_cached_fit_reused_across_horizons_matches_original(inventory_wide):
    """La app ajusta una vez por serie y pronostica con distintos horizontes:
    debe dar exactamente lo mismo que el grid search original por horizonte."""
    filtered = _metrics_inputs(inventory_wide, "ARMEX DE 15 x 20")
    series = filtered.set_index("fecha")["consumo"].sort_index()

    best_model, best_order = inventory.fit_best_arima(series)
    for horizon in (8, 29):
        expected, expected_order = ref.app_arima_forecast(series, forecast_days=horizon)
        assert best_order == expected_order
        assert_series_equal(inventory.forecast_from_fit(best_model, horizon), expected, check_exact=True)


def _concurrent_forecasts_are_correct(cache_decorator, inventory_wide):
    cached_fit = cache_decorator(max_entries=4, show_spinner=False)(inventory.fit_best_arima)
    filtered = _metrics_inputs(inventory_wide, "ALAMBRON")
    series = filtered.set_index("fecha")["consumo"].sort_index()

    best_model, _ = cached_fit(series)
    expected = {h: inventory.forecast_from_fit(best_model, h).to_numpy() for h in range(1, 31)}

    def session(horizon):
        model, _ = cached_fit(series)  # acierto de caché, como otra sesión
        return horizon, inventory.forecast_from_fit(model, horizon).to_numpy()

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(session, list(range(1, 31)) * 10))
    return all(np.array_equal(out, expected[h]) for h, out in results)


def test_cached_fit_is_safe_across_concurrent_sessions(inventory_wide):
    """Varias sesiones de Streamlit usan el modelo en caché al mismo tiempo."""
    st = pytest.importorskip("streamlit")
    assert _concurrent_forecasts_are_correct(st.cache_data, inventory_wide)
