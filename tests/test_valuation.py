import numpy as np
import pandas as pd
import pytest

import inventory
import procurement
import valuation
from tests import _reference as ref


@pytest.fixture
def lines(suppliers_raw):
    return procurement.prepare_purchase_lines(suppliers_raw)


def _valuation(inventory_wide, prices_raw, lines, lead_time_days=7, units=None):
    long = inventory.melt_inventory(inventory_wide)
    df = inventory.add_consumption(long)
    book = valuation.price_book(prices_raw, lines)
    return valuation.inventory_valuation(long, df, book, lead_time_days, units)


def test_values_match_original_calculation(inventory_wide, prices_raw, lines):
    val = _valuation(inventory_wide, prices_raw, lines).set_index("material")
    old = ref.app_inventory_value(inventory_wide, prices_raw).set_index("material")

    assert val["valor"].sum() == old["valor"].sum()
    for material in old.index:
        assert val.loc[material, "valor"] == old.loc[material, "valor"]
        assert val.loc[material, "cantidad"] == old.loc[material, "inv_al_dia"]


def test_prices_join_by_material_not_row_order(inventory_wide, prices_raw, lines):
    shuffled = prices_raw.sample(frac=1, random_state=3).reset_index(drop=True)
    a = _valuation(inventory_wide, prices_raw, lines).set_index("material")["valor"]
    b = _valuation(inventory_wide, shuffled, lines).set_index("material")["valor"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_missing_price_is_nan_not_zero(inventory_wide, prices_raw, lines):
    partial = prices_raw.iloc[1:]
    val = _valuation(inventory_wide, partial, lines).set_index("material")
    missing = prices_raw["material"].iloc[0]
    assert np.isnan(val.loc[missing, "valor"])
    assert val["participacion"].sum() == pytest.approx(1.0)


def test_shares_sorting_and_concentration(inventory_wide, prices_raw, lines):
    val = _valuation(inventory_wide, prices_raw, lines)
    assert val["valor"].is_monotonic_decreasing
    assert val["participacion"].sum() == pytest.approx(1.0)
    assert valuation.concentration(val) == pytest.approx(val["participacion"].iloc[:3].sum())


def test_maximum_inventory_matches_the_inventory_chart(inventory_wide, prices_raw, lines):
    """Mismo inventario máximo que replenishment_metrics con rango completo y días por defecto."""
    val = _valuation(inventory_wide, prices_raw, lines, lead_time_days=5).set_index("material")
    df = inventory.add_consumption(inventory.melt_inventory(inventory_wide))
    for material in inventory_wide["material"]:
        filtered = inventory.filter_inventory(df, material, df["fecha"].min().date(), df["fecha"].max().date())
        target = inventory.target_days_for(material)
        expected = inventory.replenishment_metrics(filtered, pd.Series([0.0] * (5 + target)), 5, target)
        assert val.loc[material, "inventario_maximo"] == expected["maximum_inventory"]
        assert val.loc[material, "dias_objetivo"] == target


def test_coverage_and_excess_hand_computed():
    wide = pd.DataFrame({
        "material": ["A", "B"],
        "2026-01-01": [100.0, 50.0], "2026-01-02": [90.0, 50.0], "2026-01-03": [80.0, 50.0],
    })
    prices = pd.DataFrame({"material": ["A", "B"], "fecha": ["01/01/2026"] * 2, "precio": [2.0, 3.0]})
    lines = procurement.prepare_purchase_lines(pd.DataFrame(columns=range(9)))
    long = inventory.melt_inventory(wide)
    val = valuation.inventory_valuation(long, inventory.add_consumption(long),
                                        valuation.price_book(prices, lines), 1).set_index("material")

    avg = (0 + 10 + 10) / 3
    assert val.loc["A", "cobertura_dias"] == pytest.approx(80.0 / avg)
    assert np.isinf(val.loc["B", "cobertura_dias"])  # sin consumo
    assert np.isnan(val.loc["B", "exceso_cantidad"])  # sin consumo no hay máximo: no se marca exceso
    _, _, max_a = inventory.inventory_targets(avg, pd.Series([0, 10, 10.0]).std(),
                                             pd.Series([0, 10, 10.0]).std() / avg, 1, 14)
    assert val.loc["A", "exceso_cantidad"] == pytest.approx(max(0.0, 80.0 - max_a))


def test_price_book_sources_and_dates(prices_raw, lines):
    book = valuation.price_book(prices_raw, lines)
    assert (book["fecha_precio_referencia"] == pd.Timestamp("2026-07-13")).all()
    last = lines.sort_values(["pedido", "recibido"]).groupby("material").tail(1).set_index("material")
    pd.testing.assert_series_equal(book["ultimo_costo_compra"].sort_index(),
                                   last["precio_unitario"].sort_index(), check_names=False)


def test_value_quantities_is_reusable_for_planned_purchases():
    planned = pd.Series({"A": 10.0, "B": 4.0, "C": 1.0})
    prices = pd.Series({"A": 2.5, "B": 10.0})
    out = valuation.value_quantities(planned, prices)
    assert out["A"] == 25.0 and out["B"] == 40.0 and np.isnan(out["C"])


def test_material_units():
    md = pd.DataFrame({"material": ["A ", "B"], "unit": ["kg", "pzs"]})
    assert valuation.material_units(md).to_dict() == {"A": "kg", "B": "pzs"}
    assert valuation.material_units(None).empty
    assert valuation.material_units(pd.DataFrame({"x": [1]})).empty
