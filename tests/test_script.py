from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

pytest.importorskip("matplotlib")
import script  # noqa: E402
from tests import _reference as ref  # noqa: E402

XLSX = Path(__file__).resolve().parent.parent / "inventario_aceros (1).xlsx"


def _purchases(rows):
    return pd.DataFrame(rows, columns=["Material", "Cantidad", "recibido"]).assign(
        recibido=lambda d: pd.to_datetime(d["recibido"])
    )


def _inventory(material, values, start="2026-01-01"):
    return pd.DataFrame({
        "material": material,
        "fecha": pd.date_range(start, periods=len(values), freq="D"),
        "inventario": values,
    })


def test_reconstruct_consumption_hand_computed():
    inv = pd.concat([
        _inventory("A", [100.0, 80.0, 130.0, 120.0, 90.0]),
        _inventory("B", [10.0, 12.0, 8.0]),  # sin recepciones: la subida es un ajuste
    ], ignore_index=True)
    purchases = _purchases([
        ["A", 60, "2026-01-03"],   # entra en (01-02, 01-03]
        ["A", 5, "2026-01-01"],    # == fecha del primer snapshot: no entra en ningún intervalo
        ["A", 7, "2026-01-05"],    # límite superior incluido
        ["A", np.nan, "2026-01-04"],
        ["C", 999, "2026-01-02"],  # otro material
    ])

    out = script.reconstruct_consumption(inv, purchases)
    a = out[out["material"] == "A"]
    b = out[out["material"] == "B"]

    assert a["recepciones_intervalo"].tolist()[1:] == [0, 60, 0, 7]
    assert a["consumo_intervalo"].tolist()[1:] == [20.0, 10.0, 10.0, 37.0]
    assert not a["ajuste_datos"].any()

    assert b["consumo_intervalo"].isna().tolist() == [True, True, False]
    assert b["ajuste_datos"].tolist() == [False, True, False]


def test_reconstruct_consumption_matches_original_random():
    rng = np.random.default_rng(42)
    inv = pd.concat(
        [_inventory(m, rng.integers(0, 500, 40).astype(float)) for m in ["A", "B", "C"]],
        ignore_index=True,
    )
    purchases = _purchases([
        [m, float(rng.integers(1, 300)), pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(d))]
        for m, d in zip(rng.choice(["A", "B", "D"], 50), rng.integers(-3, 45, 50))
    ])
    assert_frame_equal(
        script.reconstruct_consumption(inv, purchases),
        ref.script_reconstruct_consumption(inv, purchases, script.count_days_between),
        check_exact=True,
    )


def test_evaluate_current_policy_matches_original():
    rng = np.random.default_rng(7)
    frames = []
    for material, n in [("A", 40), ("B", 12), ("C", 11), ("D", 3)]:  # C y D: series demasiado cortas
        dates = pd.date_range("2026-03-01", periods=n, freq="D")
        keep = rng.random(n) > 0.2  # huecos que asfreq rellena con 0
        keep[[0, -1]] = True
        frames.append(pd.DataFrame({
            "material": material,
            "fecha": dates[keep],
            "consumo_estimado_dia": rng.choice([0.0, 3.0, 4.5, 10.0], keep.sum()),
        }))
    daily = pd.concat(frames, ignore_index=True)

    assert_frame_equal(
        script.evaluate_current_policy(daily),
        ref.script_evaluate_current_policy(daily),
        check_exact=True,
    )


def test_evaluate_current_policy_empty():
    assert script.evaluate_current_policy(pd.DataFrame()).empty


def test_full_pipeline_on_workbook_matches_original():
    pytest.importorskip("openpyxl")
    if not XLSX.exists():
        pytest.skip("No está el archivo de Excel")

    inv_long, purchases, _, _ = script.load_data(XLSX)
    reconstructed = script.reconstruct_consumption(inv_long, purchases)
    assert_frame_equal(
        reconstructed,
        ref.script_reconstruct_consumption(inv_long, purchases, script.count_days_between),
        check_exact=True,
    )

    daily = script.build_daily_series(reconstructed)
    assert_frame_equal(
        script.evaluate_current_policy(daily),
        ref.script_evaluate_current_policy(daily),
        check_exact=True,
    )
