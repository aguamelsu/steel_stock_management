import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
import ui  # noqa: E402


def test_spanish_dates_and_ages():
    assert ui.fecha(pd.Timestamp("2026-09-18")) == "18 sep 2026"
    assert ui.fecha(pd.Timestamp("2026-01-06 15:30")) == "6 ene 2026"
    assert ui.fecha(pd.NaT) == "—" and ui.fecha(None) == "—"
    assert ui.hace(0) == "hoy" and ui.hace(1) == "hace 1 día" and ui.hace(38) == "hace 38 días"


def test_numbers_currency_and_percentages():
    assert ui.mxn(110058) == "$110,058"
    assert ui.mxn(24.5, 2) == "$24.50"
    assert ui.numero(1750.4) == "1,750"
    assert ui.porcentaje(0.3333) == "33.3%"
    for formatter in (ui.mxn, ui.numero, ui.porcentaje):
        assert formatter(np.nan) == "—" and formatter(None) == "—"


def test_freshness_badge_marks_stale_data():
    as_of = pd.Timestamp("2026-10-26")
    stale = ui.freshness_badge("Precios al", pd.Timestamp("2026-07-13"), as_of, 30)
    fresh = ui.freshness_badge("Existencias al", pd.Timestamp("2026-10-24"), as_of, 7)
    assert stale.startswith(":orange-badge[") and "hace 105 días" in stale and "desactualizado" in stale
    assert fresh.startswith(":gray-badge[") and "desactualizado" not in fresh
    assert "sin fecha" in ui.freshness_badge("Precios al", pd.NaT, as_of, 30)
