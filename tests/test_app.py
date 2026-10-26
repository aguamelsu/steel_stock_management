"""
Pruebas de la app completa con Streamlit AppTest. Google Sheets se sustituye por
los CSV de data/, así que no se usan credenciales ni red.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import inventory  # noqa: E402
from tests import _reference as ref  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"
ARMEX = "ARMEX DE 15 x 15"
OTHER = "ALAMBRE RECOCIDO"


@pytest.fixture
def app(monkeypatch, inventory_wide, suppliers_raw, prices_raw):
    import gspread
    import google.oauth2.service_account as sa

    sheets = {
        "inventario": inventory_wide,
        "pedidos_proveedores": suppliers_raw,
        "precios": prices_raw,
    }

    class Worksheet:
        def __init__(self, title):
            self.title = title

        def get_all_records(self):
            return sheets[self.title].to_dict("records")

    class Spreadsheet:
        def worksheets(self):
            return [Worksheet(t) for t in sheets]

    class Client:
        def open_by_url(self, url):
            return Spreadsheet()

    monkeypatch.setattr(gspread, "authorize", lambda credentials: Client())
    monkeypatch.setattr(sa.Credentials, "from_service_account_info",
                        classmethod(lambda cls, info, scopes=None: object()))

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.secrets["gcp_service_account"] = {"type": "service_account"}
    at.secrets["google_sheets"] = {"spreadsheet_url": "stub"}
    return at


def _target_days_input(at):
    return next(n for n in at.number_input if n.label == "Objetivo de días de inventario")


def _lead_time_input(at):
    return next(n for n in at.number_input if n.label == "Días de anticipo para realizar pedido")


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _min_stock_trace_name(at):
    import json
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    return next(t["name"] for t in spec["data"] if t["name"].startswith("Stock mínimo"))


def _expected_original(inventory_wide, material, lead_time_days):
    """Resultados de la lógica original (target_days fijo 7/14 por material)."""
    df = ref.app_preprocess(inventory_wide)
    filtered = ref.app_filter(df, material, df["fecha"].min().date(), df["fecha"].max().date())
    series = filtered.set_index("fecha")["consumo"].sort_index()
    target_days = 7 if material == ARMEX else 14
    forecast, _ = ref.app_arima_forecast(series, forecast_days=lead_time_days + target_days)
    return ref.app_metrics(filtered, forecast, lead_time_days, material)


def _assert_rendered(at, m):
    assert not at.exception
    assert _metric(at, "Inventario al día") == f"{m['current_inventory']:,.2f}"
    assert _metric(at, "Consumo promedio") == f"{m['avg_consumption']:,.2f}"
    assert _metric(at, "ROP") == f"{m['reorder_point']:,.2f}"
    assert _metric(at, "Cantidad recomendada de compra") == f"{m['recommended_qty']:,.2f}"
    assert (len(at.error) > 0) == bool(m["buy_now"])


def test_default_target_days_preserve_original_results(app, inventory_wide):
    app.run()
    app.selectbox[0].select(OTHER).run()
    assert _target_days_input(app).value == 14
    assert _min_stock_trace_name(app) == "Stock mínimo (14 días)"
    _assert_rendered(app, _expected_original(inventory_wide, OTHER, 7))

    app.selectbox[0].select(ARMEX).run()
    assert _target_days_input(app).value == 7
    assert _min_stock_trace_name(app) == "Stock mínimo (7 días)"
    _assert_rendered(app, _expected_original(inventory_wide, ARMEX, 7))


def test_target_days_override_is_applied(app, inventory_wide):
    app.run()
    app.selectbox[0].select(ARMEX).run()
    _lead_time_input(app).set_value(5).run()
    _target_days_input(app).set_value(30).run()

    assert _target_days_input(app).value == 30
    assert _min_stock_trace_name(app) == "Stock mínimo (30 días)"

    df = inventory.add_consumption(inventory.melt_inventory(inventory_wide))
    filtered = inventory.filter_inventory(df, ARMEX, df["fecha"].min().date(), df["fecha"].max().date())
    best_model, _ = inventory.fit_best_arima(filtered.set_index("fecha")["consumo"].sort_index())
    forecast = inventory.forecast_from_fit(best_model, 5 + 30)
    _assert_rendered(app, inventory.replenishment_metrics(filtered, forecast, 5, 30))


def test_changing_material_resets_target_days_to_its_default(app):
    app.run()
    app.selectbox[0].select(ARMEX).run()
    _target_days_input(app).set_value(21).run()
    assert _target_days_input(app).value == 21

    app.selectbox[0].select(OTHER).run()
    assert _target_days_input(app).value == 14

    app.selectbox[0].select(ARMEX).run()
    assert _target_days_input(app).value == 7
