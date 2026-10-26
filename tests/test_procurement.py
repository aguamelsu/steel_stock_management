import numpy as np
import pandas as pd
import pytest

import procurement
from tests import _reference as ref

AS_OF = pd.Timestamp("2026-10-26")


def _lines(rows):
    """rows: (material, proveedor, orden, pedido dd/mm/aaaa, recibido dd/mm/aaaa)."""
    raw = pd.DataFrame([
        {"Material": m, "Proveedor (Texto)": p, "ID de pedido de compra": o, "PU Promedio": 10.0 + o,
         "Cantidad": 1.0, "Valor facturado": 10, "pedido": ped, "recibido": rec,
         "dias en llegar": (pd.to_datetime(rec, dayfirst=True) - pd.to_datetime(ped, dayfirst=True)).days}
        for m, p, o, ped, rec in rows
    ])
    return procurement.prepare_purchase_lines(raw)


def test_prepare_purchase_lines_real_data(suppliers_raw):
    lines = procurement.prepare_purchase_lines(suppliers_raw)
    assert len(lines) == len(suppliers_raw)
    assert set(lines["proveedor"]) == {"HOME DEPOT", "MEXLAM", "CEMEX", "VIMAR"}
    assert lines["pedido"].dt.month.max() <= 12 and lines["recibido"].notna().all()
    assert procurement.inconsistent_lead_times(lines).empty


def test_unknown_supplier_keeps_its_name_instead_of_disappearing():
    lines = _lines([("A", "PROVEEDOR NUEVO SA DE CV", 1, "01/03/2026", "05/03/2026")])
    assert lines["proveedor"].tolist() == ["PROVEEDOR NUEVO SA DE CV"]


def test_lead_time_stats_uses_median_and_counts_orders():
    lines = _lines([
        ("A", "HOME DEPOT MEXICO S DE RL DE CV", 1, "01/03/2026", "03/03/2026"),
        ("A", "HOME DEPOT MEXICO S DE RL DE CV", 2, "01/04/2026", "05/04/2026"),
        ("A", "HOME DEPOT MEXICO S DE RL DE CV", 3, "01/05/2026", "17/06/2026"),  # atípica 47 días
        ("A", "CEMEX", 4, "01/06/2026", "03/06/2026"),
    ])
    stats = procurement.lead_time_stats(lines, AS_OF).set_index("proveedor")

    hd = stats.loc["HOME DEPOT"]
    assert (hd["ordenes"], hd["mediana_dias"], hd["min_dias"], hd["max_dias"]) == (3, 4, 2, 47)
    assert hd["ultima_recepcion"] == pd.Timestamp("2026-06-17")
    assert hd["ultimo_precio"] == 13.0 and hd["fecha_ultimo_precio"] == pd.Timestamp("2026-05-01")
    assert hd["evidencia"] == "limited"
    assert hd["dias_desde_recepcion"] == (AS_OF - pd.Timestamp("2026-06-17")).days
    assert bool(hd["desactualizado"])  # más de 90 días
    assert stats.loc["CEMEX", "evidencia"] == "insufficient"


@pytest.mark.parametrize("orders, level", [(1, "insufficient"), (2, "insufficient"), (3, "limited"),
                                           (4, "limited"), (5, "sufficient"), (12, "sufficient")])
def test_evidence_level(orders, level):
    assert procurement.evidence_level(orders) == level


def test_compare_only_ranks_suppliers_with_enough_orders():
    rows = [("A", "CEMEX", 1, "01/03/2026", "02/03/2026")]  # 1 día, pero un solo dato
    rows += [("A", "HOME DEPOT MEXICO S DE RL DE CV", 10 + i, "01/03/2026", "08/03/2026") for i in range(3)]
    stats = procurement.lead_time_stats(_lines(rows), AS_OF)

    result = procurement.compare_suppliers(stats, "A")
    assert result["fastest"]["proveedor"] == "HOME DEPOT"  # CEMEX es más rápido pero sin evidencia
    assert not result["comparable"]


def test_compare_two_supported_suppliers():
    rows = [("A", "CEMEX", i, "01/03/2026", "04/03/2026") for i in range(3)]
    rows += [("A", "HOME DEPOT MEXICO S DE RL DE CV", 10 + i, "01/03/2026", "08/03/2026") for i in range(5)]
    result = procurement.compare_suppliers(procurement.lead_time_stats(_lines(rows), AS_OF), "A")
    assert result["comparable"]
    assert result["fastest"]["proveedor"] == "CEMEX"


def test_compare_without_evidence_or_data():
    stats = procurement.lead_time_stats(_lines([("A", "CEMEX", 1, "01/03/2026", "02/03/2026")]), AS_OF)
    assert procurement.compare_suppliers(stats, "A")["fastest"] is None
    assert procurement.compare_suppliers(stats, "B")["rows"].empty


def test_matrix_never_shows_missing_pairs_as_zero(suppliers_raw):
    stats = procurement.lead_time_stats(procurement.prepare_purchase_lines(suppliers_raw), AS_OF)
    matrix = procurement.lead_time_matrix(stats)
    old = ref.app_suppliers_pivot(suppliers_raw)

    # Donde el heatmap anterior mostraba 0 no hay entregas: ahora es "—"
    zeros = [(m, p) for m in old.index for p in old.columns if old.loc[m, p] == 0]
    assert zeros and all(matrix.loc[m, p] == "—" for m, p in zeros)
    assert not (stats["mediana_dias"] == 0).any()


def test_matrix_cell_shows_median_and_orders():
    rows = [("A", "CEMEX", i, "01/03/2026", d) for i, d in enumerate(["03/03/2026", "04/03/2026", "11/03/2026"])]
    matrix = procurement.lead_time_matrix(procurement.lead_time_stats(_lines(rows), AS_OF))
    assert matrix.loc["A", "CEMEX"] == "3 d · 3"
