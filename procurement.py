"""
Tiempos de entrega observados por material y proveedor (hoja pedidos_proveedores).

Solo describe entregas registradas: tiempo de entrega = días naturales entre la
fecha de pedido y la de recepción. La hoja no tiene fecha compromiso, así que no
se calculan puntualidad ni retrasos.
"""
import pandas as pd

SUPPLIER_COLUMNS = [
    "material",
    "proveedor",
    "orden",
    "precio_unitario",
    "cantidad",
    "importe",
    "pedido",
    "recibido",
    "dias_entrega",
]

SUPPLIER_NAMES = {
    "HOME DEPOT MEXICO S DE RL DE CV": "HOME DEPOT",
    "MEXLAM CONCEPTOS SA DE CV": "MEXLAM",
    "CEMEX": "CEMEX",
    "ACEROS VIMAR MEXICO SAPI DE CV": "VIMAR",
}

# Órdenes recibidas necesarias para que un tiempo de entrega cuente como
# evidencia y pueda compararse contra otros proveedores.
MIN_ORDERS_TO_COMPARE = 3
MIN_ORDERS_SUFFICIENT = 5

# Una última recepción más antigua que esto se marca como dato desactualizado.
STALE_AFTER_DAYS = 90

EVIDENCE_LABELS = {
    "sufficient": "Suficiente",
    "limited": "Limitada",
    "insufficient": "Insuficiente",
}


def prepare_purchase_lines(suppliers_df_raw):
    """Hoja pedidos_proveedores -> una fila por partida recibida, con tipos limpios."""
    lines = suppliers_df_raw.copy()
    lines.columns = SUPPLIER_COLUMNS

    lines["material"] = lines["material"].astype(str).str.strip()
    supplier = lines["proveedor"].astype(str).str.strip()
    # Proveedores nuevos conservan su razón social en lugar de perderse
    lines["proveedor"] = supplier.map(SUPPLIER_NAMES).fillna(supplier)

    lines["pedido"] = pd.to_datetime(lines["pedido"], format="%d/%m/%Y")
    lines["recibido"] = pd.to_datetime(lines["recibido"], format="%d/%m/%Y")
    for column in ["precio_unitario", "cantidad", "importe", "dias_entrega"]:
        lines[column] = pd.to_numeric(lines[column], errors="coerce")

    return lines


def inconsistent_lead_times(lines):
    """Partidas cuyo 'dias en llegar' no coincide con recibido - pedido."""
    calculated = (lines["recibido"] - lines["pedido"]).dt.days
    return lines[lines["dias_entrega"] != calculated]


def evidence_level(orders):
    if orders >= MIN_ORDERS_SUFFICIENT:
        return "sufficient"
    if orders >= MIN_ORDERS_TO_COMPARE:
        return "limited"
    return "insufficient"


def lead_time_stats(lines, as_of):
    """Una fila por material y proveedor con lo observado en sus entregas."""
    lines = lines.dropna(subset=["dias_entrega"])
    latest = lines.sort_values(["pedido", "recibido"]).groupby(
        ["material", "proveedor"]
    ).tail(1).set_index(["material", "proveedor"])

    stats = lines.groupby(["material", "proveedor"]).agg(
        ordenes=("orden", "nunique"),
        mediana_dias=("dias_entrega", "median"),
        min_dias=("dias_entrega", "min"),
        max_dias=("dias_entrega", "max"),
        primer_pedido=("pedido", "min"),
        ultima_recepcion=("recibido", "max"),
    )
    stats["ultimo_precio"] = latest["precio_unitario"]
    stats["fecha_ultimo_precio"] = latest["pedido"]
    stats["dias_desde_recepcion"] = (pd.Timestamp(as_of) - stats["ultima_recepcion"]).dt.days
    stats["desactualizado"] = stats["dias_desde_recepcion"] > STALE_AFTER_DAYS
    stats["evidencia"] = stats["ordenes"].map(evidence_level)

    return stats.reset_index().sort_values(["material", "mediana_dias", "ordenes"],
                                           ascending=[True, True, False])


def compare_suppliers(stats, material):
    """
    Resumen para decidir proveedor de un material. Solo se comparan proveedores
    con al menos MIN_ORDERS_TO_COMPARE órdenes recibidas.
    """
    rows = stats[stats["material"] == material]
    candidates = rows[rows["ordenes"] >= MIN_ORDERS_TO_COMPARE].sort_values(
        ["mediana_dias", "ordenes"], ascending=[True, False]
    )
    return {
        "rows": rows,
        "candidates": candidates,
        "fastest": candidates.iloc[0] if len(candidates) else None,
        "comparable": len(candidates) >= 2,
    }


def lead_time_matrix(stats):
    """Material x proveedor con 'mediana · órdenes'; sin entregas queda vacío, nunca 0."""
    cell = stats.assign(
        celda=lambda d: d["mediana_dias"].map(lambda v: f"{v:g} d")
        + " · " + d["ordenes"].astype(str)
    )
    matrix = cell.pivot(index="material", columns="proveedor", values="celda")
    return matrix.sort_index().sort_index(axis=1).fillna("—")


def order_dates(lines):
    return lines["pedido"].min(), lines["pedido"].max(), lines["recibido"].max()


def evidence_label(level):
    return EVIDENCE_LABELS[level]
