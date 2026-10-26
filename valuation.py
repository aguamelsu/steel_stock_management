"""
Valuación de inventario: valor = cantidad x precio, por material.

Pensado para reutilizarse en planeación financiera: price_book reúne cada fuente
de precio con su fecha, y value_quantities valora cualquier vector de cantidades
por material (hoy las existencias; después, compras planeadas o presupuestos).
"""
import numpy as np
import pandas as pd

import inventory

# Antigüedad a partir de la cual se advierte que el dato está desactualizado.
STALE_QUANTITIES_AFTER_DAYS = 7
STALE_PRICES_AFTER_DAYS = 30


def material_units(mat_dict_df):
    """Unidad de medida por material desde la hoja mat_dict (vacío si no existe)."""
    if mat_dict_df is None or not {"material", "unit"} <= set(mat_dict_df.columns):
        return pd.Series(dtype=str)
    units = mat_dict_df.drop_duplicates("material")
    return pd.Series(units["unit"].astype(str).str.strip().to_numpy(),
                     index=units["material"].astype(str).str.strip())


def latest_quantities(inventory_long):
    """Existencia por material en la última columna (fecha) de la hoja inventario."""
    last_date = inventory_long["fecha"].iloc[-1]
    latest = inventory_long[inventory_long["fecha"] == last_date]

    return pd.DataFrame(
        {"cantidad": latest["inventario"].to_numpy(), "fecha_cantidad": last_date},
        index=pd.Index(latest["material"].astype(str).str.strip(), name="material"),
    )


def price_book(prices_df_raw, purchase_lines):
    """
    Precios por material con su origen y fecha:
      - precio_referencia: hoja 'precios' (la fecha más reciente por material).
      - ultimo_costo_compra: precio unitario de la última orden en pedidos_proveedores.
    Ninguno es costo de reposición actual.
    """
    prices = prices_df_raw.copy()
    prices["material"] = prices["material"].astype(str).str.strip()
    prices["fecha"] = pd.to_datetime(prices["fecha"], dayfirst=True)
    prices["precio"] = pd.to_numeric(prices["precio"], errors="coerce")
    reference = (
        prices.dropna(subset=["precio"])
        .sort_values("fecha")
        .groupby("material").tail(1)
        .set_index("material")
    )

    last_purchase = (
        purchase_lines.dropna(subset=["precio_unitario"])
        .sort_values(["pedido", "recibido"])
        .groupby("material").tail(1)
        .set_index("material")
    )

    book = pd.DataFrame({
        "precio_referencia": reference["precio"],
        "fecha_precio_referencia": reference["fecha"],
    }).join(pd.DataFrame({
        "ultimo_costo_compra": last_purchase["precio_unitario"],
        "fecha_ultimo_costo": last_purchase["pedido"],
        "proveedor_ultimo_costo": last_purchase["proveedor"],
    }), how="outer")
    book.index.name = "material"

    return book


def value_quantities(quantities, prices):
    """Cantidad x precio alineado por material; sin precio -> NaN (nunca 0)."""
    return quantities * prices.reindex(quantities.index)


def inventory_valuation(inventory_long, consumption_df, book, lead_time_days, units=None):
    """
    Una fila por material: existencia, valor a precio de referencia, participación,
    cobertura y exceso sobre el inventario máximo (misma fórmula que la gráfica
    de inventario, con los días objetivo por defecto de cada material).
    """
    val = latest_quantities(inventory_long).join(book, how="left")

    # Pesos enteros, igual que el valor que mostraba la versión anterior
    val["valor"] = value_quantities(val["cantidad"], val["precio_referencia"]).round(0)
    val["participacion"] = val["valor"] / val["valor"].sum()

    # Mismas estadísticas y fórmula que la gráfica de inventario (rango completo),
    # para que el inventario máximo coincida exactamente con su línea punteada
    consumption = consumption_df.assign(
        material=consumption_df["material"].astype(str).str.strip()
    )
    stats = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for material, rows in consumption.groupby("material", sort=False):
            _, avg, std, cv = inventory.consumption_stats(rows)
            _, _, max_inv = inventory.inventory_targets(
                avg, std, cv, lead_time_days, inventory.target_days_for(material)
            )
            stats[material] = (avg, std, max_inv)
    stats = pd.DataFrame.from_dict(stats, orient="index",
                                   columns=["consumo_promedio", "consumo_desv_est", "inventario_maximo"])
    val = val.join(stats)

    with np.errstate(divide="ignore", invalid="ignore"):
        val["cobertura_dias"] = np.where(
            val["consumo_promedio"] > 0, val["cantidad"] / val["consumo_promedio"], np.inf
        )
        val["dif_costo_vs_referencia"] = val["ultimo_costo_compra"] / val["precio_referencia"] - 1
    val["dias_objetivo"] = [inventory.target_days_for(m) for m in val.index]

    has_max = val["inventario_maximo"].notna() & np.isfinite(val["inventario_maximo"])
    val["exceso_cantidad"] = (val["cantidad"] - val["inventario_maximo"]).clip(lower=0).where(has_max)
    val["exceso_valor"] = (val["exceso_cantidad"] * val["precio_referencia"]).round(0)

    val["unidad"] = (units if units is not None else pd.Series(dtype=str)).reindex(val.index)

    return val.sort_values("valor", ascending=False, na_position="last").reset_index()


def concentration(valuation, top_n=3):
    """Participación conjunta de los top_n materiales por valor."""
    return valuation["participacion"].nlargest(top_n).sum()
