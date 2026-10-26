"""
inventory_report_metrics.py

Calcula las métricas necesarias para diagnosticar la política actual de inventario
y preparar la propuesta de estrategias de compra.

Archivo esperado:
    inventario_aceros.xlsx

Hojas esperadas:
    - inventario
    - mat_dict
    - pedidos_proveedores
    - precios

Salidas:
    reporte_metricas_inventario/
        01_resumen_metricas.csv
        02_consumo_diario_reconstruido.csv
        03_consumo_semanal.csv
        04_lead_times.csv
        05_compras.csv
        06_politica_5_dias.csv
        07_calidad_datos.csv
        08_metricas_rop.csv
        graficas/*.png

REQUISITOS:
    pip install pandas numpy openpyxl matplotlib

NOTA METODOLÓGICA:
    Entre dos snapshots de inventario:

        Consumo = Inventario_anterior + Recepciones_en_intervalo - Inventario_actual

    Las recepciones se toman de 'pedidos_proveedores' usando la fecha 'recibido'.
    Esto evita interpretar una recepción como consumo negativo.

    Para tasas diarias se divide el consumo reconstruido entre los días calendario
    transcurridos entre snapshots. Si tu operación debe analizarse exclusivamente
    por días hábiles, cambia DAYS_MODE a "business".
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib

# Backend sin interfaz gráfica: las gráficas solo se guardan como PNG, y el
# backend interactivo por defecto (TkAgg) crea ventanas que nunca se muestran.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

INPUT_FILE = "inventario_aceros (1).xlsx"
OUTPUT_DIR = Path("reporte_metricas_inventario")

# "calendar" = días calendario entre snapshots
# "business" = lunes-viernes entre snapshots
DAYS_MODE = "calendar"

# Política actual
CURRENT_COVERAGE_DAYS = 5

# Horizonte usado para evaluar si la semana anterior predice los siguientes 5 días
TRAILING_DAYS = 7
FORWARD_DAYS = 5

# Nivel de servicio ilustrativo para métricas ROP
SERVICE_LEVEL_Z = 1.645  # ~95%

# Si conoces el tiempo administrativo interno promedio, colócalo aquí.
# Puede ser un solo valor para todos los materiales:
DEFAULT_ADMIN_DAYS = 0.0

# O puedes especificarlo por material:
ADMIN_DAYS_BY_MATERIAL = {
    # 'VARILLA CORRUGADA 3/8"': 2.0,
    # 'MALLA ELECTROSOLDADA 6-6-10/10': 2.0,
}

# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def safe_cv(std, mean):
    if pd.isna(mean) or mean == 0:
        return np.nan
    return std / mean


def safe_pct(num, den):
    if den == 0 or pd.isna(den):
        return np.nan
    return 100.0 * num / den


def count_days_between(start, end, mode="calendar"):
    """
    Número de días usados como denominador para transformar consumo de un
    intervalo en tasa diaria.

    Ejemplo:
        snapshot lunes -> snapshot martes = 1 día.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if end <= start:
        return np.nan

    if mode == "calendar":
        return (end - start).days

    if mode == "business":
        # Cuenta días hábiles en (start, end], sin considerar feriados.
        dates = pd.date_range(start + pd.Timedelta(days=1), end, freq="D")
        return int((dates.dayofweek < 5).sum())

    raise ValueError("DAYS_MODE debe ser 'calendar' o 'business'.")


def series_wape(actual, forecast):
    mask = actual.notna() & forecast.notna()
    a = actual[mask]
    f = forecast[mask]
    den = a.abs().sum()
    if den == 0:
        return np.nan
    return 100.0 * (a - f).abs().sum() / den


def longest_true_run(values):
    max_run = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ---------------------------------------------------------------------
# CARGA Y LIMPIEZA
# ---------------------------------------------------------------------

def load_data(path):
    # Abrir el libro una sola vez en lugar de una vez por hoja
    with pd.ExcelFile(path) as book:
        inv_raw = pd.read_excel(book, sheet_name="inventario", header=None)
        purchases = pd.read_excel(book, sheet_name="pedidos_proveedores")
        mat_dict = pd.read_excel(book, sheet_name="mat_dict")
        prices = pd.read_excel(book, sheet_name="precios")

    # ----- Inventario -----
    header = inv_raw.iloc[0].tolist()

    valid_cols = [0]
    parsed_dates = {}

    for j, value in enumerate(header[1:], start=1):
        dt = pd.to_datetime(value, errors="coerce")
        if pd.notna(dt):
            valid_cols.append(j)
            parsed_dates[j] = pd.Timestamp(dt).normalize()

    inv = inv_raw.iloc[1:, valid_cols].copy()
    inv.columns = ["material"] + [parsed_dates[j] for j in valid_cols[1:]]
    inv["material"] = inv["material"].astype(str).str.strip()
    inv = inv[inv["material"].ne("") & inv["material"].ne("nan")]

    for c in inv.columns[1:]:
        inv[c] = pd.to_numeric(inv[c], errors="coerce")

    inv_long = inv.melt(
        id_vars="material",
        var_name="fecha",
        value_name="inventario"
    )
    inv_long["fecha"] = pd.to_datetime(inv_long["fecha"]).dt.normalize()
    inv_long = inv_long.dropna(subset=["inventario"])
    inv_long = inv_long.sort_values(["material", "fecha"]).reset_index(drop=True)

    # ----- Pedidos -----
    purchases.columns = purchases.columns.astype(str).str.strip()
    purchases["Material"] = purchases["Material"].astype(str).str.strip()
    purchases["pedido"] = pd.to_datetime(purchases["pedido"], errors="coerce").dt.normalize()
    purchases["recibido"] = pd.to_datetime(purchases["recibido"], errors="coerce").dt.normalize()

    for c in ["PU Promedio", "Cantidad", "Valor facturado", "dias en llegar"]:
        purchases[c] = pd.to_numeric(purchases[c], errors="coerce")

    # Recalcular LT como validación
    purchases["lead_time_calculado"] = (
        purchases["recibido"] - purchases["pedido"]
    ).dt.days

    # ----- Diccionario de materiales -----
    mat_dict.columns = mat_dict.columns.astype(str).str.strip()
    mat_dict["material"] = mat_dict["material"].astype(str).str.strip()
    mat_dict["minimum_stock"] = pd.to_numeric(
        mat_dict["minimum_stock"], errors="coerce"
    )

    # ----- Precios -----
    prices.columns = prices.columns.astype(str).str.strip()
    prices["material"] = prices["material"].astype(str).str.strip()
    prices["fecha"] = pd.to_datetime(prices["fecha"], errors="coerce").dt.normalize()
    prices["precio"] = pd.to_numeric(prices["precio"], errors="coerce")

    return inv_long, purchases, mat_dict, prices


# ---------------------------------------------------------------------
# RECONSTRUCCIÓN DEL CONSUMO
# ---------------------------------------------------------------------

def reconstruct_consumption(inv_long, purchases):
    rows = []

    receipts = purchases.dropna(subset=["recibido", "Cantidad"]).copy()

    # Recepciones separadas por material una sola vez (mismo orden de filas),
    # en lugar de filtrar la tabla completa en cada intervalo.
    receipts_by_material = {
        m: (r["recibido"].to_numpy(), r["Cantidad"].to_numpy())
        for m, r in receipts.groupby("Material", sort=False)
    }
    no_receipts = (
        receipts["recibido"].to_numpy()[:0],
        receipts["Cantidad"].to_numpy()[:0],
    )

    for material, g in inv_long.groupby("material"):
        g = g.sort_values("fecha").reset_index(drop=True)
        receipt_dates, receipt_qtys = receipts_by_material.get(material, no_receipts)

        dates = list(g["fecha"])
        stocks = g["inventario"].to_numpy()

        for i in range(len(g)):
            date = dates[i]
            stock = stocks[i]

            if i == 0:
                rows.append({
                    "material": material,
                    "fecha": date,
                    "inventario": stock,
                    "fecha_anterior": pd.NaT,
                    "inventario_anterior": np.nan,
                    "dias_intervalo": np.nan,
                    "recepciones_intervalo": np.nan,
                    "consumo_intervalo": np.nan,
                    "consumo_diario": np.nan,
                    "ajuste_datos": False
                })
                continue

            prev_date = dates[i - 1]
            prev_stock = stocks[i - 1]

            days = count_days_between(prev_date, date, DAYS_MODE)

            in_interval = (
                (receipt_dates > prev_date.to_datetime64()) &
                (receipt_dates <= date.to_datetime64())
            )

            receipt_qty = receipt_qtys[in_interval].sum()

            raw_consumption = prev_stock + receipt_qty - stock

            # Un valor negativo significa que hubo una entrada que no pudo ser
            # conciliada con la tabla de pedidos, corrección de inventario u otra
            # inconsistencia. No se convierte silenciosamente en consumo.
            data_adjustment = raw_consumption < -1e-9

            consumption = raw_consumption if raw_consumption >= 0 else np.nan
            daily = consumption / days if pd.notna(consumption) and days and days > 0 else np.nan

            rows.append({
                "material": material,
                "fecha": date,
                "inventario": stock,
                "fecha_anterior": prev_date,
                "inventario_anterior": prev_stock,
                "dias_intervalo": days,
                "recepciones_intervalo": receipt_qty,
                "consumo_intervalo": consumption,
                "consumo_diario": daily,
                "ajuste_datos": data_adjustment
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SERIE DIARIA CALENDARIO
# ---------------------------------------------------------------------

def build_daily_series(reconstructed):
    """
    Distribuye uniformemente el consumo reconstruido de cada intervalo entre los
    días incluidos en ese intervalo. Esto no pretende inventar el día exacto de
    salida: sirve para obtener tasas, coberturas y comparaciones consistentes.

    Si tienes movimientos diarios reales, reemplaza esta aproximación por ellos.
    """
    all_rows = []

    for material, g in reconstructed.groupby("material"):
        g = g.sort_values("fecha")

        for _, r in g.iterrows():
            if pd.isna(r["fecha_anterior"]) or pd.isna(r["consumo_intervalo"]):
                continue

            start = pd.Timestamp(r["fecha_anterior"])
            end = pd.Timestamp(r["fecha"])
            total = r["consumo_intervalo"]

            dates = pd.date_range(start + pd.Timedelta(days=1), end, freq="D")

            if DAYS_MODE == "business":
                dates = dates[dates.dayofweek < 5]

            if len(dates) == 0:
                continue

            per_day = total / len(dates)

            for dt in dates:
                all_rows.append({
                    "material": material,
                    "fecha": dt.normalize(),
                    "consumo_estimado_dia": per_day
                })

    daily = pd.DataFrame(all_rows)

    if daily.empty:
        return daily

    daily = (
        daily.groupby(["material", "fecha"], as_index=False)
        ["consumo_estimado_dia"].sum()
        .sort_values(["material", "fecha"])
    )

    return daily


# ---------------------------------------------------------------------
# CONSUMO SEMANAL
# ---------------------------------------------------------------------

def weekly_consumption(daily):
    if daily.empty:
        return pd.DataFrame()

    df = daily.copy()
    iso = df["fecha"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)

    weekly = (
        df.groupby(["material", "iso_year", "iso_week"], as_index=False)
        ["consumo_estimado_dia"].sum()
        .rename(columns={"consumo_estimado_dia": "consumo_semanal"})
    )
    return weekly


# ---------------------------------------------------------------------
# EVALUACIÓN DE LA POLÍTICA: SEMANA ANTERIOR + 5 DÍAS
# ---------------------------------------------------------------------

def evaluate_current_policy(daily):
    """
    Para cada fecha:
      pronóstico 5 días = promedio diario de los 7 días previos * 5
      consumo real 5 días = suma de los siguientes 5 días

    Reporta:
      - subestimación
      - error absoluto
      - sesgo
    """
    rows = []

    if daily.empty:
        return pd.DataFrame()

    for material, g in daily.groupby("material"):
        s = (
            g.set_index("fecha")["consumo_estimado_dia"]
            .sort_index()
            .asfreq("D", fill_value=0)
        )

        # s tiene frecuencia diaria continua, así que las ventanas
        # [fecha-7d, fecha-1d] y [fecha, fecha+4d] son posiciones fijas; solo se
        # recorren las fechas que tienen ambas ventanas completas.
        for i in range(TRAILING_DAYS, len(s) - FORWARD_DAYS + 1):
            date = s.index[i]
            hist = s.iloc[i - TRAILING_DAYS:i]
            fut = s.iloc[i:i + FORWARD_DAYS]

            daily_avg = hist.mean()
            forecast_5d = daily_avg * FORWARD_DAYS
            actual_5d = fut.sum()

            rows.append({
                "material": material,
                "fecha_evaluacion": date,
                "consumo_7d_previos": hist.sum(),
                "promedio_diario_7d": daily_avg,
                "pronostico_5d": forecast_5d,
                "consumo_real_5d": actual_5d,
                "error": forecast_5d - actual_5d,
                "error_absoluto": abs(forecast_5d - actual_5d),
                "subestima": forecast_5d < actual_5d,
                "sobreestima": forecast_5d > actual_5d
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# RESUMEN POR MATERIAL
# ---------------------------------------------------------------------

def make_summary(
    inv_long,
    reconstructed,
    daily,
    weekly,
    purchases,
    mat_dict,
    prices,
    policy_eval
):
    materials = sorted(inv_long["material"].unique())
    rows = []

    minimum_map = (
        mat_dict.drop_duplicates("material")
        .set_index("material")["minimum_stock"]
        .to_dict()
    )

    latest_price_map = (
        prices.sort_values("fecha")
        .dropna(subset=["precio"])
        .groupby("material")
        .tail(1)
        .set_index("material")["precio"]
        .to_dict()
    )

    for material in materials:
        inv = inv_long[inv_long["material"] == material].sort_values("fecha")
        rec = reconstructed[reconstructed["material"] == material].sort_values("fecha")
        d = daily[daily["material"] == material].sort_values("fecha")
        w = weekly[weekly["material"] == material].copy()
        p = purchases[purchases["Material"] == material].sort_values("pedido")
        pe = policy_eval[policy_eval["material"] == material].copy()

        min_stock = minimum_map.get(material, np.nan)
        latest_price = latest_price_map.get(material, np.nan)

        # ---------------- Consumo ----------------
        total_consumption = rec["consumo_intervalo"].sum(min_count=1)

        period_start = inv["fecha"].min()
        period_end = inv["fecha"].max()
        period_days = max((period_end - period_start).days, 1)

        avg_daily = total_consumption / period_days if pd.notna(total_consumption) else np.nan

        daily_std = d["consumo_estimado_dia"].std(ddof=1) if len(d) > 1 else np.nan

        weekly_mean = w["consumo_semanal"].mean() if not w.empty else np.nan
        weekly_std = w["consumo_semanal"].std(ddof=1) if len(w) > 1 else np.nan
        weekly_cv = safe_cv(weekly_std, weekly_mean)
        weekly_max = w["consumo_semanal"].max() if not w.empty else np.nan

        nonzero_pct = (
            safe_pct((d["consumo_estimado_dia"] > 0).sum(), len(d))
            if len(d) else np.nan
        )

        # ---------------- Inventario ----------------
        avg_stock = inv["inventario"].mean()
        min_inv = inv["inventario"].min()
        max_inv = inv["inventario"].max()
        current_stock = inv.iloc[-1]["inventario"]

        pct_below_min = (
            safe_pct((inv["inventario"] < min_stock).sum(), len(inv))
            if pd.notna(min_stock) else np.nan
        )

        stockout_observations = int((inv["inventario"] <= 0).sum())

        # Cobertura usando demanda media del periodo
        if pd.notna(avg_daily) and avg_daily > 0:
            coverage = inv["inventario"] / avg_daily
            avg_coverage = coverage.mean()
            min_coverage = coverage.min()
            max_coverage = coverage.max()
            pct_under_5 = safe_pct((coverage < CURRENT_COVERAGE_DAYS).sum(), len(coverage))
            current_coverage = current_stock / avg_daily
        else:
            avg_coverage = min_coverage = max_coverage = pct_under_5 = current_coverage = np.nan

        # ---------------- Lead time ----------------
        lead = p["dias en llegar"].dropna()
        lt_mean = lead.mean() if len(lead) else np.nan
        lt_median = lead.median() if len(lead) else np.nan
        lt_std = lead.std(ddof=1) if len(lead) > 1 else np.nan
        lt_min = lead.min() if len(lead) else np.nan
        lt_max = lead.max() if len(lead) else np.nan
        lt_p90 = lead.quantile(0.90) if len(lead) else np.nan
        pct_lt_over_5 = safe_pct((lead > CURRENT_COVERAGE_DAYS).sum(), len(lead)) if len(lead) else np.nan

        admin_days = ADMIN_DAYS_BY_MATERIAL.get(material, DEFAULT_ADMIN_DAYS)
        replenishment_mean = lt_mean + admin_days if pd.notna(lt_mean) else np.nan
        replenishment_p90 = lt_p90 + admin_days if pd.notna(lt_p90) else np.nan

        # ---------------- Compras ----------------
        n_orders = len(p)

        if n_orders and p["pedido"].notna().any():
            purchase_span_days = max(
                (p["pedido"].max() - p["pedido"].min()).days,
                1
            )
            orders_per_month = n_orders / purchase_span_days * 30.44
        else:
            orders_per_month = np.nan

        avg_order_qty = p["Cantidad"].mean() if n_orders else np.nan
        total_order_qty = p["Cantidad"].sum(min_count=1) if n_orders else np.nan
        avg_order_value = p["Valor facturado"].mean() if n_orders else np.nan
        total_order_value = p["Valor facturado"].sum(min_count=1) if n_orders else np.nan

        order_dates = p["pedido"].dropna().sort_values()
        avg_days_between_orders = (
            order_dates.diff().dt.days.dropna().mean()
            if len(order_dates) > 1 else np.nan
        )

        # ---------------- Precio ----------------
        avg_purchase_price = p["PU Promedio"].mean() if n_orders else np.nan
        price_std = p["PU Promedio"].std(ddof=1) if n_orders > 1 else np.nan
        price_cv = safe_cv(price_std, avg_purchase_price)

        # ---------------- Error política actual ----------------
        if not pe.empty:
            under_pct = safe_pct(pe["subestima"].sum(), len(pe))
            over_pct = safe_pct(pe["sobreestima"].sum(), len(pe))
            mae_5d = pe["error_absoluto"].mean()
            bias_5d = pe["error"].mean()
            wape_5d = series_wape(pe["consumo_real_5d"], pe["pronostico_5d"])
        else:
            under_pct = over_pct = mae_5d = bias_5d = wape_5d = np.nan

        # ---------------- ROP ilustrativo ----------------
        # Fórmula básica:
        # SS = z * sigma_diaria * sqrt(LT)
        # ROP = demanda_diaria * LT + SS
        if (
            pd.notna(avg_daily) and avg_daily >= 0 and
            pd.notna(daily_std) and daily_std >= 0 and
            pd.notna(replenishment_mean) and replenishment_mean > 0
        ):
            safety_stock_95 = SERVICE_LEVEL_Z * daily_std * math.sqrt(replenishment_mean)
            rop_95 = avg_daily * replenishment_mean + safety_stock_95
        else:
            safety_stock_95 = np.nan
            rop_95 = np.nan

        # ROP conservador usando P90 de LT
        if (
            pd.notna(avg_daily) and
            pd.notna(daily_std) and
            pd.notna(replenishment_p90) and replenishment_p90 > 0
        ):
            ss_p90lt = SERVICE_LEVEL_Z * daily_std * math.sqrt(replenishment_p90)
            rop_p90lt_95 = avg_daily * replenishment_p90 + ss_p90lt
        else:
            rop_p90lt_95 = np.nan

        # ---------------- Capital inventario ----------------
        avg_inventory_value = avg_stock * latest_price if pd.notna(latest_price) else np.nan
        current_inventory_value = current_stock * latest_price if pd.notna(latest_price) else np.nan
        max_inventory_value = max_inv * latest_price if pd.notna(latest_price) else np.nan

        # ---------------- Calidad de datos ----------------
        adjustments = int(rec["ajuste_datos"].sum())
        adjustment_pct = safe_pct(adjustments, max(len(rec) - 1, 1))

        rows.append({
            "material": material,

            "fecha_inicio_inventario": period_start,
            "fecha_fin_inventario": period_end,
            "observaciones_inventario": len(inv),

            "consumo_total_reconstruido": total_consumption,
            "consumo_promedio_diario": avg_daily,
            "desv_est_consumo_diario": daily_std,
            "consumo_semanal_promedio": weekly_mean,
            "desv_est_consumo_semanal": weekly_std,
            "cv_semanal": weekly_cv,
            "consumo_semanal_maximo": weekly_max,
            "pct_dias_con_consumo_estimado": nonzero_pct,

            "inventario_promedio": avg_stock,
            "inventario_minimo_observado": min_inv,
            "inventario_maximo_observado": max_inv,
            "inventario_actual": current_stock,
            "stock_minimo_parametro": min_stock,
            "pct_observaciones_debajo_minimo": pct_below_min,
            "observaciones_stockout": stockout_observations,

            "cobertura_promedio_dias": avg_coverage,
            "cobertura_minima_dias": min_coverage,
            "cobertura_maxima_dias": max_coverage,
            "cobertura_actual_dias": current_coverage,
            "pct_observaciones_cobertura_menor_5d": pct_under_5,

            "numero_pedidos": n_orders,
            "pedidos_por_mes_aprox": orders_per_month,
            "dias_promedio_entre_pedidos": avg_days_between_orders,
            "cantidad_promedio_por_pedido": avg_order_qty,
            "cantidad_total_pedida": total_order_qty,
            "valor_promedio_pedido": avg_order_value,
            "valor_total_pedidos": total_order_value,

            "lead_time_promedio_dias": lt_mean,
            "lead_time_mediana_dias": lt_median,
            "lead_time_desv_est_dias": lt_std,
            "lead_time_min_dias": lt_min,
            "lead_time_max_dias": lt_max,
            "lead_time_p90_dias": lt_p90,
            "pct_pedidos_lead_time_mayor_5d": pct_lt_over_5,
            "tiempo_administrativo_dias": admin_days,
            "reposicion_total_promedio_dias": replenishment_mean,
            "reposicion_total_p90_dias": replenishment_p90,

            "pct_veces_semana_anterior_subestima_5d": under_pct,
            "pct_veces_semana_anterior_sobreestima_5d": over_pct,
            "mae_pronostico_5d": mae_5d,
            "sesgo_pronostico_5d": bias_5d,
            "wape_pronostico_5d_pct": wape_5d,

            "precio_actual": latest_price,
            "pu_promedio_historico": avg_purchase_price,
            "desv_est_precio_historico": price_std,
            "cv_precio_historico": price_cv,
            "valor_inventario_promedio_estimado": avg_inventory_value,
            "valor_inventario_actual_estimado": current_inventory_value,
            "valor_inventario_maximo_estimado": max_inventory_value,

            "stock_seguridad_95_ilustrativo": safety_stock_95,
            "rop_95_ilustrativo": rop_95,
            "rop_95_con_lead_time_p90": rop_p90lt_95,

            "intervalos_con_ajuste_no_conciliado": adjustments,
            "pct_intervalos_con_ajuste_no_conciliado": adjustment_pct
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# TABLAS SECUNDARIAS
# ---------------------------------------------------------------------

def lead_time_table(purchases):
    cols = [
        "Material", "Proveedor (Texto)", "ID de pedido de compra",
        "PU Promedio", "Cantidad", "Valor facturado",
        "pedido", "recibido", "dias en llegar", "lead_time_calculado"
    ]
    existing = [c for c in cols if c in purchases.columns]
    return purchases[existing].sort_values(["Material", "pedido"])


def purchases_summary(purchases):
    p = purchases.copy()

    summary = (
        p.groupby("Material", as_index=False)
        .agg(
            numero_pedidos=("ID de pedido de compra", "count"),
            cantidad_total=("Cantidad", "sum"),
            cantidad_promedio=("Cantidad", "mean"),
            valor_total=("Valor facturado", "sum"),
            valor_promedio=("Valor facturado", "mean"),
            pu_promedio=("PU Promedio", "mean"),
            lead_time_promedio=("dias en llegar", "mean"),
            lead_time_mediana=("dias en llegar", "median"),
            lead_time_max=("dias en llegar", "max"),
        )
    )
    return summary


def data_quality_table(reconstructed, purchases):
    rows = []

    for material, g in reconstructed.groupby("material"):
        rows.append({
            "material": material,
            "intervalos_analizados": max(len(g) - 1, 0),
            "intervalos_con_entrada_no_conciliada": int(g["ajuste_datos"].sum()),
            "pct_ajustes": safe_pct(
                g["ajuste_datos"].sum(),
                max(len(g) - 1, 1)
            )
        })

    quality = pd.DataFrame(rows)

    # Diferencia entre lead time proporcionado y calculado
    p = purchases.dropna(
        subset=["dias en llegar", "lead_time_calculado"]
    ).copy()
    p["diferencia_lt"] = p["dias en llegar"] - p["lead_time_calculado"]

    lt_check = (
        p.groupby("Material", as_index=False)
        .agg(
            pedidos_con_lt=("diferencia_lt", "count"),
            pedidos_lt_inconsistente=("diferencia_lt", lambda x: (x != 0).sum())
        )
        .rename(columns={"Material": "material"})
    )

    return quality.merge(lt_check, on="material", how="left")


# ---------------------------------------------------------------------
# GRÁFICAS
# ---------------------------------------------------------------------

def create_charts(inv_long, weekly, summary, output_dir):
    chart_dir = output_dir / "graficas"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inventario histórico por material
    for material, g in inv_long.groupby("material"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(g["fecha"], g["inventario"], marker="o", markersize=3)
        ax.set_title(f"Inventario histórico - {material}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Inventario")
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()

        filename = "".join(
            ch if ch.isalnum() or ch in " -_" else "_"
            for ch in material
        ).strip()
        fig.savefig(chart_dir / f"inventario_{filename}.png", dpi=160)
        plt.close(fig)

    # 2. Consumo semanal por material
    for material, g in weekly.groupby("material"):
        labels = [
            f"{int(y)}-W{int(w):02d}"
            for y, w in zip(g["iso_year"], g["iso_week"])
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, g["consumo_semanal"])
        ax.set_title(f"Consumo semanal estimado - {material}")
        ax.set_xlabel("Semana")
        ax.set_ylabel("Consumo")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        filename = "".join(
            ch if ch.isalnum() or ch in " -_" else "_"
            for ch in material
        ).strip()
        fig.savefig(chart_dir / f"consumo_semanal_{filename}.png", dpi=160)
        plt.close(fig)

    # 3. CV semanal
    s = summary.sort_values("cv_semanal", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(s["material"], s["cv_semanal"])
    ax.set_title("Variabilidad del consumo semanal por material")
    ax.set_xlabel("Coeficiente de variación (CV)")
    fig.tight_layout()
    fig.savefig(chart_dir / "comparativo_cv_semanal.png", dpi=180)
    plt.close(fig)

    # 4. Lead time vs política 5 días
    s = summary.sort_values("lead_time_mediana_dias", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(s["material"], s["lead_time_mediana_dias"])
    ax.axvline(
        CURRENT_COVERAGE_DAYS,
        linestyle="--",
        label=f"Política actual: {CURRENT_COVERAGE_DAYS} días"
    )
    ax.set_title("Lead time mediano vs cobertura actual")
    ax.set_xlabel("Días")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "lead_time_vs_5_dias.png", dpi=180)
    plt.close(fig)

    # 5. % de observaciones con cobertura <5 días
    s = summary.sort_values(
        "pct_observaciones_cobertura_menor_5d",
        ascending=True
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        s["material"],
        s["pct_observaciones_cobertura_menor_5d"]
    )
    ax.set_title("Frecuencia de cobertura inferior a 5 días")
    ax.set_xlabel("% de observaciones")
    fig.tight_layout()
    fig.savefig(chart_dir / "cobertura_menor_5_dias.png", dpi=180)
    plt.close(fig)

    # 6. % de subestimación de la política semana anterior
    s = summary.sort_values(
        "pct_veces_semana_anterior_subestima_5d",
        ascending=True
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(
        s["material"],
        s["pct_veces_semana_anterior_subestima_5d"]
    )
    ax.set_title("Frecuencia con que la semana anterior subestima los siguientes 5 días")
    ax.set_xlabel("% de evaluaciones")
    fig.tight_layout()
    fig.savefig(chart_dir / "subestimacion_semana_anterior.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No encontré '{INPUT_FILE}'. "
            "Coloca el script en la misma carpeta que el Excel o cambia INPUT_FILE."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/7 Cargando datos...")
    inv_long, purchases, mat_dict, prices = load_data(input_path)

    print("2/7 Reconstruyendo consumo...")
    reconstructed = reconstruct_consumption(inv_long, purchases)

    print("3/7 Construyendo serie diaria y semanal...")
    daily = build_daily_series(reconstructed)
    weekly = weekly_consumption(daily)

    print("4/7 Evaluando la política de semana anterior + 5 días...")
    policy_eval = evaluate_current_policy(daily)

    print("5/7 Calculando métricas...")
    summary = make_summary(
        inv_long=inv_long,
        reconstructed=reconstructed,
        daily=daily,
        weekly=weekly,
        purchases=purchases,
        mat_dict=mat_dict,
        prices=prices,
        policy_eval=policy_eval
    )

    lt = lead_time_table(purchases)
    purchase_summary = purchases_summary(purchases)
    quality = data_quality_table(reconstructed, purchases)

    rop_cols = [
        "material",
        "consumo_promedio_diario",
        "desv_est_consumo_diario",
        "lead_time_promedio_dias",
        "lead_time_p90_dias",
        "tiempo_administrativo_dias",
        "reposicion_total_promedio_dias",
        "reposicion_total_p90_dias",
        "stock_seguridad_95_ilustrativo",
        "rop_95_ilustrativo",
        "rop_95_con_lead_time_p90"
    ]
    rop = summary[[c for c in rop_cols if c in summary.columns]].copy()

    print("6/7 Guardando tablas...")
    summary.to_csv(
        OUTPUT_DIR / "01_resumen_metricas.csv",
        index=False,
        encoding="utf-8-sig"
    )
    reconstructed.to_csv(
        OUTPUT_DIR / "02_consumo_reconstruido_por_intervalo.csv",
        index=False,
        encoding="utf-8-sig"
    )
    daily.to_csv(
        OUTPUT_DIR / "03_consumo_diario_estimado.csv",
        index=False,
        encoding="utf-8-sig"
    )
    weekly.to_csv(
        OUTPUT_DIR / "04_consumo_semanal.csv",
        index=False,
        encoding="utf-8-sig"
    )
    lt.to_csv(
        OUTPUT_DIR / "05_lead_times_detalle.csv",
        index=False,
        encoding="utf-8-sig"
    )
    purchase_summary.to_csv(
        OUTPUT_DIR / "06_compras_resumen.csv",
        index=False,
        encoding="utf-8-sig"
    )
    policy_eval.to_csv(
        OUTPUT_DIR / "07_evaluacion_politica_5_dias.csv",
        index=False,
        encoding="utf-8-sig"
    )
    quality.to_csv(
        OUTPUT_DIR / "08_calidad_datos.csv",
        index=False,
        encoding="utf-8-sig"
    )
    rop.to_csv(
        OUTPUT_DIR / "09_metricas_rop.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("7/7 Generando gráficas...")
    create_charts(inv_long, weekly, summary, OUTPUT_DIR)

    # Mostrar las columnas más útiles directamente en consola
    key_cols = [
        "material",
        "consumo_promedio_diario",
        "cv_semanal",
        "cobertura_promedio_dias",
        "cobertura_minima_dias",
        "pct_observaciones_cobertura_menor_5d",
        "lead_time_promedio_dias",
        "lead_time_mediana_dias",
        "lead_time_p90_dias",
        "pct_pedidos_lead_time_mayor_5d",
        "pedidos_por_mes_aprox",
        "pct_veces_semana_anterior_subestima_5d",
        "wape_pronostico_5d_pct",
        "rop_95_ilustrativo"
    ]

    print("\nRESUMEN PRINCIPAL\n")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:,.2f}"
    ):
        print(summary[key_cols].to_string(index=False))

    print(f"\nListo. Archivos guardados en: {OUTPUT_DIR.resolve()}")

    print(
        "\nIMPORTANTE: Los campos relacionados con Ventas comprometidas no pueden "
        "calcularse porque el archivo actual no contiene una tabla de reservas/pedidos "
        "de Ventas. Cuando tengas esos datos, deben agregarse como demanda extraordinaria "
        "separada del consumo regular."
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()