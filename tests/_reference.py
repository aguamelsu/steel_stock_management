"""
Copias literales de la lógica ORIGINAL (antes de la optimización), usadas como
oráculo en las pruebas de regresión. No usar desde la app.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# --------------------------- app.py original ---------------------------

def app_preprocess(inventory_df):
    df_raw = inventory_df.copy()
    df_baking = df_raw.copy()
    df_baking = df_baking.melt(id_vars="material", var_name="fecha", value_name="inventario")
    df_baking['fecha'] = pd.to_datetime(df_baking['fecha'])
    df_baking["material"] = df_baking["material"].astype("category")
    df = df_baking.copy()
    df = df.sort_values(["material", "fecha"])
    if "consumo" not in df.columns:
        df["daily_change"] = df.groupby("material")["inventario"].diff()
        df["consumo"] = -df["daily_change"]
        df.loc[df["consumo"] < 0, "consumo"] = 0
        df["consumo"] = df["consumo"].fillna(0)
    return df


def app_filter(df, selected_material, start_date, end_date):
    return df[
        (df["material"] == selected_material) &
        (df["fecha"].dt.date >= start_date) &
        (df["fecha"].dt.date <= end_date)
    ].copy()


def app_arima_forecast(series, forecast_days=7):
    series = series.asfreq("D").fillna(0)
    best_aic = np.inf
    best_model = None
    best_order = None
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_model = fit
                        best_order = (p, d, q)
                except:  # noqa: E722
                    continue
    forecast = best_model.forecast(steps=forecast_days)
    forecast = forecast.clip(lower=0)
    return forecast, best_order


def app_metrics(filtered_df, forecast, lead_time_days, selected_material):
    current_inventory = filtered_df["inventario"].iloc[-1]
    avg_consumption = filtered_df["consumo"].mean()
    std_consumption = filtered_df["consumo"].std()
    cv = std_consumption / avg_consumption

    armex_materials = ["ARMEX 15X15", "ARMEX 15X20", "ARMEX DE 15 x 15",
                       "ARMEX DE 15 x 20", "ARMEX 15 x 15", "ARMEX 15 x 20"]
    if selected_material.upper().strip() in [m.upper().strip() for m in armex_materials]:
        target_days = 7
    else:
        target_days = 14
    forecast_horizon = lead_time_days + target_days

    safety_stock = 1.5 * std_consumption * np.sqrt(lead_time_days)
    lead_time_demand = forecast.iloc[:lead_time_days].sum()
    projected_inventory_arrival = current_inventory - lead_time_demand
    buffer = np.clip(cv, 0.10, 0.50)
    target_stock = avg_consumption * (lead_time_days + target_days) + safety_stock
    maximum_inventory = target_stock * (1 + buffer)
    recommended_qty = max(target_stock - projected_inventory_arrival, 0)
    recommended_qty = max(0, recommended_qty)
    reorder_point = forecast.iloc[:forecast_horizon].sum() + safety_stock
    buy_now = current_inventory <= reorder_point
    if avg_consumption > 0:
        days_until_stockout = current_inventory / avg_consumption
        estimated_stockout_date = filtered_df["fecha"].max() + pd.Timedelta(days=days_until_stockout)
    else:
        days_until_stockout = np.inf
        estimated_stockout_date = None
    minimum_stock = avg_consumption * target_days
    return {
        "target_days": target_days,
        "current_inventory": current_inventory,
        "avg_consumption": avg_consumption,
        "std_consumption": std_consumption,
        "cv": cv,
        "safety_stock": safety_stock,
        "maximum_inventory": maximum_inventory,
        "recommended_qty": recommended_qty,
        "reorder_point": reorder_point,
        "buy_now": buy_now,
        "days_until_stockout": days_until_stockout,
        "estimated_stockout_date": estimated_stockout_date,
        "minimum_stock": minimum_stock,
    }


def app_suppliers_pivot(suppliers_df_raw):
    suppliers_df_raw = suppliers_df_raw.copy()
    suppliers_df_raw.columns = ['material', 'proveedor', 'id', 'pu', 'cantidad',
                                'total', 'pedido', 'recibido', 'espera']
    suppliers_df_baking = suppliers_df_raw.copy()
    suppliers_df_baking['material'] = suppliers_df_baking['material'].astype('category')
    suppliers_df_baking['proveedor'] = suppliers_df_baking['proveedor'].astype('category')
    suppliers_df_baking['pedido'] = pd.to_datetime(suppliers_df_baking['pedido'], format="%d/%m/%Y")
    suppliers_df_baking['recibido'] = pd.to_datetime(suppliers_df_baking['recibido'], format="%d/%m/%Y")
    suppliers_df_baking = suppliers_df_baking.drop(columns='id')
    suppliers_df_baking['proveedor'] = suppliers_df_baking['proveedor'].map({
        'HOME DEPOT MEXICO S DE RL DE CV': 'HOME DEPOT',
        'MEXLAM CONCEPTOS SA DE CV': 'MEXLAM',
        'CEMEX': 'CEMEX',
        'ACEROS VIMAR MEXICO SAPI DE CV': 'VIMAR'
    })
    suppliers_df = suppliers_df_baking.copy()
    supplier_summary = (
        suppliers_df.groupby(["material", "proveedor"])
        .agg(avg_lead=("espera", "mean"), std_lead=("espera", "std"), purchases=("espera", "count"))
        .reset_index()
    )
    pivot = supplier_summary.pivot(index="material", columns="proveedor", values="avg_lead")
    pivot = pivot.fillna(0)
    return pivot.sort_index().sort_index(axis=1)


def app_inventory_value(inventory_df, prices_df_raw):
    df_raw = inventory_df.copy()
    df_baking = df_raw.copy()
    df_baking = df_baking.melt(id_vars="material", var_name="fecha", value_name="inventario")
    df_baking['fecha'] = pd.to_datetime(df_baking['fecha'])
    df_baking["material"] = df_baking["material"].astype("category")
    df_baking = df_baking.set_index('fecha')
    prices_df_baking = prices_df_raw.copy()
    prices_df_baking['fecha'] = pd.to_datetime(prices_df_baking['fecha'], dayfirst=True)
    prices_df = prices_df_baking.copy()
    prices_df['inv_al_dia'] = df_baking.loc[df_baking.index[-1]]['inventario'].reset_index().drop(columns='fecha')
    prices_df['valor'] = round(prices_df['precio'] * prices_df['inv_al_dia'])
    return prices_df


# --------------------------- script.py original ---------------------------

def script_reconstruct_consumption(inv_long, purchases, count_days_between, days_mode="calendar"):
    rows = []
    receipts = purchases.dropna(subset=["recibido", "Cantidad"]).copy()
    for material, g in inv_long.groupby("material"):
        g = g.sort_values("fecha").reset_index(drop=True)
        for i in range(len(g)):
            date = g.loc[i, "fecha"]
            stock = g.loc[i, "inventario"]
            if i == 0:
                rows.append({
                    "material": material, "fecha": date, "inventario": stock,
                    "fecha_anterior": pd.NaT, "inventario_anterior": np.nan,
                    "dias_intervalo": np.nan, "recepciones_intervalo": np.nan,
                    "consumo_intervalo": np.nan, "consumo_diario": np.nan,
                    "ajuste_datos": False
                })
                continue
            prev_date = g.loc[i - 1, "fecha"]
            prev_stock = g.loc[i - 1, "inventario"]
            days = count_days_between(prev_date, date, days_mode)
            mat_receipts = receipts[
                (receipts["Material"] == material) &
                (receipts["recibido"] > prev_date) &
                (receipts["recibido"] <= date)
            ]
            receipt_qty = mat_receipts["Cantidad"].sum()
            raw_consumption = prev_stock + receipt_qty - stock
            data_adjustment = raw_consumption < -1e-9
            consumption = raw_consumption if raw_consumption >= 0 else np.nan
            daily = consumption / days if pd.notna(consumption) and days and days > 0 else np.nan
            rows.append({
                "material": material, "fecha": date, "inventario": stock,
                "fecha_anterior": prev_date, "inventario_anterior": prev_stock,
                "dias_intervalo": days, "recepciones_intervalo": receipt_qty,
                "consumo_intervalo": consumption, "consumo_diario": daily,
                "ajuste_datos": data_adjustment
            })
    return pd.DataFrame(rows)


def script_evaluate_current_policy(daily, trailing_days=7, forward_days=5):
    rows = []
    if daily.empty:
        return pd.DataFrame()
    for material, g in daily.groupby("material"):
        s = (
            g.set_index("fecha")["consumo_estimado_dia"]
            .sort_index()
            .asfreq("D", fill_value=0)
        )
        for date in s.index:
            trailing_start = date - pd.Timedelta(days=trailing_days)
            trailing_end = date - pd.Timedelta(days=1)
            forward_start = date
            forward_end = date + pd.Timedelta(days=forward_days - 1)
            hist = s.loc[trailing_start:trailing_end]
            fut = s.loc[forward_start:forward_end]
            if len(hist) < trailing_days or len(fut) < forward_days:
                continue
            daily_avg = hist.mean()
            forecast_5d = daily_avg * forward_days
            actual_5d = fut.sum()
            rows.append({
                "material": material, "fecha_evaluacion": date,
                "consumo_7d_previos": hist.sum(), "promedio_diario_7d": daily_avg,
                "pronostico_5d": forecast_5d, "consumo_real_5d": actual_5d,
                "error": forecast_5d - actual_5d,
                "error_absoluto": abs(forecast_5d - actual_5d),
                "subestima": forecast_5d < actual_5d,
                "sobreestima": forecast_5d > actual_5d
            })
    return pd.DataFrame(rows)
