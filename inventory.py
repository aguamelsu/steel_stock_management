"""
Preprocesamiento y cálculos del dashboard (sin dependencias de Streamlit).

app.py envuelve estas funciones con st.cache_data / st.cache_resource para que
no se recalculen en cada rerun.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


ARMEX_MATERIALS = [
    "ARMEX 15X15",
    "ARMEX 15X20",
    "ARMEX DE 15 x 15",
    "ARMEX DE 15 x 20",
    "ARMEX 15 x 15",
    "ARMEX 15 x 20",
]

SUPPLIER_COLUMNS = [
    'material',
    'proveedor',
    'id',
    'pu',
    'cantidad',
    'total',
    'pedido',
    'recibido',
    'espera',
]

SUPPLIER_NAMES = {
    'HOME DEPOT MEXICO S DE RL DE CV': 'HOME DEPOT',
    'MEXLAM CONCEPTOS SA DE CV': 'MEXLAM',
    'CEMEX': 'CEMEX',
    'ACEROS VIMAR MEXICO SAPI DE CV': 'VIMAR'
}


##############################
# PREPROCESSING              #
##############################
def melt_inventory(inventory_df):
    """Hoja 'inventario' (una columna por fecha) -> formato largo."""
    df = inventory_df.melt(
        id_vars="material",
        var_name="fecha",
        value_name="inventario"
    )

    df['fecha'] = pd.to_datetime(df['fecha'])
    df["material"] = df["material"].astype("category")

    return df


def add_consumption(inventory_long):
    """Ordena por material/fecha y deriva el consumo diario de las caídas de inventario."""
    df = inventory_long.sort_values(["material", "fecha"])

    df["daily_change"] = df.groupby("material")["inventario"].diff()
    df["consumo"] = -df["daily_change"]
    df.loc[df["consumo"] < 0, "consumo"] = 0
    df["consumo"] = df["consumo"].fillna(0)

    return df


def prepare_suppliers(suppliers_df_raw):
    suppliers_df = suppliers_df_raw.copy()
    suppliers_df.columns = SUPPLIER_COLUMNS

    suppliers_df['material'] = suppliers_df['material'].astype('category')
    suppliers_df['proveedor'] = suppliers_df['proveedor'].astype('category')
    suppliers_df['pedido'] = pd.to_datetime(suppliers_df['pedido'], format="%d/%m/%Y")
    suppliers_df['recibido'] = pd.to_datetime(suppliers_df['recibido'], format="%d/%m/%Y")

    suppliers_df = suppliers_df.drop(columns='id')

    suppliers_df['proveedor'] = suppliers_df['proveedor'].map(SUPPLIER_NAMES)

    return suppliers_df


def supplier_lead_time_pivot(suppliers_df):
    """Matriz material x proveedor con el tiempo de espera promedio."""
    supplier_summary = (
        suppliers_df
        .groupby(["material", "proveedor"])
        .agg(
            avg_lead=("espera", "mean"),
            std_lead=("espera", "std"),
            purchases=("espera", "count")
        )
        .reset_index()
    )

    pivot = supplier_summary.pivot(
        index="material",
        columns="proveedor",
        values="avg_lead"
    )

    pivot = pivot.fillna(0)
    pivot = pivot.sort_index().sort_index(axis=1)

    return pivot


def inventory_value(inventory_long, prices_df_raw):
    """Precio x inventario de la última fecha de la hoja, por material."""
    by_date = inventory_long.set_index('fecha')

    prices_df = prices_df_raw.copy()
    prices_df['fecha'] = pd.to_datetime(prices_df['fecha'], dayfirst=True)

    prices_df['inv_al_dia'] = by_date.loc[by_date.index[-1]]['inventario'].reset_index().drop(columns='fecha')
    prices_df['valor'] = round(prices_df['precio'] * prices_df['inv_al_dia'])

    return prices_df


def filter_inventory(df, material, start_date, end_date):
    """Filas de un material con fecha (día calendario) en [start_date, end_date]."""
    start = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    return df[
        (df["material"] == material) &
        (df["fecha"] >= start) &
        (df["fecha"] < end_exclusive)
    ].copy()


#########
# ARIMA #
#########
def fit_best_arima(series):
    """Grid search ARIMA(p, d, q) por AIC. Devuelve (modelo ajustado, orden)."""
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

                except Exception:
                    continue

    return best_model, best_order


def forecast_from_fit(best_model, forecast_days):
    forecast = best_model.forecast(steps=forecast_days)
    return forecast.clip(lower=0)


def arima_forecast(series, forecast_days=7):
    best_model, best_order = fit_best_arima(series)
    return forecast_from_fit(best_model, forecast_days), best_order


##########################
# MATH AND STATISTICS #
##########################
def target_days_for(material):
    if material.upper().strip() in [m.upper().strip() for m in ARMEX_MATERIALS]:
        return 7
    return 14


def consumption_stats(filtered_df):
    current_inventory = filtered_df["inventario"].iloc[-1]
    avg_consumption = filtered_df["consumo"].mean()
    std_consumption = filtered_df["consumo"].std()
    cv = std_consumption / avg_consumption

    return current_inventory, avg_consumption, std_consumption, cv


def replenishment_metrics(filtered_df, forecast, lead_time_days, target_days):
    current_inventory, avg_consumption, std_consumption, cv = consumption_stats(filtered_df)

    forecast_horizon = lead_time_days + target_days

    safety_stock = 1.5 * std_consumption * np.sqrt(lead_time_days)

    lead_time_demand = forecast.iloc[:lead_time_days].sum()

    projected_inventory_arrival = current_inventory - lead_time_demand

    buffer = np.clip(cv, 0.10, 0.50)

    target_stock = avg_consumption * (lead_time_days + target_days) + safety_stock
    maximum_inventory = target_stock * (1 + buffer)

    recommended_qty = max(
        target_stock - projected_inventory_arrival,
        0
    )

    recommended_qty = max(0, recommended_qty)

    reorder_point = (
        forecast.iloc[:forecast_horizon].sum()
        + safety_stock
    )

    buy_now = current_inventory <= reorder_point

    if avg_consumption > 0:
        days_until_stockout = current_inventory / avg_consumption
        estimated_stockout_date = (
            filtered_df["fecha"].max()
            + pd.Timedelta(days=days_until_stockout)
        )
    else:
        days_until_stockout = np.inf
        estimated_stockout_date = None

    minimum_stock = avg_consumption * target_days

    return {
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
