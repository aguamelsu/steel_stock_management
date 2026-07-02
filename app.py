import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gspread 
from google.oauth2.service_account import Credentials
from statsmodels.tsa.arima.model import ARIMA

# Configurar pagina
st.set_page_config(
    page_title="Steel Inventory Dashboard",
    layout="wide"
)

st.title("Steel Inventory Forecast Dashboard")

# Auth google para data base
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    "/workspaces/steel_stock_management/credentials.json",
    scopes=scope
)

gc = gspread.authorize(creds)

sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1qAsFjW2lT8e54FsvsMkfGIHwQEDxsQuf4SzVAMXlK-g/edit?gid=824196706#gid=824196706")

inventory_ws = sheet.worksheet('inventario')
inventory_df = pd.DataFrame(inventory_ws.get_all_records())

df_raw = inventory_df.copy()

df_baking = df_raw.copy()

# Convert dates from columns back into rows
df_baking = df_baking.melt(
    id_vars="material",
    var_name="fecha",
    value_name="inventario"
)

# Preprocessing
df_baking['fecha'] = pd.to_datetime(df_baking['fecha'])
df_baking["material"] = df_baking["material"].astype("category")

df = df_baking.copy()

df = df.sort_values(["material", "fecha"])


# =========================
# CREATE CONSUMO COLUMN
# =========================

if "consumo" not in df.columns:
    df["daily_change"] = df.groupby("material")["inventario"].diff()
    df["consumo"] = -df["daily_change"]
    df.loc[df["consumo"] < 0, "consumo"] = 0
    df["consumo"] = df["consumo"].fillna(0)


# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Filters")

materials = sorted(df["material"].unique())

selected_material = st.sidebar.selectbox(
    "Select material",
    materials
)

min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Please select a start and end date.")
    st.stop()

start_date, end_date = date_range

filtered_df = df[
    (df["material"] == selected_material) &
    (df["fecha"].dt.date >= start_date) &
    (df["fecha"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data available for this filter.")
    st.stop()


# =========================
# ARIMA FUNCTION
# =========================

def arima_forecast(series, forecast_days=7):
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

                except:
                    continue

    forecast = best_model.forecast(steps=forecast_days)
    forecast = forecast.clip(lower=0)

    return forecast, best_order


# =========================
# INVENTORY PARAMETERS
# =========================

current_inventory = filtered_df["inventario"].iloc[-1]
avg_consumption = filtered_df["consumo"].mean()
std_consumption = filtered_df["consumo"].std()

lead_time_days = st.sidebar.number_input(
    "Lead time days",
    min_value=1,
    max_value=60,
    value=7
)

target_days = st.sidebar.number_input(
    "Target stock days",
    min_value=1,
    max_value=60,
    value=15
)

safety_multiplier = st.sidebar.number_input(
    "Safety stock multiplier",
    min_value=0.0,
    max_value=5.0,
    value=1.65,
    step=0.05
)

safety_stock = safety_multiplier * std_consumption * np.sqrt(lead_time_days)

reorder_point = (avg_consumption * lead_time_days) + safety_stock

target_stock = (avg_consumption * target_days) + safety_stock

recommended_qty = max(0, target_stock - current_inventory)

buy_now = current_inventory <= reorder_point


# =========================
# KPI CARDS
# =========================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Current inventory", f"{current_inventory:,.2f}")
col2.metric("Avg daily consumption", f"{avg_consumption:,.2f}")
col3.metric("Reorder point", f"{reorder_point:,.2f}")
col4.metric("Recommended quantity", f"{recommended_qty:,.2f}")

if buy_now:
    col5.error("BUY NOW")
else:
    col5.success("WAIT")


# =========================
# FORECAST BUTTON
# =========================

show_forecast = st.button("Forecast next 7 days")

forecast = None
forecast_dates = None
best_order = None

if show_forecast:
    series = (
        filtered_df
        .set_index("fecha")["consumo"]
        .sort_index()
    )

    forecast, best_order = arima_forecast(series, forecast_days=7)

    forecast_dates = pd.date_range(
        start=filtered_df["fecha"].max() + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    forecast.index = forecast_dates

    st.info(f"Forecasting next 15 days using ARIMA{best_order}")


# =========================
# INVENTORY GRAPH
# =========================

st.subheader("Inventory over time")

fig_inv = go.Figure()

fig_inv.add_trace(
    go.Scatter(
        x=filtered_df["fecha"],
        y=filtered_df["inventario"],
        mode="lines+markers",
        name="Inventory"
    )
)

fig_inv.add_trace(
    go.Scatter(
        x=filtered_df["fecha"],
        y=[reorder_point] * len(filtered_df),
        mode="lines",
        name="Reorder Point",
        line=dict(dash="dash", color="red")
    )
)

fig_inv.update_layout(
    xaxis_title="Date",
    yaxis_title="Inventory",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_inv, use_container_width=True)


# =========================
# CONSUMPTION GRAPH
# =========================

st.subheader("Consumption over time")

fig_cons = go.Figure()

fig_cons.add_trace(
    go.Bar(
        x=filtered_df["fecha"],
        y=filtered_df["consumo"],
        name="Actual consumption"
    )
)

if show_forecast and forecast is not None:
    fig_cons.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Forecasted consumption",
            line=dict(dash="dash")
        )
    )

fig_cons.update_layout(
    xaxis_title="Date",
    yaxis_title="Consumption",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_cons, use_container_width=True)


# =========================
# RECOMMENDATION PANEL
# =========================

st.subheader("Purchase recommendation")

left, right = st.columns([1, 2])

with left:
    st.metric("Recommended quantity to buy", f"{recommended_qty:,.2f}")
    st.metric("Target stock", f"{target_stock:,.2f}")
    st.metric("Safety stock", f"{safety_stock:,.2f}")
    st.metric("Lead time demand", f"{avg_consumption * lead_time_days:,.2f}")

with right:
    if buy_now:
        st.error(
            f"""
            Current inventory is below or near the reorder point.

            Recommended action: buy **{recommended_qty:,.2f}** units of **{selected_material}**.
            """
        )
    else:
        st.success(
            f"""
            Current inventory is above the reorder point.

            Recommended action: no purchase needed right now.
            """
        )


# =========================
# DATA TABLE
# =========================

with st.expander("Show filtered data"):
    st.dataframe(filtered_df)