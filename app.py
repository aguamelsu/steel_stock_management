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

st.title("Inventario Aceros ABC")


##############################
# GOOGLE AUTH, PREPROCESSING #
##############################
@st.cache_data(ttl=300)
def load_google_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
    )

    gc = gspread.authorize(credentials)

    spreadsheet = gc.open_by_url(
        st.secrets["google_sheets"]["spreadsheet_url"]
    )

    inventory_ws = spreadsheet.worksheet("inventario")
    suppliers_ws = spreadsheet.worksheet("pedidos_proveedores")
    prices_ws = spreadsheet.worksheet("precios")

    inventory_df = pd.DataFrame(inventory_ws.get_all_records())
    suppliers_df = pd.DataFrame(suppliers_ws.get_all_records())
    prices_df = pd.DataFrame(prices_ws.get_all_records())

    return inventory_df, suppliers_df, prices_df

inventory_df, suppliers_df_raw, prices_df_raw = load_google_sheet()

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


if "consumo" not in df.columns:
    df["daily_change"] = df.groupby("material")["inventario"].diff()
    df["consumo"] = -df["daily_change"]
    df.loc[df["consumo"] < 0, "consumo"] = 0
    df["consumo"] = df["consumo"].fillna(0)

# FILTROS
st.sidebar.header("Filtros")

materials = sorted(df["material"].unique())
selected_material = st.sidebar.selectbox(
    "Seleccionar material",
    materials
)

min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.warning("Selecciona fecha de inicio y de fin")
    st.stop()

start_date, end_date = date_range

filtered_df = df[
    (df["material"] == selected_material) &
    (df["fecha"].dt.date >= start_date) &
    (df["fecha"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No hay datos suficientes para este filtro")
    st.stop()


#########
# ARIMA #
#########
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


##########################
# MATH AND STATISTICS #
##########################
current_inventory = filtered_df["inventario"].iloc[-1]
avg_consumption = filtered_df["consumo"].mean()
std_consumption = filtered_df["consumo"].std()



lead_time_days = st.sidebar.number_input(
    "Días de anticipo para realizar pedido",
    min_value=1,
    max_value=15,
    value=7
)

target_days = st.sidebar.number_input(
    "Objetivo de días de inventario",
    min_value=1,
    max_value=60,
    value=15
)

# Series for ARIMA
series = (
    filtered_df
    .set_index("fecha")["consumo"]
    .sort_index()
)

forecast, best_order = arima_forecast(
    series,
    forecast_days=lead_time_days + target_days
)

# Target days by material
armex_materials = [
    "ARMEX 15X15",
    "ARMEX 15X20",
    "ARMEX DE 15 x 15",
    "ARMEX DE 15 x 20",
    "ARMEX 15 x 15",
    "ARMEX 15 x 20",
]

if selected_material.upper().strip() in [m.upper().strip() for m in armex_materials]:
    target_days = 7
else:
    target_days = 14

forecast_horizon = lead_time_days + target_days

forecast, best_order = arima_forecast(
    series,
    forecast_days=forecast_horizon
)

safety_stock = 1.5 * std_consumption * np.sqrt(lead_time_days)

lead_time_demand = forecast.iloc[:lead_time_days].sum()

projected_inventory_arrival = current_inventory - lead_time_demand

target_demand_after_arrival = forecast.iloc[
    lead_time_days:lead_time_days + target_days
].sum()

recommended_qty = (
    target_demand_after_arrival
    + safety_stock
    - projected_inventory_arrival
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

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Inventario al día", f"{current_inventory:,.2f}")
col2.metric("Consumo promedio", f"{avg_consumption:,.2f}")
col3.metric("ROP", f"{reorder_point:,.2f}")
col4.metric("Cantidad recomendada de compra", f"{recommended_qty:,.2f}")

if estimated_stockout_date is not None:
    col5.metric(
        "Stockout estimado",
        estimated_stockout_date.strftime("%Y-%m-%d")
    )
else:
    col5.metric("Stockout estimado", "Sin consumo")

if buy_now:
    col6.error("COMPRAR")
else:
    col6.success("ESPERAR")

# show_forecast = st.button("Predicción próximos 7 días")

# if show_forecast:
#     forecast_dates = pd.date_range(
#         start=filtered_df["fecha"].max() + pd.Timedelta(days=1),
#         periods=7,
#         freq="D"
#     )

#     forecast.index = forecast_dates

#     st.info(f"ARIMA{best_order}")


minimum_stock = avg_consumption * target_days

#########################
# SCATTER DE INVENTARIO #
#########################
st.subheader("INVENTARIO")

fig_inv = go.Figure()

fig_inv.add_trace(
    go.Scatter(
        x=filtered_df["fecha"],
        y=filtered_df["inventario"],
        mode="lines+markers",
        name="Inventario"
    )
)

fig_inv.add_trace(
    go.Scatter(
        x=filtered_df["fecha"],
        y=[reorder_point] * len(filtered_df),
        mode="lines",
        name="ROP",
        line=dict(dash="dash", color="red")
    )
)



fig_inv.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Inventario",
    template="plotly_white",
    height=450
)

# if show_forecast and forecast is not None:
#     last_inv = filtered_df["inventario"].iloc[-1]

#     pred_inv = last_inv - forecast.cumsum()
#     pred_inv = pred_inv.clip(lower=0)

#     fig_inv.add_trace(
#         go.Scatter(
#             x=forecast.index,
#             y=pred_inv,
#             mode="lines+markers",
#             name="Prediccion 7 dias",
#             line=dict(color="red")
#         )
#     )

all_dates = list(filtered_df["fecha"])

# if show_forecast:
#     all_dates.extend(forecast.index)

fig_inv.add_trace(
    go.Scatter(
        x=all_dates,
        y=[minimum_stock] * len(all_dates),
        mode="lines",
        name=f"Stock mínimo ({target_days} días)",
        line=dict(
            dash="dot",
            color="green",
            width=2
        )
    )
)

if estimated_stockout_date is not None:
    fig_inv.add_vline(
        x=estimated_stockout_date,
        line_dash="dot",
        line_color="orange",
        annotation_text="Stockout estimado",
        annotation_position="top"
    )

st.plotly_chart(fig_inv, use_container_width=True)



#############################
# CONSUMO VS TIEMPO BARPLOT #
#############################
st.subheader("CONSUMO A LO LARGO DEL TIEMPO")

fig_cons = go.Figure()

fig_cons.add_trace(
    go.Bar(
        x=filtered_df["fecha"],
        y=filtered_df["consumo"],
        name="Consumo actual"
    )
)

fig_cons.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Consumo",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_cons, use_container_width=True)


#################
# RECOMENDACIÓN #
#################
st.subheader("Recomendación:")

left, right = st.columns([1, 2])

with left:
    st.metric("Cantidad recomendada", f"{recommended_qty:,.2f}")

with right:
    if buy_now:
        st.error(
            f"""
            Inventario bajo.

            Comprar **{recommended_qty:,.2f}** unidades de **{selected_material}**.
            """
        )
    else:
        st.success(
            f"""
            Sobreinventario o inventario justo.

            No comprar.
            """
        )


##########################
# HEATMAP DE PROVEEDORES #
##########################
suppliers_df_raw.columns= ['material', 
                        'proveedor', 
                        'id', 
                        'pu', 
                        'cantidad', 
                        'total', 
                        'pedido', 
                        'recibido', 
                        'espera']

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

st.subheader("T/E PROVEEDORES 2026")

fig_sup = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="balance",
        colorbar_title="T/E (dias)",
        hoverongaps=False,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        hovertemplate=(
            "Proveedor: %{x}<br>"
            "Material: %{y}<br>"
            "T/E Promedio: %{z:.1f} días"
            "<extra></extra>"
        ),
        connectgaps=True,

    )
)

fig_sup.update_layout(
    xaxis_title="PROVEEDOR",
    yaxis_title="MATERIAL",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_sup, use_container_width=True)


#######################
# VALOR DE INVENTARIO #
#######################
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

df_baking = df_baking.set_index('fecha')

df = df_baking.copy()

# Preprocessing
prices_df_baking = prices_df_raw.copy()
prices_df_baking['fecha'] = pd.to_datetime(prices_df_baking['fecha'], dayfirst=True)
prices_df = prices_df_baking.copy()

prices_df['inv_al_dia'] = df_baking.loc[df_baking.index[-1]]['inventario'].reset_index().drop(columns='fecha')
prices_df['valor'] = round(prices_df['precio'] * prices_df['inv_al_dia'])

# Graph
st.subheader("Valor Inventario")

fig_dinero = go.Figure()

fig_dinero.add_trace(
    go.Bar(
        x=prices_df["material"],
        y=prices_df["valor"],
        name="INVENTARIO EN DINERO",
        hovertemplate=(
            "Material: %{x}<br>"
            "Valor total: $%{y}<br>"
            "<extra></extra>"
        )
    )
)

fig_dinero.update_layout(
    xaxis_title="Material",
    yaxis_title="Dinero",
    template="plotly_white",
    height=450
)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)

col_p1, col_p2 = st.columns(2)

col_p1.metric(label="Costo Total Inventario",
        value=f"${prices_df['valor'].sum():,.2f}")

max_row = prices_df.loc[prices_df["valor"].idxmax()]
col_p2.metric(
    "Más costoso",
    f"{max_row['material'].lower()}",
    f"${max_row['valor']:,.2f}"
)


st.plotly_chart(fig_dinero, use_container_width=True)




with st.expander("Datos filtrados"):
    st.dataframe(filtered_df)