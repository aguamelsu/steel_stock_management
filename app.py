import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

import inventory

# Configurar pagina
st.set_page_config(
    page_title="Steel Inventory Dashboard",
    layout="wide"
)

st.title("Inventario Aceros ABC")


##############################
# GOOGLE AUTH, PREPROCESSING #
##############################
@st.cache_resource
def get_gspread_client():
    # La app solo lee: solo lectura de Sheets, sin acceso a Drive
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ]

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
    )

    return gspread.authorize(credentials)


def load_google_sheet():
    spreadsheet = get_gspread_client().open_by_url(
        st.secrets["google_sheets"]["spreadsheet_url"]
    )

    # Una sola llamada de metadatos en lugar de una por spreadsheet.worksheet()
    worksheets = {}
    for ws in spreadsheet.worksheets():
        worksheets.setdefault(ws.title, ws)

    def records(title):
        if title not in worksheets:
            raise gspread.exceptions.WorksheetNotFound(title)
        return pd.DataFrame(worksheets[title].get_all_records())

    return records("inventario"), records("pedidos_proveedores"), records("precios")


@st.cache_data(ttl=300)
def load_data():
    """Descarga y preprocesa todo una vez por TTL, no en cada rerun."""
    inventory_df, suppliers_df_raw, prices_df_raw = load_google_sheet()

    inventory_long = inventory.melt_inventory(inventory_df)
    df = inventory.add_consumption(inventory_long)
    supplier_pivot = inventory.supplier_lead_time_pivot(
        inventory.prepare_suppliers(suppliers_df_raw)
    )
    prices_df = inventory.inventory_value(inventory_long, prices_df_raw)

    return df, supplier_pivot, prices_df


# El grid search de ARIMA (32 ajustes) es lo más costoso de la app: se ajusta
# una vez por serie y se reutiliza entre reruns y sesiones.
# cache_data (no cache_resource): cada rerun recibe su propia copia del modelo,
# porque forecast() sobre un mismo objeto desde varios hilos (sesiones
# simultáneas) devuelve resultados incorrectos.
fit_best_arima = st.cache_data(max_entries=64, show_spinner=False)(
    inventory.fit_best_arima
)

df, pivot, prices_df = load_data()

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

filtered_df = inventory.filter_inventory(df, selected_material, start_date, end_date)

if filtered_df.empty:
    st.warning("No hay datos suficientes para este filtro")
    st.stop()


##########################
# MATH AND STATISTICS #
##########################
lead_time_days = st.sidebar.number_input(
    "Días de anticipo para realizar pedido",
    min_value=1,
    max_value=15,
    value=7
)

# Por defecto: 7 días para ARMEX y 14 para el resto; el usuario puede cambiarlo.
# La key por material hace que al cambiar de material vuelva a su valor por defecto.
target_days = st.sidebar.number_input(
    "Objetivo de días de inventario",
    min_value=1,
    max_value=60,
    value=inventory.target_days_for(selected_material),
    key=f"target_days_{selected_material}"
)

# Series for ARIMA
series = (
    filtered_df
    .set_index("fecha")["consumo"]
    .sort_index()
)

forecast_horizon = lead_time_days + target_days

best_model, best_order = fit_best_arima(series)
forecast = inventory.forecast_from_fit(best_model, forecast_horizon)

metrics = inventory.replenishment_metrics(
    filtered_df, forecast, lead_time_days, target_days
)
current_inventory = metrics["current_inventory"]
avg_consumption = metrics["avg_consumption"]
maximum_inventory = metrics["maximum_inventory"]
recommended_qty = metrics["recommended_qty"]
reorder_point = metrics["reorder_point"]
buy_now = metrics["buy_now"]
estimated_stockout_date = metrics["estimated_stockout_date"]
minimum_stock = metrics["minimum_stock"]

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
        y=[maximum_inventory] * len(filtered_df),
        mode="lines",
        name="Inventario Máximo",
        line=dict(dash="dash", color="black")
    )
)


fig_inv.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Inventario",
    template="plotly_white",
    height=450
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