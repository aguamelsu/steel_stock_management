"""
Secciones del dashboard: tiempos de entrega por proveedor y valor del inventario.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import procurement
import ui
import valuation

EVIDENCE_ICONS = {"sufficient": "●", "limited": "◐", "insufficient": "○"}


def _ordenes(n):
    return f"{n} orden" if n == 1 else f"{n} órdenes"


##################################
# TIEMPOS DE ENTREGA POR PROVEEDOR #
##################################
def _lead_time_strip(material_lines, rows):
    """Cada orden recibida es un punto; la raya vertical es la mediana del proveedor."""
    rows = rows.sort_values(["mediana_dias", "ordenes"], ascending=[False, True])
    position = {p: i for i, p in enumerate(rows["proveedor"])}
    labels = [f"{p} · {_ordenes(n)}" for p, n in zip(rows["proveedor"], rows["ordenes"])]
    evidence = dict(zip(rows["proveedor"], rows["evidencia"]))

    points = material_lines.dropna(subset=["dias_entrega"]).copy()
    # Separa verticalmente las órdenes con el mismo número de días para que no se oculten
    rank = points.groupby(["proveedor", "dias_entrega"]).cumcount()
    size = points.groupby(["proveedor", "dias_entrega"])["orden"].transform("size")
    points["y"] = points["proveedor"].map(position) + (rank - (size - 1) / 2) * 0.12
    points["suficiente"] = points["proveedor"].map(evidence) != "insufficient"

    fig = go.Figure()
    for sufficient, color, name in [
        (True, ui.SERIES, "Orden recibida"),
        (False, ui.MUTED, "Orden recibida · evidencia insuficiente"),
    ]:
        subset = points[points["suficiente"] == sufficient]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["dias_entrega"], y=subset["y"], mode="markers", name=name,
            marker=dict(size=10, color=color, opacity=0.85,
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([
                subset["proveedor"], subset["orden"].astype(str),
                subset["pedido"].map(ui.fecha), subset["recibido"].map(ui.fecha),
            ], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> · orden %{customdata[1]}<br>"
                           "Pedido: %{customdata[2]}<br>Recibido: %{customdata[3]}<br>"
                           "Tiempo de entrega: %{x} días<extra></extra>"),
        ))

    # Una mediana de 1–2 órdenes no significa nada: solo se marca con evidencia limitada o mejor
    with_median = rows[rows["evidencia"] != "insufficient"]
    if len(with_median):
        fig.add_trace(go.Scatter(
            x=with_median["mediana_dias"], y=with_median["proveedor"].map(position), mode="markers",
            name="Mediana",
            marker=dict(symbol="line-ns", size=22, line=dict(width=3, color=ui.SERIES_DARK)),
            customdata=np.stack([with_median["proveedor"], with_median["ordenes"],
                                 with_median["min_dias"], with_median["max_dias"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>Mediana: %{x:g} días<br>"
                           "Rango: %{customdata[2]}–%{customdata[3]} días · %{customdata[1]} órdenes"
                           "<extra></extra>"),
        ))

    ui.style_figure(fig, height=70 + 46 * len(rows), legend=True)
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=labels,
                     range=[-0.6, len(rows) - 0.4])
    fig.update_xaxes(rangemode="tozero", ticksuffix=" d")
    return fig


def _supplier_insight(comparison):
    rows, fastest, candidates = comparison["rows"], comparison["fastest"], comparison["candidates"]
    minimum = procurement.MIN_ORDERS_TO_COMPARE

    if fastest is None:
        st.markdown(
            f"Ningún proveedor tiene al menos **{minimum} órdenes recibidas** de este material. "
            "Los tiempos que se muestran son datos aislados: no permiten elegir proveedor "
            "ni estimar el tiempo de entrega con confianza."
        )
    else:
        stats = (f"mediana **{fastest['mediana_dias']:g} días**, rango "
                 f"{fastest['min_dias']:g}–{fastest['max_dias']:g} días en {_ordenes(fastest['ordenes'])}")
        if comparison["comparable"]:
            st.markdown(f"**{fastest['proveedor']}** tiene el menor tiempo de entrega: {stats}.")
            for _, other in candidates.iloc[1:].iterrows():
                st.markdown(f"Le sigue **{other['proveedor']}**: mediana {other['mediana_dias']:g} días "
                            f"en {_ordenes(other['ordenes'])}.")
        elif len(rows) == 1:
            st.markdown(f"**{fastest['proveedor']}** es el único proveedor con entregas registradas: {stats}.")
        else:
            st.markdown(
                f"Solo **{fastest['proveedor']}** tiene {minimum} o más órdenes recibidas: {stats}. "
                "Los demás proveedores tienen muy pocas entregas para compararlos."
            )

        if fastest["max_dias"] >= 3 * fastest["mediana_dias"] and fastest["max_dias"] - fastest["mediana_dias"] >= 7:
            st.markdown(f":orange-badge[:material/warning: Entrega atípica] Una orden tardó "
                        f"**{fastest['max_dias']:g} días**; planear con la mediana puede quedarse corto.")

    age = rows["dias_desde_recepcion"].min()
    if age > procurement.STALE_AFTER_DAYS:
        newest = rows["ultima_recepcion"].max()
        st.markdown(f":orange-badge[:material/schedule: Datos históricos] La entrega más reciente de este "
                    f"material fue el {ui.fecha(newest)} ({ui.hace(age)}). Los tiempos actuales pueden ser distintos.")


def _supplier_table(rows):
    table = pd.DataFrame({
        "Proveedor": rows["proveedor"],
        "Órdenes": rows["ordenes"],
        "Mediana (días)": rows["mediana_dias"],
        "Rango (días)": [f"{a:g}–{b:g}" for a, b in zip(rows["min_dias"], rows["max_dias"])],
        "Última recepción": [f"{ui.fecha(d)} · {ui.hace(n)}" for d, n in
                             zip(rows["ultima_recepcion"], rows["dias_desde_recepcion"])],
        "Último precio pagado": rows["ultimo_precio"],
        "Evidencia": [f"{EVIDENCE_ICONS[e]} {procurement.evidence_label(e)}" for e in rows["evidencia"]],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Órdenes": st.column_config.NumberColumn(
                help="Órdenes de compra recibidas de este material con este proveedor."),
            "Mediana (días)": st.column_config.NumberColumn(
                format="%g", help="Días naturales entre pedido y recepción; la mediana no se distorsiona por una sola entrega atípica."),
            "Último precio pagado": st.column_config.NumberColumn(
                format="dollar", help="Precio unitario (MXN) de la orden más reciente con este proveedor. Es histórico."),
            "Evidencia": st.column_config.TextColumn(
                help=(f"● Suficiente: {procurement.MIN_ORDERS_SUFFICIENT}+ órdenes · "
                      f"◐ Limitada: {procurement.MIN_ORDERS_TO_COMPARE}–{procurement.MIN_ORDERS_SUFFICIENT - 1} · "
                      f"○ Insuficiente: menos de {procurement.MIN_ORDERS_TO_COMPARE}. "
                      "Solo se comparan proveedores con evidencia limitada o suficiente.")),
        },
    )


def render_supplier_lead_times(purchase_lines, selected_material, as_of):
    st.subheader("Tiempos de entrega por proveedor")

    if purchase_lines.empty:
        st.info("La hoja *pedidos_proveedores* no tiene órdenes registradas.", icon=":material/inventory_2:")
        return

    first_order, last_order, last_receipt = procurement.order_dates(purchase_lines)
    st.caption(
        f"Fuente: hoja *pedidos_proveedores* · {len(purchase_lines)} partidas de "
        f"{purchase_lines['orden'].nunique()} órdenes de compra pedidas entre el "
        f"{ui.fecha(first_order)} y el {ui.fecha(last_order)}. "
        "Tiempo de entrega = días naturales entre la fecha de pedido y la de recepción."
    )
    st.markdown(ui.freshness_badge("Última recepción registrada:", last_receipt, as_of,
                                   procurement.STALE_AFTER_DAYS))

    inconsistent = procurement.inconsistent_lead_times(purchase_lines)
    if len(inconsistent):
        st.warning(f"{len(inconsistent)} partidas tienen «dias en llegar» distinto de la diferencia "
                   "entre recepción y pedido. Se usa el valor capturado en la hoja.",
                   icon=":material/rule:")

    stats = procurement.lead_time_stats(purchase_lines, as_of)
    comparison = procurement.compare_suppliers(stats, selected_material)
    rows = comparison["rows"]

    with st.container(border=True):
        st.markdown(f"##### {selected_material}")
        if rows.empty:
            st.info("No hay entregas registradas de este material en *pedidos_proveedores*, "
                    "así que no se puede estimar su tiempo de entrega.", icon=":material/inventory_2:")
        else:
            chart_col, text_col = st.columns([3, 2], gap="large")
            with chart_col:
                material_lines = purchase_lines[purchase_lines["material"] == selected_material]
                ui.plotly_chart(_lead_time_strip(material_lines, rows))
            with text_col:
                st.markdown("**¿Qué proveedor conviene?**")
                _supplier_insight(comparison)
            _supplier_table(rows)

    with st.expander("Tiempos de entrega de todos los materiales"):
        st.caption("Cada celda: mediana de días · órdenes recibidas. "
                   "«—» significa que no hay entregas registradas (no es una entrega de 0 días).")
        st.dataframe(procurement.lead_time_matrix(stats).rename_axis(index="Material", columns=None),
                     width="stretch")

    with st.expander("¿Por qué no se muestra la puntualidad de cada proveedor?"):
        st.markdown(
            "La hoja solo registra la fecha de pedido y la de recepción. Para medir puntualidad "
            "y retrasos hace falta:\n"
            "- **Fecha compromiso de entrega** de cada orden (la que prometió el proveedor).\n"
            "- **Cantidad pedida vs. recibida** por partida, para detectar entregas parciales.\n"
            "- **Órdenes abiertas** (pedidas y aún no recibidas): hoy solo aparecen órdenes ya recibidas, "
            "así que las entregas más lentas en curso no se ven.\n"
            "- **Registro continuo**: la última orden capturada es del "
            f"{ui.fecha(last_order)}; las compras posteriores no están en la hoja."
        )


#######################
# VALOR DEL INVENTARIO #
#######################
def _value_bars(priced):
    """Barras horizontales ordenadas; ámbar = existencia por encima del inventario máximo."""
    data = priced.sort_values("valor")
    over = data["exceso_cantidad"].fillna(0) > 0
    customdata = np.stack([
        data["cantidad"].map(ui.numero) + " " + data["unidad"].fillna(""),
        data["precio_referencia"].map(lambda v: ui.mxn(v, 2)),
        data["participacion"].map(ui.porcentaje),
        data["cobertura_dias"].map(lambda v: "sin consumo registrado" if np.isinf(v) else f"{ui.numero(v)} días"),
        data["inventario_maximo"].map(ui.numero),
        data["exceso_valor"].map(ui.mxn),
    ], axis=-1)

    fig = go.Figure()
    for mask, color, name in [(~over, ui.SERIES, "Dentro del máximo"),
                              (over, ui.WARNING, "Por encima del máximo")]:
        if not mask.any():
            continue
        subset = data[mask]
        fig.add_trace(go.Bar(
            x=subset["valor"], y=subset["material"], orientation="h", name=name,
            marker=dict(color=color, cornerradius=4),
            text=[f"{ui.mxn(v)} · {ui.porcentaje(s)}" for v, s in zip(subset["valor"], subset["participacion"])],
            textposition="outside", cliponaxis=False,
            customdata=customdata[mask.to_numpy()],
            hovertemplate=("<b>%{y}</b><br>Valor: $%{x:,.0f} (%{customdata[2]})<br>"
                           "Existencia: %{customdata[0]} · precio ref. %{customdata[1]}<br>"
                           "Cobertura: %{customdata[3]}<br>"
                           "Inventario máximo: %{customdata[4]} · exceso %{customdata[5]}"
                           "<extra></extra>"),
        ))

    legend = bool(over.any())
    ui.style_figure(fig, height=24 + 32 * len(data) + (28 if legend else 0), legend=legend)
    fig.update_layout(bargap=0.38, barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=list(data["material"]))
    fig.update_xaxes(tickprefix="$", tickformat="~s", range=[0, data["valor"].max() * 1.28])
    return fig


def _attention_list(priced):
    flagged = priced[priced["exceso_valor"].fillna(0) > 0].sort_values("exceso_valor", ascending=False)
    st.markdown("**Requieren atención**")
    if flagged.empty:
        st.markdown(":gray-badge[:material/check_circle: Sin excesos] Ningún material supera su "
                    "inventario máximo con los parámetros actuales.")
        return
    for _, row in flagged.iterrows():
        coverage = ("sin consumo registrado" if np.isinf(row["cobertura_dias"])
                    else f"cobertura de {ui.numero(row['cobertura_dias'])} días")
        st.markdown(
            f":orange-badge[:material/warning: Exceso] **{row['material']}**: "
            f"{ui.mxn(row['exceso_valor'])} por encima del máximo · {coverage}"
        )


def _coverage_list(priced, top_n=3):
    """Materiales cuyo inventario dura más al ritmo de consumo registrado."""
    st.markdown("**Mayor cobertura**")
    st.caption("Días que alcanzaría la existencia actual al consumo promedio registrado.")
    ranked = priced.sort_values("cobertura_dias", ascending=False).head(top_n)
    for _, row in ranked.iterrows():
        days = ("sin consumo registrado" if np.isinf(row["cobertura_dias"])
                else f"{ui.numero(row['cobertura_dias'])} días")
        st.markdown(f"**{row['material']}** · {days} · {ui.mxn(row['valor'])} "
                    f"({ui.porcentaje(row['participacion'])} del valor)")


def _valuation_table(val):
    table = pd.DataFrame({
        "Material": val["material"],
        "Existencia": val["cantidad"],
        "Unidad": val["unidad"].fillna("—"),
        "Precio de referencia": val["precio_referencia"],
        "Valor (MXN)": val["valor"],
        "Participación": val["participacion"],
        "Cobertura (días)": val["cobertura_dias"].replace(np.inf, np.nan),
        "Inventario máximo": val["inventario_maximo"],
        "Exceso (MXN)": val["exceso_valor"],
        "Último costo de compra": val["ultimo_costo_compra"],
        "Fecha último costo": val["fecha_ultimo_costo"].map(ui.fecha),
        "Costo vs. referencia": val["dif_costo_vs_referencia"],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Existencia": st.column_config.NumberColumn(format="localized"),
            "Precio de referencia": st.column_config.NumberColumn(
                format="dollar", help="Hoja precios (MXN por unidad)."),
            "Valor (MXN)": st.column_config.NumberColumn(format="localized", help="Existencia × precio de referencia."),
            "Participación": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1, color="#2a78d6",
                help="Porcentaje del valor total del inventario."),
            "Cobertura (días)": st.column_config.NumberColumn(
                format="%.0f",
                help="Existencia ÷ consumo promedio por registro diario (lunes a sábado). Vacío = sin consumo registrado."),
            "Inventario máximo": st.column_config.NumberColumn(
                format="%.0f", help="Mismo cálculo que la línea de inventario máximo de la gráfica de inventario."),
            "Exceso (MXN)": st.column_config.NumberColumn(
                format="localized", help="Valor de la existencia por encima del inventario máximo."),
            "Último costo de compra": st.column_config.NumberColumn(
                format="dollar", help="Precio unitario de la orden más reciente en pedidos_proveedores. Histórico."),
            "Costo vs. referencia": st.column_config.NumberColumn(
                format="percent", help="Último costo de compra respecto al precio de referencia."),
        },
    )


def render_inventory_value(inventory_long, consumption_df, book, units, lead_time_days, as_of):
    st.subheader("Valor del inventario")
    st.caption(
        "Valor = existencia del último registro de la hoja *inventario* × precio de referencia "
        "de la hoja *precios* (MXN). Es un valor de referencia: no equivale al costo de reposición actual."
    )

    val = valuation.inventory_valuation(inventory_long, consumption_df, book, lead_time_days, units)
    st.markdown(
        ui.freshness_badge("Existencias al", val["fecha_cantidad"].iloc[0], as_of,
                           valuation.STALE_QUANTITIES_AFTER_DAYS)
        + " "
        + ui.freshness_badge("Precios de referencia al", val["fecha_precio_referencia"].max(), as_of,
                             valuation.STALE_PRICES_AFTER_DAYS)
    )

    priced = val[val["valor"].notna()]
    unpriced = val[val["valor"].isna()]
    if priced.empty:
        st.info("No hay precios de referencia para ningún material en la hoja *precios*; "
                "no es posible valorar el inventario.", icon=":material/price_change:")
        return
    if len(unpriced):
        st.warning("Sin precio de referencia (no incluidos en el total): "
                   + ", ".join(unpriced["material"]), icon=":material/price_change:")

    total = priced["valor"].sum()
    top = priced.nlargest(3, "valor")
    over = priced[priced["exceso_valor"].fillna(0) > 0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor total (MXN)", ui.mxn(total), border=True,
              help="Suma de existencia × precio de referencia de todos los materiales con precio.")
    k2.metric("Concentración en 3 materiales", ui.porcentaje(valuation.concentration(priced)), border=True,
              help="Participación conjunta de: " + ", ".join(top["material"]) + ".")
    k3.metric("Valor por encima del máximo", ui.mxn(over["exceso_valor"].sum()), border=True,
              help=("Existencia por encima del inventario máximo, valuada a precio de referencia. "
                    f"Usa {lead_time_days} días de anticipo (barra lateral) y los días objetivo por "
                    "defecto de cada material (7 ARMEX, 14 el resto)."))
    k4.metric("Materiales por encima del máximo", f"{len(over)} de {len(priced)}", border=True)

    chart_col, list_col = st.columns([3, 2], gap="large")
    with chart_col:
        ui.plotly_chart(_value_bars(priced))
    with list_col:
        _attention_list(priced)
        _coverage_list(priced)

    with st.expander("Detalle por material"):
        _valuation_table(val)
        st.caption(
            "Precio de referencia: hoja *precios*. Último costo de compra: orden más reciente en "
            "*pedidos_proveedores*. Ninguno de los dos es un costo de reposición vigente."
        )
