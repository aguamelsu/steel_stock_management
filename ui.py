"""
Formato y componentes visuales compartidos (español de México).
"""
import math
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

TIMEZONE = ZoneInfo("America/Mexico_City")

# Colores con significado (no decorativos)
SERIES = "#2a78d6"      # serie principal
SERIES_DARK = "#104281"  # marcador de mediana
MUTED = "#898781"       # dato con evidencia insuficiente
WARNING = "#fab219"     # estado: requiere atención (siempre con ícono y texto)
GRID = "rgba(137, 135, 129, 0.25)"
# Texto de gráficas: gris medio legible sobre fondo claro y oscuro
INK = "#6f6e69"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_MONTHS = ["ene", "feb", "mar", "abr", "may", "jun",
           "jul", "ago", "sep", "oct", "nov", "dic"]


def now():
    """Hora actual en la Ciudad de México (el servidor puede estar en UTC)."""
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)


def today():
    return now().normalize()


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT


def fecha(value):
    if _missing(value) or pd.isna(value):
        return "—"
    ts = pd.Timestamp(value)
    return f"{ts.day} {_MONTHS[ts.month - 1]} {ts.year}"


def hace(days):
    if _missing(days) or pd.isna(days):
        return "—"
    days = int(days)
    if days <= 0:
        return "hoy"
    if days == 1:
        return "hace 1 día"
    return f"hace {days} días"


def mxn(value, decimals=0):
    if _missing(value) or pd.isna(value):
        return "—"
    return f"${value:,.{decimals}f}"


def numero(value, decimals=0):
    if _missing(value) or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def porcentaje(value, decimals=1):
    if _missing(value) or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def freshness_badge(label, date, as_of, stale_after_days):
    """
    Insignia en Markdown 'Etiqueta 18 sep 2026 · hace 38 días': ámbar con reloj si
    el dato tiene más de stale_after_days días, gris si está al corriente.
    """
    if _missing(date) or pd.isna(date):
        return f":gray-badge[:material/help: {label}: sin fecha]"
    age = (pd.Timestamp(as_of) - pd.Timestamp(date).normalize()).days
    text = f"{label} {fecha(date)} · {hace(age)}"
    if age > stale_after_days:
        return f":orange-badge[:material/schedule: {text} · desactualizado]"
    return f":gray-badge[:material/check_circle: {text}]"


def style_figure(fig, height, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8 if not legend else 36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    title=None, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     showline=False, ticks="", title=None, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="", title=None,
                     automargin=True, ticksuffix="  ")
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}


def plotly_chart(fig):
    # theme=None: el tema de Streamlit fuerza la leyenda en vertical
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG)


def inject_css():
    st.markdown(
        """
<style>
[data-testid="stMetricValue"] {
    font-size: 25px;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.85rem;
}
[data-testid="stCaptionContainer"] {
    line-height: 1.45;
}
</style>
""",
        unsafe_allow_html=True,
    )
