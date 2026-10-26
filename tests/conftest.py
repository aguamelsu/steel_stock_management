from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def inventory_wide():
    """Hoja 'inventario' tal como la entrega gspread: una columna de texto por fecha."""
    long = pd.read_csv(DATA_DIR / "inventory_df.csv")
    wide = long.pivot(index="material", columns="fecha", values="inventario")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


@pytest.fixture(scope="session")
def suppliers_raw():
    """Hoja 'pedidos_proveedores' con fechas dd/mm/aaaa, como en Google Sheets."""
    return pd.read_csv(DATA_DIR / "pedidos_proveedores.csv")


@pytest.fixture(scope="session")
def prices_raw(inventory_wide):
    materials = inventory_wide["material"].tolist()
    return pd.DataFrame({
        "material": materials,
        "fecha": ["13/07/2026"] * len(materials),
        "precio": [100.0 + 7.3 * i for i in range(len(materials))],
    })
