# steel_stock_management

## Dashboard

```
pip install -r requirements.txt
streamlit run app.py
```

Requiere `.streamlit/secrets.toml` con `gcp_service_account` y `google_sheets.spreadsheet_url`.

## Reporte de métricas y pruebas

```
pip install -r requirements-dev.txt
python script.py      # escribe reporte_metricas_inventario/
python -m pytest
```
