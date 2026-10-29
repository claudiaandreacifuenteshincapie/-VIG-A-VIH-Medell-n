# VIGÍA-VIH Medellín

Vigilancia de la **presentación avanzada del VIH al momento de la notificación** (estadio SIDA o muerto) con datos **reales** de SIVIGILA, evento 850, Medellín 2012–2023.
Reemplaza al proyecto DULCINEA v2 (cohorte sintética y modelos de abandono/deterioro), que no puede construirse con SIVIGILA porque no contiene fechas de dispensación.

## Estructura (Arquitectura Medallion + Notebooks EDA)
```
config.py              nombre, rutas Medallion (DIR_BRONZE, DIR_SILVER, DIR_GOLD), umbrales
run_pipeline.py        Bronze -> Silver -> Gold en un comando
src/etl.py             limpieza, Silver sin identificadores, población de análisis
src/estadistica.py     Wilson, regresión logística (IRLS), embudo
src/analitica.py       cubo, series, territorio, regresiones, hipótesis H1-H3
src/modelo.py          modelo predictivo con validación temporal, bootstrap, H4
app.py                 dashboard Streamlit (lee solo data/gold)
notebooks/             Notebooks Jupyter de EDA estructurados por capa Medallion:
  ├── 01_EDA_Capa_Bronze_Auditoria_Cruda.ipynb
  ├── 02_EDA_Capa_Silver_Limpieza_y_Cohorte.ipynb
  └── 03_EDA_Capa_Gold_Hipotesis_Modelo_y_Paridad.ipynb
data/
  ├── bronze/          sivigila_vih.csv (crudo) y _metadata_bronze.json (huella SHA-256)
  ├── silver/          notificaciones_silver.csv, poblacion_analisis.csv y _reporte_calidad.json
  └── gold/            17 tablas y JSON agregados que consume el dashboard app.py
tests/test_pipeline.py pruebas unitarias del pipeline
docs/                  diccionario de variables y estado de verificación
```

## Uso
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # ~25 s; procesa data/bronze -> data/silver -> data/gold
python tests/test_pipeline.py   # o: python -m pytest -q
jupyter lab notebooks/          # explorar los 3 Notebooks EDA por capa Medallion
streamlit run app.py            # abrir el tablero interactivo
```

## Diseño analítico y Reglas de Negocio (resumen)
* **Capa Bronze (`15.470` notificaciones × `116` columnas):** Registro de huella `SHA-256`, reparación de *mojibake* (`cp1252` -> `utf-8`) y auditoría de anomalías crudas (475 retrasos negativos laboratorio-notificación, 269 diferencias entre año epidemiológico y calendario, 1 fecha de nacimiento < 1910 y 3 edades > 105 años).
* **Capa Silver (`14.586` personas en población de análisis):** Seudonimización de `ID` con `SHA-256` + sal (`vigia-v1|`), eliminación de identificadores directos (Ley 1581 de 2012), estandarización de edad a años cumplidos (`[0, 105]`), deduplicación por primera notificación cronológica (`14.881` personas únicas; `569` con re-notificaciones) y filtro de residencia en Medellín con estadio clínico conocido (`295` excluidos).
* **Capa Gold (`17` artefactos agregados e inferenciales):**
  * **Desenlace (`tardio`):** Estadio SIDA o Muerto en la 1.ª notificación (`1.491` casos = `10,22 %`).
  * **H1 (Tendencia decreciente):** Se sostiene (`OR/año ajustado = 0,862`, IC95 `0,847–0,878`).
  * **H2 (Variación territorial entre comunas):** No se sostiene tras ajustar (`LRT p = 0,253`).
  * **H3 (Edad >= 40 y aseguramiento no contributivo):** Se sostiene (`OR` entre `3,32` y `4,35` para >=40 años; `2,07` subsidiado; `2,54` no asegurado).
  * **H4 (Modelo predictivo en validación temporal 2012–2018 -> 2019–2023):** Se sostiene en discriminación (`AUC = 0,724` vs. `0,682` solo-edad; `Lift@20% = 2,43`), no en calibración global (`intercepto = -0,89` por caída secular de prevalencia de `14,3 %` a `6,2 %`).
