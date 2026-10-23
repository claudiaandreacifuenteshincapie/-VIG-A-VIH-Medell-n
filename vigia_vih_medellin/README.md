# VIGÍA-VIH Medellín (nombre de trabajo)

Vigilancia de la **presentación avanzada del VIH al momento de la notificación** (estadio SIDA o muerto) con datos **reales** de SIVIGILA, evento 850, Medellín 2012–2023.
Reemplaza al proyecto DULCINEA v2 (cohorte sintética y modelos de abandono/deterioro), que no puede construirse con SIVIGILA porque no contiene fechas de dispensación.

> El nombre es una propuesta (la evaluación recomendó cambiar «DULCINEA»). No verifiqué colisiones con otros proyectos o marcas. Se cambia en `config.py` (`NOMBRE`).

## Estructura
```
config.py              nombre, rutas, umbrales (MIN_CELDA, año de corte, semilla)
run_pipeline.py        Bronze -> Silver -> Gold en un comando
src/etl.py             limpieza, Silver sin identificadores, población de análisis
src/estadistica.py     Wilson, regresión logística (IRLS), embudo
src/analitica.py       cubo, series, territorio, regresiones, hipótesis H1-H3
src/modelo.py          modelo predictivo con validación temporal, bootstrap, H4
app.py                 dashboard Streamlit (lee solo data/gold)
tests/test_pipeline.py pruebas
docs/                  diccionario de variables y estado de verificación
data/landing/          AQUÍ va sivigila_vih.csv (no se versiona)
data/silver/           datos limpios a nivel de notificación (no se versiona)
data/gold/             tablas agregadas que consume el dashboard
```

## Uso
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copiar sivigila_vih.csv en data/landing/
python run_pipeline.py          # ~25 s; regenera data/silver y data/gold
python tests/test_pipeline.py   # o: python -m pytest -q
streamlit run app.py
```

## Diseño analítico (resumen)
* **Población de análisis:** primera notificación por persona, residente en Medellín, con estadio conocido (14.586 de 15.470 notificaciones).
* **Desenlace:** `tardio` = estadio SIDA o muerto. Aproximación; SIVIGILA no registra CD4.
* **H1** tendencia decreciente · **H2** variación territorial · **H3** edad y aseguramiento · **H4** modelo vs. línea base de solo edad (validación temporal 2012-2018 → 2019-2023). Las reglas de decisión están en `data/gold/hipotesis.csv` y en el documento de entrega.
* **No se calculan tasas** (no hay denominadores DANE en el proyecto) y **no se puntúan personas** en el dashboard.

## Privacidad
* El CSV crudo y `data/silver/` contienen cuasi-identificadores (edad, barrio-comuna, fechas): **no los suba a GitHub ni a un despliegue público**. `.gitignore` ya los excluye.
* `data/gold/cubo_casos.csv` es agregado pero fino (año × comuna × sexo × edad × aseguramiento × mecanismo × nacionalidad). Si va a publicar el dashboard, suba `MIN_CELDA` o retire dimensiones del cubo, y obtenga la aprobación de la Secretaría de Salud / comité de ética.
* Seguimiento pendiente de la ruta de aprobación y los acuerdos de uso de datos con la fuente.

## Estado de verificación
Probado en este entorno: ETL, estadística, modelo, hipótesis y 6 pruebas unitarias (incluye comparación de la regresión propia con scikit-learn).
**No probado con Streamlit/Plotly reales** (no estaban instalados en el entorno de desarrollo): la lógica de datos de `app.py` se ejecutó con módulos simulados. Si algo falla al abrir el tablero, revise primero la versión de `streamlit`/`plotly`.
