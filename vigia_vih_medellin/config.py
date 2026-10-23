"""Configuración central del proyecto. Cambiar aquí el nombre, rutas y umbrales."""
from pathlib import Path

BASE = Path(__file__).resolve().parent

# --- Identidad del proyecto (el nombre es un NOMBRE DE TRABAJO propuesto; ver docs/README) ---
NOMBRE = "VIGÍA-VIH Medellín"
SUBTITULO = ("Presentación avanzada del VIH al momento de la notificación · "
             "SIVIGILA evento 850 · Medellín")

# --- Rutas ---
RAW_CSV = BASE / "data" / "landing" / "sivigila_vih.csv"
DIR_SILVER = BASE / "data" / "silver"
DIR_GOLD = BASE / "data" / "gold"

# --- Parámetros analíticos ---
SEED = 2026
MIN_CELDA = 10          # no se muestran porcentajes de estratos con n < MIN_CELDA
ANIO_BASE = 2012        # año de referencia para la tendencia (OR por año)
CORTE_ENTRENAMIENTO = 2018   # entrenamiento: años <= 2018 ; prueba temporal: años >= 2019
N_BOOT = 500            # remuestreos bootstrap para IC del modelo
