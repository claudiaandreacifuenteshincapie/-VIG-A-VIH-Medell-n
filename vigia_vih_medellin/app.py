"""Dashboard VIGÍA-VIH — lee SOLO tablas agregadas de data/gold (nunca registros individuales).
Ejecutar:  streamlit run app.py        (antes:  python run_pipeline.py)
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from src.estadistica import limites_embudo, wilson

NAVY, TEAL, RED, GRIS = "#1F4E78", "#148F77", "#B03A2E", "#8A94A6"
ORDEN_EDAD = ["<18", "18-28", "29-39", "40-49", "50-59", "60+", "Sin dato"]

st.set_page_config(page_title=config.NOMBRE, page_icon="🩺", layout="wide")
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
.hdr {{ background: linear-gradient(90deg,{NAVY},#2E75B6); padding: 1.2rem 1.6rem; border-radius: 10px; color: white; }}
.hdr h1 {{ margin:0; font-size:1.6rem; color:white; }} .hdr p {{ margin:.25rem 0 0; color:#D9E2F3; font-size:.92rem; }}
.badge {{ display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.4);
         padding:2px 10px; border-radius:12px; font-size:.72rem; margin-top:.5rem; margin-right:.4rem; }}
div[data-testid="stMetric"] {{ background:#F7F9FC; border:1px solid #E3E8F0; border-radius:10px; padding:.6rem .9rem; }}
button[data-baseweb="tab"] {{ font-weight:600; }}
.nota {{ color:#6B7589; font-size:.8rem; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- carga
@st.cache_data
def leer(nombre):
    return pd.read_csv(config.DIR_GOLD / nombre)


@st.cache_data
def leer_json(nombre):
    return json.loads((config.DIR_GOLD / nombre).read_text(encoding="utf-8"))


if not (config.DIR_GOLD / "kpis.json").exists():
    st.error("Todavía no hay resultados para mostrar. Comprueba que el archivo sivigila_vih.csv esté en la carpeta data/bronze/ y ejecuta python run_pipeline.py para preparar la información.")
    st.stop()

K, RES = leer_json("kpis.json"), leer_json("resumen_estadistico.json")
CUBO = leer("cubo_casos.csv")
CUBO["grupo_edad"] = CUBO["grupo_edad"].astype(str)
ANIO_PARCIAL = int(K["anio_parcial"])


def agregar(cubo, por):
    """Agrega el cubo, calcula % e IC95 de Wilson y oculta estratos con n < MIN_CELDA."""
    g = cubo.groupby(por, dropna=False).agg(n=("n", "sum"), n_tardio=("n_tardio", "sum")).reset_index()
    g["pct"] = np.where(g["n"] > 0, g["n_tardio"] / g["n"] * 100, np.nan)
    lo, hi = wilson(g["n_tardio"], g["n"])
    g["ic_inf"], g["ic_sup"] = lo * 100, hi * 100
    sup = g["n"] < config.MIN_CELDA
    g.loc[sup, ["pct", "ic_inf", "ic_sup"]] = np.nan
    g["suprimido"] = sup
    return g


def miles(n):
    """Entero con punto como separador de miles (formato es-CO)."""
    return f"{int(n):,}".replace(",", ".")


def fmt(v, d=1, suf=""):
    return "—" if pd.isna(v) else f"{v:,.{d}f}{suf}"


# ---------------------------------------------------------------- encabezado y filtros
st.markdown(f"""<div class="hdr"><h1>🩺 {config.NOMBRE}</h1><p>Personas con VIH notificadas en Medellín: qué muestran los registros y cómo interpretar sus resultados</p>
<span class="badge">Registros de SIVIGILA</span><span class="badge">Del {K['fecha_min']} al {K['fecha_max']}</span>
<span class="badge">{miles(K['poblacion_analisis'])} personas incluidas, residentes en Medellín</span></div>""",
            unsafe_allow_html=True)
st.caption("Este tablero resume los registros de VIH enviados al Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA) en Medellín. VIH significa virus de la inmunodeficiencia humana y SIDA, síndrome de inmunodeficiencia adquirida. Cada persona se cuenta una sola vez, usando su primera notificación. Las cifras describen a las personas que aparecen en estos registros y cumplen los criterios del análisis; no representan necesariamente todos los casos de VIH que existen en la ciudad.")

st.sidebar.header("Elige qué información consultar")
st.sidebar.caption("Puedes combinar varias opciones. Al cambiar una selección, se actualizan las tres primeras pestañas.")
a_min, a_max = int(CUBO["anio"].min()), int(CUBO["anio"].max())
rango = st.sidebar.slider("Años de notificación", a_min, a_max, (a_min, a_max),
                          help="Muestra los registros cuya primera notificación está dentro de los años elegidos.")
incluir_parcial = st.sidebar.checkbox(f"Incluir {ANIO_PARCIAL} (información disponible hasta {K['fecha_max']})", value=True,
                                      help="Este año todavía no está completo. Déjalo marcado para incluir lo recibido hasta la fecha indicada; desmárcalo para comparar solo años completos.")
sexo_sel = st.sidebar.multiselect("Sexo registrado", sorted(CUBO["sexo"].unique()), default=sorted(CUBO["sexo"].unique()),
                                  help="Permite mostrar uno o varios de los sexos registrados en la notificación.")
edad_op = [e for e in ORDEN_EDAD if e in set(CUBO["grupo_edad"])]
edad_sel = st.sidebar.multiselect("Edad al momento de la notificación", edad_op, default=edad_op,
                                  help="Agrupa a las personas por rangos de edad. Puedes elegir uno o varios rangos.")
aseg_op = sorted(CUBO["aseguramiento"].unique())
aseg_sel = st.sidebar.multiselect("Afiliación al sistema de salud", aseg_op, default=aseg_op,
                                  help="Muestra los registros según el tipo de afiliación reportado, por ejemplo contributivo, subsidiado o sin afiliación.")
com_op = sorted(CUBO["comuna"].unique())
com_sel = st.sidebar.multiselect("Comuna o corregimiento de residencia", com_op, default=com_op,
                                 help="Permite consultar una o varias zonas de residencia. Los registros sin comuna aparecen como 'Sin información'.")
st.sidebar.caption(f"Por privacidad, no se muestran porcentajes de grupos con menos de {config.MIN_CELDA} personas. La cifra pequeña puede aparecer en el conteo, pero no su porcentaje.")
st.sidebar.caption("Importante: estos filtros solo cambian las pestañas 1, 2 y 3. Las pestañas 4, 5, 6 y 7 presentan resultados calculados con todas las personas incluidas en el estudio, sin aplicar las selecciones de esta barra lateral.")

m = (CUBO["anio"].between(*rango) & CUBO["sexo"].isin(sexo_sel) & CUBO["grupo_edad"].isin(edad_sel)
     & CUBO["aseguramiento"].isin(aseg_sel) & CUBO["comuna"].isin(com_sel))
if not incluir_parcial:
    m &= CUBO["anio"] != ANIO_PARCIAL
F = CUBO[m]
if F.empty:
    st.warning("Los filtros no dejan ningún caso.")
    st.stop()

tabs = st.tabs(["1 · Vista general", "2 · Características de las personas", "3 · Comunas y corregimientos",
                "4 · Factores relacionados", "5 · Evaluación de una herramienta", "6 · Calidad de la información",
                "7 · Preguntas y alcance del estudio"])

# ---------------------------------------------------------------- 1 Resumen
with tabs[0]:
    tot = agregar(F.assign(todo="t"), "todo").iloc[0]
    sx = F.groupby("sexo")["n"].sum()
    sin_com = F.loc[F["comuna"] == "Sin información", "n"].sum()
    c = st.columns(4)
    c[0].metric("Personas incluidas", miles(tot['n']),
                help="Cantidad de personas únicas que quedan después de aplicar las selecciones de la barra lateral. Cada persona se cuenta una sola vez.")
    c[1].metric("En etapa avanzada al notificar", fmt(tot["pct"], 1, " %"),
                help=f"Porcentaje de personas cuya primera notificación ya registraba SIDA o muerte. El rango de referencia del 95 % es {fmt(tot['ic_inf'])}–{fmt(tot['ic_sup'])} %.")
    c[2].metric("Registradas como mujeres", fmt(sx.get("Mujer", 0) / tot["n"] * 100, 1, " %"),
                help="Porcentaje de las personas seleccionadas que aparecen registradas con sexo mujer.")
    c[3].metric("Sin comuna identificada", fmt(sin_com / tot["n"] * 100, 1, " %"),
                help="Porcentaje de las personas seleccionadas para quienes no se informó una comuna de residencia.")
    st.caption("Las cifras resumen cuántas personas cumplen la selección actual y qué proporción tenía un registro de SIDA o muerte al ser notificada por primera vez. El total ayuda a conocer el tamaño del grupo; el porcentaje permite comparar grupos de tamaños distintos; y el dato de comuna faltante advierte cuánto puede limitarse la lectura territorial.")

    por_anio = agregar(F, "anio")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=por_anio["anio"], y=por_anio["n"], name="Personas registradas", marker_color="#C9D6E8", secondary_y=False)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["ic_sup"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", secondary_y=True)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["ic_inf"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(176,58,46,.18)", name="Rango de referencia (95 %)", hoverinfo="skip", secondary_y=True)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["pct"], mode="lines+markers", name="% en etapa avanzada",
                    line=dict(color=RED, width=3), secondary_y=True)
    fig.add_vline(x=2018.5, line_dash="dot", line_color=GRIS,
                  annotation_text="Desde 2019 se registra nacionalidad", annotation_position="top left")
    fig.update_yaxes(title_text="Número de personas", secondary_y=False)
    fig.update_yaxes(title_text="Personas en etapa avanzada (%)", rangemode="tozero", secondary_y=True)
    fig.update_xaxes(dtick=1, title_text="Año de la primera notificación")
    fig.update_layout(height=430, margin=dict(t=30), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cómo leerlo: cada barra muestra cuántas personas fueron notificadas por primera vez en ese año. La línea roja muestra qué porcentaje de ellas ya estaba registrado en etapa avanzada (SIDA o muerte). La franja clara muestra un rango posible alrededor de ese porcentaje: si es ancha, hay más incertidumbre; si es estrecha, la estimación es más precisa.")
    st.caption(f"En este tablero, 'etapa avanzada al notificar' significa que la primera notificación ya registraba SIDA o muerte. Es una aproximación para estudiar la presentación tardía; no permite saber cuándo se produjo la infección. SIVIGILA no contiene el recuento de células de defensa llamadas CD4, un examen que ayuda a valorar el estado del sistema inmunitario. El año {ANIO_PARCIAL} está incompleto y solo incluye registros hasta {K['fecha_max']}.")
    st.caption("Estas cifras cuentan notificaciones recibidas; no son el total de personas con VIH en Medellín ni permiten calcular qué proporción de toda la población tiene VIH, porque no se dispone aquí del total de habitantes como referencia.")

# ---------------------------------------------------------------- 2 Perfil
with tabs[1]:
    var = st.selectbox("¿Qué característica deseas revisar?", ["sexo", "grupo_edad", "aseguramiento", "mecanismo", "migrante"],
                       format_func=lambda v: {"sexo": "Sexo registrado", "grupo_edad": "Edad al notificar", "aseguramiento": "Afiliación al sistema de salud",
                                              "mecanismo": "Forma probable de transmisión", "migrante": "Nacionalidad registrada"}[v],
                       help="Selecciona una característica para comparar los grupos de personas que aparecen en los registros.")
    st.caption("Esta pestaña responde dos preguntas distintas: cuántas personas hay en cada grupo y qué porcentaje de ese grupo aparece en etapa avanzada. Por eso conviene leer juntos los dos gráficos.")
    g = agregar(F, var)
    if var == "grupo_edad":
        g["o"] = g[var].map({e: i for i, e in enumerate(ORDEN_EDAD)})
        g = g.sort_values("o")
    if var == "migrante":
        st.info("SIVIGILA empezó a registrar la nacionalidad en 2019. Antes de ese año, 'Sin dato' no significa que la persona no fuera migrante: significa que esa información no se recogía de forma comparable. Para comparar nacionalidades, selecciona 2019 en adelante.")
    c1, c2 = st.columns(2)
    f1 = px.bar(g, x=var, y="n", color_discrete_sequence=[NAVY], title="Cantidad de personas en cada grupo")
    f1.update_layout(height=380, xaxis_title="Grupo", yaxis_title="Número de personas")
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure(go.Bar(x=g[var], y=g["pct"], marker_color=RED,
                          error_y=dict(type="data", symmetric=False, array=(g["ic_sup"] - g["pct"]).fillna(0),
                                       arrayminus=(g["pct"] - g["ic_inf"]).fillna(0))))
    f2.update_layout(height=380, title="Porcentaje de cada grupo en etapa avanzada", xaxis_title="Grupo", yaxis_title="Personas en etapa avanzada (%)")
    c2.plotly_chart(f2, use_container_width=True)
    st.caption("Cómo leerlos: el gráfico de la izquierda muestra cuántas personas hay en cada grupo. El de la derecha muestra qué parte de ese mismo grupo estaba en etapa avanzada. Las pequeñas líneas sobre las barras del segundo gráfico indican el rango de incertidumbre de cada porcentaje. Un porcentaje más alto no significa necesariamente que haya más personas: revisa también el conteo del gráfico izquierdo.")
    st.caption(f"Por privacidad, el porcentaje y su rango se ocultan cuando un grupo tiene menos de {config.MIN_CELDA} personas. Esto evita que grupos muy pequeños puedan identificarse a partir de los resultados.")
    t = g[[var, "n", "n_tardio", "pct", "ic_inf", "ic_sup", "suprimido"]].rename(
        columns={var: "Grupo", "n": "Personas en el grupo", "n_tardio": "Personas en etapa avanzada", "pct": "Porcentaje en etapa avanzada", "ic_inf": "Rango posible (95 %), desde", "ic_sup": "Rango posible (95 %), hasta", "suprimido": "Porcentaje oculto por privacidad"})
    st.dataframe(t.round(1), use_container_width=True, hide_index=True)
    st.caption("En la tabla, 'Personas en el grupo' es el total de la categoría seleccionada; 'Personas en etapa avanzada' es el número que tenía SIDA o muerte en la primera notificación. El porcentaje se calcula dentro de cada grupo, no sobre toda la población del tablero. Los dos últimos valores muestran el rango de incertidumbre; cuando aparece vacío, el porcentaje se ocultó por privacidad.")

# ---------------------------------------------------------------- 3 Territorio
with tabs[2]:
    Ft = F[F["comuna"] != "Sin información"]
    sin_info = int(F.loc[F["comuna"] == "Sin información", "n"].sum())
    st.warning(f"{miles(sin_info)} personas ({sin_info / F['n'].sum() * 100:.1f} %) no tienen comuna registrada. No aparecen en las comparaciones de esta pestaña. Por eso, los resultados describen solo a quienes sí tienen una comuna identificada y podrían no representar por completo a la población incluida.")
    T = agregar(Ft, "comuna")
    T = T[T["n"] > 0].sort_values("pct", ascending=True)
    fig = go.Figure(go.Bar(y=T["comuna"], x=T["pct"], orientation="h", marker_color=TEAL,
                           error_x=dict(type="data", symmetric=False, array=(T["ic_sup"] - T["pct"]).fillna(0),
                                        arrayminus=(T["pct"] - T["ic_inf"]).fillna(0))))
    p0 = Ft["n_tardio"].sum() / max(Ft["n"].sum(), 1) * 100
    fig.add_vline(x=p0, line_dash="dash", line_color=GRIS, annotation_text=f"Medellín: {p0:.1f} %")
    fig.update_layout(height=620, title="Porcentaje en etapa avanzada según comuna", xaxis_title="Personas en etapa avanzada (%)", margin=dict(l=240))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, use_container_width=True)
    lim = limites_embudo(p0 / 100, int(max(T["n"].max() * 1.15, 50)))
    fe = go.Figure()
    for nv, col, nom in [(0.998, "#C9D6E8", "Rango poco frecuente (99,8 %)"), (0.95, "#8FA9CC", "Rango habitual (95 %) ")]:
        l = lim[lim["nivel"] == nv]
        fe.add_scatter(x=l["n"], y=l["sup"] * 100, mode="lines", line=dict(color=col), name=nom)
        fe.add_scatter(x=l["n"], y=l["inf"] * 100, mode="lines", line=dict(color=col), showlegend=False)
    fe.add_scatter(x=T["n"], y=T["pct"], mode="markers", text=T["comuna"], marker=dict(color=RED, size=9), name="Comunas")
    fe.add_hline(y=p0, line_dash="dash", line_color=GRIS)
    fe.update_layout(height=620, title="¿La diferencia de una comuna podría deberse al azar?", xaxis_title="Número de personas registradas", yaxis_title="Personas en etapa avanzada (%)")
    c2.plotly_chart(fe, use_container_width=True)
    st.caption("Gráfico de barras (izquierda): cada barra muestra qué porcentaje de las personas registradas en una comuna estaba en etapa avanzada. La línea vertical marca el porcentaje general de Medellín entre las personas con comuna conocida. Las líneas cortas muestran el rango de incertidumbre del porcentaje de cada comuna.")
    st.caption("Gráfico de puntos (derecha): cada punto representa una comuna. Cuanto más a la derecha, más personas tiene esa comuna en los registros; su altura indica el porcentaje en etapa avanzada. Las bandas azules muestran el rango en el que podrían ubicarse muchas comunas por variación natural. La banda del 95 % funciona como una primera señal para revisar; la del 99,8 % indica un resultado mucho menos habitual si las comunas no tuvieran diferencias reales. Un punto fuera de una banda merece revisión, pero no prueba por sí solo que exista un problema en esa comuna. Las comunas con pocos registros tienen bandas más amplias.")
    r = RES["regresion"]
    st.markdown(f"*Comprobación adicional:* al comparar las comunas y tener en cuenta también el año, el sexo, la edad, la afiliación y la forma probable de transmisión, el resultado fue p = {r['p_lrt']:.3f}.")
    st.caption("En palabras sencillas: con estos datos no se encontró evidencia suficiente para afirmar que las diferencias entre comunas se mantienen después de considerar esas otras características. El valor p no es la probabilidad de que una hipótesis sea verdadera ni mide qué tan grande es una diferencia. Tampoco demuestra que todas las comunas sean iguales.")
    with st.expander("Ver el detalle de la comprobación estadística"):
        st.write(f"La comparación estadística dio χ² = {r['lrt_chi2']:.1f}, con {r['lrt_gl']} grados de libertad y p = {r['p_lrt']:.3f}. Se incluyeron {miles(r['n_modelo_B'])} personas con comuna conocida. El valor χ² resume cuánto cambió el resultado al añadir la comuna a la comparación; los grados de libertad corresponden al número de diferencias entre categorías que se evaluaron. El valor p indica si una diferencia de ese tamaño sería poco habitual cuando no hay diferencias reales entre comunas, después de considerar las demás características.")
    tabla_territorio = T.sort_values("pct", ascending=False)[["comuna", "n", "n_tardio", "pct", "ic_inf", "ic_sup", "suprimido"]].rename(
        columns={"comuna": "Comuna o corregimiento", "n": "Personas registradas", "n_tardio": "Personas en etapa avanzada", "pct": "Porcentaje en etapa avanzada",
                 "ic_inf": "Rango posible (95 %), desde", "ic_sup": "Rango posible (95 %), hasta", "suprimido": "Porcentaje oculto por privacidad"})
    st.dataframe(tabla_territorio.round(1), use_container_width=True, hide_index=True)
    st.caption("La tabla permite revisar el número de personas y el porcentaje de cada comuna. El porcentaje se calcula sobre quienes sí tienen comuna registrada. El rango del 95 % expresa la incertidumbre de ese porcentaje; si queda vacío, el porcentaje se ocultó porque el grupo es pequeño.")

# ---------------------------------------------------------------- 4 Factores relacionados

def bonito(t):
    """Traduce los nombres internos de las categorías a rótulos legibles."""
    return (t.replace("anio_c", "Cambio por cada año adicional")
             .replace("grupo_edad: ", "Edad ").replace("aseguramiento: ", "Afiliación: ")
             .replace("mecanismo: ", "Forma probable de transmisión: ")
             .replace("sexo: ", "Sexo: ").replace("comuna: ", "Comuna: "))


def forest(df, etiqueta, titulo, alto=520):
    df = df.copy()
    fig = go.Figure(go.Scatter(x=df["OR"], y=df[etiqueta], mode="markers", marker=dict(size=9, color=NAVY),
                               error_x=dict(type="data", symmetric=False, array=df["IC95_sup"] - df["OR"], arrayminus=df["OR"] - df["IC95_inf"])))
    fig.add_vline(x=1, line_dash="dash", line_color=GRIS)
    fig.update_xaxes(type="log", title="Comparación relativa frente al grupo de referencia (OR)")
    fig.update_layout(height=alto, title=titulo, yaxis=dict(autorange="reversed"))
    return fig


with tabs[3]:
    A = leer("regresion_principal.csv")
    st.caption("Esta pestaña revisa si algunas características de las personas aparecen relacionadas con una mayor o menor presencia de SIDA o muerte en la primera notificación. La comparación tiene en cuenta varias características al mismo tiempo; así se evita atribuir a una sola característica diferencias que podrían estar relacionadas con otra. Una relación estadística no demuestra que esa característica sea la causa.")
    st.markdown("*¿Con qué grupos se compara?* Para cada comparación se toma como punto de referencia a los hombres, las personas de 18 a 28 años, quienes pertenecen al régimen contributivo y quienes tienen como forma probable de transmisión la heterosexual. Los demás grupos se comparan con esas referencias, teniendo en cuenta las otras características del análisis.")
    st.caption("Cómo leer el resultado: OR (odds ratio, o razón de posibilidades) es una medida que compara cuánto aparece la etapa avanzada en un grupo frente al grupo de referencia, teniendo en cuenta las demás características incluidas. OR = 1 significa que no se observa diferencia; por encima de 1 indica un valor mayor y por debajo de 1, menor. Por ejemplo, OR = 2 significa el doble de posibilidades relativas, no el doble de probabilidad ni 2 puntos porcentuales más. El rango del 95 % (intervalo de confianza) muestra la incertidumbre: si es estrecho, la estimación es más precisa; si es amplio, hay más incertidumbre. El 95 % describe el nivel de confianza del método, no la probabilidad de una persona. Si el rango incluye 1, los datos no permiten descartar que no haya diferencia.")
    anio_row = A[A["termino"] == "anio_c"].iloc[0]
    c = st.columns(3)
    c[0].metric("Comparación de un año con el siguiente", f"{anio_row['OR']:.3f}", help="Compara las posibilidades relativas de presentar SIDA o muerte entre años consecutivos, manteniendo iguales las demás características del análisis.")
    c[1].metric("Rango posible de esa comparación (95 %)", f"{anio_row['IC95_inf']:.3f} – {anio_row['IC95_sup']:.3f}", help="Muestra el rango de valores compatibles con los datos para la comparación anual. Como el rango está por debajo de 1, los resultados apuntan a una disminución con los años.")
    c[2].metric("Disminución relativa estimada por año", f"{(1 - anio_row['OR']) * 100:.1f} %", help="Es la disminución de las posibilidades relativas por año, no una reducción de igual número de puntos porcentuales en la probabilidad de una persona.")
    st.caption("La comparación anual resume la tendencia general después de tener en cuenta las demás características incluidas. No significa que la probabilidad de cada persona baje ese porcentaje cada año.")
    A2 = A[A["termino"] != "anio_c"].copy()
    A2["etiqueta"] = A2["termino"].map(bonito)
    st.plotly_chart(forest(A2, "etiqueta", "Comparación de cada grupo con su grupo de referencia"), use_container_width=True)
    st.caption("Cada punto muestra el resultado de un grupo y cada línea horizontal muestra su rango de incertidumbre. La escala horizontal se organiza alrededor del valor 1: a la derecha están los grupos con posibilidades relativas mayores que su referencia y a la izquierda, los que tienen posibilidades menores. La línea vertical en 1 significa 'sin diferencia frente al grupo de referencia'. Un punto a la derecha indica posibilidades relativas mayores; a la izquierda, menores. Cuando la línea horizontal cruza la línea vertical, no se puede descartar que no haya diferencia.")
    S = leer("sensibilidad_tendencia.csv")
    nombres_analisis = {
        "Crudo (solo año)": "Solo se compara el año",
        "Ajustado (modelo principal)": "Año y otras características consideradas",
        "Ajustado, sin 2023 (año parcial)": "Sin incluir 2023, que está incompleto",
        "Ajustado, solo notificación rutinaria + sin dato de fuente": "Notificaciones habituales, sin fuente informada",
        "Ajustado, solo notificación rutinaria": "Solo notificaciones habituales",
        "Ajustado, resultado = solo SIDA (excluye 'Muerto')": "Solo casos en etapa SIDA; no incluye fallecimientos",
        "Ajustado, solo 2012-2018": "Solo años 2012 a 2018",
        "Ajustado, solo 2019 en adelante": "Solo años 2019 en adelante",
    }
    S["analisis"] = S["analisis"].replace(nombres_analisis)
    S["OR"] = S["OR_por_anio"]
    st.plotly_chart(forest(S, "analisis", "¿La tendencia se mantiene al revisar los datos de distintas maneras?", 380), use_container_width=True)
    st.caption("Este gráfico repite la comparación anual usando distintas versiones de los datos: primero se compara únicamente el año; luego se tienen en cuenta otras características; también se excluye el año incompleto y se aplican otros criterios de selección. Si los puntos y sus rangos se parecen, la tendencia no depende de una sola forma de hacer el análisis. Esto no elimina todos los posibles errores ni los cambios que haya tenido el registro.")
    st.caption("Una advertencia importante: durante el periodo cambiaron algunas prácticas de registro y búsqueda de casos. Por eso, aunque la tendencia se mantiene en estas comprobaciones, no es posible asegurar que todo el cambio observado se deba a una transformación real de la situación de salud.")
    with st.expander("Ver las tablas y cifras estadísticas detalladas"):
        tabla_regresion = A.copy()
        tabla_regresion["termino"] = tabla_regresion["termino"].map(bonito)
        tabla_regresion["modelo"] = "Comparación que tiene en cuenta año, sexo, edad, afiliación y forma probable de transmisión"
        tabla_regresion = tabla_regresion.rename(columns={"termino": "Característica comparada", "OR": "Comparación relativa (OR)",
                                                           "IC95_inf": "Rango posible (95 %), desde", "IC95_sup": "Rango posible (95 %), hasta",
                                                           "p": "Señal estadística (valor p)", "modelo": "Características consideradas"})
        tabla_sensibilidad = S.drop(columns="OR").copy()
        tabla_sensibilidad = tabla_sensibilidad.rename(columns={"analisis": "Forma de revisar la tendencia", "OR_por_anio": "Comparación relativa por año",
                                                                    "IC95_inf": "Rango posible (95 %), desde", "IC95_sup": "Rango posible (95 %), hasta",
                                                                    "p": "Señal estadística (valor p)", "n": "Personas analizadas", "eventos": "Personas en etapa avanzada"})
        st.dataframe(tabla_regresion.round(4), use_container_width=True, hide_index=True)
        st.caption("La primera tabla muestra cuánto difiere cada grupo frente al grupo de referencia, después de considerar las otras características. En la columna OR, 1 significa que no hay diferencia; por encima de 1, el valor es mayor; por debajo de 1, es menor. El rango del 95 % indica qué tan precisa es la estimación. El valor p ayuda a evaluar si la diferencia observada podría explicarse por variación aleatoria; no mide su importancia práctica.")
        st.dataframe(tabla_sensibilidad.round(4), use_container_width=True, hide_index=True)
        st.caption("La segunda tabla comprueba si la comparación anual cambia al usar distintas selecciones de datos. El valor p es una señal estadística: cuanto más pequeño, menos compatible sería observar una diferencia así si no existiera una relación. No indica la importancia práctica del resultado ni la probabilidad de que una hipótesis sea verdadera.")

# ---------------------------------------------------------------- 5 Evaluación de una herramienta de apoyo
with tabs[4]:
    mi = RES["modelo"]
    st.error("*Límite de uso:* esta pestaña evalúa el comportamiento general de un programa con datos históricos. No muestra ni califica personas. No debe utilizarse para tomar decisiones sobre pacientes: todavía no se ha comprobado su funcionamiento en otras fuentes de datos y, en los años recientes, sus estimaciones de riesgo son más altas que lo observado.")
    st.caption("La herramienta aprende patrones a partir de registros de años anteriores y luego se revisa con registros de años posteriores que no se usaron para enseñarle. Así se comprueba si lo aprendido funciona cuando se aplica a información más nueva.")
    C = leer("modelos_comparacion.csv")
    c = st.columns(4)
    c[0].metric("Años usados para enseñar al programa", f"{mi['anios_entrenamiento']}", help=f"Periodo con {miles(mi['n_entrenamiento'])} personas, de las cuales {miles(mi['eventos_entrenamiento'])} tenían SIDA o muerte en la primera notificación. El programa usa estos registros para reconocer patrones.")
    c[1].metric("Años usados para comprobarlo", f"{mi['anios_prueba']}", help=f"Periodo posterior con {miles(mi['n_prueba'])} personas y {miles(mi['eventos_prueba'])} casos en etapa avanzada. Estos años no se usaron para enseñar al programa.")
    c[2].metric("Casos por cada relación que aprende el programa", f"{mi['eventos_por_parametro']:.0f}", help="Indica cuántos casos en etapa avanzada hubo, en promedio, por cada peso o relación que el programa tuvo que calcular a partir de los datos. Ayuda a revisar si había suficientes ejemplos; no mide si el programa acierta.")
    c[3].metric("Personas en etapa avanzada: enseñanza → comprobación", f"{mi['prevalencia_entrenamiento'] * 100:.1f} % → {mi['prevalencia_prueba'] * 100:.1f} %", help="Muestra qué porcentaje estaba en etapa avanzada en cada periodo. La diferencia entre los periodos importa porque el programa aprendió con una frecuencia mayor a la que encontró después.")
    st.caption("En los años usados para enseñar al programa, el porcentaje de personas en etapa avanzada era mayor que en los años usados para comprobarlo. Este cambio en los datos puede hacer que el programa calcule riesgos demasiado altos cuando se usa en años posteriores.")
    nombres_metodos = {
        "Prevalencia (sin modelo)": "Referencia: promedio general",
        "Regla: edad ≥ 40 años": "Regla sencilla: edad de 40 años o más",
        "Regresión logística: solo edad": "Método estadístico: solo edad",
        "Regresión logística (modelo primario)": "Método estadístico: varias características",
        "Gradient boosting (comparador)": "Programa automatizado de comparación",
    }
    C_grafico = C.copy()
    C_grafico["modelo"] = C_grafico["modelo"].replace(nombres_metodos)
    f = go.Figure()
    f.add_bar(x=C_grafico["modelo"], y=C_grafico["auc"], name="Capacidad para distinguir", marker_color=NAVY,
              error_y=dict(type="data", symmetric=False, array=C["auc_sup"] - C["auc"], arrayminus=C["auc"] - C["auc_inf"]))
    f.add_hline(y=0.5, line_dash="dash", line_color=GRIS, annotation_text="Resultado similar al azar")
    f.update_layout(height=380, title="Capacidad del programa para distinguir entre grupos", yaxis_title="Resultado (0,5 = azar; 1 = separación perfecta)",
                    yaxis_range=[0.4, 0.85], xaxis_title="Método comparado")
    st.plotly_chart(f, use_container_width=True)
    st.caption("Este resultado se llama AUC. Expresa qué tan bien ordena el programa a las personas: una persona que estaba en etapa avanzada debería recibir una puntuación mayor que una persona que no lo estaba. El AUC resume con qué frecuencia logra ordenar correctamente esos dos casos. 0,5 equivale a hacerlo como al azar; 1 significa separar perfectamente ambos grupos. Un resultado de 0,724 no significa que el programa acierte el 72,4 % de las veces ni que sus porcentajes de riesgo sean exactos. Las líneas sobre las barras muestran el rango de incertidumbre del 95 %.")
    cols = ["modelo", "auc", "auc_inf", "auc_sup", "pr_auc", "lift_top20", "recall_top20", "brier", "cal_pendiente", "cal_intercepto"]
    tabla_modelos = C[cols].copy()
    tabla_modelos["modelo"] = tabla_modelos["modelo"].replace(nombres_metodos)
    tabla_modelos["recall_top20"] = tabla_modelos["recall_top20"] * 100
    tabla_modelos = tabla_modelos.rename(columns={"modelo": "Método comparado", "auc": "Capacidad para distinguir (AUC)",
                                              "auc_inf": "AUC: rango 95 %, desde", "auc_sup": "AUC: rango 95 %, hasta",
                                              "pr_auc": "Detección de casos menos frecuentes (PR-AUC)",
                                              "lift_top20": "Concentración de casos en el 20 % de mayor puntuación (Lift)",
                                              "recall_top20": "Casos avanzados encontrados dentro de ese 20 % (%)",
                                              "brier": "Error de las estimaciones de riesgo (Brier)",
                                              "cal_pendiente": "Ajuste de las diferencias entre riesgos (pendiente)",
                                              "cal_intercepto": "Desviación general de los riesgos calculados (intercepto)"})
    st.dataframe(tabla_modelos.round(3), use_container_width=True, hide_index=True)
    st.caption("Cómo leer la tabla: AUC resume qué tan bien el programa separa a quienes estaban y no estaban en etapa avanzada; 0,5 equivale al azar y 1 sería una separación perfecta. PR-AUC resume qué tan bien encuentra los casos de etapa avanzada, algo especialmente importante cuando son pocos. 'Concentración en el 20 %' compara el grupo al que el programa asigna las puntuaciones más altas con el promedio general: un valor de 2 significa que allí se concentra el doble de casos que en el promedio. 'Casos encontrados' indica qué parte de todos los casos avanzados quedó dentro de ese 20 %. El error Brier resume qué tan lejos quedan las estimaciones de lo que ocurrió: cuanto más cerca de 0, mejor. En la columna de casos encontrados, por ejemplo, 48,6 significa que el programa ubicó cerca de 49 de cada 100 casos avanzados dentro del grupo del 20 % con puntuación más alta. En los dos indicadores que revisan si los porcentajes calculados se parecen a los observados, el valor ideal de la pendiente es cercano a 1: significa que los cambios en la estimación siguen de cerca los cambios reales. La desviación general ideal es cercana a 0: significa que, en promedio, el programa no calcula porcentajes sistemáticamente más altos o más bajos.")
    st.caption("La tabla compara una regla sencilla basada en la edad, un método estadístico que combina varias características, otro programa automatizado y una referencia que asigna a todas las personas el promedio general. En el método estadístico, el programa calcula el peso de cada característica para producir una estimación; el programa automatizado aprende combinaciones de patrones en los datos. Los valores permiten revisar sus diferencias; no significan que alguno esté listo para usarse en la atención de personas.")
    st.markdown(f"*Diferencia frente al método que solo usa la edad:* {mi['delta_auc']:+.3f} (rango de incertidumbre del 95 %: {mi['delta_auc_inf']:+.3f} a {mi['delta_auc_sup']:+.3f}).")
    st.caption("La diferencia es positiva: el método principal distingue algo mejor que el que usa únicamente la edad. Sin embargo, la mejora es moderada y no convierte al programa en una herramienta lista para tomar decisiones individuales. El rango mostrado expresa la incertidumbre de esa diferencia.")
    c1, c2 = st.columns(2)
    cal = leer("calibracion_modelo.csv")
    fc = go.Figure()
    fc.add_scatter(x=[0, 0.4], y=[0, 0.4], mode="lines", line=dict(dash="dash", color=GRIS), name="Estimación igual a lo observado")
    fc.add_scatter(x=cal["pred_media"], y=cal["obs"], mode="lines+markers", marker_color=RED, name="Grupos de puntuación similar")
    fc.update_layout(height=380, title="¿Los riesgos calculados se parecen a lo que ocurrió?", xaxis_title="Riesgo estimado por el programa (%)", yaxis_title="Porcentaje que realmente estuvo en etapa avanzada (%)")
    fc.update_xaxes(tickformat=".0%")
    fc.update_yaxes(tickformat=".0%")
    c1.plotly_chart(fc, use_container_width=True)
    pred_media = C.loc[C["modelo"] == "Regresión logística (modelo primario)", "prediccion_media"]
    pred_media = float(pred_media.iloc[0]) if not pred_media.empty else float("nan")
    st.caption(f"Este gráfico revisa si los porcentajes que calcula el programa se parecen a lo que después ocurrió. A esta revisión se le llama calibración. Cada punto reúne personas con puntuaciones similares. La línea diagonal representa el resultado ideal: si el programa estima 10 %, debería observarse cerca de 10 % de personas en etapa avanzada. Los puntos por debajo indican que el programa calculó un riesgo mayor al que se observó. En esta prueba estimó, en promedio, {pred_media * 100:.1f} %, pero se observó {mi['prevalencia_prueba'] * 100:.1f} %. Por eso, sus estimaciones son demasiado altas para este periodo.")
    der = leer("deriva_temporal.csv")
    fd = go.Figure()
    fd.add_scatter(x=der["anio"], y=der["observado"] * 100, mode="lines+markers", name="Lo que ocurrió", line_color=RED)
    fd.add_scatter(x=der["anio"], y=der["predicho_medio"] * 100, mode="lines+markers", name="Promedio calculado por el programa", line_color=NAVY)
    fd.add_vline(x=config.CORTE_ENTRENAMIENTO + 0.5, line_dash="dot", line_color=GRIS,
                 annotation_text="Aquí empieza el periodo de comprobación")
    fd.update_layout(height=380, title="Cambios a lo largo de los años: resultado real y estimación", yaxis_title="Personas en etapa avanzada (%)", xaxis_title="Año")
    c2.plotly_chart(fd, use_container_width=True)
    st.caption("La línea roja muestra el porcentaje que realmente se observó cada año. La azul muestra el porcentaje promedio que calculó el programa. Cuanto más separadas estén, menos se parece la estimación a lo que ocurrió. La línea vertical marca el cambio entre los años usados para enseñar al programa y los años usados para comprobarlo.")
    imp = leer("importancia_variables.csv")
    imp = imp.copy()
    imp["variable"] = imp["variable"].replace({"edad_anios": "Edad", "aseguramiento": "Afiliación al sistema de salud", "mecanismo": "Forma probable de transmisión"})
    st.plotly_chart(px.bar(imp, x="caida_auc", y="variable", orientation="h", color_discrete_sequence=[TEAL],
                           title="Qué información utiliza más el programa").update_layout(height=320, yaxis=dict(autorange="reversed"),
                                                                                         xaxis_title="Caída en la capacidad para distinguir", yaxis_title="Información utilizada"),
                    use_container_width=True)
    st.caption("Para elaborar este gráfico se desordena cada dato por separado y se revisa cuánto empeora la capacidad del programa para distinguir entre grupos. Si la barra es más larga, esa información le resulta más útil al programa para ordenar los casos. Esto no significa que esa característica cause la presentación avanzada ni que sea más importante desde el punto de vista de salud pública.")

# ---------------------------------------------------------------- 6 Calidad de la información
with tabs[5]:
    st.caption("Esta pestaña permite revisar si los datos necesarios están registrados y cuánto tiempo pasa entre la consulta y la notificación a SIVIGILA. Una información incompleta o tardía puede dificultar la vigilancia y cambiar la interpretación de los resultados.")
    comp = leer("calidad_completitud.csv").pivot(index="variable", columns="anio", values="pct_completo")
    comp.index.name = "Dato revisado"
    comp.columns.name = "Año"
    fh = px.imshow(comp, text_auto=".0f", color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=100,
                   title="Porcentaje de registros que tienen cada dato, por año")
    fh.update_layout(height=400, xaxis_title="Año", yaxis_title="Dato revisado", coloraxis_colorbar_title="Porcentaje con dato")
    st.plotly_chart(fh, use_container_width=True)
    st.caption("Cada fila corresponde a un dato que se revisó y cada columna a un año. El número dentro de la casilla indica qué porcentaje de los registros de ese año tiene ese dato diligenciado. 100 % significa que está presente en todos; 70 % significa que falta en aproximadamente 3 de cada 10 registros. Los tonos más oscuros indican que hay más información registrada; los más claros, que falta más información.")
    op = leer("oportunidad_anual.csv")
    fo = make_subplots(specs=[[{"secondary_y": True}]])
    fo.add_scatter(x=op["anio"], y=op["mediana_dias"], name="Retraso del caso intermedio (días)", line_color=NAVY, secondary_y=False)
    fo.add_scatter(x=op["anio"], y=op["p90_dias"], name="Retraso de 9 de cada 10 casos (días)", line_color=RED, secondary_y=False)
    fo.add_scatter(x=op["anio"], y=op["pct_hasta_7d"], name="Notificados en 7 días o menos (%)", line=dict(color=TEAL, dash="dot"), secondary_y=True)
    fo.update_layout(height=380, title="Tiempo entre la consulta y la notificación", xaxis_title="Año")
    fo.update_yaxes(title_text="Días transcurridos", secondary_y=False)
    fo.update_yaxes(title_text="Notificados en 7 días o menos (%)", secondary_y=True)
    st.plotly_chart(fo, use_container_width=True)
    st.caption("La primera línea indica el retraso del caso que queda en la mitad de los registros: la mitad se notificó en ese número de días o menos y la otra mitad tardó igual o más. La segunda línea muestra el tiempo dentro del cual se notificaron 9 de cada 10 casos; por eso permite observar los retrasos más largos. La línea punteada muestra qué porcentaje se notificó en los primeros 7 días. Menos días y un porcentaje mayor dentro de 7 días indican una notificación más rápida.")
    cal = RES["calidad"]
    st.subheader("Revisiones encontradas en los registros")
    st.caption("Las cifras siguientes no son indicadores de la situación del VIH en la ciudad; son revisiones de la información que llegó al sistema. Cada frase indica si cuenta personas o registros.")
    st.markdown(f"- *Notificaciones repetidas:* {miles(cal['personas_con_renotificacion'])} personas ({K['pct_personas_renotificadas']:.1f} %) aparecen más de una vez. Para el análisis se conserva la primera notificación de cada persona.\n"
                f"- *Comuna sin informar:* {K['pct_sin_comuna']:.1f} % de las personas incluidas no tiene comuna de residencia identificada.\n"
                f"- *Fechas o edades que requieren revisión:* {miles(cal['retrasos_resultado_notif_negativos'])} registros tienen una fecha de resultado posterior a la fecha de notificación; {cal['fechas_nacimiento_invalidas_antes_1910']} registro tiene una fecha de nacimiento inválida y {cal['edades_fuera_de_rango_0_105']} muestran una edad fuera del rango revisado.\n"
                f"- *Año epidemiológico diferente al año de la fecha:* {cal['anio_epi_distinto_del_anio_calendario']} registros tienen un año de vigilancia distinto al año calendario de la fecha de notificación. El año epidemiológico es la forma en que el sistema agrupa las notificaciones por periodos de vigilancia, que no siempre coinciden exactamente con enero a diciembre. Esta diferencia debe tenerse en cuenta al comparar años.\n"
                f"- *Infecciones asociadas registradas:* {K['pct_con_oportunista_sida']:.1f} % de las personas registradas en etapa SIDA tiene al menos una infección oportunista anotada, frente a {K['pct_con_oportunista_vih']:.1f} % de las registradas en etapa VIH.\n"
                f"- *Hospitalización registrada:* {K['pct_hospitalizados_sida']:.1f} % de las personas en etapa SIDA aparece hospitalizada, frente a {K['pct_hospitalizados_vih']:.1f} % de las personas en etapa VIH.")
    st.caption("Una infección oportunista es una infección que puede aparecer con mayor facilidad cuando las defensas del cuerpo están debilitadas. Las dos últimas comparaciones sirven para revisar si los registros de personas en etapa SIDA contienen con más frecuencia señales clínicas relacionadas con una enfermedad avanzada. No demuestran que una condición cause la otra y no tienen en cuenta todas las diferencias entre las personas. La fecha de resultado posterior a la notificación se señala porque, en la secuencia esperada, el resultado de la prueba debería estar disponible antes de enviar la notificación.")

# ---------------------------------------------------------------- 7 Preguntas y alcance del estudio
with tabs[6]:
    st.subheader("¿Qué buscó responder este estudio?")
    st.markdown("Se revisó cómo han cambiado, entre 2012 y 2023, los registros de personas que ya aparecían en etapa SIDA o fallecidas cuando fueron notificadas por primera vez en Medellín. También se estudió si esa situación variaba según características de las personas y su comuna de residencia.")
    st.markdown("Además, se examinó si la información disponible al momento de la notificación permitía construir un programa que reconociera patrones relacionados con la presentación avanzada. La evaluación se hizo con años posteriores a los usados para enseñar al programa.")
    H = leer("hipotesis.csv")
    st.subheader("Preguntas revisadas y resultados")
    st.caption("Cada apartado presenta una pregunta que se quiso comprobar, el resultado encontrado y lo que significa para la lectura del tablero. 'Los resultados respaldan la pregunta' indica que se encontró la señal esperada; 'no se encontró evidencia suficiente' significa que los datos no permitieron confirmarla. No encontrar evidencia no demuestra que lo contrario sea cierto.")
    explicaciones = {
        "H1": ("¿Ha disminuido con los años el porcentaje de personas que ya estaban en etapa SIDA o habían fallecido al ser notificadas?",
               "Los resultados muestran una disminución en la comparación anual, incluso al repetir el análisis con distintas selecciones de datos. La medida que compara un año con el anterior fue 0,862 (rango del 95 %: 0,847 a 0,878; el valor 1 representaría ausencia de cambio). Esto equivale a una disminución aproximada del 13,8 % por año en esa medida. No significa una reducción de 13,8 puntos porcentuales en el porcentaje de personas ni demuestra por sí sola por qué ocurrió el cambio.",
               "Los resultados respaldan esta pregunta."),
        "H2": ("¿Hay diferencias entre comunas que no se expliquen solo por la cantidad de personas registradas o por sus características?",
               "Después de tener en cuenta el año, el sexo, la edad, la afiliación y la forma probable de transmisión, no se encontró evidencia suficiente para afirmar que las diferencias entre comunas fueran mayores que las esperadas por variación natural (p = 0,253). El valor p ayuda a saber si una diferencia como esta sería poco habitual cuando no hay diferencias reales entre comunas; no es la probabilidad de que la hipótesis sea verdadera. Esto no demuestra que todas las comunas sean iguales.",
               "No se encontró evidencia suficiente para confirmar esta pregunta."),
        "H3": ("¿La presentación avanzada aparece con mayor frecuencia en algunos grupos de edad o de afiliación?",
               "Al comparar personas con características similares en las demás variables, la medida de presentación avanzada fue 3,3 veces la del grupo de 18 a 28 años para quienes tenían entre 40 y 49 años; 3,8 veces para quienes tenían entre 50 y 59; y 4,3 veces para quienes tenían 60 años o más. Frente al régimen contributivo, fue 2,1 veces en el régimen subsidiado y 2,5 veces entre personas sin afiliación. Son comparaciones entre grupos, no multiplicadores exactos de la probabilidad de cada persona. Estas relaciones no demuestran que la edad o la afiliación sean la causa.",
               "Los resultados respaldan esta pregunta."),
        "H4": ("¿El programa reconoce mejor los casos de presentación avanzada que una regla sencilla basada solo en la edad?",
               "El programa principal distinguió mejor entre personas con y sin etapa avanzada que la regla basada únicamente en la edad. Su resultado fue 0,724 en la medida de capacidad para distinguir (AUC: 0,5 equivale al azar y 1 a una separación perfecta); la diferencia frente a la regla de solo edad fue +0,042. Sin embargo, sus estimaciones de riesgo fueron mayores que lo observado: calculó en promedio 13,8 % frente a 6,2 % reales en los años de comprobación. Por tanto, reconoce cierto patrón, pero sus porcentajes no representan bien lo que ocurre y no debe usarse para decidir sobre personas.",
               "La capacidad para distinguir fue mejor; los porcentajes de riesgo no quedaron bien ajustados."),
    }
    for _, h in H.iterrows():
        pregunta, lectura, estado = explicaciones.get(h["id"], (h["enunciado"], h["resultado"], h["veredicto"]))
        if h["id"] in ("H1", "H3"):
            st.success(f"*{h['id']} · {estado}*")
        elif h["id"] == "H4":
            st.warning(f"*{h['id']} · {estado}*")
        else:
            st.info(f"*{h['id']} · {estado}*")
        with st.expander(f"{h['id']}: leer la pregunta y su resultado"):
            st.markdown(f"*Pregunta:* {pregunta}")
            st.write(lectura)
            st.caption("Los cálculos detallados se conservan en los archivos de resultados del proyecto; aquí se muestra su significado en lenguaje sencillo.")
    st.subheader("Qué aspectos limitan la interpretación")
    st.markdown("- *Los registros no son todos los casos de la ciudad.* El tablero cuenta las notificaciones que llegaron al sistema y cumplen los criterios del estudio. Como no incluye un total de habitantes con el que comparar, no permite calcular qué proporción de toda la población de Medellín tiene VIH.\n"
                "- *La etapa avanzada es una aproximación.* Se considera que una persona estaba en etapa avanzada cuando su primera notificación ya indicaba SIDA o muerte. No se cuenta con resultados de CD4 ni con la fecha de infección para precisar cuándo ocurrió el contagio o establecer por completo si hubo diagnóstico tardío.\n"
                "- *La forma de registrar datos cambió.* Por ejemplo, la nacionalidad empezó a registrarse en 2019. Por eso, algunas diferencias entre años pueden deberse a cambios en el registro y no solo a cambios en la situación de salud.\n"
                f"- *Falta la comuna en {K['pct_sin_comuna']:.1f} % de las personas.* Las comparaciones territoriales incluyen únicamente a quienes sí tienen comuna identificada.\n"
                "- *Algunas categorías de transmisión se agruparon.* Los códigos disponibles requieren confirmación con el documento oficial que explica qué significa cada código, para asegurar que cada categoría esté interpretada correctamente.\n"
                "- *El programa todavía no es una herramienta para la atención.* En los años de comprobación asignó porcentajes de riesgo mayores a los observados y no se ha probado con otras fuentes de información o en otros territorios.")
    st.caption(f"Fuente de los registros: SIVIGILA, evento 850 (código usado para agrupar las notificaciones de VIH, SIDA y mortalidad por SIDA). Periodo disponible en este análisis: {K['fecha_min']} a {K['fecha_max']}. Los resultados se presentan agrupados; el tablero no muestra fichas individuales.")
