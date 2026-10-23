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
    st.error("Aún no hay resultados. Copie `sivigila_vih.csv` en `data/landing/` y ejecute `python run_pipeline.py`.")
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
st.markdown(f"""<div class="hdr"><h1>🩺 {config.NOMBRE}</h1><p>{config.SUBTITULO}</p>
<span class="badge">Datos reales SIVIGILA</span><span class="badge">{K['fecha_min']} → {K['fecha_max']}</span>
<span class="badge">{miles(K['poblacion_analisis'])} personas (1.ª notificación, residentes en Medellín)</span></div>""",
            unsafe_allow_html=True)

st.sidebar.header("Filtros (pestañas 1–3)")
a_min, a_max = int(CUBO["anio"].min()), int(CUBO["anio"].max())
rango = st.sidebar.slider("Años", a_min, a_max, (a_min, a_max))
incluir_parcial = st.sidebar.checkbox(f"Incluir {ANIO_PARCIAL} (año parcial, hasta {K['fecha_max']})", value=True)
sexo_sel = st.sidebar.multiselect("Sexo", sorted(CUBO["sexo"].unique()), default=sorted(CUBO["sexo"].unique()))
edad_op = [e for e in ORDEN_EDAD if e in set(CUBO["grupo_edad"])]
edad_sel = st.sidebar.multiselect("Grupo de edad", edad_op, default=edad_op)
aseg_op = sorted(CUBO["aseguramiento"].unique())
aseg_sel = st.sidebar.multiselect("Aseguramiento", aseg_op, default=aseg_op)
com_op = sorted(CUBO["comuna"].unique())
com_sel = st.sidebar.multiselect("Comuna / corregimiento", com_op, default=com_op)
st.sidebar.caption(f"Porcentajes de estratos con n < {config.MIN_CELDA} se ocultan (protección de confidencialidad). "
                   "Las pestañas 4–7 usan toda la población: no responden a estos filtros.")

m = (CUBO["anio"].between(*rango) & CUBO["sexo"].isin(sexo_sel) & CUBO["grupo_edad"].isin(edad_sel)
     & CUBO["aseguramiento"].isin(aseg_sel) & CUBO["comuna"].isin(com_sel))
if not incluir_parcial:
    m &= CUBO["anio"] != ANIO_PARCIAL
F = CUBO[m]
if F.empty:
    st.warning("Los filtros no dejan ningún caso.")
    st.stop()

tabs = st.tabs(["1 · Resumen", "2 · Perfil", "3 · Territorio", "4 · Determinantes",
                "5 · Modelo predictivo", "6 · Calidad y oportunidad", "7 · Proyecto e hipótesis"])

# ---------------------------------------------------------------- 1 Resumen
with tabs[0]:
    tot = agregar(F.assign(todo="t"), "todo").iloc[0]
    sx = F.groupby("sexo")["n"].sum()
    sin_com = F.loc[F["comuna"] == "Sin información", "n"].sum()
    c = st.columns(4)
    c[0].metric("Personas notificadas (filtro)", miles(tot['n']))
    c[1].metric("Presentación avanzada (SIDA o muerto)", fmt(tot["pct"], 1, " %"),
                help=f"IC95 {fmt(tot['ic_inf'])}–{fmt(tot['ic_sup'])} %")
    c[2].metric("Mujeres", fmt(sx.get("Mujer", 0) / tot["n"] * 100, 1, " %"))
    c[3].metric("Sin comuna registrada", fmt(sin_com / tot["n"] * 100, 1, " %"))

    por_anio = agregar(F, "anio")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=por_anio["anio"], y=por_anio["n"], name="Personas notificadas", marker_color="#C9D6E8", secondary_y=False)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["ic_sup"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", secondary_y=True)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["ic_inf"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(176,58,46,.18)", name="IC95 %", hoverinfo="skip", secondary_y=True)
    fig.add_scatter(x=por_anio["anio"], y=por_anio["pct"], mode="lines+markers", name="% presentación avanzada",
                    line=dict(color=RED, width=3), secondary_y=True)
    fig.add_vline(x=2018.5, line_dash="dot", line_color=GRIS,
                  annotation_text="desde 2019 se registra nacionalidad", annotation_position="top left")
    fig.update_yaxes(title_text="Personas", secondary_y=False)
    fig.update_yaxes(title_text="% presentación avanzada", rangemode="tozero", secondary_y=True)
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=430, margin=dict(t=30), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<p class='nota'>Presentación avanzada = estadio clínico SIDA o muerto en la primera notificación (aproximación al diagnóstico tardío: "
                f"SIVIGILA no registra CD4). {ANIO_PARCIAL} es un año parcial. Las notificaciones no equivalen a incidencia: "
                f"no se calculan tasas porque el proyecto no incluye denominadores poblacionales.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 2 Perfil
with tabs[1]:
    var = st.selectbox("Variable", ["sexo", "grupo_edad", "aseguramiento", "mecanismo", "migrante"],
                       format_func=lambda v: {"sexo": "Sexo", "grupo_edad": "Grupo de edad", "aseguramiento": "Aseguramiento",
                                              "mecanismo": "Mecanismo probable de transmisión", "migrante": "Nacionalidad (migrante)"}[v])
    g = agregar(F, var)
    if var == "grupo_edad":
        g["o"] = g[var].map({e: i for i, e in enumerate(ORDEN_EDAD)})
        g = g.sort_values("o")
    if var == "migrante":
        st.info("La nacionalidad solo se registra desde 2019: use el filtro de años (2019 →) para comparar de forma válida; "
                "'Sin dato' corresponde casi por completo a 2012–2018.")
    c1, c2 = st.columns(2)
    f1 = px.bar(g, x=var, y="n", color_discrete_sequence=[NAVY], title="Personas notificadas")
    f1.update_layout(height=380, xaxis_title=None)
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure(go.Bar(x=g[var], y=g["pct"], marker_color=RED,
                          error_y=dict(type="data", symmetric=False, array=(g["ic_sup"] - g["pct"]).fillna(0),
                                       arrayminus=(g["pct"] - g["ic_inf"]).fillna(0))))
    f2.update_layout(height=380, title="% presentación avanzada (IC95)", xaxis_title=None, yaxis_title="%")
    c2.plotly_chart(f2, use_container_width=True)
    t = g[[var, "n", "n_tardio", "pct", "ic_inf", "ic_sup", "suprimido"]].rename(
        columns={var: "Categoría", "n": "n", "n_tardio": "Avanzados", "pct": "%", "ic_inf": "IC95 inf", "ic_sup": "IC95 sup", "suprimido": "Oculto (n pequeño)"})
    st.dataframe(t.round(1), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 3 Territorio
with tabs[2]:
    Ft = F[F["comuna"] != "Sin información"]
    sin_info = int(F.loc[F["comuna"] == "Sin información", "n"].sum())
    st.warning(f"{miles(sin_info)} personas ({sin_info / F['n'].sum() * 100:.1f} %) no tienen comuna registrada y se excluyen de esta pestaña. "
               "Si esa omisión no es aleatoria, el ranking puede sesgarse.")
    T = agregar(Ft, "comuna")
    T = T[T["n"] > 0].sort_values("pct", ascending=True)
    fig = go.Figure(go.Bar(y=T["comuna"], x=T["pct"], orientation="h", marker_color=TEAL,
                           error_x=dict(type="data", symmetric=False, array=(T["ic_sup"] - T["pct"]).fillna(0),
                                        arrayminus=(T["pct"] - T["ic_inf"]).fillna(0))))
    p0 = Ft["n_tardio"].sum() / max(Ft["n"].sum(), 1) * 100
    fig.add_vline(x=p0, line_dash="dash", line_color=GRIS, annotation_text=f"Medellín {p0:.1f} %")
    fig.update_layout(height=620, title="% presentación avanzada por comuna (IC95 de Wilson)", xaxis_title="%", margin=dict(l=240))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, use_container_width=True)
    lim = limites_embudo(p0 / 100, int(max(T["n"].max() * 1.15, 50)))
    fe = go.Figure()
    for nv, col, nom in [(0.998, "#C9D6E8", "límite 99,8 %"), (0.95, "#8FA9CC", "límite 95 %")]:
        l = lim[lim["nivel"] == nv]
        fe.add_scatter(x=l["n"], y=l["sup"] * 100, mode="lines", line=dict(color=col), name=nom)
        fe.add_scatter(x=l["n"], y=l["inf"] * 100, mode="lines", line=dict(color=col), showlegend=False)
    fe.add_scatter(x=T["n"], y=T["pct"], mode="markers", text=T["comuna"], marker=dict(color=RED, size=9), name="Comunas")
    fe.add_hline(y=p0, line_dash="dash", line_color=GRIS)
    fe.update_layout(height=620, title="Gráfico de embudo: ¿variación mayor que el azar?", xaxis_title="Personas notificadas (n)", yaxis_title="%")
    c2.plotly_chart(fe, use_container_width=True)
    r = RES["regresion"]
    st.markdown(f"**Prueba formal (H2, toda la población):** razón de verosimilitud del bloque comuna, ajustado por año, sexo, edad, "
                f"aseguramiento y mecanismo: χ²({r['lrt_gl']}) = {r['lrt_chi2']:.1f}, **p = {r['p_lrt']:.3f}**. "
                "Con 20 comunas es esperable que ~1 caiga fuera del límite del 95 % solo por azar.")
    st.dataframe(T.sort_values("pct", ascending=False)[["comuna", "n", "n_tardio", "pct", "ic_inf", "ic_sup", "suprimido"]].round(1),
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 4 Determinantes
def bonito(t):
    return (t.replace("grupo_edad: ", "Edad ").replace("aseguramiento: ", "").replace("mecanismo: ", "Mecanismo: ")
             .replace("sexo: ", "").replace("comuna: ", ""))


def forest(df, etiqueta, titulo, alto=520):
    df = df.copy()
    fig = go.Figure(go.Scatter(x=df["OR"], y=df[etiqueta], mode="markers", marker=dict(size=9, color=NAVY),
                               error_x=dict(type="data", symmetric=False, array=df["IC95_sup"] - df["OR"], arrayminus=df["OR"] - df["IC95_inf"])))
    fig.add_vline(x=1, line_dash="dash", line_color=GRIS)
    fig.update_xaxes(type="log", title="Odds ratio (escala log)")
    fig.update_layout(height=alto, title=titulo, yaxis=dict(autorange="reversed"))
    return fig


with tabs[3]:
    A = leer("regresion_principal.csv")
    st.markdown("**Modelo A** (regresión logística, toda la población): año + sexo + edad + aseguramiento + mecanismo. "
                "Referencias: hombre, 18–28 años, contributivo, heterosexual. Asociaciones, no causalidad.")
    anio_row = A[A["termino"] == "anio_c"].iloc[0]
    c = st.columns(3)
    c[0].metric("OR por cada año calendario", f"{anio_row['OR']:.3f}", help="OR < 1: la presentación avanzada disminuye cada año")
    c[1].metric("IC95", f"{anio_row['IC95_inf']:.3f} – {anio_row['IC95_sup']:.3f}")
    c[2].metric("Reducción relativa anual de las odds", f"{(1 - anio_row['OR']) * 100:.1f} %")
    A2 = A[A["termino"] != "anio_c"].copy()
    A2["etiqueta"] = A2["termino"].map(bonito)
    st.plotly_chart(forest(A2, "etiqueta", "Odds ratio ajustados (IC95)"), use_container_width=True)
    S = leer("sensibilidad_tendencia.csv")
    S["OR"] = S["OR_por_anio"]
    st.plotly_chart(forest(S, "analisis", "Robustez de la tendencia anual (OR por año)", 380), use_container_width=True)
    st.markdown("<p class='nota'>El descenso persiste en todos los análisis de sensibilidad, pero su magnitud no puede separarse de cambios en el "
                "registro (p. ej., búsqueda activa comunitaria casi sin casos avanzados; nacionalidad registrada desde 2019).</p>", unsafe_allow_html=True)
    with st.expander("Tablas"):
        st.dataframe(A.round(4), use_container_width=True, hide_index=True)
        st.dataframe(S.drop(columns="OR").round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 5 Modelo
with tabs[4]:
    mi = RES["modelo"]
    st.error("**Uso permitido:** evaluación poblacional del desempeño de un modelo. Este tablero NO puntúa ni muestra personas, y el modelo "
             "no debe usarse para decisiones individuales: ni está validado externamente ni calibrado para años recientes.")
    C = leer("modelos_comparacion.csv")
    c = st.columns(4)
    c[0].metric("Entrenamiento", f"{mi['anios_entrenamiento']}", help=f"{miles(mi['n_entrenamiento'])} personas · {miles(mi['eventos_entrenamiento'])} eventos")
    c[1].metric("Prueba temporal", f"{mi['anios_prueba']}", help=f"{miles(mi['n_prueba'])} personas · {miles(mi['eventos_prueba'])} eventos")
    c[2].metric("Eventos por parámetro (EPV)", f"{mi['eventos_por_parametro']:.0f}", help="Regla práctica: ≥ 10–20")
    c[3].metric("Prevalencia entrenamiento → prueba", f"{mi['prevalencia_entrenamiento'] * 100:.1f} % → {mi['prevalencia_prueba'] * 100:.1f} %")
    f = go.Figure()
    f.add_bar(x=C["modelo"], y=C["auc"], name="AUC", marker_color=NAVY,
              error_y=dict(type="data", symmetric=False, array=C["auc_sup"] - C["auc"], arrayminus=C["auc"] - C["auc_inf"]))
    f.add_hline(y=0.5, line_dash="dash", line_color=GRIS, annotation_text="azar")
    f.update_layout(height=380, title="AUC en la prueba temporal (IC95 bootstrap)", yaxis_range=[0.4, 0.85], xaxis_title=None)
    st.plotly_chart(f, use_container_width=True)
    cols = ["modelo", "auc", "auc_inf", "auc_sup", "pr_auc", "lift_top20", "recall_top20", "brier", "cal_pendiente", "cal_intercepto"]
    st.dataframe(C[cols].round(3), use_container_width=True, hide_index=True)
    st.markdown(f"ΔAUC modelo primario vs. solo-edad: **{mi['delta_auc']:+.3f}** (IC95 {mi['delta_auc_inf']:+.3f} a {mi['delta_auc_sup']:+.3f}). "
                f"El gradient boosting no mejora a la regresión logística.")
    c1, c2 = st.columns(2)
    cal = leer("calibracion_modelo.csv")
    fc = go.Figure()
    fc.add_scatter(x=[0, 0.4], y=[0, 0.4], mode="lines", line=dict(dash="dash", color=GRIS), name="calibración perfecta")
    fc.add_scatter(x=cal["pred_media"], y=cal["obs"], mode="lines+markers", marker_color=RED, name="deciles de riesgo")
    fc.update_layout(height=380, title="Calibración (prueba): el modelo sobrestima", xaxis_title="Riesgo predicho", yaxis_title="Observado")
    c1.plotly_chart(fc, use_container_width=True)
    der = leer("deriva_temporal.csv")
    fd = go.Figure()
    fd.add_scatter(x=der["anio"], y=der["observado"] * 100, mode="lines+markers", name="Observado", line_color=RED)
    fd.add_scatter(x=der["anio"], y=der["predicho_medio"] * 100, mode="lines+markers", name="Predicho (modelo)", line_color=NAVY)
    fd.add_vline(x=config.CORTE_ENTRENAMIENTO + 0.5, line_dash="dot", line_color=GRIS)
    fd.update_layout(height=380, title="Deriva temporal: la prevalencia cae y el modelo no la sigue", yaxis_title="%")
    c2.plotly_chart(fd, use_container_width=True)
    imp = leer("importancia_variables.csv")
    st.plotly_chart(px.bar(imp, x="caida_auc", y="variable", orientation="h", color_discrete_sequence=[TEAL],
                           title="Importancia por permutación (caída de AUC en prueba)").update_layout(height=320, yaxis=dict(autorange="reversed")),
                    use_container_width=True)

# ---------------------------------------------------------------- 6 Calidad
with tabs[5]:
    comp = leer("calidad_completitud.csv").pivot(index="variable", columns="anio", values="pct_completo")
    fh = px.imshow(comp, text_auto=".0f", color_continuous_scale="Blues", aspect="auto", zmin=0, zmax=100,
                   title="Completitud por variable y año (%)")
    fh.update_layout(height=400)
    st.plotly_chart(fh, use_container_width=True)
    op = leer("oportunidad_anual.csv")
    fo = make_subplots(specs=[[{"secondary_y": True}]])
    fo.add_scatter(x=op["anio"], y=op["mediana_dias"], name="Mediana (días)", line_color=NAVY, secondary_y=False)
    fo.add_scatter(x=op["anio"], y=op["p90_dias"], name="Percentil 90 (días)", line_color=RED, secondary_y=False)
    fo.add_scatter(x=op["anio"], y=op["pct_hasta_7d"], name="% notificado ≤ 7 días", line=dict(color=TEAL, dash="dot"), secondary_y=True)
    fo.update_layout(height=380, title="Oportunidad: días entre la consulta y la notificación")
    fo.update_yaxes(title_text="Días", secondary_y=False)
    fo.update_yaxes(title_text="%", secondary_y=True)
    st.plotly_chart(fo, use_container_width=True)
    cal = RES["calidad"]
    st.markdown("**Hallazgos de calidad del dato**")
    st.markdown(f"- {miles(cal['personas_con_renotificacion'])} personas ({K['pct_personas_renotificadas']:.1f} %) tienen más de una notificación; el análisis usa la primera.\n"
                f"- {K['pct_sin_comuna']:.1f} % de la población de análisis no tiene comuna.\n"
                f"- {miles(cal['retrasos_resultado_notif_negativos'])} registros con fecha de resultado posterior a la notificación; "
                f"{cal['fechas_nacimiento_invalidas_antes_1910']} fecha de nacimiento inválida; {cal['edades_fuera_de_rango_0_105']} edades fuera de rango.\n"
                f"- {cal['anio_epi_distinto_del_anio_calendario']} registros cuyo año epidemiológico difiere del año calendario de la fecha de notificación.\n"
                f"- Validez de constructo del desenlace: {K['pct_con_oportunista_sida']:.0f} % de los casos en estadio SIDA tiene ≥ 1 infección oportunista "
                f"registrada frente a {K['pct_con_oportunista_vih']:.1f} % en estadio VIH; hospitalización {K['pct_hospitalizados_sida']:.0f} % vs. {K['pct_hospitalizados_vih']:.0f} %.")

# ---------------------------------------------------------------- 7 Proyecto
with tabs[6]:
    st.subheader("Pregunta y objetivos")
    st.markdown("**Pregunta.** ¿Cómo ha evolucionado entre 2012 y 2023 la presentación avanzada del VIH (SIDA o muerte) al momento de la notificación en "
                "Medellín, y qué factores sociodemográficos y territoriales se asocian con ella?")
    st.markdown("**Objetivo general.** Caracterizar la tendencia, los determinantes y la distribución territorial de la presentación avanzada del VIH en Medellín "
                "con datos reales de SIVIGILA, y evaluar con validación temporal si variables disponibles al notificar permiten priorizar a quienes tienen mayor riesgo.")
    H = leer("hipotesis.csv")
    st.subheader("Hipótesis, reglas de decisión y veredictos")
    for _, h in H.iterrows():
        ok = h["veredicto"].startswith("Se sostiene") and "no en" not in h["veredicto"]
        (st.success if ok else st.warning if "parcial" in h["veredicto"] or "no en" in h["veredicto"] else st.error)(
            f"**{h['id']} — {h['veredicto']}.** {h['enunciado']}")
        with st.expander(f"{h['id']}: regla de decisión y resultado"):
            st.markdown(f"**Regla:** {h['regla']}\n\n**Resultado:** {h['resultado']}")
    st.subheader("Paralelo con el estudio de Nigeria (por tareas, no por problemas)")
    st.dataframe(pd.DataFrame([
        ["Medir un desenlace en datos reales de rutina", "IIT (>28 días sin dispensación): 33,7 %", "Presentación avanzada al notificar: medida con SIVIGILA"],
        ["Evaluar calidad e integridad de los datos", "EMR incompleto e inconsistente (limitación declarada)", "Completitud, duplicados y consistencia (pestaña 6)"],
        ["Comparar modelos de ML con una referencia simple", "LR, AdaBoost, RF; partición por paciente", "LR, boosting y líneas base; partición TEMPORAL"],
        ["Predecir la interrupción del tratamiento (IIT)", "Posible por tener fechas de dispensación", "NO abordado: SIVIGILA no tiene dispensación"]],
        columns=["Tarea", "Nigeria (Ijaiya et al., 2025)", "VIGÍA-VIH Medellín"]), use_container_width=True, hide_index=True)
    st.subheader("Limitaciones")
    st.markdown("- Notificación ≠ incidencia ni prevalencia; sin denominadores no hay tasas.\n"
                "- El estadio SIDA/muerto es una aproximación al diagnóstico tardío (sin CD4 ni fecha de infección).\n"
                "- Cambios de registro en el tiempo (nacionalidad desde 2019; fuente de notificación) pueden mover la tendencia.\n"
                f"- {K['pct_sin_comuna']:.0f} % sin comuna: el análisis territorial es de casos completos.\n"
                "- Los códigos de mecanismo 5–13 se agrupan; los 1–4 se verificaron empíricamente y deben confirmarse con el diccionario oficial del INS.\n"
                "- El modelo está descalibrado en los años recientes y no ha sido validado externamente.")
    st.markdown(f"<p class='nota'>Fuente: SIVIGILA, evento 850 (VIH/SIDA/mortalidad por SIDA), registros seudonimizados. Corte de datos: {K['fecha_max']}.</p>", unsafe_allow_html=True)
