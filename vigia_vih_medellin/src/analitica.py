"""Capa Gold descriptiva e inferencial: cubo, series, territorio, regresiones e hipótesis H1-H3."""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config
from src import estadistica as es

DIMS_CUBO = ["anio", "comuna", "sexo", "grupo_edad", "aseguramiento", "mecanismo", "migrante"]
REF_MODELO = {"sexo": "Hombre", "grupo_edad": "18-28", "aseguramiento": "Contributivo", "mecanismo": "Heterosexual"}


def _guardar(df, nombre):
    config.DIR_GOLD.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DIR_GOLD / nombre, index=False, encoding="utf-8")


def cubo_casos(pob):
    d = pob.copy()
    d["grupo_edad"] = d["grupo_edad"].fillna("Sin dato")
    d["sexo"] = d["sexo"].fillna("Sin dato")
    cubo = (d.groupby(DIMS_CUBO, observed=True, dropna=False)
              .agg(n=("tardio", "size"), n_tardio=("tardio", "sum")).reset_index())
    cubo["n_tardio"] = cubo["n_tardio"].astype(int)
    _guardar(cubo, "cubo_casos.csv")
    return cubo


def serie_anual(pob):
    g = pob.groupby("anio").agg(n=("tardio", "size"), n_tardio=("tardio", "sum"),
                                n_mujeres=("sexo", lambda s: (s == "Mujer").sum()),
                                n_migrantes=("migrante", lambda s: (s == "Sí").sum()),
                                n_sin_comuna=("comuna_conocida", lambda s: (~s).sum())).reset_index()
    g["n_tardio"] = g["n_tardio"].astype(int)
    g["pct_tardio"] = g["n_tardio"] / g["n"] * 100
    lo, hi = es.wilson(g["n_tardio"], g["n"])
    g["ic_inf"], g["ic_sup"] = lo * 100, hi * 100
    g["pct_mujeres"] = g["n_mujeres"] / g["n"] * 100
    g["pct_sin_comuna"] = g["n_sin_comuna"] / g["n"] * 100
    g["anio_parcial"] = g["anio"] == pob["anio"].max()
    _guardar(g, "serie_anual.csv")
    return g


def territorio(pob):
    k = pob[pob["comuna_conocida"]]
    t = k.groupby("comuna").agg(n=("tardio", "size"), n_tardio=("tardio", "sum")).reset_index()
    t["n_tardio"] = t["n_tardio"].astype(int)
    t["pct_tardio"] = t["n_tardio"] / t["n"] * 100
    lo, hi = es.wilson(t["n_tardio"], t["n"])
    t["ic_inf"], t["ic_sup"] = lo * 100, hi * 100
    p0 = t["n_tardio"].sum() / t["n"].sum()
    t["z_embudo"] = (t["n_tardio"] / t["n"] - p0) / np.sqrt(p0 * (1 - p0) / t["n"])
    t["fuera_embudo_95"] = t["z_embudo"].abs() > 1.96
    t["fuera_embudo_998"] = t["z_embudo"].abs() > 3.09
    t["suprimido"] = es.suprimir(t["n"], config.MIN_CELDA)
    t = t.sort_values("pct_tardio", ascending=False)
    _guardar(t, "territorio_comuna.csv")
    _guardar(es.limites_embudo(p0, int(t["n"].max() * 1.15)), "funnel_limites.csv")
    grande = t[t["n"] >= 30]
    chi2, p_chi, gl, _ = stats.chi2_contingency(
        np.array([grande["n_tardio"], grande["n"] - grande["n_tardio"]]))
    return t, {"p0_pct": p0 * 100, "chi2": float(chi2), "gl": int(gl), "p_chi2": float(p_chi)}


def perfil(pob):
    filas = []
    for var in ["sexo", "grupo_edad", "aseguramiento", "mecanismo", "zona", "migrante", "fuente"]:
        g = pob.groupby(var, dropna=False).agg(n=("tardio", "size"), n_tardio=("tardio", "sum")).reset_index()
        g = g.rename(columns={var: "categoria"})
        g.insert(0, "variable", var)
        filas.append(g)
    p = pd.concat(filas, ignore_index=True)
    p["n_tardio"] = p["n_tardio"].astype(int)
    p["pct_tardio"] = p["n_tardio"] / p["n"] * 100
    lo, hi = es.wilson(p["n_tardio"], p["n"])
    p["ic_inf"], p["ic_sup"] = lo * 100, hi * 100
    _guardar(p, "perfil_tardio.csv")
    return p


def regresiones(pob):
    d = pob.dropna(subset=["grupo_edad", "sexo"]).copy()
    d["anio_c"] = d["anio"].astype(float) - config.ANIO_BASE
    y = d["tardio"].astype(float).values
    X = es.construir_X(d, ["anio_c"], {**REF_MODELO})
    f = es.logit_fit(X.values, y)
    A = es.tabla_or(f, list(X.columns))
    A["modelo"] = "A: año + sexo + edad + aseguramiento + mecanismo"
    _guardar(A, "regresion_principal.csv")

    # Modelo B: + comuna (casos con comuna conocida) y prueba de razón de verosimilitud del bloque
    k = d[d["comuna_conocida"]].copy()
    k_ref = k["comuna"].value_counts().idxmax()
    yk = k["tardio"].astype(float).values
    Xa = es.construir_X(k, ["anio_c"], {**REF_MODELO})
    Xb = es.construir_X(k, ["anio_c"], {**REF_MODELO, "comuna": k_ref})
    fa, fb = es.logit_fit(Xa.values, yk), es.logit_fit(Xb.values, yk)
    gl = Xb.shape[1] - Xa.shape[1]
    lrt = 2 * (fb["loglik"] - fa["loglik"])
    p_lrt = float(stats.chi2.sf(lrt, gl))
    B = es.tabla_or(fb, list(Xb.columns))
    B = B[B["termino"].str.startswith("comuna")].copy()
    B["modelo"] = f"B: modelo A + comuna (ref: {k_ref})"
    _guardar(B, "regresion_comuna.csv")

    # Sensibilidad de la tendencia anual (H1)
    def fila(nombre, df, **kw):
        r = es.or_por_anio(df, anio_base=config.ANIO_BASE, **kw)
        r["analisis"] = nombre
        return r
    anio_parcial = int(d["anio"].max())
    sens = [
        fila("Crudo (solo año)", pob, ajustar=False),
        fila("Ajustado (modelo principal)", pob),
        fila(f"Ajustado, sin {anio_parcial} (año parcial)", pob[pob["anio"] < anio_parcial]),
        fila("Ajustado, solo notificación rutinaria + sin dato de fuente",
             pob[pob["fuente"].isin(["Notificación rutinaria", "Sin dato"])]),
        fila("Ajustado, solo notificación rutinaria", pob[pob["fuente"] == "Notificación rutinaria"]),
        fila("Ajustado, resultado = solo SIDA (excluye 'Muerto')",
             pob[pob["estadio"].isin(["VIH", "SIDA"])].assign(tardio=lambda x: (x["estadio"] == "SIDA"))),
        fila("Ajustado, solo 2012-2018", pob[pob["anio"] <= 2018]),
        fila("Ajustado, solo 2019 en adelante", pob[pob["anio"] >= 2019]),
    ]
    S = pd.DataFrame(sens)[["analisis", "OR_por_anio", "IC95_inf", "IC95_sup", "p", "n", "eventos"]]
    _guardar(S, "sensibilidad_tendencia.csv")
    return A, B, S, {"lrt_chi2": float(lrt), "lrt_gl": int(gl), "p_lrt": p_lrt, "comuna_ref": k_ref, "n_modelo_B": int(len(k))}


def calidad_completitud(pob):
    d = pob.copy()
    comp = pd.DataFrame({
        "Comuna de residencia": d["comuna_conocida"],
        "Nacionalidad": d["migrante"] != "Sin dato",
        "Fuente de notificación": d["fuente"] != "Sin dato",
        "Aseguramiento determinado": d["aseguramiento"] != "Indeterminado",
        "Mecanismo de transmisión": ~d["mecanismo"].eq("Otro/sin dato"),
        "Edad válida": d["edad_anios"].notna(),
    })
    comp["anio"] = d["anio"].values
    largo = comp.groupby("anio").mean().mul(100).reset_index().melt("anio", var_name="variable", value_name="pct_completo")
    _guardar(largo, "calidad_completitud.csv")
    return largo


def oportunidad(pob):
    def agr(s):
        s = s.dropna()
        return pd.Series({"n_validos": len(s), "mediana_dias": s.median(),
                          "p90_dias": s.quantile(0.9), "pct_hasta_7d": (s <= 7).mean() * 100})
    o = pob.groupby("anio")["dias_consulta_notif"].apply(agr).unstack().reset_index()
    _guardar(o, "oportunidad_anual.csv")
    return o


def kpis(silver, pob, cal, serie, terr_info):
    prim = serie[~serie["anio_parcial"]]
    ult, pri = prim.iloc[-1], prim.iloc[0]
    renot = silver[silver["n_notificaciones"] > 1].drop_duplicates("id_persona")
    prog = (renot["estadio_max_persona"] > renot.groupby("id_persona")["estadio_max_persona"].transform("min")).mean() \
        if len(renot) else np.nan
    k = {
        "notificaciones": cal["filas_crudas"], "personas": cal["personas_unicas"],
        "poblacion_analisis": cal["poblacion_analisis"],
        "pct_tardio_global": float(pob["tardio"].astype(float).mean() * 100),
        "pct_tardio_primer_anio": float(pri["pct_tardio"]), "primer_anio": int(pri["anio"]),
        "pct_tardio_ultimo_anio_completo": float(ult["pct_tardio"]), "ultimo_anio_completo": int(ult["anio"]),
        "pct_mujeres": float((pob["sexo"] == "Mujer").mean() * 100),
        "mediana_edad": float(pob["edad_anios"].median()),
        "pct_sin_comuna": cal["pct_comuna_sin_informacion"],
        "fecha_min": cal["fecha_min_notificacion"], "fecha_max": cal["fecha_max_notificacion"],
        "anio_parcial": cal["anio_parcial"],
        "pct_personas_renotificadas": float(cal["personas_con_renotificacion"] / cal["personas_unicas"] * 100),
        "pct_hospitalizados_vih": float(pob.loc[pob["estadio"] == "VIH", "hospitalizado"].astype(float).mean() * 100),
        "pct_hospitalizados_sida": float(pob.loc[pob["estadio"] == "SIDA", "hospitalizado"].astype(float).mean() * 100),
        "pct_con_oportunista_sida": float((pob.loc[pob["estadio"] == "SIDA", "n_oportunistas"] > 0).mean() * 100),
        "pct_con_oportunista_vih": float((pob.loc[pob["estadio"] == "VIH", "n_oportunistas"] > 0).mean() * 100),
        "mediana_dias_consulta_notif": float(pob["dias_consulta_notif"].median()),
    }
    (config.DIR_GOLD / "kpis.json").write_text(json.dumps(k, ensure_ascii=False, indent=2), encoding="utf-8")
    return k


def hipotesis_descriptivas(A, B, S, info_reg, terr_info):
    """Evalúa H1-H3 con reglas de decisión fijadas ANTES de ver los resultados (ver documento)."""
    filas = []
    # H1: tendencia decreciente robusta
    principal = S[S["analisis"].str.startswith("Ajustado (modelo")].iloc[0]
    reglas = S[S["analisis"].str.startswith(("Ajustado (modelo", "Crudo", "Ajustado, sin ", "Ajustado, solo notificación rutinaria", "Ajustado, resultado = solo SIDA"))
               & ~S["analisis"].str.contains("sin dato de fuente")]
    ok = bool((reglas["IC95_sup"] < 1).all())
    per = S[S["analisis"].str.contains("solo 2012-2018|solo 2019")]
    filas.append({
        "id": "H1", "enunciado": "La proporción de presentación avanzada (SIDA o muerto) disminuye entre 2012 y 2023.",
        "regla": "OR anual < 1 con IC95 superior < 1 en el modelo principal y en las 4 sensibilidades (crudo, sin año parcial, solo notificación rutinaria, solo SIDA).",
        "resultado": f"OR/año = {principal['OR_por_anio']:.3f} (IC95 {principal['IC95_inf']:.3f}–{principal['IC95_sup']:.3f}); "
                     + "; ".join(f"{r['analisis'].replace('Ajustado, ', '')}: {r['OR_por_anio']:.3f}" for _, r in per.iterrows()),
        "veredicto": "Se sostiene" if ok else "No se sostiene"})
    # H2: heterogeneidad territorial ajustada
    p = info_reg["p_lrt"]
    filas.append({
        "id": "H2", "enunciado": "La presentación avanzada varía entre comunas más de lo esperable por azar, aun ajustando por año, sexo, edad, aseguramiento y mecanismo.",
        "regla": "Prueba de razón de verosimilitud del bloque 'comuna' con p < 0,01 (modelo B vs. A, casos con comuna conocida).",
        "resultado": f"LRT χ²({info_reg['lrt_gl']}) = {info_reg['lrt_chi2']:.1f}; p = {p:.4f}. Descriptivo: χ²({terr_info['gl']}) = {terr_info['chi2']:.1f}, p = {terr_info['p_chi2']:.4f}.",
        "veredicto": "Se sostiene" if p < 0.01 else "No se sostiene"})
    # H3: determinantes
    def orr(term):
        r = A[A["termino"] == term]
        return r.iloc[0] if len(r) else None
    chequeos = []
    for term, texto in [("grupo_edad: 40-49", "40-49 años"), ("grupo_edad: 50-59", "50-59 años"), ("grupo_edad: 60+", "60+ años"),
                        ("aseguramiento: Subsidiado", "subsidiado"), ("aseguramiento: No asegurado", "no asegurado")]:
        r = orr(term)
        if r is not None:
            chequeos.append((texto, r["OR"], r["IC95_inf"], r["IC95_sup"], r["IC95_inf"] > 1))
    ok3 = all(c[4] for c in chequeos)
    filas.append({
        "id": "H3", "enunciado": "La edad ≥ 40 años y no estar en régimen contributivo se asocian con mayor probabilidad de presentación avanzada.",
        "regla": "OR ajustado > 1 con IC95 inferior > 1 para 40-49, 50-59, 60+, subsidiado y no asegurado (ref.: 18-28 años y contributivo).",
        "resultado": "; ".join(f"{t}: OR {o:.2f} ({lo:.2f}–{hi:.2f})" for t, o, lo, hi, _ in chequeos),
        "veredicto": "Se sostiene" if ok3 else ("Se sostiene parcialmente" if any(c[4] for c in chequeos) else "No se sostiene")})
    return filas
