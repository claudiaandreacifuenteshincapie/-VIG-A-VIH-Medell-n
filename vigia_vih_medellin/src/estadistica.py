"""Utilidades estadísticas sin dependencias extra (numpy/scipy/pandas)."""
import numpy as np
import pandas as pd
from scipy import stats


def wilson(k, n, z=1.96):
    """Intervalo de Wilson para una proporción (vectorizado). Devuelve (lo, hi) en proporción."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z ** 2 / n
        centro = (p + z ** 2 / (2 * n)) / den
        mitad = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    lo, hi = centro - mitad, centro + mitad
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def logit_fit(X, y, max_iter=100, tol=1e-9, ridge=1e-10):
    """Regresión logística por IRLS (Newton-Raphson). Devuelve beta, se, loglik, convergencia."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]
    beta = np.zeros(k)
    convergio = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(k)
        paso = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + paso
        if np.max(np.abs(paso)) < tol:
            convergio = True
            break
    eta = np.clip(X @ beta, -35, 35)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    w = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(k))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "loglik": ll, "convergio": convergio}


def construir_X(df, numericas=(), categoricas=None):
    """Matriz de diseño con intercepto. categoricas: {columna: nivel_de_referencia}."""
    X = pd.DataFrame({"Intercepto": np.ones(len(df))}, index=df.index)
    for c in numericas:
        X[c] = df[c].astype(float)
    for c, ref in (categoricas or {}).items():
        for v in sorted(df[c].dropna().unique()):
            if v != ref:
                X[f"{c}: {v}"] = (df[c] == v).astype(float)
    return X.loc[:, (X != 0).any(axis=0)]


def tabla_or(ajuste, nombres, z=1.96):
    b, se = ajuste["beta"], ajuste["se"]
    t = pd.DataFrame({"termino": nombres, "OR": np.exp(b), "IC95_inf": np.exp(b - z * se),
                      "IC95_sup": np.exp(b + z * se), "p": 2 * stats.norm.sf(np.abs(b / se))})
    return t[t["termino"] != "Intercepto"].reset_index(drop=True)


def or_por_anio(df, resultado="tardio", ajustar=True, anio_base=2012):
    """OR por cada año calendario. Devuelve dict con OR, IC95, p, n y eventos."""
    d = df.dropna(subset=[resultado, "anio"]).copy()
    d["anio_c"] = d["anio"].astype(float) - anio_base
    if ajustar:
        d = d.dropna(subset=["grupo_edad", "sexo"])
        X = construir_X(d, ["anio_c"], {"sexo": "Hombre", "grupo_edad": "18-28",
                                        "aseguramiento": "Contributivo", "mecanismo": "Heterosexual"})
    else:
        X = construir_X(d, ["anio_c"])
    f = logit_fit(X.values, d[resultado].astype(float).values)
    i = list(X.columns).index("anio_c")
    b, se = f["beta"][i], f["se"][i]
    return {"OR_por_anio": float(np.exp(b)), "IC95_inf": float(np.exp(b - 1.96 * se)),
            "IC95_sup": float(np.exp(b + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(b / se))),
            "n": int(len(d)), "eventos": int(d[resultado].sum())}


def limites_embudo(p0, n_max=1500, nivel=(0.95, 0.998)):
    """Límites de control de un gráfico de embudo (aprox. normal) alrededor de la proporción global p0."""
    n = np.arange(10, n_max + 1)
    filas = []
    for nv in nivel:
        z = stats.norm.ppf(0.5 + nv / 2)
        h = z * np.sqrt(p0 * (1 - p0) / n)
        filas.append(pd.DataFrame({"n": n, "nivel": nv, "inf": np.clip(p0 - h, 0, 1), "sup": np.clip(p0 + h, 0, 1)}))
    return pd.concat(filas, ignore_index=True)


def suprimir(n, minimo):
    return np.asarray(n) < minimo
