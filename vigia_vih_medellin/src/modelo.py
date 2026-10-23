"""Modelo predictivo de presentación avanzada con validación TEMPORAL y comparación con líneas base.

Diseño (fijado antes de mirar el desempeño en prueba):
  * Entrenamiento: notificaciones con año <= CORTE_ENTRENAMIENTO. Prueba: años posteriores (se evalúa UNA vez).
  * Predictores = SOLO lo disponible al notificar y no definitorio del desenlace:
    edad, sexo, aseguramiento, mecanismo, zona, comuna.  EXCLUIDOS por circularidad/fuga: hospitalización,
    infecciones oportunistas, condición final, año (para no aprender la deriva).
  * Modelo primario preespecificado: regresión logística L2 (C elegido por CV 5 pliegues SOLO en entrenamiento).
    Comparadores: gradient boosting, LR solo-edad, regla edad>=40, prevalencia.
  * IC95 por bootstrap sobre la muestra de prueba; diferencias con bootstrap pareado.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from src import estadistica as es

CAT = ["sexo", "aseguramiento", "mecanismo", "zona", "comuna"]


class Preparador:
    def fit(self, df):
        self.mediana = float(df["edad_anios"].median())
        e = df["edad_anios"].fillna(self.mediana)
        self.mu, self.sd = float(e.mean()), float(e.std())
        self.columnas = self._x(df).columns
        return self

    def _x(self, df):
        d = df[["edad_anios"] + CAT].copy()
        z = (d["edad_anios"].fillna(self.mediana) - self.mu) / self.sd
        d = d.drop(columns="edad_anios")
        d[CAT] = d[CAT].fillna("Sin dato")
        X = pd.get_dummies(d, columns=CAT, dtype=float)
        X.insert(0, "edad_z", z.values)
        X.insert(1, "edad_z2", (z ** 2).values)
        return X

    def transform(self, df):
        return self._x(df).reindex(columns=self.columnas, fill_value=0.0)


def _metricas_punto(y, p):
    k = int(np.ceil(0.2 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    prev = y.mean()
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "recall_top20": y[top].sum() / y.sum(), "precision_top20": y[top].mean(),
            "lift_top20": y[top].mean() / prev}


def calibracion_logit(y, p):
    """Pendiente e intercepto de calibración: logit(y) ~ a + b*logit(p)."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc))
    X = np.column_stack([np.ones(len(lp)), lp])
    f = es.logit_fit(X, y)
    return {"intercepto": float(f["beta"][0]), "pendiente": float(f["beta"][1]),
            "pend_ic_inf": float(f["beta"][1] - 1.96 * f["se"][1]), "pend_ic_sup": float(f["beta"][1] + 1.96 * f["se"][1])}


def entrenar_y_evaluar(pob):
    rng = np.random.default_rng(config.SEED)
    d = pob.dropna(subset=["tardio"]).copy()
    d["y"] = d["tardio"].astype(int)
    tr, te = d[d["anio"] <= config.CORTE_ENTRENAMIENTO], d[d["anio"] > config.CORTE_ENTRENAMIENTO]
    prep = Preparador().fit(tr)
    Xtr, Xte = prep.transform(tr), prep.transform(te)
    ytr, yte = tr["y"].values, te["y"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)

    cols_edad = ["edad_z", "edad_z2"]
    lr_c = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1, 10]}, scoring="average_precision", cv=cv).fit(Xtr, ytr)
    lr_e = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [1.0]}, scoring="average_precision", cv=cv).fit(Xtr[cols_edad], ytr)
    hgb = GridSearchCV(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=config.SEED),
                       {"max_iter": [100, 250], "min_samples_leaf": [20, 100]}, scoring="average_precision", cv=cv).fit(Xtr, ytr)

    preds = {
        "Prevalencia (sin modelo)": np.full(len(yte), ytr.mean()),
        "Regla: edad ≥ 40 años": (te["edad_anios"].fillna(0).values >= 40).astype(float),
        "Regresión logística: solo edad": lr_e.predict_proba(Xte[cols_edad])[:, 1],
        "Regresión logística (modelo primario)": lr_c.predict_proba(Xte)[:, 1],
        "Gradient boosting (comparador)": hgb.predict_proba(Xte)[:, 1],
    }
    # La prevalencia constante no discrimina: se le añade ruido de desempate mínimo para evitar empates en AUC
    preds["Prevalencia (sin modelo)"] = preds["Prevalencia (sin modelo)"] + rng.normal(0, 1e-9, len(yte))

    puntual = {k: _metricas_punto(yte, p) for k, p in preds.items()}
    # Bootstrap pareado
    B, n = config.N_BOOT, len(yte)
    nombres = list(preds)
    acum = {k: {"auc": [], "pr_auc": []} for k in nombres}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = yte[idx]
        if yb.min() == yb.max():
            continue
        for k in nombres:
            acum[k]["auc"].append(roc_auc_score(yb, preds[k][idx]))
            acum[k]["pr_auc"].append(average_precision_score(yb, preds[k][idx]))
    prim, base = "Regresión logística (modelo primario)", "Regresión logística: solo edad"
    d_auc = np.array(acum[prim]["auc"]) - np.array(acum[base]["auc"])
    d_ap = np.array(acum[prim]["pr_auc"]) - np.array(acum[base]["pr_auc"])

    filas = []
    for k in nombres:
        c = calibracion_logit(yte, np.clip(preds[k], 1e-6, 1 - 1e-6)) if k not in ("Prevalencia (sin modelo)", "Regla: edad ≥ 40 años") else {}
        filas.append({"modelo": k, **puntual[k],
                      "auc_inf": np.percentile(acum[k]["auc"], 2.5), "auc_sup": np.percentile(acum[k]["auc"], 97.5),
                      "pr_auc_inf": np.percentile(acum[k]["pr_auc"], 2.5), "pr_auc_sup": np.percentile(acum[k]["pr_auc"], 97.5),
                      "cal_intercepto": c.get("intercepto", np.nan), "cal_pendiente": c.get("pendiente", np.nan),
                      "cal_pend_inf": c.get("pend_ic_inf", np.nan), "cal_pend_sup": c.get("pend_ic_sup", np.nan),
                      "prevalencia_prueba": yte.mean(), "prediccion_media": float(np.mean(preds[k])) if "Regla" not in k else np.nan})
    comp = pd.DataFrame(filas)
    comp.to_csv(config.DIR_GOLD / "modelos_comparacion.csv", index=False)

    # Calibración por deciles del modelo primario
    p = preds[prim]
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"decil": dec, "p": p, "y": yte}).groupby("decil").agg(pred_media=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    cal.to_csv(config.DIR_GOLD / "calibracion_modelo.csv", index=False)

    # Deriva temporal: observado vs. predicho por año (entrenamiento + prueba)
    todo = d.copy()
    todo["pred"] = lr_c.predict_proba(prep.transform(todo))[:, 1]
    der = todo.groupby("anio").agg(observado=("y", "mean"), predicho_medio=("pred", "mean"), n=("y", "size")).reset_index()
    der["conjunto"] = np.where(der["anio"] <= config.CORTE_ENTRENAMIENTO, "entrenamiento", "prueba")
    der.to_csv(config.DIR_GOLD / "deriva_temporal.csv", index=False)

    # Importancia por permutación (caída de AUC) del modelo primario, por variable original
    base_auc = roc_auc_score(yte, preds[prim])
    imp = []
    for v in ["edad_anios"] + CAT:
        caidas = []
        for _ in range(15):
            t2 = te.copy()
            t2[v] = rng.permutation(t2[v].values)
            caidas.append(base_auc - roc_auc_score(yte, lr_c.predict_proba(prep.transform(t2))[:, 1]))
        imp.append({"variable": v, "caida_auc": float(np.mean(caidas)), "sd": float(np.std(caidas))})
    pd.DataFrame(imp).sort_values("caida_auc", ascending=False).to_csv(config.DIR_GOLD / "importancia_variables.csv", index=False)

    n_par = Xtr.shape[1] + 1
    info = {"n_entrenamiento": int(len(tr)), "eventos_entrenamiento": int(ytr.sum()), "n_prueba": int(len(te)),
            "eventos_prueba": int(yte.sum()), "n_parametros": int(n_par), "eventos_por_parametro": float(ytr.sum() / n_par),
            "prevalencia_entrenamiento": float(ytr.mean()), "prevalencia_prueba": float(yte.mean()),
            "C_elegido": float(lr_c.best_params_["C"]), "hgb_params": hgb.best_params_,
            "delta_auc": float(np.mean(d_auc)), "delta_auc_inf": float(np.percentile(d_auc, 2.5)), "delta_auc_sup": float(np.percentile(d_auc, 97.5)),
            "delta_pr_auc": float(np.mean(d_ap)), "delta_pr_inf": float(np.percentile(d_ap, 2.5)), "delta_pr_sup": float(np.percentile(d_ap, 97.5)),
            "anios_entrenamiento": f"{int(tr['anio'].min())}-{int(tr['anio'].max())}", "anios_prueba": f"{int(te['anio'].min())}-{int(te['anio'].max())}"}
    return comp, info


def hipotesis_modelo(comp, info):
    r = comp[comp["modelo"] == "Regresión logística (modelo primario)"].iloc[0]
    a = r["auc_inf"] > 0.60
    b = info["delta_auc_inf"] > 0
    c = (0.8 <= r["cal_pendiente"] <= 1.2) and (r["cal_pend_inf"] <= 1.2) and (r["cal_pend_sup"] >= 0.8)
    dd = r["lift_top20"] >= 1.5
    # (e) NO estaba en la regla original: se añadió tras ver que el intercepto delataba descalibración global.
    e = abs(r["cal_intercepto"]) <= 0.3
    if all([a, b, c, dd]) and e:
        veredicto = "Se sostiene"
    elif all([a, b, c, dd]):
        veredicto = "Se sostiene en discriminación; no en calibración global (hallazgo post hoc)"
    else:
        veredicto = "Se sostiene parcialmente" if any([a, b, c, dd]) else "No se sostiene"
    return {
        "id": "H4",
        "enunciado": "Con variables disponibles al notificar, un modelo discrimina la presentación avanzada mejor que una línea base de solo edad, en validación temporal.",
        "regla": "(a) límite inferior IC95 del AUC > 0,60; (b) ΔAUC vs. solo-edad con IC95 > 0; (c) pendiente de calibración entre 0,8 y 1,2; (d) lift en el 20 % de mayor riesgo ≥ 1,5. [(e) |intercepto de calibración| ≤ 0,3: criterio añadido POST HOC, se reporta por transparencia.]",
        "resultado": (f"(a) AUC {r['auc']:.3f} (IC95 {r['auc_inf']:.3f}–{r['auc_sup']:.3f}): {'cumple' if a else 'no cumple'}. "
                      f"(b) ΔAUC {info['delta_auc']:+.3f} (IC95 {info['delta_auc_inf']:+.3f} a {info['delta_auc_sup']:+.3f}): {'cumple' if b else 'no cumple'}. "
                      f"(c) pendiente {r['cal_pendiente']:.2f} (IC95 {r['cal_pend_inf']:.2f}–{r['cal_pend_sup']:.2f}): {'cumple' if c else 'no cumple'}. "
                      f"(d) lift@20 % = {r['lift_top20']:.2f}: {'cumple' if dd else 'no cumple'}. "
                      f"(e, post hoc) intercepto {r['cal_intercepto']:+.2f}: {'cumple' if e else 'no cumple'}; "
                      f"prevalencia predicha {r['prediccion_media']*100:.1f} % vs. observada {r['prevalencia_prueba']*100:.1f} % en prueba."),
        "veredicto": veredicto}
