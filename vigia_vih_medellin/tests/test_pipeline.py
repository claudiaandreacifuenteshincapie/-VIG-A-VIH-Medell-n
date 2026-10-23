"""Pruebas mínimas del pipeline.  Ejecutar:  python -m pytest -q   (o  python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import estadistica as es  # noqa: E402
from src import etl  # noqa: E402


def test_reparar_texto():
    assert etl.reparar_texto("MedellÃ­n") == "Medellín"
    assert etl.reparar_texto("ESPAÃ‘A") == "ESPAÑA"
    assert etl.reparar_texto("Medellín") == "Medellín"      # ya correcto: no se daña
    assert etl.reparar_texto(np.nan) is np.nan or pd.isna(etl.reparar_texto(np.nan))


def test_parse_fecha_incluye_serial_excel_y_vacios():
    s = pd.Series(["29/01/2022", "43850", "-   -", None])
    f = etl.parse_fecha(s)
    assert f.iloc[0] == pd.Timestamp("2022-01-29")
    assert f.iloc[1] == pd.Timestamp("2020-01-20")           # serial Excel 43850
    assert pd.isna(f.iloc[2]) and pd.isna(f.iloc[3])


def test_wilson_valores_conocidos():
    lo, hi = es.wilson(10, 100)
    assert abs(lo - 0.0552) < 1e-3 and abs(hi - 0.1744) < 1e-3
    lo0, hi0 = es.wilson(0, 0)
    assert np.isnan(lo0) and np.isnan(hi0)


def test_logit_irls_coincide_con_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.8 * X[:, 0] - 0.5 * X[:, 1]))))
    f = es.logit_fit(np.column_stack([np.ones(len(X)), X]), y)
    ref = LogisticRegression(C=1e10, max_iter=5000).fit(X, y)
    assert f["convergio"]
    assert np.allclose(f["beta"][1:], ref.coef_[0], atol=1e-3)
    assert abs(f["beta"][0] - ref.intercept_[0]) < 1e-3


def _mini_raw():
    cols = ["ID", "fec_not", "año", "edad_", "uni_med_", "sexo_", "tip_ss_", "mec_pro_t", "area_", "nombre_comuna",
            "nacionali_", "fuente", "est_cli", "pac_hos_", "con_fin_", "fec_con_", "fec_res", "nmun_resi"] + etl.OPORTUNISTAS
    base = {c: None for c in cols}
    filas = []
    for i, (id_, fecha, est, mun) in enumerate([("A", "01/03/2015", "1", "MEDELLIN"), ("A", "01/03/2016", "2", "MEDELLIN"),
                                                ("B", "05/05/2020", "2", "MEDELLIN"), ("C", "09/09/2019", "1", "BELLO")]):
        r = dict(base, ID=id_, fec_not=fecha, año=fecha[-4:], edad_="30", uni_med_="1", sexo_="M", tip_ss_="C",
                 mec_pro_t="2", area_="1", nombre_comuna="10  La Candelaria", est_cli=est, pac_hos_="2", con_fin_="1",
                 fec_con_=fecha, fec_res=fecha, nmun_resi=mun)
        for o in etl.OPORTUNISTAS:
            r[o] = "2"
        filas.append(r)
    return pd.DataFrame(filas)


def test_silver_primera_notificacion_y_poblacion():
    silver = etl.construir_silver(_mini_raw())
    assert silver["primera_notificacion"].sum() == 3          # A, B, C
    assert silver.loc[silver["id_persona"] == silver.iloc[0]["id_persona"], "n_notificaciones"].iloc[0] == 2
    pob = etl.poblacion_analisis(silver)
    assert len(pob) == 2                                      # A (1.ª) y B; C no reside en Medellín
    assert set(pob["comuna"]) == {"10 La Candelaria"}         # espacios normalizados
    assert pob["tardio"].tolist() == [False, True]
    assert "ID" not in silver.columns and "fecha_nto_" not in silver.columns   # sin identificadores directos


def test_supresion_celdas_pequenas():
    assert es.suprimir([3, 10, 50], 10).tolist() == [True, False, False]


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK ", nombre)
