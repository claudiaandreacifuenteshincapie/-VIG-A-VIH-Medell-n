"""Ejecuta todo el pipeline: Bronze -> Silver -> Gold.   Uso:  python run_pipeline.py"""
import json
import time

import pandas as pd

import config
from src import analitica as an
from src import etl, modelo


def main():
    t0 = time.time()
    if not config.RAW_CSV.exists():
        raise SystemExit(f"No se encontró {config.RAW_CSV}. Copie allí el archivo sivigila_vih.csv.")
    config.DIR_GOLD.mkdir(parents=True, exist_ok=True)
    print("1/4 Bronze -> Silver ...")
    raw, silver, pob, cal = etl.ejecutar_bronze_silver()
    print(f"    {cal['filas_crudas']:,} notificaciones · {cal['personas_unicas']:,} personas · población de análisis {len(pob):,}")

    print("2/4 Gold descriptivo ...")
    an.cubo_casos(pob)
    serie = an.serie_anual(pob)
    terr, terr_info = an.territorio(pob)
    an.perfil(pob)
    an.calidad_completitud(pob)
    an.oportunidad(pob)

    print("3/4 Regresiones e hipótesis H1-H3 ...")
    A, B, S, info_reg = an.regresiones(pob)
    hip = an.hipotesis_descriptivas(A, B, S, info_reg, terr_info)

    print("4/4 Modelo predictivo (validación temporal) e hipótesis H4 ...")
    comp, info_mod = modelo.entrenar_y_evaluar(pob)
    hip.append(modelo.hipotesis_modelo(comp, info_mod))
    pd.DataFrame(hip).to_csv(config.DIR_GOLD / "hipotesis.csv", index=False, encoding="utf-8")

    k = an.kpis(silver, pob, cal, serie, terr_info)
    extra = {"territorio": terr_info, "regresion": info_reg, "modelo": info_mod, "calidad": cal}
    (config.DIR_GOLD / "resumen_estadistico.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s. Resultados en {config.DIR_GOLD}")
    for h in hip:
        print(f"  {h['id']}: {h['veredicto']}")


if __name__ == "__main__":
    main()
