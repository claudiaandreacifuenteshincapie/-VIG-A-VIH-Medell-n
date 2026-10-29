"""ETL Bronze -> Silver sobre la base real SIVIGILA (evento 850, Medellín).

Bronze : se registra la huella (SHA-256) y metadatos de la fuente. NO se duplica el archivo crudo.
Silver : una fila por NOTIFICACIÓN, con variables limpias, sin identificadores directos
         (se eliminan: ID original, certificado de defunción, fecha de nacimiento, nº de registro,
         ocupación, código de prestador y texto libre de dirección).
"""
import hashlib
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config

ESTADIO = {"1": "VIH", "2": "SIDA", "3": "Muerto"}
ASEGURAMIENTO = {"C": "Contributivo", "S": "Subsidiado", "N": "No asegurado",
                 "E": "Especial/Excepción", "P": "Especial/Excepción", "I": "Indeterminado"}
# Códigos 1-4 verificados empíricamente (edad mediana del código 4 = 4 años -> perinatal).
# Los códigos 5-13 se agrupan: verificar contra el diccionario oficial del INS antes de desagregar.
MECANISMO = {"1": "Heterosexual", "2": "Homosexual", "3": "Bisexual", "4": "Perinatal"}
ZONA = {"1": "Cabecera", "2": "Centro poblado", "3": "Rural disperso"}
FUENTE = {"1": "Notificación rutinaria", "2": "Búsqueda activa institucional",
          "3": "Vigilancia intensificada", "4": "Búsqueda activa comunitaria", "5": "Investigación"}
OPORTUNISTAS = ["can_eso", "can_va", "tub_pul", "can_cer", "tub_exp", "coccidiodo", "citomegalo",
                "ren_cit", "encefalopa", "otras_micr", "his_ext", "iso_cro", "erp_zos", "his_dis",
                "lin_bur", "neu_pne", "neu_rec", "lin_inm", "cri_cro", "cri_ext", "sar_kap",
                "sin_ema", "leu_mul", "sep_rec", "tox_cer", "meningitis"]
ORDEN_EDAD = ["<18", "18-28", "29-39", "40-49", "50-59", "60+"]


def reparar_texto(s):
    """Corrige mojibake (UTF-8 leído como cp1252): 'MedellÃ­n' -> 'Medellín'."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def parse_fecha(serie: pd.Series) -> pd.Series:
    """Fechas dd/mm/aaaa; los seriales de Excel (p. ej. 43850) se convierten; '-   -' -> NaT."""
    s = serie.astype("string").str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    m = out.isna() & num.between(30000, 60000)
    out[m] = pd.Timestamp("1899-12-30") + pd.to_timedelta(num[m], unit="D")
    return out


def sha256_archivo(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_crudo(ruta=None) -> pd.DataFrame:
    ruta = ruta or config.RAW_CSV
    df = pd.read_csv(ruta, sep=";", dtype=str, encoding="utf-8")
    df.columns = [reparar_texto(c) for c in df.columns]
    return df


def normalizar_comuna(x):
    if not isinstance(x, str):
        return "Sin información"
    x = re.sub(r"\s+", " ", reparar_texto(x).strip())
    return "Sin información" if x.lower().startswith("sin inf") else x


def construir_silver(raw: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    d["id_persona"] = raw["ID"].map(lambda v: hashlib.sha256(f"vigia-v1|{v}".encode()).hexdigest()[:12])
    d["fec_not"] = parse_fecha(raw["fec_not"])
    d["anio"] = pd.to_numeric(raw["año"], errors="coerce").astype("Int64")
    d["mes"] = d["fec_not"].dt.month

    edad = pd.to_numeric(raw["edad_"], errors="coerce")
    um = raw["uni_med_"]
    edad_anios = np.select([um == "1", um == "2", um == "3"], [edad, edad / 12, edad / 365], np.nan)
    edad_anios = pd.Series(edad_anios, index=raw.index).where(lambda s: (s >= 0) & (s <= 105))
    d["edad_anios"] = edad_anios
    d["grupo_edad"] = pd.cut(edad_anios, [-1, 18, 29, 40, 50, 60, 200], right=False,
                             labels=ORDEN_EDAD).astype("object")
    d["sexo"] = raw["sexo_"].map({"M": "Hombre", "F": "Mujer"})
    d["aseguramiento"] = raw["tip_ss_"].map(ASEGURAMIENTO).fillna("Indeterminado")
    d["mecanismo"] = raw["mec_pro_t"].map(MECANISMO).fillna("Otro/sin dato")
    d["zona"] = raw["area_"].map(ZONA)
    d["comuna"] = raw["nombre_comuna"].map(normalizar_comuna)
    d["comuna_conocida"] = ~d["comuna"].isin(["Sin información", "Otro Municipio"])
    nac = raw["nacionali_"]
    d["migrante"] = np.where(nac.isna(), "Sin dato", np.where(nac == "170", "No", "Sí"))
    d["fuente"] = raw["fuente"].map(FUENTE).fillna("Sin dato")
    d["estadio"] = raw["est_cli"].map(ESTADIO)
    d["tardio"] = d["estadio"].isin(["SIDA", "Muerto"]).where(d["estadio"].notna())
    d["hospitalizado"] = raw["pac_hos_"].map({"1": True, "2": False})
    d["fallecido"] = raw["con_fin_"].map({"2": True, "1": False})
    d["n_oportunistas"] = (raw[OPORTUNISTAS] == "1").sum(axis=1)

    f_con, f_res = parse_fecha(raw["fec_con_"]), parse_fecha(raw["fec_res"])
    dias = (d["fec_not"] - f_con).dt.days
    d["dias_consulta_notif"] = dias.where(dias.between(0, 3650))
    dias_r = (d["fec_not"] - f_res).dt.days
    d["dias_resultado_notif"] = dias_r.where(dias_r.between(0, 3650))

    resid = raw["nmun_resi"].map(reparar_texto).astype("string").str.upper()
    d["residente_medellin"] = resid.eq("MEDELLIN").fillna(False).astype(bool)

    # Persona: primera notificación y re-notificaciones
    d = d.sort_values(["id_persona", "fec_not"], kind="stable")
    d["n_notificaciones"] = d.groupby("id_persona")["id_persona"].transform("size")
    d["primera_notificacion"] = ~d.duplicated("id_persona", keep="first")
    rank = d["estadio"].map({"VIH": 1, "SIDA": 2, "Muerto": 3})
    d["estadio_max_persona"] = rank.groupby(d["id_persona"]).transform("max")
    return d.sort_index()


def poblacion_analisis(silver: pd.DataFrame) -> pd.DataFrame:
    """Primera notificación por persona, residente en Medellín y con estadio clínico conocido."""
    m = silver["primera_notificacion"] & silver["residente_medellin"] & silver["estadio"].notna()
    return silver.loc[m].copy()


def reporte_calidad(raw: pd.DataFrame, silver: pd.DataFrame, pob: pd.DataFrame) -> dict:
    fnac = parse_fecha(raw["fecha_nto_"])
    return {
        "filas_crudas": int(len(raw)),
        "personas_unicas": int(silver["id_persona"].nunique()),
        "personas_con_renotificacion": int((silver.groupby("id_persona").size() > 1).sum()),
        "poblacion_analisis": int(len(pob)),
        "excluidos_no_residentes_o_sin_estadio": int(silver["primera_notificacion"].sum() - len(pob)),
        "fecha_min_notificacion": str(silver["fec_not"].min().date()),
        "fecha_max_notificacion": str(silver["fec_not"].max().date()),
        "anio_parcial": int(silver["fec_not"].max().year),
        "pct_comuna_sin_informacion": round(float((~pob["comuna_conocida"]).mean() * 100), 1),
        "fechas_nacimiento_invalidas_antes_1910": int((fnac < "1910-01-01").sum()),
        "edades_fuera_de_rango_0_105": int(raw["edad_"].notna().sum() - silver["edad_anios"].notna().sum()),
        "retrasos_resultado_notif_negativos": int(((silver["fec_not"] - parse_fecha(raw["fec_res"])).dt.days < 0).sum()),
        "anio_epi_distinto_del_anio_calendario": int((silver["fec_not"].dt.year != silver["anio"]).sum()),
        "generado": datetime.now().isoformat(timespec="seconds"),
    }


def ejecutar_bronze_silver(ruta=None):
    ruta = ruta or config.RAW_CSV
    raw = leer_crudo(ruta)
    getattr(config, "DIR_BRONZE", config.DIR_SILVER).mkdir(parents=True, exist_ok=True)
    config.DIR_SILVER.mkdir(parents=True, exist_ok=True)
    meta = {"archivo": ruta.name if hasattr(ruta, "name") else str(ruta),
            "sha256": sha256_archivo(ruta), "n_filas": int(len(raw)), "n_columnas": int(raw.shape[1]),
            "fecha_ingesta": datetime.now().isoformat(timespec="seconds"),
            "aviso": "Datos reales de vigilancia (SIVIGILA). Contienen cuasi-identificadores: no publicar."}
    if hasattr(config, "DIR_BRONZE"):
        (config.DIR_BRONZE / "_metadata_bronze.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.DIR_SILVER / "_metadata_bronze.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    silver = construir_silver(raw)
    pob = poblacion_analisis(silver)
    silver.to_csv(config.DIR_SILVER / "notificaciones_silver.csv", index=False)
    pob.to_csv(config.DIR_SILVER / "poblacion_analisis.csv", index=False)
    cal = reporte_calidad(raw, silver, pob)
    (config.DIR_SILVER / "_reporte_calidad.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw, silver, pob, cal
