# Diccionario de variables derivadas (Silver) y estado de verificación

Fuente: SIVIGILA, evento 850 (VIH/SIDA/mortalidad por SIDA), archivo `sivigila_vih.csv` (separador `;`, 116 columnas, 15.470 filas).

| Variable derivada | Origen (columna SIVIGILA) | Regla | Verificación |
|---|---|---|---|
| `anio` | `año` | Año epidemiológico reportado | Verificado (269 filas difieren del año calendario de `fec_not`: se informa) |
| `estadio` | `est_cli` | 1 = VIH, 2 = SIDA, 3 = Muerto | Coherente con el boletín del INS (estadios VIH / sida / muerto) y con los datos: 91 % de "SIDA" tiene ≥ 1 infección oportunista, 0,15 % de "VIH" |
| `tardio` | `est_cli` | SIDA o Muerto en la **primera** notificación de la persona | Definición propia (aproximación; no hay CD4) |
| `sexo` | `sexo_` | M = Hombre, F = Mujer | Verificado |
| `edad_anios` | `edad_`, `uni_med_` | 1 = años, 2 = meses (÷12), 3 = días (÷365); se descartan edades > 105 | Verificado |
| `grupo_edad` | `edad_anios` | <18, 18-28, 29-39, 40-49, 50-59, 60+ | Propio |
| `aseguramiento` | `tip_ss_` | C Contributivo · S Subsidiado · N No asegurado · E/P Especial-Excepción · I Indeterminado | Verificar contra el diccionario del INS |
| `mecanismo` | `mec_pro_t` | 1 Heterosexual · 2 Homosexual · 3 Bisexual · 4 Perinatal · resto "Otro/sin dato" | Códigos 1-4 deducidos de los datos (el código 4 tiene edad mediana de 4 años y 2 sexos balanceados). **Confirmar con el diccionario oficial**; 5-13 no se desagregan |
| `zona` | `area_` | 1 Cabecera · 2 Centro poblado · 3 Rural disperso | Verificar |
| `comuna` | `nombre_comuna` | Se corrige mojibake, se normalizan espacios; "Sin información" = no registrada | Verificado |
| `migrante` | `nacionali_` | vacío = "Sin dato"; 170 (Colombia) = No; otro = Sí | Solo disponible desde 2019 (77 % de 2019; ~100 % desde 2020) |
| `fuente` | `fuente` | 1 rutinaria · 2 búsqueda activa institucional · 3 vigilancia intensificada · 4 búsqueda activa comunitaria · 5 investigación | Verificar |
| `hospitalizado`, `fallecido` | `pac_hos_`, `con_fin_` | 1 = sí / 2 = no (hospitalización); 2 = fallecido (condición final) | Verificar. **No se usan como predictores** (consecuencia/definición del desenlace) |
| `n_oportunistas` | 26 columnas de enfermedades definitorias | Conteo de respuestas = 1 | Solo para validar el desenlace; no es predictor |
| `dias_consulta_notif` | `fec_con_`, `fec_not` | Días; se descartan negativos y > 3650 | Verificado |
| `id_persona` | `ID` | SHA-256 con sal de proyecto, 12 caracteres | Seudonimización adicional |

**Columnas eliminadas en Silver** (identificadores o cuasi-identificadores): `ID`, `cer_def_`, `fecha_nto_`, `nreg`, `ocupacion_`, `cod_pre`, `localidad_`, `cen_pobla_`, `vereda_`, `bar_ver_`, `codigo_barrio`, `nombre_barrio`, `nom_upgd`, `cod_ase_`, campos de fuerzas militares.

**Pendientes de verificación con el INS / Secretaría de Salud de Medellín:** significado exacto de `ajuste_` (valores 0/3/7), `tip_pru` (aquí 2 = prueba con resultado numérico, probablemente carga viral), `periodo_epid`=28 (3 casos) y códigos 5-13 de `mec_pro_t`.
