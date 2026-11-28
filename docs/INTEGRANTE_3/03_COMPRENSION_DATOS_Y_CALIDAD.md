# Comprensión de datos y calidad

## Fuente y estructura

El notebook genera, cuando no existe, `Dataset_10000_Bolivia_Sintetico.csv` ejecutando `Dataset_10000_Bolivia_Sintetico.py`, que fija la semilla 42. La corrida analizada produjo 10.000 filas y 13 columnas:

`id_registro`, `departamento`, `precipitacion_24h_mm`, `precipitacion_3d_mm`, `precipitacion_7d_mm`, `pendiente_grados`, `elevacion_m`, `humedad_pct`, `tipo_suelo`, `geologia`, `score_riesgo_bruto`, `indice_riesgo_deslizamiento`, `deslizamiento_ocurrido`.

El objetivo tiene 5.618 ceros (56,18 %) y 4.382 unos (43,82 %). No se encontraron nulos ni filas duplicadas. Las variables categóricas observadas tienen seis departamentos, cuatro tipos de suelo y tres categorías geológicas.

## Rangos y coherencia

| Variable | Mínimo observado | Máximo observado | Fuera del rango revisado |
|---|---:|---:|---:|
| `precipitacion_24h_mm` | 0,00 | 137,09 | 0 |
| `precipitacion_3d_mm` | 0,00 | 410,61 | 0 |
| `precipitacion_7d_mm` | 0,00 | 2.066,28 | 0 |
| `pendiente_grados` | 0,00 | 82,13 | 0 |
| `humedad_pct` | 0,00 | 100,00 | 0 |
| `elevacion_m` | 500,21 | 4.999,53 | 0 |

Se encontraron 2.532 casos con precipitación de 24 horas mayor que la de 3 días y 2.449 con precipitación de 3 días mayor que la de 7 días. Estos conteos pueden solaparse. Las ventanas se generan independientemente, por lo que su orden temporal no es físicamente coherente. No se modificaron los registros; esta limitación debe tratarse en una futura versión del generador.

## Procedencia del objetivo y advertencia de fuga

El código generador construye `score_riesgo_bruto` usando precipitación acumulada de 7 días, pendiente, humedad y ruido; escala el score a `indice_riesgo_deslizamiento` y utiliza el índice para generar `deslizamiento_ocurrido`. Por ello, score e índice y el identificador fueron excluidos de los predictores.

La variable objetivo es una simulación estocástica basada en la fórmula del generador. No constituye un registro observado de deslizamientos en Bolivia ni representa la prevalencia real de eventos.
