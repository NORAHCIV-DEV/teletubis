# Informe de K-Means — GeoRisk AI (H4)

## Objetivo y alcance

Se utiliza K-Means, un algoritmo no supervisado, para agrupar **10.000 observaciones geoclimáticas sintéticas** de Bolivia en perfiles con características parecidas. Estos grupos complementan, pero **no sustituyen**, la predicción supervisada de riesgo de deslizamiento. No representan alertas ni resultados validados en campo.

Este informe documenta el notebook `H4_Entrega_KMeans_GeoRisk_AI.ipynb` entregado por el equipo. Los valores siguientes corresponden a **sus resultados de ejecución guardados**; pueden variar si cambia la generación o el preprocesamiento de datos.

## Descripción de las variables

| Variable utilizada en K-Means | Unidad | Descripción |
|---|---|---|
| `precipitacion_24h_mm` | mm | Lluvia acumulada en las últimas 24 horas; representa precipitación inmediata. |
| `precipitacion_3d_mm` | mm | Lluvia acumulada en tres días; caracteriza episodios recientes. |
| `precipitacion_7d_mm` | mm | Lluvia acumulada en siete días; caracteriza precipitación prolongada. |
| `pendiente_grados` | grados | Ángulo de inclinación del terreno. |
| `elevacion_m` | m | Elevación sobre el nivel del mar. |
| `humedad_pct` | % | Humedad del suelo en el registro sintético. |

Variables contextuales como departamento, tipo de suelo y geología no intervienen en el K-Means del notebook. `indice_riesgo_deslizamiento` y `deslizamiento_ocurrido` se analizan **después** de agrupar y no se emplean para entrenar K-Means.

## Preparación y metodología

1. Generación reproducible del dataset sintético con la semilla 42 mediante `Dataset_10000_Bolivia_Sintetico.py`.
2. Selección de seis variables numéricas; conversión, eliminación de nulos y duplicados.
3. Recorte de extremos a los percentiles 1 y 99 de cada variable.
4. Estandarización con `StandardScaler`, para que las escalas de mm, grados, m y % no distorsionen las distancias.
5. Comparación de valores de K entre 2 y 6 con WCSS e índice Silhouette.
6. Fijación de **K=4** en la entrega H4 para comunicar cuatro perfiles, aun cuando no es el mayor Silhouette.
7. Visualización PCA de dos dimensiones y mapa de calor de perfiles; los clusters se entrenan con las seis variables, no con el PCA.

## Comparación de K (ejecución guardada)

| K | Silhouette | WCSS |
|---:|---:|---:|
| 2 | 0,2300 | 52.306,7534 |
| 3 | 0,1546 | 45.790,8546 |
| 4 | 0,1625 | 40.768,1168 |
| 5 | 0,1721 | 36.183,4411 |
| 6 | 0,1585 | 33.399,5808 |

**Interpretación:** K=2 tiene la mayor puntuación Silhouette entre las alternativas comparadas. K=4 se usa como una decisión descriptiva de H4, no como óptimo estadístico demostrado. La baja puntuación de K=4 indica que la separación de grupos es limitada.

## Perfiles y nombres descriptivos (K=4)

| Cluster | Nombre descriptivo | Observaciones | Lluvia 3 días (mm) | Lluvia 7 días (mm) | Pendiente (°) | Elevación (m) | Humedad (%) | Deslizamientos simulados (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Saturación por lluvia acumulada | 1.295 | 37,81 | 654,99 | 30,90 | 2.801,04 | 59,60 | 58,0 |
| 1 | Valles de menor elevación | 3.706 | 29,71 | 104,36 | 29,78 | 1.618,71 | 59,46 | 41,0 |
| 2 | Lluvia reciente de alta intensidad | 1.381 | 130,79 | 128,06 | 30,87 | 2.714,44 | 59,52 | 43,0 |
| 3 | Altiplano de gran elevación | 3.618 | 30,35 | 106,55 | 30,57 | 3.935,57 | 59,88 | 42,0 |

**Cluster 0 — Saturación por lluvia acumulada.** Presenta la mayor precipitación media en siete días, por lo que se distingue por lluvias prolongadas. La expresión «saturación» es una etiqueta interpretativa, no una medición directa de saturación del suelo.

**Cluster 1 — Valles de menor elevación.** Tiene la menor elevación media entre los cuatro grupos, con precipitaciones semanales moderadas en este dataset.

**Cluster 2 — Lluvia reciente de alta intensidad.** Tiene la mayor precipitación acumulada de tres días entre los grupos restantes, diferenciándose por episodios de lluvia reciente.

**Cluster 3 — Altiplano de gran elevación.** Tiene la mayor elevación media y no presenta las precipitaciones semanales extremas del grupo 0. La denominación «Altiplano» es interpretativa y no una clasificación geográfica validada por coordenadas.

Las etiquetas 0, 1, 2 y 3 **no representan una jerarquía de riesgo**. Cada nombre se asigna después del entrenamiento según los centroides y puede cambiar si se reentrena el modelo.

## Hallazgos y limitaciones

En el dataset simulado, el cluster 0 presenta 58,0 % de ocurrencia de deslizamientos y el cluster 1, 41,0 %, una diferencia de 17 puntos porcentuales. Son asociaciones dentro de datos **sintéticos**, no probabilidades de campo ni evidencia causal. Se requiere validación con mediciones reales, temporalidad, información espacial y revisión por especialistas antes de un uso operativo.

## Reproducir en Visual Studio Code o Jupyter

Desde la raíz del repositorio en **Git Bash**, con Python 3.11 disponible:

```bash
"/c/Users/ASUS TUF/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install jupyter seaborn matplotlib
./.venv/Scripts/python.exe Dataset_10000_Bolivia_Sintetico.py
./.venv/Scripts/python.exe -m jupyter notebook H4_Entrega_KMeans_GeoRisk_AI.ipynb
```

La ruta de Python es un **ejemplo específico de la computadora del equipo**; en otras computadoras se debe utilizar `python` o `py -3.11`. Si el notebook está en otra carpeta, ajustar la ubicación del CSV o abrir Jupyter desde la raíz del proyecto. En Jupyter, seleccionar **Run All** para ejecutar todas las celdas. Se deben guardar evidencias de la tabla de variables, comparación de K, PCA, mapa de calor, nombres y conclusiones.

### Verificaciones recomendadas antes de la entrega

- Confirmar que el notebook `.ipynb` y este informe estén publicados en la rama `main` del repositorio correcto.
- Generar el CSV antes de ejecutar el notebook, porque el repositorio no incluye el CSV.
- Comprobar que las 16 celdas del notebook se ejecuten en un ambiente limpio y que los nombres descriptivos correspondan a sus nuevos centroides.
- Comprobar que la interfaz web y el notebook utilizan el mismo entrenamiento si se comparan sus etiquetas y valores; podrían diferir por su preprocesamiento.
- Completar autores, fecha, versión del repositorio y referencia al material del ingeniero en el informe final.
