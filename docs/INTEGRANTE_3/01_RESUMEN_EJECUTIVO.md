# Resumen ejecutivo — Integrante 3

## Alcance

Se realizó el EDA, el análisis de sensibilidad y la comparación entre Árbol de Decisión y Regresión Logística para `deslizamiento_ocurrido`, utilizando el generador sintético del repositorio y las fases de CRISP-DM. El notebook reproducible es [Analisis_Sensibilidad_GeoRisk_AI.ipynb](../../notebooks/Analisis_Sensibilidad_GeoRisk_AI.ipynb).

## Resultados principales

- Dataset: 10.000 filas y 13 columnas; 0 valores nulos y 0 filas duplicadas.
- Balance objetivo: 5.618 registros de clase 0 (56,18 %) y 4.382 de clase 1 (43,82 %).
- Partición estratificada: 8.000 observaciones para train y 2.000 para test, `random_state=42`.
- Mejor ROC-AUC en test: Regresión Logística, 0,6091. Mejor F1 y recall en test: Árbol de Decisión, 0,4541 y 0,4292, respectivamente.

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Árbol de Decisión | 0,5480 | 0,4821 | 0,4292 | 0,4541 | 0,5400 |
| Regresión Logística | 0,6010 | 0,5789 | 0,3265 | 0,4175 | 0,6091 |

No hay un ganador para todos los criterios. Regresión Logística ofrece mayor ROC-AUC, accuracy y precision; el Árbol de Decisión obtiene mayor recall y F1. Los resultados son modestos y corresponden a una etiqueta sintética probabilística, no a eventos verificados en terreno.

## Recomendación

Mantener ambos modelos como comparación académica y no desplegar sus puntuaciones como alertas reales. Si se requiere continuar la evaluación, definir primero el costo de falsos negativos y falsas alarmas, mejorar la coherencia de los datos y validar con registros observados independientes.
