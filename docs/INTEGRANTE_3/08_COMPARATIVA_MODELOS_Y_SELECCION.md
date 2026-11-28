# Comparativa y selección de modelos

## Evaluación común en test

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Árbol de Decisión | 0,5480 | 0,4821 | **0,4292** | **0,4541** | 0,5400 |
| Regresión Logística | **0,6010** | **0,5789** | 0,3265 | 0,4175 | **0,6091** |

El Árbol produjo TN=720, FP=404, FN=500, TP=376. Regresión Logística produjo TN=916, FP=208, FN=590, TP=286. Los dos modelos utilizan los mismos predictores, split y evaluación.

## Validación cruzada (media ± desviación estándar)

| Métrica | Árbol de Decisión | Regresión Logística |
|---|---:|---:|
| Accuracy | 0,5309 ± 0,0165 | 0,5851 ± 0,0092 |
| Precision | 0,4625 ± 0,0193 | 0,5481 ± 0,0199 |
| Recall | 0,4310 ± 0,0180 | 0,3052 ± 0,0123 |
| F1-Score | 0,4461 ± 0,0165 | 0,3920 ± 0,0134 |
| ROC-AUC | 0,5304 ± 0,0092 | 0,6024 ± 0,0076 |

## Decisión

Si el criterio de comparación es la capacidad de ordenamiento de probabilidades, Regresión Logística es la candidata académica por su mayor ROC-AUC. Si se priorizan recall o F1 al umbral predeterminado, el Árbol obtuvo valores superiores, a costa de más falsas alarmas. No se recomienda presentar un modelo como seleccionado para despliegue: falta una función de costo acordada, datos reales representativos y validación externa. Las curvas ROC y las matrices de confusión están incluidas en el notebook ejecutado.
