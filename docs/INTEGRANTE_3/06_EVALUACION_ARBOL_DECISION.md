# Evaluación — Árbol de Decisión

## Configuración

El modelo es `DecisionTreeClassifier(random_state=42, min_samples_leaf=10)`, dentro de un pipeline con imputación y One-Hot Encoding. Se entrenó en train y se evaluó en el test estratificado común, con umbral de clasificación nativo del estimador. La validación cruzada estratificada usa cinco folds de train.

## Métricas

| Métrica | Test | CV media | CV desviación estándar |
|---|---:|---:|---:|
| Accuracy | 0,5480 | 0,5309 | 0,0165 |
| Precision | 0,4821 | 0,4625 | 0,0193 |
| Recall | 0,4292 | 0,4310 | 0,0180 |
| F1-Score | 0,4541 | 0,4461 | 0,0165 |
| ROC-AUC | 0,5400 | 0,5304 | 0,0092 |

## Matriz de confusión de test

| | Predice 0 | Predice 1 |
|---|---:|---:|
| Real 0 | 720 | 404 |
| Real 1 | 500 | 376 |

El modelo no detectó 500 de los 876 positivos del test (57,1 % de falsos negativos sobre los positivos).

## Importancia de características

Las seis variables numéricas principales por importancia de impureza fueron `pendiente_grados` (0,1846), `precipitacion_7d_mm` (0,1836), `elevacion_m` (0,1490), `humedad_pct` (0,1477), `precipitacion_3d_mm` (0,1288) y `precipitacion_24h_mm` (0,1269). La importancia por impureza describe el uso interno del árbol y no es una medida causal; también puede favorecer predictores con más puntos de corte.

## Interpretación

El Árbol superó a la Regresión Logística en recall y F1 en test, pero su ROC-AUC de 0,5400 es bajo. El resultado no demuestra capacidad suficiente para una alerta real.
