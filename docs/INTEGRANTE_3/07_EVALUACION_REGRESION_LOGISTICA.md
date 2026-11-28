# Evaluación — Regresión Logística

## Configuración

Se ajustó `LogisticRegression(max_iter=2000, random_state=42)` dentro de un pipeline que imputa, estandariza las seis variables numéricas y aplica One-Hot Encoding a las tres categóricas. La partición y folds son idénticos a los del Árbol.

## Métricas

| Métrica | Test | CV media | CV desviación estándar |
|---|---:|---:|---:|
| Accuracy | 0,6010 | 0,5851 | 0,0092 |
| Precision | 0,5789 | 0,5481 | 0,0199 |
| Recall | 0,3265 | 0,3052 | 0,0123 |
| F1-Score | 0,4175 | 0,3920 | 0,0134 |
| ROC-AUC | 0,6091 | 0,6024 | 0,0076 |

## Matriz de confusión de test

| | Predice 0 | Predice 1 |
|---|---:|---:|
| Real 0 | 916 | 208 |
| Real 1 | 590 | 286 |

El modelo no detectó 590 de los 876 positivos del test (67,4 % de falsos negativos sobre los positivos). Tuvo menos falsos positivos que el Árbol (208 frente a 404) y también menos verdaderos positivos (286 frente a 376).

## Coeficientes

Los coeficientes de las variables numéricas en su escala estandarizada más altos fueron pendiente (0,2779), precipitación 7d (0,2432) y humedad (0,1640). Para categorías, cada coeficiente compara con la categoría de referencia definida por `drop='first'`. Son relaciones aprendidas por el clasificador, no efectos causales.

## Interpretación

Regresión Logística obtuvo el mejor ROC-AUC de test (0,6091), accuracy (0,6010) y precision (0,5789) de la comparación. El recall de 0,3265 y el nivel general de discriminación siguen siendo limitados para una alerta, además de estar medidos sobre un objetivo sintético.
