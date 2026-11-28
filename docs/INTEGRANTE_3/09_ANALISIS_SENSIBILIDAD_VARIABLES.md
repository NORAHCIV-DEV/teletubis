# Análisis de sensibilidad de variables

## Método

Sobre cada fila del conjunto test se aumentó una sola variable en una desviación estándar calculada en train, sin cambiar las demás. Se comparó la probabilidad positiva antes y después y la proporción clasificada como positiva al umbral 0,5. Es un análisis de sensibilidad del modelo, no una intervención causal. La alteración simultánea puede crear combinaciones que no sean físicamente coherentes.

## Cambio medio en probabilidad positiva

| Variable aumentada en +1 desviación estándar | Incremento (unidad) | Árbol | Regresión Logística |
|---|---:|---:|---:|
| Precipitación 24 h | 14,7626 mm | -0,0071 | -0,0051 |
| Precipitación 3 días | 44,2340 mm | -0,0075 | -0,0024 |
| Precipitación 7 días | 215,2328 mm | +0,0781 | +0,0581 |
| Pendiente | 14,8457° | +0,0481 | +0,0665 |
| Humedad del suelo | 19,3958 puntos porcentuales | +0,0318 | +0,0391 |

Para el Árbol, el cambio absoluto medio de probabilidad fue 0,0961 para lluvia 24h, 0,1032 para lluvia 3d, 0,2407 para lluvia 7d, 0,2712 para pendiente y 0,2041 para humedad. La respuesta local del árbol puede no ser monótona por sus divisiones discretas; por eso cambios medios pequeños o negativos no contradicen sus importancias de características.

## Cambio de proporción de predicciones positivas

| Modelo | Base | +1 DE precip. 7d | +1 DE pendiente | +1 DE humedad |
|---|---:|---:|---:|---:|
| Árbol de Decisión | 43,70 % | 54,15 % | 52,45 % | 48,85 % |
| Regresión Logística | 24,70 % | 46,00 % | 49,75 % | 37,25 % |

Las variables con señal más consistente dentro de este experimento son precipitación 7d, pendiente y humedad. Los cambios reflejan el comportamiento de los modelos entrenados con este generador, no el impacto esperado en una ladera real.

## Importancia y coeficientes

El Árbol asignó sus mayores importancias a pendiente (0,1846) y precipitación 7d (0,1836), seguidas de elevación (0,1490) y humedad (0,1477). Regresión Logística presentó los mayores coeficientes numéricos estandarizados para pendiente (0,2779), precipitación 7d (0,2432) y humedad (0,1640). La convergencia de las tres variables con los patrones de etiqueta sintética debe interpretarse teniendo en cuenta que intervinieron en la construcción del objetivo.
