# Regresión Logística - Clasificación de Deslizamientos

## De qué trata

Usamos regresión logística para clasificar si ocurre un deslizamiento (variable binaria: 0 o 1). Sirve como **línea base** para comparar con el Random Forest que usamos en el sistema final.

## Datos

El dataset sintético de 10.000 registros de Bolivia. Variables: precipitación (24 h, 3 días, 7 días), pendiente, elevación, humedad, tipo de suelo, departamento y geología.

## Pasos que seguimos

1. **Exploración**: distribuciones y relación de cada variable con el objetivo (sección 5 del notebook).
2. **Preparación**: One-Hot Encoding con `pd.get_dummies(drop_first=True)`. Se excluyen `id_registro`, `score_riesgo_bruto` e `indice_riesgo_deslizamiento` (data leakage: se usaron para generar el objetivo).
3. **División**: 80 % entrenamiento y 20 % prueba, estratificada, `random_state=42`.
4. **Escalado**: `StandardScaler` ajustado solo con el entrenamiento, para que los coeficientes sean comparables.
5. **Entrenamiento**: `LogisticRegression` de scikit-learn.
6. **Evaluación**: accuracy, recall, precision, F1 y AUC-ROC. El script `regresion_logistica.py` reproduce todas las cifras.

## Resultados (conjunto de prueba, 2.000 registros)

| Métrica | Valor |
|---|---|
| Exactitud (accuracy) | 60,2 % |
| Sensibilidad (recall de *Ocurre*) | 32,8 % |
| Precisión (de *Ocurre*) | 58,0 % |
| F1 (de *Ocurre*) | 0,419 |
| AUC-ROC | 0,609 |

- **Sin sobreajuste:** la exactitud en entrenamiento (58,8 %) es similar a la de prueba (60,2 %).
- **Variables más importantes** (aumento de las odds por cada aumento de 1 desviación estándar): pendiente +32 % (≈ 15°), lluvia a 7 días +28 % (≈ 216 mm) y humedad +18 % (≈ 20 puntos). La lluvia de 24 h casi no influye (−2 %), y departamento, suelo y geología cambian las odds menos de 6 %.

## Por qué no usamos este modelo en el sistema

La regresión logística asume relaciones lineales entre las variables. Probamos también el Random Forest (con SMOTE, notebook secciones 8 y 9): ambos modelos rinden prácticamente igual (accuracy ≈ 0,58 y F1-macro ≈ 0,57), por lo que la limitación está en la señal de los datos sintéticos y no en el algoritmo. El Random Forest se mantiene como modelo principal del sistema por su mayor flexibilidad, y este análisis es la línea base de comparación.

## Limitaciones

- Cifras de una versión anterior (accuracy ≈ 83 %) no se pudieron reproducir y fueron reemplazadas por las de esta versión.
- Datos sintéticos: hace falta validar con datos reales.

## Librerías usadas
- pandas, numpy
- scikit-learn (LogisticRegression, StandardScaler, train_test_split, métricas)
