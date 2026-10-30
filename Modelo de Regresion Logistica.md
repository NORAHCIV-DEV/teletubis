# Regresión Logística - Clasificación de Deslizamientos

## De qué trata

Usamos regresión logística para clasificar si un terreno tiene riesgo de deslizamiento o no (variable binaria: 0 o 1). La idea es comparar este modelo con el Random Forest que usamos en el sistema final.

## Datos

Trabajamos con el dataset sintético de 10,000 registros que generamos para Bolivia. Tiene variables como precipitación (24h, 3 días, 7 días), pendiente, elevación, humedad, tipo de suelo, departamento y geología.

## Pasos que seguimos

1. **Carga y exploración**: Revisamos la distribución de las variables con gráficos de barras y boxplots
2. **Preparación**: Convertimos las variables categóricas con One-Hot Encoding (`pd.get_dummies`). Sacamos las columnas que podían causar data leakage como `score_riesgo_bruto`
3. **División**: 80% train, 20% test con estratificación
4. **Entrenamiento**: Usamos `statsmodels` para tener los p-valores y coeficientes, y así ver qué variables son más significativas
5. **Evaluación**: Calculamos accuracy, recall, precision, F1 y AUC-ROC

## Resultados

- Accuracy en test: ~83%
- El delta entre train y test fue bajo (~0.002), así que no hay sobreajuste
- Las variables más importantes fueron `precipitacion_7d_mm` y `pendiente_grados`

## Por qué no usamos este modelo en el sistema

La regresión logística asume relaciones lineales entre las variables. Para nuestro caso, donde la interacción entre lluvia y pendiente es más compleja, el Random Forest captura mejor esos patrones no lineales. Pero este análisis nos sirvió como línea base para comparar.

## Librerías usadas
- pandas, numpy, matplotlib, seaborn
- scikit-learn (train_test_split, métricas)
- statsmodels (modelo logit con MLE)
