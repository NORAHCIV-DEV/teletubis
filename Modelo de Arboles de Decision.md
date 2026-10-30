# Árboles de Decisión - Clasificación de Deslizamientos

## De qué trata

Entrenamos un árbol de decisión (CART) para clasificar el riesgo de deslizamiento. Este modelo es la base del Random Forest que usamos en el sistema, porque un Random Forest es básicamente un conjunto de muchos árboles de decisión.

## Datos

El mismo dataset de 10,000 registros sintéticos de Bolivia con variables topográficas, climáticas y geológicas.

## Pasos que seguimos

1. **Exploración**: Gráficos de distribución y boxplots para entender los datos
2. **Preparación**: Variables dummy con `pd.get_dummies`, eliminamos columnas con leakage
3. **División**: 80/20 con estratificación
4. **Entrenamiento**: `DecisionTreeClassifier` con `max_depth=5` y `min_samples_leaf=15` para controlar el sobreajuste
5. **Evaluación**: Métricas estándar + curva de complejidad para elegir la profundidad correcta

## Resultados

- Accuracy en test: ~83.35%
- AUC-ROC: 0.885
- La curva de complejidad mostró que con profundidad mayor a 8 el modelo se sobreajusta sin mejorar en test
- Las variables que más pesan: precipitación a 7 días y pendiente

## Reglas del árbol

Lo interesante de este modelo es que genera reglas tipo IF-THEN que se pueden leer. Por ejemplo: si la lluvia a 7 días supera 175mm y la pendiente es mayor a 27°, la probabilidad de deslizamiento supera el 90%.

## Relación con el sistema

Nuestro sistema usa Random Forest (conjunto de árboles). Este análisis individual nos ayudó a entender cómo funciona cada árbol por separado antes de combinarlos.

## Librerías
- pandas, numpy, matplotlib, seaborn
- scikit-learn (DecisionTreeClassifier, plot_tree, métricas)
