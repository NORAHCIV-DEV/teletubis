# Árboles de Decisión - Clasificación de Deslizamientos

## De qué trata

Entrenamos un árbol de decisión (CART) para clasificar si ocurre un deslizamiento (0 = no ocurre, 1 = ocurre). Este modelo es la base del Random Forest que usamos en el sistema, porque un Random Forest es un conjunto de muchos árboles de decisión.

## Datos

El mismo dataset de 10.000 registros sintéticos de Bolivia (`Dataset_10000_Bolivia_Sintetico.py`, semilla 42) con variables topográficas, climáticas y geológicas.

## Pasos que seguimos

1. **Exploración**: distribuciones, boxplots y correlaciones (ver sección 5 del notebook `Alerta_Deslizamientos_CRISP_DM.ipynb`).
2. **Preparación**: variables dummy con `pd.get_dummies(drop_first=True)`. Se excluyen `id_registro`, `score_riesgo_bruto` e `indice_riesgo_deslizamiento` porque se usaron para generar el objetivo (data leakage).
3. **División**: 80 % entrenamiento y 20 % prueba, estratificada y con `random_state=42`.
4. **Entrenamiento**: `DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42)` para controlar el sobreajuste.
5. **Evaluación**: métricas en el conjunto de prueba y curva de complejidad para elegir la profundidad.

## Resultados (conjunto de prueba, 2.000 registros)

| Métrica | Valor |
|---|---|
| Exactitud (accuracy) | 59,6 % |
| AUC-ROC | 0,585 |
| F1-macro | 0,543 |
| Recall de la clase *Ocurre* | 0,291 |

- **Curva de complejidad:** con profundidad 3 la exactitud en prueba es 59,1 %; con 5 llega a 59,6 %; con 8 baja a 56,7 % y con 12 a 54,9 %, mientras la exactitud en entrenamiento sube de 60,8 % (profundidad 5) a 68,6 % (profundidad 12). Por eso elegimos profundidad 5: más allá, el árbol memoriza el entrenamiento y empeora en prueba.
- **Variables que más pesan en el árbol:** pendiente (43,3 %), lluvia a 7 días (29,2 %) y humedad (17,6 %). La lluvia de 24 h pesa solo 3,1 %.

## Reglas del árbol

El árbol genera reglas tipo IF-THEN que se pueden leer. La primera división es la lluvia a 7 días (≈ 362 mm) y luego la pendiente y la humedad. Ejemplos con los datos reales:

- Lluvia a 7 días > 175 mm **y** pendiente > 27°: ocurre deslizamiento en el 54,8 % de los 1.860 registros (el promedio general es 43,8 %).
- Lluvia a 7 días > 500 mm **y** pendiente > 40°: ocurre en el 65,9 % de los 340 registros.

Las probabilidades son moderadas, no cercanas al 100 %, porque en los datos sintéticos el deslizamiento se simuló con mucho azar.

## Limitaciones

- Desempeño modesto: el árbol detecta menos de 3 de cada 10 deslizamientos. Las cifras de una versión anterior de este documento (accuracy ≈ 83 %, AUC 0,885) **no se pudieron reproducir** con este dataset y esta configuración y fueron reemplazadas por las de esta versión.
- Los datos son sintéticos: hace falta validar con datos reales (SENAMHI, Defensa Civil).

## Relación con el sistema

Nuestro sistema usa Random Forest (conjunto de árboles; ver el notebook, secciones 8 a 10). Este análisis individual ayudó a entender cómo funciona cada árbol antes de combinarlos.

## Librerías
- pandas, numpy, matplotlib, seaborn
- scikit-learn (DecisionTreeClassifier, plot_tree, métricas)
