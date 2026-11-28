# Preparación de datos sin fuga

## Predictores y objetivo

El objetivo es `deslizamiento_ocurrido`. Los nueve predictores usados son seis numéricos (`precipitacion_24h_mm`, `precipitacion_3d_mm`, `precipitacion_7d_mm`, `pendiente_grados`, `elevacion_m`, `humedad_pct`) y tres categóricos (`departamento`, `tipo_suelo`, `geologia`).

Se excluyen `id_registro`, `score_riesgo_bruto` e `indice_riesgo_deslizamiento`: el primero es identificador y los dos últimos formaron parte del proceso que produjo la etiqueta. No se aplica SMOTE ni se elimina o recorta observaciones.

## Partición y transformación

`train_test_split` se aplica primero, con `test_size=0.20`, `stratify=y` y `random_state=42`: 8.000 filas para train y 2.000 para test. La proporción de clase 1 fue 43,82 % en total, 43,82 % en train y 43,80 % en test.

Imputación, One-Hot Encoding y escalado se implementan dentro de `Pipeline` y `ColumnTransformer`. El imputador numérico usa mediana; el categórico, moda. One-Hot Encoding utiliza `drop='first'` y `handle_unknown='ignore'`. Regresión Logística estandariza las variables numéricas con `StandardScaler`; el Árbol no requiere escalado. Aunque la revisión de calidad no encontró nulos, el imputador queda dentro del pipeline para que cualquier ajuste futuro se aprenda solo del fold de entrenamiento.

La validación cruzada estratificada de cinco folds se ejecuta en el conjunto train; cada fold vuelve a ajustar sus transformaciones. El test no interviene en la selección ni el ajuste del preprocesamiento y se utiliza para la evaluación final y análisis descriptivos de umbral/sensibilidad.

## Verificación

La corrida guardada confirma dimensiones `(10000, 13)`, 0 nulos, 0 duplicados, 8.000 filas train, 2.000 test y las tres columnas excluidas enumeradas arriba. La misma partición y variables se utilizaron para los dos modelos.
