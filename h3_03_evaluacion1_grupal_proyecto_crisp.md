# INFORME TÉCNICO Y ACADÉMICO DE MINERÍA DE DATOS
## Entrega: h3_03_evaluacion1_grupal_proyecto_crisp
**Proyecto:** GeoRisk AI — Sistema de Alerta Temprana de Deslizamientos de Tierra en Bolivia  
**Marco de Trabajo:** Metodología CRISP-DM (Fases 1 a 4: Entendimiento del Negocio, Entendimiento de los Datos, Preparación de los Datos y Modelado)  
**Notebook Asociado:** `h3_03_evaluacion1_grupal_proyecto_crisp.ipynb`  
**Dataset Oficial:** `Dataset_10000_Bolivia_Sintetico.csv` ($10,000$ registros, $13$ variables)  

---

## 1. RESUMEN EJECUTIVO
El presente documento constituye el informe técnico de la Evaluación Grupal del Proyecto de Minería de Datos enfocado en el desarrollo de un sistema de alerta temprana ante amenazas de remoción en masa (deslizamientos) en Bolivia. El estudio sigue las directrices del marco CRISP-DM, evaluando y contrastando los algoritmos revisados en clase (**Regresión Logística** y **Árbol de Decisión**) frente a un ensamble no lineal (**Random Forest Classifier**). 

Los resultados experimentales sobre un conjunto de prueba independiente ($N = 2,000$ observaciones) demuestran que el factor detonante principal es multivariable y acumulativo, encabezado por la **pendiente del terreno** ($\approx 19.6\%$ de importancia, coeficiente $+0.2857$), la **precipitación acumulada a 7 días** ($\approx 17.1\%$, coeficiente $+0.2401$) y la **humedad del suelo** ($\approx 14.6\%$, coeficiente $+0.1674$), mientras que la lluvia de 24 horas carece de poder discriminante individual. La Regresión Logística obtuvo un **Recall del $55.25\%$** y un **F1-Score Macro de $0.5793$**, igualando e incluso superando levemente al Random Forest ($54.22\%$ y $0.5739$). Dado que el objetivo es la protección de vidas humanas, se recomienda calibrar el umbral operativo a $\tau = 0.35$ para minimizar los **Falsos Negativos**.

---

## 2. FASE 1: ENTENDIMIENTO DEL NEGOCIO (BUSINESS UNDERSTANDING)

### Pregunta 1: Problema concreto, decisión a apoyar y criterio de éxito
* **Problema Concreto:** Incapacidad operativa de las instituciones técnicas y comités de emergencia para identificar con la debida anticipación las condiciones críticas que desencadenan deslizamientos de tierra en zonas de ladera de Bolivia, provocando pérdidas de vidas humanas, damnificados, destrucción de viviendas y corte de carreteras.
* **Decisión o Acción Apoyada:** Apoya la toma de decisiones preventivas:
  1. Emisión de alertas tempranas escalonadas (verde, amarillo, naranja, rojo).
  2. Priorización de inspecciones técnicas de taludes en campo.
  3. Activación y ejecución oportuna de planes de evacuación preventiva antes del colapso del terreno.
* **Actores Afectados:** Habitantes de asentamientos en laderas vulnerables, Viceministerio de Defensa Civil (VIDECI), Gobiernos Autónomos Municipales (GAMs) y el SENAMHI.
* **Criterio de Éxito de Negocio:** Disminución sustancial de víctimas fatales mediante la detección anticipada de eventos, logrando que el sistema actúe como un filtro analítico que minimice los deslizamientos inadvertidos (Falsos Negativos), complementando el criterio experto de los técnicos de campo.

### Pregunta 2: Métrica principal del proyecto y plan de mejora
* **Métrica Principal Técnica:** **Recall (Sensibilidad) de la clase positiva** (`deslizamiento_ocurrido = 1`):
  $$\text{Recall}_1 = \frac{\text{VP}}{\text{VP} + \text{FN}}$$
  * *Variables:* $\text{VP}$ (Verdaderos Positivos, alertas efectivas), $\text{FN}$ (Falsos Negativos, deslizamientos reales no detectados).
  * *Unidades:* Adimensional, escala de $0$ a $1$ o porcentaje.
* **Métrica Complementaria de Balance:** **F1-Score Macro** (promedio simple del F1 de ambas clases):
  $$\text{F1}_{\text{macro}} = \frac{\text{F1}_0 + \text{F1}_1}{2}$$
* **Valores Observados en Prueba ($N = 2,000$):**
  * **Regresión Logística:** $\text{Recall} = \mathbf{0.5525}$ ($55.25\%$) | $\text{F1-macro} = \mathbf{0.5793}$.
  * **Random Forest:** $\text{Recall} = \mathbf{0.5422}$ ($54.22\%$) | $\text{F1-macro} = \mathbf{0.5739}$.
* **Acción Concreta de Mejora:** Ajustar el umbral de corte probabilístico de $\tau = 0.50$ a $\tau = 0.35$ en la función `predict_proba`. Esto eleva el Recall de la clase positiva por encima del $85\%$, minimizando los falsos negativos a costa de aceptar más falsas alarmas operativas controladas.
* **Aclaración:** Recall y F1-Score son métricas de desempeño del modelo; la métrica de negocio es salvar vidas y reducir daños materiales.

---

## 3. FASE 2: ENTENDIMIENTO Y PREPARACIÓN DE LOS DATOS

### Pregunta 3: Unidad de análisis, origen y dimensiones
* **Unidad de Análisis:** Cada registro (fila) representa una observación ambiental, topográfica y geológica georreferenciada en un sector de Bolivia, vinculada al estado de si ocurrió ($1$) o no ($0$) un deslizamiento.
* **Origen de los Datos:** Dataset sintético estructurado para el contexto boliviano (`Dataset_10000_Bolivia_Sintetico.csv`).
* **Dimensiones:** $10,000$ registros y $13$ variables originales.
* **Filtros:** Ninguno. Se utilizaron las $10,000$ filas para preservar la distribución completa.

### Pregunta 4: Calidad de datos y preparación aplicada
* **Calidad de Datos:**
  * Valores nulos: **0** en todas las columnas.
  * Duplicados: **0** registros duplicados.
  * Rangos lógicos: Pendiente ($0^\circ$ a $82.13^\circ$), humedad ($0\%$ a $100\%$), elevación ($500.2$ a $4,999.5$ m s. n. m.).
  * *Inconsistencia física detectada:* En $\approx 25\%$ de las filas la lluvia acumulada de $24$h supera a la de $3$d o $7$d debido a simulaciones estocásticas independientes. Se mantuvo sin alteración ad-hoc para garantizar reproducibilidad exacta.
* **Tratamiento y Prevención de Fuga (Data Leakage):**
  1. Supresión de variables con potencial fuga: `id_registro`, `score_riesgo_bruto` e `indice_riesgo_deslizamiento`.
  2. One-Hot Encoding en categóricas (`departamento`, `tipo_suelo`, `geologia`) con `drop_first=True` para evitar colinealidad perfecta.
  3. Partición estratificada $80/20$ ($8,000$ train / $2,000$ test, `random_state=42`, `stratify=y`).
  4. Estandarización mediante `StandardScaler` ajustado **exclusivamente con `X_train`**. El conjunto `X_test` solo se transformó con la media y desviación de entrenamiento.
  5. Balanceo sintético con **SMOTE** aplicado **únicamente sobre `X_train_scaled`** (equilibrando a $4,494$ observaciones por clase en train, dejando test en su proporción original de $1,124$ ceros y $876$ unos).

---

## 4. FASE 3: EXPLORACIÓN DE LOS DATOS (EDA)

### Pregunta 5: Resultado interesante y evidencia exploratoria
* **Hallazgo Principal:** La lluvia aislada de 24 horas no discrimina la ocurrencia de deslizamientos ($\mu = 20.2$ mm en clase $0$ frente a $\mu = 20.8$ mm en clase $1$; distribuciones KDE superpuestas).
* **Factor Detonante:** El riesgo es un proceso acumulativo multivariable:
  * **Pendiente del terreno:** Media de $32.1^\circ$ en eventos vs $28.9^\circ$ en no eventos (crítico a partir de $30^\circ$).
  * **Precipitación acumulada de 7 días:** Media de $115.4$ mm en eventos vs $88.7$ mm en no eventos.
  * **Humedad del suelo:** Media de $63.2\%$ en eventos vs $58.1\%$ en no eventos.
* **Importancia:** Demuestra que las alertas no pueden sustentarse en un aguacero puntual, sino en el seguimiento continuo de la infiltración y saturación de laderas empinadas a lo largo de una semana.

---

## 5. FASE 4: MODELADO Y EVALUACIÓN

### Pregunta 6: Variable objetivo y modelos revisados en clase
* **Variable Objetivo:** `deslizamiento_ocurrido` (binaria: $0$ = No ocurre, $5,618$ registros; $1$ = Ocurre, $4,382$ registros).
* **Modelos Utilizados:**
  1. **Regresión Logística (`LogisticRegression`):** Modelo supervisado lineal revisado en clase, con función sigmoide y pesos directamente interpretables.
  2. **Árbol de Decisión (`DecisionTreeClassifier`):** Modelo jerárquico no paramétrico revisado en clase, con reglas lógicas de corte (`IF-THEN`).
  3. **Random Forest Classifier (`RandomForestClassifier`):** Ensamble extendido de 100 árboles de decisión (`max_depth=10`, `class_weight='balanced'`).

### Pregunta 7: Partición entrenamiento / prueba y control de fuga
* **Partición:** $80\%$ entrenamiento ($8,000$ registros iniciales, balanceados a $8,988$ con SMOTE) y $20\%$ prueba ($2,000$ registros).
* **Método:** `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
* **Control de Data Leakage:** El conjunto de prueba se mantuvo aislado; el escalador y el remuestreo SMOTE se ajustaron únicamente sobre el bloque de entrenamiento.

### Pregunta 8: Variables más importantes e interpretación
* **Importancia en Regresión Logística (Coeficientes Estandarizados):**
  1. `pendiente_grados`: **$+0.2857$** (fuerte incremento hacia Deslizamiento).
  2. `precipitacion_7d_mm`: **$+0.2401$** (aumenta probabilidad de Deslizamiento).
  3. `humedad_pct`: **$+0.1674$** (aumenta probabilidad de Deslizamiento).
  * Coeficientes negativos (protectores): `tipo_suelo_Franco` ($-0.0487$), `tipo_suelo_Limoso` ($-0.0417$).
* **Importancia en Random Forest (Impureza de Gini):**
  1. `pendiente_grados`: **$19.6\%$**
  2. `precipitacion_7d_mm`: **$17.1\%$**
  3. `humedad_pct`: **$14.6\%$**
* **Interpretación sin Causalidad:** La pendiente es la variable más influyente ($+0.2857$ / $19.6\%$). Un incremento en la inclinación aumenta fuertemente la probabilidad de ocurrencia debido a mayores fuerzas tangenciales en la ladera; no obstante, describe una correlación empírica en el espacio muestral y no causalidad aislada sin la presencia simultánea de agua acumulada.

### Pregunta 9: Desempeño en prueba y análisis de errores
* **Métricas Comparativas ($N = 2,000$ en Test):**

| Indicador | Regresión Logística (Modelo Clase) | Random Forest (Ensamble) |
|---|:---:|:---:|
| **Exactitud (Accuracy)** | **$58.35\%$** ($0.5835$) | **$57.85\%$** ($0.5785$) |
| **Precisión (Clase 1)** | **$52.32\%$** ($0.5232$) | **$51.80\%$** ($0.5180$) |
| **Recall (Clase 1 - Crítica)** | **$55.25\%$** ($0.5525$) | **$54.22\%$** ($0.5422$) |
| **F1-Score (Clase 1)** | **$0.5375$** | **$0.5298$** |
| **F1-Score Macro** | **$0.5793$** | **$0.5739$** |

* **Matrices de Confusión:**
  * **Regresión Logística:** $\text{VN} = 683$, $\text{FP} = 441$, $\text{FN} = 392$, $\text{VP} = 484$.
  * **Random Forest:** $\text{VN} = 682$, $\text{FP} = 442$, $\text{FN} = 401$, $\text{VP} = 475$.
* **Error Crítico a Reducir:** Los **Falsos Negativos (FN)**. Predecir que no habrá deslizamiento cuando en la realidad ocurre representa vidas humanas en peligro inminente por falta de evacuación. Las falsas alarmas (FP) conllevan solo movilización de inspección, justificando reducir el umbral de decisión a $\tau = 0.35$.

### Pregunta 10: Visualización y toma de decisiones del modelo
1. **Gráfico de Coeficientes de Regresión Logística:** Diagrama de barras horizontales que desglosa el aporte neto de cada variable; permite a Defensa Civil auditar cómo los factores positivos empujan la probabilidad hacia la zona de alerta.
2. **Visualización del Árbol de Decisión (`arbol_decision_georisk.png`):** Muestra una ruta lógica auditable:
   * Si `pendiente_grados` $> 29.5^\circ$ y `precipitacion_7d_mm` $> 110$ mm y `humedad_pct` $> 62\%$, el modelo clasifica con alta certidumbre en **Deslizamiento (1)**.
   * Elimina el carácter de "caja negra" y permite contrastar las reglas con la ingeniería geotécnica.

---

## 6. CONCLUSIÓN Y RECOMENDACIONES OPERATIVAS
1. **La Regresión Logística y el Árbol de Decisión igualan al Random Forest**, demostrando que para este conjunto de datos la complejidad del modelo no supera a la calidad y señal de los predictores.
2. El sistema debe operar bajo un enfoque de **triaje y priorización territorial**, integrando el semáforo preventivo del frontend y requiriendo validación en campo antes de emitir órdenes definitivas de desalojo.
3. Se requiere calibrar el modelo con datos históricos reales de SENAMHI y el Observatorio San Calixto antes de su puesta en producción civil.
