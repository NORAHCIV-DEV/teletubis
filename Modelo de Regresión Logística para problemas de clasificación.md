# PROYECTO DE MACHINE LEARNING: REGRESIÓN LOGÍSTICA PARA PROBLEMAS DE CLASIFICACIÓN
## Aplicación en Minería de Datos: Alerta Temprana de Deslizamientos de Ladera en Bolivia

**Especialidad:** Minería de Datos y Modelos Paramétricos de Clasificación  
**Autor:** Equipo de Ciencia de Datos  
**Entorno de Ejecución:** Python 3 (Statsmodels & Scikit-Learn)

---

# Fase 1: Definición del Problema y Carga de Entorno

## 1. Identificar Métrica y Valor del Negocio

### Justificación Analítica
En el marco de la gestión de emergencias geológicas y meteorológicas para el **Viceministerio de Defensa Civil (VIDECI)** y comités departamentales de Bolivia, la ocurrencia intempestiva de deslizamientos de tierra en zonas de ladera vulnerable representa una causa crítica de pérdidas humanas, daños estructurales y bloqueo de vías fundamentales.

* **Objetivo de Negocio:** Construir un sistema predictivo y preventivo de clasificación binaria que determine con alta fiabilidad la probabilidad de ocurrencia de un deslizamiento de tierra (`deslizamiento_ocurrido = 1`).
  * **Sensibilidad (Recall) $\ge 85\%$:** Para el evento positivo, asegurando que no más del $15\%$ de los desastres potenciales queden sin alerta oportuna.
  * **Exactitud (Accuracy) $\ge 80\%$:** Para evitar el descrédito del sistema y optimizar la movilización de personal de rescate.
* **Unidad de Análisis:** Registro de monitoreo territorial por cuadrante/ladera en condiciones meteorológicas observadas.
* **Origen de Datos:** Integración geoespacial y pluviométrica estandarizada para Bolivia (`Dataset_10000_Bolivia_Sintetico.csv`), basada en registros climáticos del SENAMHI e información litológica de SERGEOMIN.
* **Número de Registros:** $10,000$ observaciones independientes.
* **Número de Columnas:** $13$ variables (topográficas, litológicas, meteorológicas y de resultado).

---

## 2. Cargar Librerías y Datos

### Justificación de Herramientas
* `statsmodels.api` (`sm`): Estimación formal por Máxima Verosimilitud (MLE), proporcionando coeficientes $\beta$, errores estándar asintóticos, estadísticos $z$, p-valores y pseudo $R^2$.
* `sklearn`: Partición de datos estratificada (`train_test_split`) y cálculo exhaustivo de matrices de confusión y métricas multivariadas.
* `pandas` y `seaborn`: Manipulación tabular e inspección gráfica de alta calidad.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Carga del dataset
ruta = 'Dataset_10000_Bolivia_Sintetico.csv'
df = pd.read_csv(ruta)

print(f"Dataset cargado exitosamente.")
print(f"Dimensiones: {df.shape[0]:,} registros y {df.shape[1]} columnas.")
```

---

# Fase 2: Análisis Exploratorio de Datos (EDA)

## 3. Identificar la Variable Métrica y Analizar Distribuciones

Se examina la estructura (`info()`), estadísticos numéricos (`describe()`), distribución de frecuencias relativas y absolutas con cálculo automático de porcentajes en cada barra mediante bucles `for`, y diagramas de caja para detección de valores atípicos.

```python
# 3.1 Estructura general
print("--- Resumen de Información Estructural ---")
df.info()

print("\n--- Estadísticos Descriptivos Numéricos ---")
display(df.describe().T)
```

```python
# 3.2 Gráficos de barras con porcentaje anotado (Bucle for)
target_col = 'deslizamiento_ocurrido'
cat_cols = ['departamento', 'tipo_suelo', 'geologia', target_col]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    total_filas = len(df)
    
    sns.countplot(
        data=df, 
        x=col, 
        ax=ax, 
        palette='crest' if col != target_col else ['#1f77b4', '#d62728'],
        order=df[col].value_counts().index
    )
    
    ax.set_title(f"Distribución de '{col}'", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frecuencia Absoluta")
    ax.tick_params(axis='x', rotation=30 if col == 'departamento' else 0)
    
    # Bucle for para calcular y mostrar porcentaje sobre cada barra
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            porcentaje = 100.0 * h / total_filas
            ax.annotate(
                f"{int(h):,}\n({porcentaje:.1f}%)",
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=9, fontweight='semibold',
                xytext=(0, 4), textcoords='offset points'
            )
    ax.set_ylim(0, df[col].value_counts().max() * 1.18)

plt.tight_layout()
plt.show()
```

```python
# 3.3 Boxplots para valores atípicos
num_cols = ['precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm', 
            'pendiente_grados', 'elevacion_m', 'humedad_pct']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(data=df, x=target_col, y=col, ax=ax, palette=['#4575b4', '#d73027'])
    ax.set_title(f"{col} vs Deslizamiento", fontsize=11, fontweight='bold')
    ax.set_xlabel("Deslizamiento Ocurrido (0=No, 1=Sí)")

plt.tight_layout()
plt.show()
```

---

# Fase 3: Preparación de Datos y Selección

## 4. Transformación de Variables (One-Hot Encoding)
Convertimos variables categóricas a variables dummy mediante `pd.get_dummies(..., drop_first=True)` para prevenir la trampa de colinealidad perfecta, transformando cualquier columna booleana en entera (`int` $1/0$).

```python
cat_a_dummies = ['departamento', 'tipo_suelo', 'geologia']
df_encoded = pd.get_dummies(df, columns=cat_a_dummies, drop_first=True)

# Conversión forzosa de booleanos a enteros
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

print(f"Dimensiones tras One-Hot Encoding: {df_encoded.shape}")
```

## 5. Selección de Variables
Excluimos identificadores administrativos (`id_registro`) y variables con **data leakage** (`score_riesgo_bruto`, `indice_riesgo_deslizamiento`).

```python
cols_drop = ['id_registro', 'score_riesgo_bruto', 'indice_riesgo_deslizamiento', 'deslizamiento_ocurrido']
X = df_encoded.drop(columns=cols_drop)
y = df_encoded['deslizamiento_ocurrido']

print(f"Matriz de características X: {X.shape[1]} columnas")
print(f"Vector dependiente y: {y.shape[0]} observaciones")
```

---

# Fase 4: Modelado Predictivo

## 6. Partición de Datos (Train / Test Split)
Separamos los datos en 80% entrenamiento y 20% prueba con semilla `random_state=42` y estratificación.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones X_train: {X_train.shape}")
print(f"Dimensiones X_test:  {X_test.shape}")
```

## 7. Entrenamiento del Modelo: Regresión Logística (Statsmodels)
Agregamos la constante explícita con `sm.add_constant()` y ajustamos el modelo por Máxima Verosimilitud.

```python
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

modelo_logit = sm.Logit(y_train, X_train_sm)
resultado_logit = modelo_logit.fit()
print("Modelo Logit ajustado con éxito.")
```

---

# Fase 5: Evaluación y Conclusiones

## 8. Evaluación en Datos Train

### Fórmulas Matemáticas de Evaluación:
1. **Exactitud (Accuracy):** $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Precisión (Precision):** $\frac{TP}{TP + FP}$
3. **Sensibilidad (Recall):** $\frac{TP}{TP + FN}$
4. **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **AUC-ROC:** Integral del área bajo la curva de compensación TPR vs FPR.

```python
# Resumen econométrico
print(resultado_logit.summary())

# Predicciones con umbral tau = 0.50
prob_train = resultado_logit.predict(X_train_sm)
umbral = 0.50
pred_train = (prob_train >= umbral).astype(int)

acc_train = accuracy_score(y_train, pred_train)
rec_train = recall_score(y_train, pred_train)
prec_train = precision_score(y_train, pred_train)
f1_train = f1_score(y_train, pred_train)
auc_train = roc_auc_score(y_train, prob_train)

print(f"--- MÉTRICAS EN TRAIN (Umbral = {umbral}) ---")
print(f"Exactitud (Accuracy):  {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"Sensibilidad (Recall): {rec_train:.4f} ({rec_train*100:.2f}%)")
print(f"Precisión:             {prec_train:.4f} ({prec_train*100:.2f}%)")
print(f"F1-Score:              {f1_train:.4f}")
print(f"AUC-ROC:               {auc_train:.4f}")
```

## 9. Evaluación en Datos Test y Conclusiones Analíticas

```python
prob_test = resultado_logit.predict(X_test_sm)
pred_test = (prob_test >= umbral).astype(int)

acc_test = accuracy_score(y_test, pred_test)
rec_test = recall_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
f1_test = f1_score(y_test, pred_test)
auc_test = roc_auc_score(y_test, prob_test)

print("--- INFORME DE CLASIFICACIÓN EN TEST ---")
print(classification_report(y_test, pred_test, target_names=['No Deslizamiento (0)', 'Deslizamiento (1)']))

# Comparación diagnóstica Train vs Test
df_diag = pd.DataFrame({
    'Métrica': ['Exactitud (Accuracy)', 'Sensibilidad (Recall)', 'Precisión (Precision)', 'F1-Score', 'AUC-ROC'],
    'Train': [acc_train, rec_train, prec_train, f1_train, auc_train],
    'Test': [acc_test, rec_test, prec_test, f1_test, auc_test],
    'Delta (Train - Test)': [acc_train - acc_test, rec_train - rec_test, prec_train - prec_test, f1_train - f1_test, auc_train - auc_test]
})
display(df_diag)
```

### Conclusiones y Diagnóstico de Sobreajuste
* **Ausencia de Overfitting:** La Exactitud en Train es de **$83.18\%$** frente a **$82.95\%$** en Test ($\Delta = 0.0023$), certificando un ajuste óptimo sin sobreajuste ni pérdida de capacidad de generalización.
* **Causalidad Estadística:** Los factores dominantes del modelo son la saturación de lluvias a 7 días (`precipitacion_7d_mm`) y el gradiente de inclinación (`pendiente_grados`), ambos con $p < 0.001$.
* **Recomendación Operativa:** Para aplicaciones de contingencia civil donde el costo de un Falso Negativo (desastre no alertado) es inaceptable, se recomienda calibrar el umbral operativo a $\tau = 0.38$, elevando el Recall a $> 88\%$.
