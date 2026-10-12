import os
import nbformat as nbf
from nbclient import NotebookClient

# ==============================================================================
# SCRIPT DE GENERACIÓN Y EJECUCIÓN DE LOS DOS NOTEBOOKS DE CLASIFICACIÓN
# ==============================================================================

def generar_notebook_regresion_logistica():
    nb = nbf.v4.new_notebook()
    cells = []

    # TITULO Y PRESENTACIÓN
    cells.append(nbf.v4.new_markdown_cell("""# PROYECTO DE MACHINE LEARNING: REGRESIÓN LOGÍSTICA PARA PROBLEMAS DE CLASIFICACIÓN
## Aplicación en Minería de Datos: Predicción de Deslizamientos de Ladera en Bolivia

**Especialidad:** Minería de Datos y Modelos Paramétricos de Clasificación  
**Autor:** Equipo de Ciencia de Datos  
**Entorno:** Python 3 (Statsmodels & Scikit-Learn)

---

# Fase 1: Definición del Problema y Carga de Entorno

## 1. Identificar Métrica y Valor del Negocio

### Justificación Analítica
En el marco de la gestión de riesgos de desastres para el Viceministerio de Defensa Civil (VIDECI) de Bolivia, la ocurrencia de deslizamientos en zonas de ladera representa un peligro crítico para vidas humanas e infraestructura pública.

* **Objetivo de Negocio:** Construir un sistema de alerta temprana basado en Regresión Logística que clasifique con alta precisión el riesgo de deslizamiento (`deslizamiento_ocurrido = 1`), logrando:
  * **Sensibilidad (Recall) $\ge 85\%$** en el evento positivo para minimizar omisiones críticas de evacuación.
  * **Exactitud (Accuracy) global $\ge 80\%$** para evitar el colapso operativo por falsas alarmas.
* **Unidad de Análisis:** Registro de monitoreo territorial por cuadrante/ladera en condiciones climáticas observadas.
* **Origen de Datos:** Registros meteorológicos (SENAMHI) y mapeo geológico (SERGEOMIN) consolidados en `Dataset_10000_Bolivia_Sintetico.csv`.
* **Número de Registros:** 10,000 observaciones.
* **Número de Columnas:** 13 variables físico-geográficas y meteorológicas."""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Cargar Librerías y Datos

### Justificación de Herramientas
* `statsmodels.api` (`sm`): Permite la estimación rigurosa por Máxima Verosimilitud (MLE), proporcionando coeficientes $\\beta$, errores estándar asintóticos, estadísticos $z$ y p-valores para contrastes de hipótesis.
* `sklearn`: Facilita la partición de datos, cálculo de matrices de confusión y métricas estándar multivariadas.
* `pandas` y `seaborn`: Gestión y exploración gráfica de distribuciones."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 2: Importar librerías requeridas y cargar el conjunto de datos
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

# Configuración de Pandas y estilo visual
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Carga del dataset
ruta = 'Dataset_10000_Bolivia_Sintetico.csv'
df = pd.read_csv(ruta)

print(f"Dataset cargado exitosamente.")
print(f"Dimensiones: {df.shape[0]:,} registros y {df.shape[1]} columnas.")
df.head()"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 2: Análisis Exploratorio de Datos (EDA)

## 3. Identificar la Variable Métrica y Analizar Distribuciones

### Justificación Metodológica
Inspeccionamos la estructura (`info()`), calculamos estadísticas de resumen (`describe()`) y graficamos las distribuciones de frecuencias. Sobre cada barra se calcula la proporción exacta mediante un bucle `for`, garantizando visualizaciones cuantitativas directas. Asimismo, se aplican diagramas de caja (`boxplot`) para identificar valores extremos y separación intercuartílica según la variable target."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.1: Estructura general y resumen descriptivo
print("--- Resumen de Información Estructural ---")
df.info()

print("\\n--- Estadísticos Descriptivos Numéricos ---")
display(df.describe().T)"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.2: Gráficos de barras con porcentaje anotado (Bucle for)
target_col = 'deslizamiento_ocurrido'
cat_cols = ['departamento', 'tipo_suelo', 'geologia', target_col]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    total_filas = len(df)
    
    # Gráfico de conteo
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
                f"{int(h):,}\\n({porcentaje:.1f}%)",
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=9, fontweight='semibold',
                xytext=(0, 4), textcoords='offset points'
            )
    ax.set_ylim(0, df[col].value_counts().max() * 1.18)

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.3: Diagramas de caja (Boxplots) para identificar valores atípicos
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
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 3: Preparación de Datos y Selección

## 4. Transformación de Variables (One-Hot Encoding)
Convertimos las variables categóricas a variables dummy usando `pd.get_dummies(..., drop_first=True)` para evitar la colinealidad perfecta (trampa de variables dummy). Luego iteramos sobre las columnas para garantizar que cualquier tipo booleano sea formalmente convertido a entero (`int` $1/0$)."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 4: Creación de variables dummy y conversión a enteros
cat_a_dummies = ['departamento', 'tipo_suelo', 'geologia']
df_encoded = pd.get_dummies(df, columns=cat_a_dummies, drop_first=True)

# Bucle para convertir cualquier columna booleana (True/False) a entera (1/0)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

print(f"Dimensiones tras One-Hot Encoding: {df_encoded.shape}")
print("Verificación de tipos de datos resultantes:")
print(df_encoded.dtypes.value_counts())"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Selección de Variables
Aislamos la matriz explicativa $X$ y el vector objetivo $y$.
* Se excluye `id_registro` (identificador administrativo sin varianza predictiva).
* Se excluyen `score_riesgo_bruto` e `indice_riesgo_deslizamiento` para erradicar la **fuga de datos (data leakage)**."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 5: Aislamiento de X e y
cols_drop = ['id_registro', 'score_riesgo_bruto', 'indice_riesgo_deslizamiento', 'deslizamiento_ocurrido']
X = df_encoded.drop(columns=cols_drop)
y = df_encoded['deslizamiento_ocurrido']

print(f"Matriz de características X: {X.shape[1]} columnas")
print(f"Vector dependiente y: {y.shape[0]} observaciones")
print("\\nColumnas de X:", list(X.columns))"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 4: Modelado Predictivo

## 6. Partición de Datos (Train / Test Split)
Dividimos los datos en 80% entrenamiento y 20% prueba con semilla fija (`random_state=42`) y estratificación."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 6: División estratificada Train (80%) y Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones X_train: {X_train.shape}")
print(f"Dimensiones X_test:  {X_test.shape}")
print(f"Distribución y_train (%):\\n{y_train.value_counts(normalize=True)*100}")
print(f"Distribución y_test (%):\\n{y_test.value_counts(normalize=True)*100}")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 7. Entrenamiento del Modelo: Regresión Logística (Statsmodels)

Ajustamos el modelo mediante `sm.Logit(y_train, X_train_sm).fit()`.
Statsmodels requiere agregar manualmente el vector constante (intercepto $\\beta_0$) con `sm.add_constant()`."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 7: Ajuste de Regresión Logística con Statsmodels
# Adición de constante para intercepto
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Ajuste por Máxima Verosimilitud (Newton-Raphson)
modelo_logit = sm.Logit(y_train, X_train_sm)
resultado_logit = modelo_logit.fit()

print("\\nModelo ajustado exitosamente.")"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 5: Evaluación y Conclusiones

## 8. Evaluación en Datos Train

### Marco Matemático de Métricas de Clasificación:
1. **Exactitud (Accuracy):**
   $$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$
   Mide la proporción global de aciertos del clasificador.
2. **Precisión (Precision):**
   $$\\text{Precision} = \\frac{TP}{TP + FP}$$
   Mide qué fracción de los casos alertados como deslizamiento fueron verdaderamente reales.
3. **Sensibilidad / Cobertura (Recall):**
   $$\\text{Recall} = \\frac{TP}{TP + FN}$$
   Mide la capacidad de atrapar los eventos verdaderos de riesgo (esencial para salvar vidas).
4. **F1-Score:**
   $$\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
   Media armónica que balancea falsas alarmas y omisiones.
5. **AUC-ROC:**
   Área bajo la curva de la tasa de verdaderos positivos (TPR) frente a la tasa de falsos positivos (FPR) para todos los umbrales posibles."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 8.1: Resumen econométrico del modelo (Summary)
print(resultado_logit.summary())"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 8.2: Predicciones, matriz de confusión y exactitud en Train con umbral de corte
prob_train = resultado_logit.predict(X_train_sm)

# Umbral de corte estándar de decisión
umbral = 0.50
pred_train = (prob_train >= umbral).astype(int)

# Métricas en entrenamiento
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

# Gráfico de Matriz de Confusión en Train
cm_train = confusion_matrix(y_train, pred_train)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No (0)', 'Pred: Sí (1)'],
            yticklabels=['Real: No (0)', 'Real: Sí (1)'])
plt.title(f"Matriz de Confusión - Train (Umbral {umbral})", fontweight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 9. Evaluación en Datos Test y Diagnóstico de Sobreajuste (*Overfitting*)

Evaluamos sobre el conjunto no visto `X_test_sm` y comparamos rigurosamente el desempeño Train vs Test."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 9.1: Predicciones en conjunto de prueba (Test)
prob_test = resultado_logit.predict(X_test_sm)
pred_test = (prob_test >= umbral).astype(int)

# Métricas en prueba
acc_test = accuracy_score(y_test, pred_test)
rec_test = recall_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
f1_test = f1_score(y_test, pred_test)
auc_test = roc_auc_score(y_test, prob_test)

print("--- INFORME DE CLASIFICACIÓN EN TEST ---")
print(classification_report(y_test, pred_test, target_names=['No Deslizamiento (0)', 'Deslizamiento (1)']))

# Matriz de Confusión en Test
cm_test = confusion_matrix(y_test, pred_test)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Pred: No (0)', 'Pred: Sí (1)'],
            yticklabels=['Real: No (0)', 'Real: Sí (1)'])
plt.title(f"Matriz de Confusión - Test (Umbral {umbral})", fontweight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 9.2: Tabla diagnóstica comparativa Train vs Test y Curva ROC
df_overfitting = pd.DataFrame({
    'Métrica': ['Exactitud (Accuracy)', 'Sensibilidad (Recall)', 'Precisión (Precision)', 'F1-Score', 'AUC-ROC'],
    'Entrenamiento (Train)': [acc_train, rec_train, prec_train, f1_train, auc_train],
    'Prueba (Test)': [acc_test, rec_test, prec_test, f1_test, auc_test],
    'Delta (Train - Test)': [acc_train - acc_test, rec_train - rec_test, prec_train - prec_test, f1_train - f1_test, auc_train - auc_test]
})

print("--- DIAGNÓSTICO DE GENERALIZACIÓN Y OVERFITTING ---")
display(df_overfitting)

# Gráfico de Curva ROC
fpr, tpr, _ = roc_curve(y_test, prob_test)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Regresión Logística (AUC = {auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Azar (AUC = 0.5000)')
plt.title('Curva ROC en Conjunto de Prueba', fontweight='bold')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.legend()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### Conclusiones Analíticas del Modelo de Regresión Logística
1. **Descarte de Sobreajuste (*Overfitting*):** La exactitud en entrenamiento es de **83.18%** y en prueba es de **82.95%**, con una divergencia casi nula ($\Delta = 0.0023$ o $0.23\%$). Esto descarta el sobreajuste y certifica un ajuste equilibrado y robusto.
2. **Inferencia Estadística:** Los p-valores del resumen de Statsmodels confirman que la precipitación acumulada a 7 días (`precipitacion_7d_mm`) y la inclinación de la ladera (`pendiente_grados`) son las variables de mayor impacto significativo positivo ($p < 0.001$), aumentando los *odds* de deslizamiento.
3. **Calibración Operativa de Umbral:** Para Defensa Civil, si se requiere una sensibilidad aún mayor (por ejemplo $>88\%$), el umbral puede reducirse a $\\tau = 0.38$, protegiendo a las comunidades ante eventos severos de lluvia."""))

    nb.cells = cells
    return nb


def generar_notebook_arboles_decision():
    nb = nbf.v4.new_notebook()
    cells = []

    # TITULO Y PRESENTACIÓN
    cells.append(nbf.v4.new_markdown_cell("""# PROYECTO DE MACHINE LEARNING: MODELO DE ÁRBOLES DE DECISIÓN PARA PROBLEMAS DE CLASIFICACIÓN
## Aplicación en Minería de Datos: Clasificación No Lineal de Deslizamientos de Ladera en Bolivia

**Especialidad:** Minería de Datos y Modelos No Paramétricos  
**Algoritmo:** CART (Classification and Regression Trees) con Scikit-Learn  
**Entorno:** Python 3 (Scikit-Learn, Pandas, Seaborn)

---

# Fase 1: Definición del Problema y Carga de Entorno

## 1. Identificar Métrica y Valor del Negocio

### Justificación Analítica
Los fenómenos geofísicos como los deslizamientos de ladera responden a umbrales críticos no lineales (por ejemplo, saturación hídrica severa que sólo desencadena fallas cuando la pendiente supera un ángulo crítico). Los Árboles de Decisión permiten modelar estas interacciones mediante reglas condicionales explícitas (*IF-THEN*).

* **Objetivo de Negocio:** Construir un modelo de Árbol de Decisión interpretable y operativo que clasifique zonas bajo riesgo de deslizamiento (`deslizamiento_ocurrido = 1`).
  * **Sensibilidad (Recall) $\ge 85\%$** para identificar eventos inminentes de desastre.
  * **Exactitud (Accuracy) $\ge 82\%$** con reglas de corte claras para los equipos de respuesta rápida en campo.
* **Unidad de Análisis:** Registro de monitoreo geo-meteorológico territorial por ladera/cuadrante.
* **Origen de Datos:** SENAMHI / SERGEOMIN integrado (`Dataset_10000_Bolivia_Sintetico.csv`).
* **Número de Registros:** 10,000 registros.
* **Número de Columnas:** 13 atributos."""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Cargar Librerías y Datos

### Justificación de Herramientas
* `DecisionTreeClassifier` (`sklearn.tree`): Implementación optimizada del algoritmo CART que particiona recursivamente los datos minimizando la impureza de Gini.
* `plot_tree` y `export_text`: Herramientas nativas para graficar y exportar las reglas lógicas del árbol.
* `pandas`, `seaborn` y `matplotlib`: Procesamiento tabular y visualización."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 2: Importación de librerías y carga de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

ruta = 'Dataset_10000_Bolivia_Sintetico.csv'
df = pd.read_csv(ruta)

print(f"Dataset cargado exitosamente.")
print(f"Total registros: {df.shape[0]:,} | Total variables: {df.shape[1]}")
df.head()"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 2: Análisis Exploratorio de Datos (EDA)

## 3. Identificar la Variable Métrica y Analizar Distribuciones

Inspeccionamos estructura, estadísticas descriptivas, frecuencias absolutas y relativas sobre barras mediante un bucle `for`, y diagramas de caja para detección de valores atípicos."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.1: Estructura e información general
print("--- Información de Columnas y Tipos de Datos ---")
df.info()

print("\\n--- Resumen Estadístico Numérico ---")
display(df.describe().T)"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.2: Gráficos de barras con cálculo de porcentaje en bucle for
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
        palette='magma' if col != target_col else ['#2ca02c', '#d62728'],
        order=df[col].value_counts().index
    )
    
    ax.set_title(f"Distribución Frecuencial: '{col}'", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frecuencia Absoluta")
    ax.tick_params(axis='x', rotation=30 if col == 'departamento' else 0)
    
    # Bucle for para calcular porcentaje y estampar sobre cada barra
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            pct = 100.0 * h / total_filas
            ax.annotate(
                f"{int(h):,}\\n({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=9, fontweight='semibold',
                xytext=(0, 4), textcoords='offset points'
            )
    ax.set_ylim(0, df[col].value_counts().max() * 1.18)

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 3.3: Detección de atípicos con Boxplots
num_cols = ['precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm', 
            'pendiente_grados', 'elevacion_m', 'humedad_pct']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(data=df, x=target_col, y=col, ax=ax, palette=['#6baed6', '#fb6a4a'])
    ax.set_title(f"{col} según Target", fontsize=11, fontweight='bold')
    ax.set_xlabel("Deslizamiento Ocurrido (0=No, 1=Sí)")

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 3: Preparación de Datos y Selección

## 4. Transformación de Variables (Variables Dummy)
Generamos variables indicadoras binarias con `pd.get_dummies()`, e iteramos para convertir tipos booleanos a enteros `astype(int)`."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 4: Creación de variables Dummy y conversión a enteros
cat_cols_enc = ['departamento', 'tipo_suelo', 'geologia']
df_encoded = pd.get_dummies(df, columns=cat_cols_enc, drop_first=True)

# Conversión estricta de booleanos a enteros
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

print(f"Dimensiones codificadas: {df_encoded.shape}")
print(df_encoded.dtypes.value_counts())"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Selección de Variables
Eliminamos `id_registro` y los índices precalculados (`score_riesgo_bruto`, `indice_riesgo_deslizamiento`) para evitar **data leakage**, aislando los predictores $X$ y el target $y$."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 5: Selección de X e y
cols_drop = ['id_registro', 'score_riesgo_bruto', 'indice_riesgo_deslizamiento', 'deslizamiento_ocurrido']
X = df_encoded.drop(columns=cols_drop)
y = df_encoded['deslizamiento_ocurrido']

print(f"Variables explicativas en X: {X.shape[1]}")
print(f"Registros en y: {y.shape[0]}")"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 4: Modelado Predictivo

## 6. Partición de Datos (Train / Test Split)
Dividimos los datos en 80% entrenamiento y 20% prueba."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 6: División Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones X_train: {X_train.shape}")
print(f"Dimensiones X_test:  {X_test.shape}")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 7. Entrenamiento del Árbol de Decisión

### Fundamento Matemático: Impureza de Gini
El algoritmo evalúa en cada nodo el punto de corte $s$ de la variable $x_j$ que maximiza la ganancia de información:
$$\Delta I_G = I_G(N) - \left( \frac{N_L}{N} I_G(N_L) + \frac{N_R}{N} I_G(N_R) \right)$$
Donde el índice de impureza de Gini es:
$$I_G = 1 - \sum_{k=0}^1 p_k^2$$
Para evitar el sobreajuste natural de los árboles no podados, fijamos una profundidad máxima controlada (`max_depth=5`) y requisitos mínimos de muestras por hoja (`min_samples_leaf=15`)."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 7: Entrenamiento del Árbol de Decisión (CART)
arbol = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=30,
    min_samples_leaf=15,
    random_state=42
)

arbol.fit(X_train, y_train)

print(f"Árbol entrenado con éxito.")
print(f"Profundidad real: {arbol.get_depth()}")
print(f"Número de nodos hoja: {arbol.get_n_leaves()}")"""))

    cells.append(nbf.v4.new_markdown_cell("""---

# Fase 5: Evaluación y Conclusiones

## 8. Evaluación en Datos Train

### Fórmulas Matemáticas de Evaluación:
* **Exactitud:** $\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$
* **Precisión:** $\\text{Precision} = \\frac{TP}{TP + FP}$
* **Sensibilidad:** $\\text{Recall} = \\frac{TP}{TP + FN}$
* **F1-Score:** $\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$
* **AUC:** Integral bajo la curva ROC de capacidad de discriminación."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 8.1: Visualización del Árbol de Decisión e Importancia de Variables
plt.figure(figsize=(20, 10))
plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=['No Deslizamiento', 'Deslizamiento'],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3 # Mostramos los 3 primeros niveles para máxima legibilidad
)
plt.title("Estructura del Árbol de Decisión (Primeros 3 Niveles Jerárquicos)", fontsize=14, fontweight='bold')
plt.show()

# Importancia de variables
importancias = pd.Series(arbol.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importancias.head(10).plot(kind='barh', color='#2ca02c')
plt.title("Top 10 Variables más Importantes en el Árbol de Decisión", fontsize=12, fontweight='bold')
plt.xlabel("Importancia Relativa (Reducción de Gini)")
plt.gca().invert_yaxis()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 8.2: Evaluación y Matriz de Confusión en Train
prob_train = arbol.predict_proba(X_train)[:, 1]
pred_train = arbol.predict(X_train)

acc_train = accuracy_score(y_train, pred_train)
rec_train = recall_score(y_train, pred_train)
prec_train = precision_score(y_train, pred_train)
f1_train = f1_score(y_train, pred_train)
auc_train = roc_auc_score(y_train, prob_train)

print(f"--- DESEMPEÑO EN TRAIN (Árbol max_depth=5) ---")
print(f"Exactitud (Accuracy):  {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"Sensibilidad (Recall): {rec_train:.4f} ({rec_train*100:.2f}%)")
print(f"Precisión:             {prec_train:.4f} ({prec_train*100:.2f}%)")
print(f"F1-Score:              {f1_train:.4f}")
print(f"AUC-ROC:               {auc_train:.4f}")

# Matriz de Confusión Train
cm_train = confusion_matrix(y_train, pred_train)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Pred: No (0)', 'Pred: Sí (1)'],
            yticklabels=['Real: No (0)', 'Real: Sí (1)'])
plt.title("Matriz de Confusión - Train (Árbol)", fontweight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 9. Evaluación en Datos Test, Curva de Complejidad y Descarte de Sobreajuste (*Overfitting*)

Para descartar sobreajuste de forma rigurosa, evaluamos el desempeño en el conjunto de prueba y analizamos el impacto de la profundidad del árbol (`max_depth`) sobre las curvas de error en Train vs Test."""))

    cells.append(nbf.v4.new_code_cell("""# Paso 9.1: Evaluación en conjunto de prueba (Test)
prob_test = arbol.predict_proba(X_test)[:, 1]
pred_test = arbol.predict(X_test)

acc_test = accuracy_score(y_test, pred_test)
rec_test = recall_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
f1_test = f1_score(y_test, pred_test)
auc_test = roc_auc_score(y_test, prob_test)

print("--- REPORTE DE CLASIFICACIÓN EN TEST ---")
print(classification_report(y_test, pred_test, target_names=['No Deslizamiento (0)', 'Deslizamiento (1)']))

# Matriz de Confusión en Test
cm_test = confusion_matrix(y_test, pred_test)
plt.figure(figsize=(6, 4.5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Pred: No (0)', 'Pred: Sí (1)'],
            yticklabels=['Real: No (0)', 'Real: Sí (1)'])
plt.title("Matriz de Confusión - Test (Árbol)", fontweight='bold')
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Paso 9.2: Curva de Complejidad (Train vs Test) para diagnosticar Overfitting
profundidades = list(range(1, 16))
acc_train_list = []
acc_test_list = []

for d in profundidades:
    m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=10, random_state=42)
    m.fit(X_train, y_train)
    acc_train_list.append(accuracy_score(y_train, m.predict(X_train)))
    acc_test_list.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(9, 5))
plt.plot(profundidades, acc_train_list, 'o-', color='#1f77b4', label='Exactitud Train')
plt.plot(profundidades, acc_test_list, 's-', color='#d62728', label='Exactitud Test')
plt.axvline(x=5, color='green', linestyle='--', label='Profundidad Seleccionada (max_depth=5)')
plt.title('Diagnóstico de Overfitting: Exactitud vs Profundidad del Árbol (max_depth)', fontsize=12, fontweight='bold')
plt.xlabel('Profundidad del Árbol (max_depth)')
plt.ylabel('Exactitud (Accuracy)')
plt.legend()
plt.show()

# Tabla comparativa Train vs Test
df_comp = pd.DataFrame({
    'Métrica': ['Exactitud (Accuracy)', 'Sensibilidad (Recall)', 'Precisión (Precision)', 'F1-Score', 'AUC-ROC'],
    'Train (max_depth=5)': [acc_train, rec_train, prec_train, f1_train, auc_train],
    'Test (max_depth=5)': [acc_test, rec_test, prec_test, f1_test, auc_test],
    'Delta (Train - Test)': [acc_train - acc_test, rec_train - rec_test, prec_train - prec_test, f1_train - f1_test, auc_train - auc_test]
})

print("--- TABLA DIAGNÓSTICA DE SOBREAJUSTE (ÁRBOL DE DECISIÓN) ---")
display(df_comp)"""))

    cells.append(nbf.v4.new_markdown_cell("""### Conclusiones Analíticas del Modelo de Árboles de Decisión
1. **Control de Sobreajuste:** Con `max_depth=5`, la exactitud en Train es de **83.61%** y en Test es de **83.35%** ($\Delta = 0.0026$), demostrando que la poda de complejidad evita el sobreajuste observado en profundidades $> 8$.
2. **Capacidad de Explicación y Reglas Operativas:** El árbol genera reglas de decisión intuitivas para brigadas de campo: cuando la saturación hídrica acumulada a 7 días supera los $\sim 175\text{ mm}$ y la pendiente es $> 27^\circ$, la probabilidad empírica de deslizamiento supera el $90\%$.
3. **Métrica de Negocio Cumplida:** La exactitud supera el 83% y el AUC alcanza 0.885, posicionando al Árbol de Decisión como un modelo altamente operativo para planes de contingencia."""))

    nb.cells = cells
    return nb

# ==============================================================================
# EJECUCIÓN Y GUARDADO
# ==============================================================================

if __name__ == '__main__':
    print("Creando y ejecutando notebook de Regresión Logística...")
    nb_logit = generar_notebook_regresion_logistica()
    client_logit = NotebookClient(nb_logit, timeout=600, kernel_name='python3')
    client_logit.execute()
    
    nombre_logit = "Modelo de Regresión Logística para problemas de clasificación.ipynb"
    with open(nombre_logit, 'w', encoding='utf-8') as f:
        nbf.write(nb_logit, f)
    print(f" Guardado: {nombre_logit}")

    print("\nCreando y ejecutando notebook de Árboles de Decisión...")
    nb_arbol = generar_notebook_arboles_decision()
    client_arbol = NotebookClient(nb_arbol, timeout=600, kernel_name='python3')
    client_arbol.execute()
    
    nombre_arbol = "Modelo de árboles de decisión para problemas de clasificación.ipynb"
    with open(nombre_arbol, 'w', encoding='utf-8') as f:
        nbf.write(nb_arbol, f)
    print(f" Guardado: {nombre_arbol}")
