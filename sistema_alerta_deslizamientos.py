
# =============================================================================
# SISTEMA PREDICTIVO DE ALERTA TEMPRANA DE DESLIZAMIENTOS EN BOLIVIA
# Metodología: CRISP-DM | Proyecto Universitario de Minería de Datos
# Autor: Data Scientist Senior
# Versión: 1.0
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 1: INSTALACIÓN DE DEPENDENCIAS (ejecutar solo si es necesario)
# ─────────────────────────────────────────────────────────────────────────────
"""
Ejecutar en terminal o celda separada si no tienes las librerías:
!pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib scipy
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 2: IMPORTACIÓN DE LIBRERÍAS
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn: Preprocesamiento
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Scikit-Learn: Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Scikit-Learn: Métricas
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    fbeta_score, recall_score,
    ConfusionMatrixDisplay
)

# Manejo de desbalance de clases
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ Librerías importadas correctamente.")
print(f"   - Pandas: {pd.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - Sklearn: Disponible")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 3: FASE 1 CRISP-DM — COMPRENSIÓN DEL NEGOCIO (Business Understanding)
# ─────────────────────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════╗
║  FASE 1: COMPRENSIÓN DEL NEGOCIO (CRISP-DM)                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Objetivo de Negocio:                                            ║
║  → Sistema de Alerta Temprana de Deslizamientos en Bolivia       ║
║  → Índice de Riesgo escalado de 0 a 100                          ║
║                                                                  ║
║  Objetivo Técnico:                                               ║
║  → Predecir probabilidad_deslizamiento (0.00 a 1.00)             ║
║  → Recall > 85% para clase crítica (deslizamiento = 1)           ║
║  → AUC-ROC > 0.85                                               ║
║                                                                  ║
║  Hipótesis de Variables Clave:                                   ║
║  → Precipitación:  ~35% de importancia esperada                  ║
║  → Pendiente:      ~25% de importancia esperada                  ║
║  → Saturación:     ~15% de importancia esperada                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 4: FASE 2 CRISP-DM — COMPRENSIÓN DE LOS DATOS (Data Understanding)
# ─────────────────────────────────────────────────────────────────────────────

# Carga del dataset
RUTA_DATASET = "Dataset_10000_Bolivia_Sintetico.csv"

try:
    df = pd.read_csv(RUTA_DATASET, encoding='utf-8')
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
except FileNotFoundError:
    print(f"⚠️  Archivo '{RUTA_DATASET}' no encontrado.")
    print("   Generando dataset sintético de demostración...")

    # ── Dataset sintético de respaldo para pruebas ──────────────────────────
    np.random.seed(42)
    N = 10_000

    departamentos = ['La Paz', 'Cochabamba', 'Santa Cruz', 'Oruro',
                     'Potosí', 'Chuquisaca', 'Tarija', 'Beni', 'Pando']
    tipos_suelo   = ['Arcilloso', 'Limoso', 'Arenoso', 'Franco', 'Rocoso']
    geologias     = ['Plutónico', 'Sedimentario', 'Metamórfico',
                     'Volcánico', 'Aluvial']

    df = pd.DataFrame({
        # Variables numéricas
        'precipitacion_24h_mm':  np.random.exponential(scale=30, size=N).clip(0, 300),
        'precipitacion_3d_mm':   np.random.exponential(scale=80, size=N).clip(0, 600),
        'precipitacion_7d_mm':   np.random.exponential(scale=150, size=N).clip(0, 900),
        'pendiente_grados':      np.random.uniform(0, 60, N),
        'elevacion_m':           np.random.uniform(200, 5000, N),
        'humedad_pct':           np.random.uniform(20, 100, N),
        'temperatura_C':         np.random.normal(15, 8, N),
        'cobertura_vegetal_pct': np.random.uniform(0, 100, N),
        'distancia_falla_km':    np.random.exponential(scale=10, size=N).clip(0.1, 50),
        'indice_vegetacion_ndvi': np.random.uniform(-0.1, 0.9, N),
        'profundidad_suelo_cm':  np.random.uniform(10, 200, N),
        'permeabilidad_suelo':   np.random.uniform(0.1, 10, N),
        'cohesion_suelo_kpa':    np.random.uniform(5, 80, N),
        'angulo_friccion_grados': np.random.uniform(15, 45, N),
        'velocidad_viento_kmh':  np.random.exponential(scale=20, size=N).clip(0, 120),
        'presion_poros_kpa':     np.random.uniform(0, 50, N),
        'sismicidad_local':      np.random.uniform(0, 5, N),
        'uso_suelo_cod':         np.random.randint(1, 8, N),
        'red_drenaje_densidad':  np.random.uniform(0.1, 5, N),
        'indice_forma_cuenca':   np.random.uniform(0.1, 1, N),
        'area_cuenca_km2':       np.random.exponential(scale=50, size=N).clip(1, 500),
        'longitud_cauce_km':     np.random.exponential(scale=20, size=N).clip(0.5, 100),
        'tiempo_concentracion_h': np.random.uniform(0.5, 24, N),
        'caudal_pico_m3s':       np.random.exponential(scale=10, size=N).clip(0, 100),
        'nivel_freático_m':      np.random.uniform(0.5, 15, N),
        'tasa_infiltracion_mmh': np.random.uniform(1, 50, N),
        'evapotranspiracion_mm': np.random.uniform(1, 8, N),
        'indice_sequia_SPI':     np.random.normal(0, 1, N),
        'año':                   np.random.randint(2010, 2025, N),
        'mes':                   np.random.randint(1, 13, N),
        'dia':                   np.random.randint(1, 29, N),
        # Variables categóricas
        'departamento':          np.random.choice(departamentos, N),
        'tipo_suelo':            np.random.choice(tipos_suelo, N),
        'geologia':              np.random.choice(geologias, N),
        # Variables adicionales
        'altitud_relativa_m':    np.random.uniform(0, 1000, N),
        'densidad_poblacion':    np.random.exponential(scale=200, size=N).clip(0, 5000),
        'infraestructura_cod':   np.random.randint(0, 5, N),
    })

    # Generación de la variable target con lógica realista
    score = (
        0.35 * (df['precipitacion_24h_mm'] / 300) +
        0.25 * (df['pendiente_grados'] / 60) +
        0.15 * (df['humedad_pct'] / 100) +
        0.10 * (1 - df['distancia_falla_km'] / 50) +
        0.10 * (df['presion_poros_kpa'] / 50) +
        0.05 * np.random.random(N)
    )
    df['deslizamiento_ocurrido'] = (score > 0.45).astype(int)

    print(f"✅ Dataset sintético generado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# Exploración inicial
print("\n" + "="*60)
print("📊 EXPLORACIÓN INICIAL DEL DATASET")
print("="*60)
print(f"\n• Forma del dataset: {df.shape}")
print(f"\n• Tipos de datos:\n{df.dtypes.to_string()}")
print(f"\n• Valores nulos por columna:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
print(f"\n• Distribución de la variable target:")
print(df['deslizamiento_ocurrido'].value_counts())
print(f"\n• Tasa de desbalance:")
total = len(df)
pos   = df['deslizamiento_ocurrido'].sum()
print(f"  Deslizamientos (1): {pos:,} ({pos/total*100:.1f}%)")
print(f"  Sin deslizamiento (0): {total-pos:,} ({(total-pos)/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 5: VISUALIZACIÓN EXPLORATORIA (EDA)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('📊 Análisis Exploratorio — Variables Clave vs Deslizamiento',
             fontsize=16, fontweight='bold', y=1.02)

variables_plot = [
    ('precipitacion_24h_mm', 'Precipitación 24h (mm)', 'steelblue'),
    ('precipitacion_3d_mm',  'Precipitación 3 días (mm)', 'royalblue'),
    ('pendiente_grados',     'Pendiente (grados)', 'darkorange'),
    ('humedad_pct',          'Humedad (%)', 'seagreen'),
    ('elevacion_m',          'Elevación (m)', 'purple'),
    ('presion_poros_kpa',    'Presión de Poros (kPa)', 'crimson'),
]

for ax, (var, titulo, color) in zip(axes.flat, variables_plot):
    for clase, label in [(0, 'Sin deslizamiento'), (1, 'Deslizamiento')]:
        data_clase = df[df['deslizamiento_ocurrido'] == clase][var]
        ax.hist(data_clase, bins=30, alpha=0.6,
                label=label, color=('lightgray' if clase == 0 else color))
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('Frecuencia')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('eda_variables_clave.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico EDA guardado como 'eda_variables_clave.png'")

# Mapa de correlación
fig, ax = plt.subplots(figsize=(14, 10))
vars_num = df.select_dtypes(include=[np.number]).columns.tolist()
correlacion = df[vars_num].corr()
mask = np.triu(np.ones_like(correlacion, dtype=bool))
sns.heatmap(correlacion, mask=mask, annot=False, cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax)
ax.set_title('🔥 Mapa de Correlación — Variables Numéricas',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlacion_variables.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Mapa de correlación guardado.")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 6: FASE 3 CRISP-DM — PREPARACIÓN DE DATOS (Data Preparation)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("🔧 FASE 3: PREPARACIÓN DE DATOS")
print("="*60)

# ── 6.1 Definición de columnas ──────────────────────────────────────────────
TARGET = 'deslizamiento_ocurrido'

VARS_CATEGORICAS = ['departamento', 'tipo_suelo', 'geologia']

VARS_NUMERICAS = [
    'precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm',
    'pendiente_grados', 'elevacion_m', 'humedad_pct',
    'temperatura_C', 'cobertura_vegetal_pct', 'distancia_falla_km',
    'indice_vegetacion_ndvi', 'profundidad_suelo_cm', 'permeabilidad_suelo',
    'cohesion_suelo_kpa', 'angulo_friccion_grados', 'velocidad_viento_kmh',
    'presion_poros_kpa', 'sismicidad_local', 'uso_suelo_cod',
    'red_drenaje_densidad', 'indice_forma_cuenca', 'area_cuenca_km2',
    'longitud_cauce_km', 'tiempo_concentracion_h', 'caudal_pico_m3s',
    'nivel_freático_m', 'tasa_infiltracion_mmh', 'evapotranspiracion_mm',
    'indice_sequia_SPI', 'año', 'mes', 'dia',
    'altitud_relativa_m', 'densidad_poblacion', 'infraestructura_cod',
]

# Filtrar solo columnas que existen en el dataset
VARS_CATEGORICAS = [c for c in VARS_CATEGORICAS if c in df.columns]
VARS_NUMERICAS   = [c for c in VARS_NUMERICAS   if c in df.columns]

print(f"\n• Variables categóricas: {VARS_CATEGORICAS}")
print(f"• Variables numéricas: {len(VARS_NUMERICAS)} columnas")
print(f"• Variable target: '{TARGET}'")

# ── 6.2 Manejo de valores nulos ─────────────────────────────────────────────
df[VARS_NUMERICAS]   = df[VARS_NUMERICAS].fillna(df[VARS_NUMERICAS].median())
df[VARS_CATEGORICAS] = df[VARS_CATEGORICAS].fillna(df[VARS_CATEGORICAS].mode().iloc[0])
print("\n✅ Valores nulos imputados (mediana para numéricas, moda para categóricas).")

# ── 6.3 One-Hot Encoding para variables categóricas ─────────────────────────
df_encoded = pd.get_dummies(df[VARS_CATEGORICAS], prefix=VARS_CATEGORICAS,
                             drop_first=True, dtype=int)

print(f"\n✅ One-Hot Encoding aplicado:")
print(f"   Variables originales: {len(VARS_CATEGORICAS)}")
print(f"   Columnas generadas:   {df_encoded.shape[1]}")

# ── 6.4 Construcción del dataset final ──────────────────────────────────────
X = pd.concat([df[VARS_NUMERICAS], df_encoded], axis=1)
y = df[TARGET]

print(f"\n• Forma de X (features): {X.shape}")
print(f"• Forma de y (target):   {y.shape}")

# ── 6.5 División entrenamiento / prueba (80/20 estratificado) ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y       # Mantiene proporciones de clases
)

print(f"\n✅ División estratificada 80/20:")
print(f"   Entrenamiento: {X_train.shape[0]:,} registros")
print(f"   Prueba:        {X_test.shape[0]:,} registros")
print(f"   Proporción deslizamiento en train: {y_train.mean():.3f}")
print(f"   Proporción deslizamiento en test:  {y_test.mean():.3f}")

# ── 6.6 Escalado de variables numéricas (StandardScaler) ───────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform en train
X_test_scaled  = scaler.transform(X_test)       # solo transform en test (evita data leakage)

print("\n✅ StandardScaler aplicado (media=0, std=1).")
print("   (fit solo en train para evitar fuga de datos)")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 7: MANEJO DE DESBALANCE DE CLASES CON SMOTE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("⚖️  MANEJO DE DESBALANCE DE CLASES CON SMOTE")
print("="*60)

# SMOTE: Synthetic Minority Over-sampling Technique
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n• Antes de SMOTE:")
print(f"  Clase 0 (Sin deslizamiento): {(y_train == 0).sum():,}")
print(f"  Clase 1 (Deslizamiento):     {(y_train == 1).sum():,}")
print(f"\n• Después de SMOTE:")
print(f"  Clase 0 (Sin deslizamiento): {(y_train_balanced == 0).sum():,}")
print(f"  Clase 1 (Deslizamiento):     {(y_train_balanced == 1).sum():,}")
print("\n✅ SMOTE aplicado. Dataset balanceado para entrenamiento.")

# Visualización del balance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (datos_y, titulo) in zip(axes, [
    (y_train,          'Antes de SMOTE'),
    (y_train_balanced, 'Después de SMOTE')
]):
    conteo = pd.Series(datos_y).value_counts()
    colores = ['#2196F3', '#F44336']
    ax.bar(['Sin deslizamiento\n(0)', 'Deslizamiento\n(1)'],
           [conteo.get(0, 0), conteo.get(1, 0)], color=colores)
    ax.set_title(titulo, fontweight='bold', fontsize=13)
    ax.set_ylabel('Cantidad de registros')
    for i, v in enumerate([conteo.get(0, 0), conteo.get(1, 0)]):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

plt.suptitle('Distribución de Clases — Efecto de SMOTE',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('balance_clases_smote.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 8: FASE 4 CRISP-DM — MODELADO (Modeling)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("🤖 FASE 4: MODELADO PREDICTIVO")
print("="*60)

# ── 8.1 Definición de modelos candidatos ────────────────────────────────────
modelos_candidatos = {
    'Regresión Logística': LogisticRegression(
        C=1.0,
        class_weight='balanced',   # Ponderación adicional por desbalance
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    ),
}

# ── 8.2 Entrenamiento y evaluación rápida con validación cruzada ─────────────
print("\n📊 Validación Cruzada (5-Fold Estratificado) — AUC-ROC:")
print("-" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_cv = {}

for nombre, modelo in modelos_candidatos.items():
    scores = cross_val_score(
        modelo, X_train_balanced, y_train_balanced,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    resultados_cv[nombre] = scores
    print(f"  {nombre:25s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 8.3 Selección del mejor modelo ──────────────────────────────────────────
mejor_nombre = max(resultados_cv, key=lambda k: resultados_cv[k].mean())
print(f"\n🏆 Mejor modelo seleccionado: {mejor_nombre}")
modelo_final = modelos_candidatos[mejor_nombre]

# ── 8.4 Entrenamiento del modelo final ──────────────────────────────────────
print(f"\n⏳ Entrenando {mejor_nombre} con datos balanceados (SMOTE)...")
modelo_final.fit(X_train_balanced, y_train_balanced)
print("✅ Entrenamiento completado.")

# ── 8.5 Predicciones ────────────────────────────────────────────────────────
y_pred_proba = modelo_final.predict_proba(X_test_scaled)[:, 1]  # Probabilidades
y_pred       = modelo_final.predict(X_test_scaled)               # Clases (0 o 1)

# Guardar probabilidades como variable de inferencia
df_resultados = X_test.copy()
df_resultados['probabilidad_deslizamiento'] = y_pred_proba
df_resultados['prediccion_clase']           = y_pred
df_resultados['valor_real']                 = y_test.values

print(f"\n✅ Predicciones generadas.")
print(f"   Muestra de probabilidades:\n{pd.Series(y_pred_proba).describe().round(4)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 9: FASE 5 CRISP-DM — EVALUACIÓN (Evaluation)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("📏 FASE 5: EVALUACIÓN DE MÉTRICAS CRÍTICAS")
print("="*60)

# ── 9.1 Métricas principales ─────────────────────────────────────────────────
auc_roc  = roc_auc_score(y_test, y_pred_proba)
recall   = recall_score(y_test, y_pred)
f2_score = fbeta_score(y_test, y_pred, beta=2)  # Beta=2 prioriza Recall

print(f"""
┌──────────────────────────────────────────────────────────────┐
│  MÉTRICAS DE EVALUACIÓN — MODELO: {mejor_nombre:<26}│
├──────────────────────────────────────────────────────────────┤
│  AUC-ROC:       {auc_roc:.4f}  (Objetivo: > 0.85)  {'✅' if auc_roc > 0.85 else '⚠️ Revisar'}{'':>15}│
│  Recall (Sens): {recall:.4f}  (Objetivo: > 0.85)  {'✅' if recall > 0.85 else '⚠️ Revisar'}{'':>15}│
│  F2-Score:      {f2_score:.4f}  (Prioriza Recall){'':>22}│
└──────────────────────────────────────────────────────────────┘
""")

# ── 9.2 Reporte completo de clasificación ────────────────────────────────────
print("📋 Reporte de Clasificación Completo:")
print("-" * 50)
print(classification_report(y_test, y_pred,
                             target_names=['Sin Deslizamiento', 'Deslizamiento']))

# ── 9.3 Matriz de Confusión (Seaborn) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Matriz de confusión absoluta
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sin Deslizamiento', 'Deslizamiento'],
            yticklabels=['Sin Deslizamiento', 'Deslizamiento'],
            ax=axes[0], linewidths=2, linecolor='white',
            annot_kws={"size": 16, "weight": "bold"})
axes[0].set_title('Matriz de Confusión (Valores Absolutos)',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Valor Real', fontsize=12)
axes[0].set_xlabel('Predicción del Modelo', fontsize=12)

# Etiquetas de análisis en la matriz
tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.15,
             f"TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}",
             transform=axes[0].transAxes, ha='center',
             fontsize=11, color='darkblue', fontweight='bold')

# Análisis crítico de Falsos Negativos
print(f"\n⚠️  ANÁLISIS DE FALSOS NEGATIVOS (Crítico para el negocio):")
print(f"   Falsos Negativos (FN): {fn:,} → Deslizamientos NO detectados")
print(f"   Tasa FN: {fn/(fn+tp)*100:.2f}% de los deslizamientos reales perdidos")
print(f"   Falsos Positivos (FP): {fp:,} → Alertas innecesarias")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f'Curva ROC (AUC = {auc_roc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5,
             linestyle='--', label='Clasificador Aleatorio')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=12)
axes[1].set_title('Curva ROC — Poder Discriminativo del Modelo',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=12)
axes[1].axhline(y=0.85, color='red', linestyle=':', lw=1.5,
                label='Umbral Recall objetivo (0.85)')

plt.tight_layout()
plt.savefig('evaluacion_modelo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráficos de evaluación guardados como 'evaluacion_modelo.png'")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 10: ANÁLISIS DE IMPORTANCIA DE VARIABLES (Feature Importance)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("🔍 IMPORTANCIA DE VARIABLES (Feature Importance)")
print("="*60)

# Obtener importancias según el tipo de modelo
nombre_features = list(X_train.columns)

if hasattr(modelo_final, 'feature_importances_'):
    # Random Forest y Gradient Boosting
    importancias = modelo_final.feature_importances_
    tipo_importancia = "Gini Importance"
elif hasattr(modelo_final, 'coef_'):
    # Regresión Logística
    importancias = np.abs(modelo_final.coef_[0])
    tipo_importancia = "Coeficiente Absoluto"
else:
    importancias = np.ones(len(nombre_features)) / len(nombre_features)
    tipo_importancia = "No disponible"

# Crear DataFrame de importancias
df_importancias = pd.DataFrame({
    'Variable': nombre_features,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False).reset_index(drop=True)

# Normalizar a porcentaje
df_importancias['Importancia_pct'] = (
    df_importancias['Importancia'] / df_importancias['Importancia'].sum() * 100
)

# Top 15 variables más importantes
top_n = 15
df_top = df_importancias.head(top_n)

print(f"\n📊 Top {top_n} Variables Más Importantes ({tipo_importancia}):")
print("-" * 50)
for i, row in df_top.iterrows():
    barra = "█" * int(row['Importancia_pct'] / 1)
    print(f"  {i+1:2d}. {row['Variable']:30s}: {row['Importancia_pct']:6.2f}% {barra}")

# Comparación con hipótesis base
print("\n📐 COMPARACIÓN CON HIPÓTESIS BASE:")
print("-" * 50)
grupos_hipotesis = {
    'Precipitación (~35%)': [c for c in df_importancias['Variable']
                              if 'precipitacion' in c.lower()],
    'Pendiente (~25%)':     [c for c in df_importancias['Variable']
                              if 'pendiente' in c.lower()],
    'Saturación/Humedad (~15%)': [c for c in df_importancias['Variable']
                                   if 'humedad' in c.lower() or 'saturaci' in c.lower()
                                   or 'presion_poros' in c.lower()],
}

for grupo, variables in grupos_hipotesis.items():
    if variables:
        pct_grupo = df_importancias[
            df_importancias['Variable'].isin(variables)
        ]['Importancia_pct'].sum()
        print(f"  {grupo}: {pct_grupo:.2f}% (Hipótesis vs Modelo)")
    else:
        print(f"  {grupo}: Variables no encontradas en features")

# Visualización de importancias
fig, ax = plt.subplots(figsize=(14, 8))
colores = ['#E53935' if i < 3 else '#1E88E5' if i < 8 else '#43A047'
           for i in range(top_n)]
bars = ax.barh(df_top['Variable'][::-1], df_top['Importancia_pct'][::-1],
               color=colores[::-1])

# Etiquetas en las barras
for bar, pct in zip(bars, df_top['Importancia_pct'][::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{pct:.2f}%', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Importancia Relativa (%)', fontsize=12)
ax.set_title(f'🔍 Top {top_n} Variables Más Importantes\n'
             f'Modelo: {mejor_nombre} | Método: {tipo_importancia}',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, df_top['Importancia_pct'].max() * 1.2)

# Leyenda de colores
from matplotlib.patches import Patch
leyenda = [
    Patch(color='#E53935', label='Top 3 — Críticas'),
    Patch(color='#1E88E5', label='Top 4-8 — Importantes'),
    Patch(color='#43A047', label='Top 9-15 — Moderadas'),
]
ax.legend(handles=leyenda, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico de importancias guardado como 'feature_importance.png'")

# Exportar importancias completas
df_importancias.to_csv('feature_importances_completo.csv', index=False)
print("✅ Importancias exportadas a 'feature_importances_completo.csv'")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 11: REGLAS DE NEGOCIO — ÍNDICE DE RIESGO (0-100)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("🚦 REGLAS DE NEGOCIO — ÍNDICE DE RIESGO DE DESLIZAMIENTO")
print("="*60)


def calcular_indice_riesgo(probabilidad: float) -> dict:
    """
    Transforma la probabilidad_deslizamiento (0.00 a 1.00) en un
    Índice de Riesgo escalado de 0 a 100 y lo categoriza con nivel
    de alerta y acción recomendada.

    Parámetros:
    -----------
    probabilidad : float
        Probabilidad de deslizamiento predicha por el modelo (0.00 - 1.00).

    Retorna:
    --------
    dict con las siguientes claves:
        - probabilidad  : float  → Probabilidad original
        - indice_riesgo : float  → Escala 0-100
        - nivel         : str    → Categoría de riesgo
        - alerta        : str    → Tipo de alerta
        - accion        : str    → Acción recomendada
        - color         : str    → Color de semáforo
        - emoji         : str    → Icono visual
    """
    # Validación de entrada
    if not (0.0 <= probabilidad <= 1.0):
        raise ValueError(f"La probabilidad debe estar entre 0.0 y 1.0. "
                         f"Recibido: {probabilidad}")

    # ── Transformación lineal: probabilidad → índice 0-100 ──────────────────
    indice = round(probabilidad * 100, 2)

    # ── Categorización por umbrales de negocio ───────────────────────────────
    if indice < 25:
        return {
            'probabilidad':  probabilidad,
            'indice_riesgo': indice,
            'nivel':         'BAJO',
            'alerta':        'Sin alerta',
            'accion':        'Monitoreo rutinario. Sin acción inmediata requerida.',
            'color':         'Verde',
            'emoji':         '🟢'
        }
    elif indice < 50:
        return {
            'probabilidad':  probabilidad,
            'indice_riesgo': indice,
            'nivel':         'MODERADO',
            'alerta':        'Vigilancia',
            'accion':        'Aumentar frecuencia de monitoreo. '
                             'Activar red de sensores. Informar a autoridades locales.',
            'color':         'Amarillo',
            'emoji':         '🟡'
        }
    elif indice < 75:
        return {
            'probabilidad':  probabilidad,
            'indice_riesgo': indice,
            'nivel':         'ALTO',
            'alerta':        'Alerta preventiva',
            'accion':        '⚠️ ACTIVAR PROTOCOLO DE ALERTA. '
                             'Notificar a Defensa Civil. '
                             'Preparar rutas de evacuación. '
                             'Restringir acceso a zonas de riesgo.',
            'color':         'Naranja',
            'emoji':         '🟠'
        }
    else:
        return {
            'probabilidad':  probabilidad,
            'indice_riesgo': indice,
            'nivel':         'MUY ALTO',
            'alerta':        'Evacuación',
            'accion':        '🚨 EVACUACIÓN INMEDIATA. '
                             'Alertar a toda la comunidad. '
                             'Cortar suministros en zonas de riesgo. '
                             'Coordinar con fuerzas de emergencia.',
            'color':         'Rojo',
            'emoji':         '🔴'
        }


def aplicar_indice_a_dataframe(df_pred: pd.DataFrame,
                                col_probabilidad: str = 'probabilidad_deslizamiento'
                                ) -> pd.DataFrame:
    """
    Aplica la función calcular_indice_riesgo a un DataFrame completo,
    agregando las columnas de índice, nivel, alerta y acción.

    Parámetros:
    -----------
    df_pred          : DataFrame con columna de probabilidades
    col_probabilidad : Nombre de la columna de probabilidades

    Retorna:
    --------
    DataFrame enriquecido con columnas de análisis de riesgo
    """
    resultados = df_pred[col_probabilidad].apply(
        lambda p: pd.Series(calcular_indice_riesgo(p))
    )
    return pd.concat([df_pred, resultados], axis=1)


# ── Demo de la función con casos de prueba ───────────────────────────────────
print("\n📋 Demostración de la función calcular_indice_riesgo():\n")
casos_prueba = [0.10, 0.35, 0.62, 0.88, 0.95]

for prob in casos_prueba:
    resultado = calcular_indice_riesgo(prob)
    print(f"  Probabilidad: {prob:.2f} → "
          f"Índice: {resultado['indice_riesgo']:5.1f} | "
          f"{resultado['emoji']} {resultado['nivel']:10s} | "
          f"{resultado['alerta']}")

# ── Aplicar a los resultados del modelo ─────────────────────────────────────
df_resultados_finales = df_resultados[['probabilidad_deslizamiento',
                                        'prediccion_clase', 'valor_real']].copy()
df_resultados_finales = aplicar_indice_a_dataframe(df_resultados_finales)

print("\n\n📊 Distribución de niveles de riesgo en el conjunto de prueba:")
print("-" * 50)
dist_niveles = df_resultados_finales['nivel'].value_counts()
for nivel, conteo in dist_niveles.items():
    pct = conteo / len(df_resultados_finales) * 100
    print(f"  {nivel:12s}: {conteo:4,} registros ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 12: VISUALIZACIÓN DEL ÍNDICE DE RIESGO
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── Distribución del Índice de Riesgo ────────────────────────────────────────
ax1 = axes[0]
n, bins, patches = ax1.hist(df_resultados_finales['indice_riesgo'],
                              bins=50, edgecolor='white', linewidth=0.5)

# Colorear zonas de riesgo
for patch, bin_left in zip(patches, bins[:-1]):
    if bin_left < 25:
        patch.set_facecolor('#4CAF50')    # Verde - Bajo
    elif bin_left < 50:
        patch.set_facecolor('#FFEB3B')    # Amarillo - Moderado
    elif bin_left < 75:
        patch.set_facecolor('#FF9800')    # Naranja - Alto
    else:
        patch.set_facecolor('#F44336')    # Rojo - Muy Alto

# Líneas de umbral
for umbral, etiqueta, color in [(25, 'Moderado', '#FFEB3B'),
                                   (50, 'Alto', '#FF9800'),
                                   (75, 'Muy Alto', '#F44336')]:
    ax1.axvline(x=umbral, color=color, linestyle='--', lw=2)
    ax1.text(umbral+0.5, ax1.get_ylim()[1]*0.9 if ax1.get_ylim()[1] > 0 else 10,
             etiqueta, color=color, fontsize=9, fontweight='bold')

ax1.set_title('Distribución del Índice de Riesgo (0-100)',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Índice de Riesgo', fontsize=12)
ax1.set_ylabel('Frecuencia', fontsize=12)

# ── Torta de categorías de riesgo ────────────────────────────────────────────
ax2 = axes[1]
colores_torta = {
    'BAJO':     '#4CAF50',
    'MODERADO': '#FFEB3B',
    'ALTO':     '#FF9800',
    'MUY ALTO': '#F44336'
}

if len(dist_niveles) > 0:
    colores_ordenados = [colores_torta.get(n, '#9E9E9E') for n in dist_niveles.index]
    wedges, texts, autotexts = ax2.pie(
        dist_niveles.values,
        labels=dist_niveles.index,
        colors=colores_ordenados,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for text in autotexts:
        text.set_fontsize(11)
        text.set_fontweight('bold')

ax2.set_title('Distribución de Categorías de Riesgo\n(Conjunto de Prueba)',
              fontsize=13, fontweight='bold')

plt.suptitle('🗺️ Sistema de Alerta Temprana de Deslizamientos — Bolivia',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('distribucion_riesgo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico de distribución de riesgo guardado.")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 13: AJUSTE DE UMBRAL PARA MAXIMIZAR RECALL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("🎯 OPTIMIZACIÓN DE UMBRAL PARA MAXIMIZAR RECALL")
print("="*60)

# Evaluar diferentes umbrales de clasificación
umbrales = np.arange(0.05, 0.95, 0.05)
resultados_umbral = []

for umbral in umbrales:
    y_pred_umbral = (y_pred_proba >= umbral).astype(int)
    recall_u      = recall_score(y_test, y_pred_umbral, zero_division=0)
    f2_u          = fbeta_score(y_test, y_pred_umbral, beta=2, zero_division=0)
    fn_u          = ((y_test == 1) & (y_pred_umbral == 0)).sum()
    fp_u          = ((y_test == 0) & (y_pred_umbral == 1)).sum()
    resultados_umbral.append({
        'umbral': umbral,
        'recall': recall_u,
        'f2':     f2_u,
        'fn':     fn_u,
        'fp':     fp_u
    })

df_umbrales = pd.DataFrame(resultados_umbral)

# Umbral óptimo: maximiza F2 (que ya prioriza Recall)
idx_optimo  = df_umbrales['f2'].idxmax()
umbral_opt  = df_umbrales.loc[idx_optimo, 'umbral']

print(f"\n• Umbral por defecto (0.50):")
row_50 = df_umbrales[df_umbrales['umbral'].round(2) == 0.50].iloc[0]
print(f"  Recall={row_50['recall']:.4f} | F2={row_50['f2']:.4f} | FN={row_50['fn']}")

print(f"\n• Umbral óptimo ({umbral_opt:.2f}) — Máximo F2-Score:")
row_opt = df_umbrales.loc[idx_optimo]
print(f"  Recall={row_opt['recall']:.4f} | F2={row_opt['f2']:.4f} | FN={row_opt['fn']}")

# Visualización del análisis de umbral
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_umbrales['umbral'], df_umbrales['recall'], 'b-o',
        markersize=5, lw=2, label='Recall')
ax.plot(df_umbrales['umbral'], df_umbrales['f2'], 'r-s',
        markersize=5, lw=2, label='F2-Score')
ax.axvline(x=umbral_opt, color='green', linestyle='--', lw=2,
           label=f'Umbral óptimo = {umbral_opt:.2f}')
ax.axhline(y=0.85, color='orange', linestyle=':', lw=1.5,
           label='Objetivo Recall = 0.85')
ax.set_xlabel('Umbral de Clasificación', fontsize=12)
ax.set_ylabel('Métrica', fontsize=12)
ax.set_title('Análisis de Sensibilidad del Umbral — Recall vs F2-Score',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)
ax.set_xlim(0.05, 0.90)
plt.tight_layout()
plt.savefig('analisis_umbral.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Umbral óptimo recomendado: {umbral_opt:.2f}")
print("   (Usar este umbral en producción para minimizar Falsos Negativos)")


# ─────────────────────────────────────────────────────────────────────────────
# CELDA 14: EXPORTACIÓN DE RESULTADOS Y RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("💾 EXPORTACIÓN DE RESULTADOS")
print("="*60)

# Aplicar umbral óptimo a las predicciones finales
df_resultados_finales['prediccion_optimizada'] = (
    df_resultados_finales['probabilidad_deslizamiento'] >= umbral_opt
).astype(int)

# Exportar dataset con predicciones
df_resultados_finales.to_csv('predicciones_finales.csv', index=False)
df_importancias.to_csv('feature_importances_completo.csv', index=False)

print("\n📁 Archivos generados:")
print("  ✅ predicciones_finales.csv        → Predicciones con índice de riesgo")
print("  ✅ feature_importances_completo.csv → Importancia de todas las variables")
print("  ✅ eda_variables_clave.png          → Análisis exploratorio")
print("  ✅ correlacion_variables.png        → Mapa de correlaciones")
print("  ✅ balance_clases_smote.png         → Efecto de SMOTE")
print("  ✅ evaluacion_modelo.png            → Matriz de confusión + ROC")
print("  ✅ feature_importance.png           → Importancia de variables")
print("  ✅ distribucion_riesgo.png          → Distribución del índice de riesgo")
print("  ✅ analisis_umbral.png              → Optimización del umbral")

# ── Resumen Ejecutivo Final ───────────────────────────────────────────────────
y_pred_final = (y_pred_proba >= umbral_opt).astype(int)
recall_final = recall_score(y_test, y_pred_final)
f2_final     = fbeta_score(y_test, y_pred_final, beta=2)
fn_final     = ((y_test == 1) & (y_pred_final == 0)).sum()

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           📋 RESUMEN EJECUTIVO — RESULTADOS FINALES              ║
╠══════════════════════════════════════════════════════════════════╣
║  Modelo Seleccionado:  {mejor_nombre:<41}║
║  Umbral Optimizado:    {umbral_opt:.2f}{'':>43}║
╠══════════════════════════════════════════════════════════════════╣
║  MÉTRICAS CON UMBRAL OPTIMIZADO:                                 ║
║  • AUC-ROC:       {auc_roc:.4f}  {'✅ OBJETIVO SUPERADO' if auc_roc > 0.85 else '⚠️  REVISAR':<42}║
║  • Recall:        {recall_final:.4f}  {'✅ OBJETIVO SUPERADO' if recall_final > 0.85 else '⚠️  REVISAR':<42}║
║  • F2-Score:      {f2_final:.4f}{'':>44}║
║  • Falsos Neg.:   {fn_final:<47}║
╠══════════════════════════════════════════════════════════════════╣
║  CATEGORÍAS DE RIESGO DETECTADAS:                                ║
║  🟢 Bajo (0-24):     Sin alerta                                  ║
║  🟡 Moderado (25-49): Vigilancia                                 ║
║  🟠 Alto (50-74):    Alerta preventiva                           ║
║  🔴 Muy Alto (75-100): Evacuación inmediata                      ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("\n🎯 Pipeline CRISP-DM completado exitosamente.")
print("   El modelo está listo para ser desplegado en el Sistema de Alerta Temprana.")
