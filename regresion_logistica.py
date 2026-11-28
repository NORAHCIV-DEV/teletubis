"""Regresión logística de GeoRisk AI (H4 - Data Storytelling).

Reproduce las cifras de docs/Diapositiva_GeoRisk_Regresion_Logistica.pptx
y de "Modelo de Regresion Logistica.md".
Uso (desde la carpeta del proyecto):
    python Dataset_10000_Bolivia_Sintetico.py
    python regresion_logistica.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Dataset_10000_Bolivia_Sintetico.csv")
y = df["deslizamiento_ocurrido"]

# Se excluyen el score y el índice de riesgo: se usaron para generar el objetivo (data leakage)
X = df.drop(columns=["id_registro", "deslizamiento_ocurrido",
                     "score_riesgo_bruto", "indice_riesgo_deslizamiento"])
X = pd.get_dummies(X, drop_first=True).astype(float)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr)
modelo = LogisticRegression(max_iter=2000).fit(scaler.transform(X_tr), y_tr)

pred = modelo.predict(scaler.transform(X_te))
prob = modelo.predict_proba(scaler.transform(X_te))[:, 1]

print(f"Exactitud (test):   {accuracy_score(y_te, pred):.3f}")
print(f"Sensibilidad:       {recall_score(y_te, pred):.3f}")
print(f"Precisión:          {precision_score(y_te, pred):.3f}")
print(f"F1:                 {f1_score(y_te, pred):.3f}")
print(f"AUC-ROC:            {roc_auc_score(y_te, prob):.3f}")
print("Matriz de confusión:\n", confusion_matrix(y_te, pred))

# Cambio de las odds de deslizamiento por cada aumento de 1 desviación estándar de la variable
odds = pd.Series(np.exp(modelo.coef_[0]), index=X.columns)
tabla = pd.DataFrame({
    "aumento_odds_pct": ((odds - 1) * 100).round(1),
    "desviacion_estandar": X_tr.std().round(1),
}).sort_values("aumento_odds_pct", ascending=False)
print("\nCambio en las odds por +1 desviación estándar:\n", tabla)
