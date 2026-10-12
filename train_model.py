import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

def entrenar_y_guardar():
    print("Entrenando modelo para el backend...")
    df = pd.read_csv("Dataset_10000_Bolivia_Sintetico.csv")

    TARGET = 'deslizamiento_ocurrido'
    VARS_CATEGORICAS = ['departamento', 'tipo_suelo', 'geologia']
    VARS_NUMERICAS = ['precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm',
                      'pendiente_grados', 'elevacion_m', 'humedad_pct']

    # OHE
    df_encoded = pd.get_dummies(df[VARS_CATEGORICAS], prefix=VARS_CATEGORICAS, drop_first=True, dtype=int)
    X = pd.concat([df[VARS_NUMERICAS], df_encoded], axis=1)
    y = df[TARGET]

    columnas_finales = X.columns.tolist()
    joblib.dump(columnas_finales, "columnas_modelo.joblib")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, "scaler.joblib")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    modelo.fit(X_train_bal, y_train_bal)

    joblib.dump(modelo, "modelo_deslizamientos.joblib")
    print("Modelo, scaler y columnas guardados con exito (.joblib).")

if __name__ == "__main__":
    entrenar_y_guardar()
