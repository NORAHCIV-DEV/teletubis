from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "Dataset_10000_Bolivia_Sintetico.csv"


def entrenar_y_guardar(dataset_path=DATASET, output_dir="."):
    print("Entrenando modelo para el backend...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)

    TARGET = 'deslizamiento_ocurrido'
    VARS_CATEGORICAS = ['departamento', 'tipo_suelo', 'geologia']
    VARS_NUMERICAS = ['precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm',
                      'pendiente_grados', 'elevacion_m', 'humedad_pct']

    # OHE
    df_encoded = pd.get_dummies(df[VARS_CATEGORICAS], prefix=VARS_CATEGORICAS, drop_first=True, dtype=int)
    X = pd.concat([df[VARS_NUMERICAS], df_encoded], axis=1)
    y = df[TARGET]

    columnas_finales = X.columns.tolist()
    joblib.dump(columnas_finales, output_dir / "columnas_modelo.joblib")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    modelo = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced",
    )
    modelo.fit(X_train_bal, y_train_bal)

    predicciones = modelo.predict(scaler.transform(X_test))
    metricas = {
        "accuracy": accuracy_score(y_test, predicciones),
        "f1": f1_score(y_test, predicciones, zero_division=0),
        "criterio": modelo.criterion,
        "max_depth": modelo.max_depth,
    }

    joblib.dump(modelo, output_dir / "modelo_deslizamientos.joblib")
    print(
        "Modelo, scaler y columnas guardados con exito (.joblib). "
        f"Accuracy={metricas['accuracy']:.4f}, F1={metricas['f1']:.4f}"
    )
    return metricas

if __name__ == "__main__":
    entrenar_y_guardar()
