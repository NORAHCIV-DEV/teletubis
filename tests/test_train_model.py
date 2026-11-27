import pandas as pd

from train_model import entrenar_y_guardar


def test_entrenamiento_gini_genera_artefactos_y_metricas(tmp_path):
    filas = []
    for i in range(80):
        alto = i >= 40
        filas.append(
            {
                "departamento": "La Paz" if i % 2 else "Cochabamba",
                "tipo_suelo": "Arcilloso" if alto else "Arenoso",
                "geologia": "Sedimentaria" if alto else "Ignea",
                "precipitacion_24h_mm": 90 if alto else 8,
                "precipitacion_3d_mm": 210 if alto else 25,
                "precipitacion_7d_mm": 390 if alto else 50,
                "pendiente_grados": 48 if alto else 12,
                "elevacion_m": 3200 if alto else 1500,
                "humedad_pct": 92 if alto else 35,
                "deslizamiento_ocurrido": int(alto),
            }
        )

    dataset = tmp_path / "datos.csv"
    salida = tmp_path / "artefactos"
    pd.DataFrame(filas).to_csv(dataset, index=False)

    metricas = entrenar_y_guardar(dataset, salida)

    assert metricas["criterio"] == "gini"
    assert metricas["max_depth"] == 5
    assert 0 <= metricas["accuracy"] <= 1
    assert 0 <= metricas["f1"] <= 1
    assert (salida / "modelo_deslizamientos.joblib").exists()
    assert (salida / "scaler.joblib").exists()
    assert (salida / "columnas_modelo.joblib").exists()
