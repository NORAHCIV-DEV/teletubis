import pandas as pd
import pytest
from kmeans_service import FEATURES, Segmenter, train


def sample(rows=60):
    records = []
    for i in range(rows):
        high = i >= rows // 2
        records.append({"precipitacion_24h_mm": (80 if high else 5) + i % 5,
                        "precipitacion_3d_mm": (180 if high else 20) + i % 7,
                        "precipitacion_7d_mm": (350 if high else 40) + i % 9,
                        "pendiente_grados": (50 if high else 10) + i % 4,
                        "elevacion_m": (3000 if high else 1000) + i,
                        "humedad_pct": (90 if high else 30) + i % 3,
                        "indice_riesgo_deslizamiento": 90 if high else 10,
                        "deslizamiento_ocurrido": int(high)})
    return pd.DataFrame(records)


def test_train_persists_and_predicts(tmp_path):
    dataset = tmp_path / "data.csv"
    sample().to_csv(dataset, index=False)
    metadata = train(dataset, tmp_path / "artifacts", k_max=4)
    assert 2 <= metadata["selected_k"] <= 4
    assert metadata["silhouette"] > 0
    assert sum(s["cantidad"] for s in metadata["segments"]) == 60
    segment = Segmenter(tmp_path / "artifacts").predict(sample(1).iloc[0][FEATURES].to_dict())
    assert segment["segmento"]


def test_rejects_missing_columns(tmp_path):
    dataset = tmp_path / "bad.csv"
    pd.DataFrame({"humedad_pct": range(20)}).to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="Faltan columnas"):
        train(dataset, tmp_path / "artifacts")


def test_rejects_insufficient_data(tmp_path):
    dataset = tmp_path / "small.csv"
    sample(5).to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="al menos 10"):
        train(dataset, tmp_path / "artifacts")
