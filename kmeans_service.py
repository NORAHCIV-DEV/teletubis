"""Entrenamiento y consumo del modelo K-Means para perfiles geoclimaticos."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["precipitacion_24h_mm", "precipitacion_3d_mm", "precipitacion_7d_mm",
            "pendiente_grados", "elevacion_m", "humedad_pct"]
PROFILE_COLUMNS = ["indice_riesgo_deslizamiento", "deslizamiento_ocurrido"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    data = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    if len(data) < 10:
        raise ValueError("Se requieren al menos 10 registros validos para entrenar K-Means")
    variable = data.nunique() > 1
    if not variable.all():
        raise ValueError(f"Caracteristicas constantes: {data.columns[~variable].tolist()}")
    return data.clip(data.quantile(0.01), data.quantile(0.99), axis=1)


def _names(profiles: list[dict[str, Any]]) -> dict[int, tuple[str, str]]:
    order = np.argsort([float(p.get("indice_riesgo_deslizamiento", 0)) for p in profiles])
    # Nombres relativos: la alerta operativa siempre proviene del clasificador supervisado.
    labels = ["Exposicion geoclimatica menor", "Exposicion geoclimatica intermedia",
              "Exposicion geoclimatica elevada", "Exposicion geoclimatica maxima"]
    result = {}
    for rank, pos in enumerate(order):
        profile = profiles[int(pos)]
        name = labels[round(rank * (len(labels) - 1) / max(len(order) - 1, 1))]
        description = (f"Lluvia 7d promedio: {profile['precipitacion_7d_mm']:.1f} mm; "
                       f"pendiente: {profile['pendiente_grados']:.1f} grados; "
                       f"humedad: {profile['humedad_pct']:.1f}%.")
        result[int(profile["cluster"])] = (name, description)
    return result


def train(dataset_path: Path, artifact_dir: Path, k_min: int = 2, k_max: int = 6) -> dict[str, Any]:
    raw = pd.read_csv(dataset_path)
    clean = _prepare(raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean)
    candidates, models = [], {}
    for k in range(k_min, min(k_max, len(clean) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            continue
        candidates.append({"k": k, "silhouette": float(silhouette_score(scaled, labels)),
                           "wcss": float(model.inertia_)})
        models[k] = model
    if not candidates:
        raise ValueError("No fue posible obtener una segmentacion valida")
    best = max(candidates, key=lambda item: item["silhouette"])
    model = models[int(best["k"])]
    extra = [c for c in PROFILE_COLUMNS if c in raw]
    profiled = raw.loc[clean.index, FEATURES + extra].copy()
    profiled["cluster"] = model.labels_
    profiles = (profiled.groupby("cluster").mean(numeric_only=True)
                .join(profiled.groupby("cluster").size().rename("cantidad"))
                .reset_index().to_dict(orient="records"))
    names = _names(profiles)
    for profile in profiles:
        profile["cluster"] = int(profile["cluster"])
        profile["cantidad"] = int(profile["cantidad"])
        profile["nombre"], profile["descripcion"] = names[profile["cluster"]]
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {"version": trained_at.replace(":", "").replace("-", ""),
                "algorithm": "KMeans", "trained_at": trained_at, "dataset": dataset_path.name,
                "records": int(len(clean)), "features": FEATURES, "selected_k": int(best["k"]),
                "silhouette": best["silhouette"], "wcss": best["wcss"],
                "candidates": candidates, "segments": profiles}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "kmeans_model.joblib")
    joblib.dump(scaler, artifact_dir / "kmeans_scaler.joblib")
    (artifact_dir / "kmeans_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class Segmenter:
    def __init__(self, artifact_dir: Path):
        self.model = joblib.load(artifact_dir / "kmeans_model.joblib")
        self.scaler = joblib.load(artifact_dir / "kmeans_scaler.joblib")
        self.metadata = json.loads((artifact_dir / "kmeans_metadata.json").read_text(encoding="utf-8"))
        self._segments = {int(item["cluster"]): item for item in self.metadata["segments"]}

    def predict(self, values: dict[str, float]) -> dict[str, Any]:
        frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        cluster = int(self.model.predict(self.scaler.transform(frame))[0])
        segment = self._segments[cluster]
        return {"cluster": cluster, "segmento": segment["nombre"],
                "descripcion": segment["descripcion"], "modelo_version": self.metadata["version"]}
