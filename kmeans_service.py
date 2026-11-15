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
FEATURE_DETAILS = [
    {"key": "precipitacion_24h_mm", "nombre": "Precipitacion en 24 horas", "unidad": "mm",
     "origen": "Distribucion exponencial con escala de 15 mm.",
     "uso": "Representa la lluvia inmediata que puede saturar rapidamente el suelo."},
    {"key": "precipitacion_3d_mm", "nombre": "Precipitacion en 3 dias", "unidad": "mm",
     "origen": "Distribucion exponencial con escala de 45 mm.",
     "uso": "Resume episodios recientes de lluvia que se acumulan durante varios dias."},
    {"key": "precipitacion_7d_mm", "nombre": "Precipitacion en 7 dias", "unidad": "mm",
     "origen": "Distribucion exponencial con escala de 120 mm y eventos extremos de 500 u 800 mm.",
     "uso": "Permite identificar saturacion prolongada y diferencia el perfil de lluvia extrema."},
    {"key": "pendiente_grados", "nombre": "Pendiente del terreno", "unidad": "grados",
     "origen": "Distribucion normal centrada en 30 grados, limitada al rango de 0 a 90.",
     "uso": "Describe la inclinacion del terreno y su influencia en el movimiento de suelo."},
    {"key": "elevacion_m", "nombre": "Elevacion", "unidad": "m s. n. m.",
     "origen": "Distribucion uniforme entre 500 y 5.000 metros.",
     "uso": "Separa observaciones de valles y zonas de gran elevacion."},
    {"key": "humedad_pct", "nombre": "Humedad del suelo", "unidad": "%",
     "origen": "Distribucion normal centrada en 60%, limitada entre 0% y 100%.",
     "uso": "Aproxima el nivel de agua retenida por el suelo."},
]


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


def _names(profiles: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Asigna nombres unicos segun la variable que realmente diferencia cada perfil."""
    by_cluster = {int(profile["cluster"]): profile for profile in profiles}
    available = set(by_cluster)
    labels: dict[int, tuple[str, str]] = {}

    weekly = max(available, key=lambda c: by_cluster[c]["precipitacion_7d_mm"])
    labels[weekly] = ("Saturacion por lluvia acumulada",
                      "Concentra la mayor precipitacion semanal y representa episodios prolongados de saturacion.")
    available.remove(weekly)

    recent = max(available, key=lambda c: by_cluster[c]["precipitacion_3d_mm"])
    labels[recent] = ("Lluvia reciente de alta intensidad",
                      "Se diferencia por el mayor acumulado de lluvia en tres dias, aunque no alcanza el extremo semanal.")
    available.remove(recent)

    low = min(available, key=lambda c: by_cluster[c]["elevacion_m"])
    labels[low] = ("Valles de menor elevacion",
                   "Agrupa las observaciones de menor altitud y lluvias acumuladas moderadas.")
    available.remove(low)

    if available:
        high = available.pop()
        labels[high] = ("Altiplano de gran elevacion",
                        "Agrupa las observaciones de mayor altitud, sin lluvia acumulada extrema.")

    result = {}
    for cluster, profile in by_cluster.items():
        name, differentiator = labels[cluster]
        description = (f"{differentiator} Promedios: lluvia 3d {profile['precipitacion_3d_mm']:.1f} mm, "
                       f"lluvia 7d {profile['precipitacion_7d_mm']:.1f} mm, "
                       f"elevacion {profile['elevacion_m']:.0f} m y humedad {profile['humedad_pct']:.1f}%.")
        result[cluster] = {"nombre": name, "descripcion": description,
                           "diferenciador": differentiator}
    return result


def train(dataset_path: Path, artifact_dir: Path, k_min: int = 2, k_max: int = 6,
          target_k: int = 4) -> dict[str, Any]:
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
    if target_k not in models:
        raise ValueError(f"El numero de clusters solicitado (k={target_k}) no es valido")
    selected = next(item for item in candidates if int(item["k"]) == target_k)
    model = models[target_k]
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
        profile.update(names[profile["cluster"]])
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {"version": trained_at.replace(":", "").replace("-", ""),
                "algorithm": "KMeans", "trained_at": trained_at, "dataset": dataset_path.name,
                "records": int(len(clean)), "features": FEATURES, "selected_k": target_k,
                "dataset_origin": {
                    "tipo": "sintetico",
                    "descripcion": "10.000 observaciones reproducibles generadas con semilla aleatoria 42 para simular condiciones geoclimaticas de Bolivia.",
                    "alcance": "Seis variables numericas forman los clusters. Departamento, suelo y geologia contextualizan el dataset; el indice y la ocurrencia se usan solo para interpretar resultados."
                },
                "feature_details": FEATURE_DETAILS,
                "selection_reason": "Cuatro perfiles para el analisis H4 de Data Storytelling y Gestalt",
                "silhouette": selected["silhouette"], "wcss": selected["wcss"],
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
                "descripcion": segment["descripcion"],
                "diferenciador": segment.get("diferenciador"),
                "modelo_version": self.metadata["version"]}
