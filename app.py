from functools import lru_cache
from pathlib import Path
import unicodedata

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Sistema de Alerta de Deslizamientos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y dependencias (rutas absolutas: funciona desde cualquier carpeta)
try:
    modelo = joblib.load(BASE_DIR / "modelo_deslizamientos.joblib")
    scaler = joblib.load(BASE_DIR / "scaler.joblib")
    columnas = joblib.load(BASE_DIR / "columnas_modelo.joblib")
except Exception as e:
    raise RuntimeError(
        f"No se pudieron cargar los modelos ({e}). Ejecuta train_model.py primero."
    ) from e



def _norm(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar categorías."""
    s = unicodedata.normalize("NFD", texto.strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Categorías conocidas por el modelo (prefijo -> {texto normalizado: nombre de columna})
CATEGORIAS = {
    pref: {_norm(c[len(pref) + 1:]): c for c in columnas if c.startswith(pref + "_")}
    for pref in ("departamento", "tipo_suelo", "geologia")
}
# Categoría de referencia que quedó fuera por drop_first=True (todas sus dummies = 0)
CATEGORIA_BASE = {"departamento": "chuquisaca", "tipo_suelo": "arcilloso", "geologia": "metamorfica"}


class DatosIngreso(BaseModel):
    precipitacion_24h_mm: float = Field(ge=0)
    precipitacion_3d_mm: float = Field(ge=0)
    precipitacion_7d_mm: float = Field(ge=0)
    pendiente_grados: float = Field(ge=0, le=90)
    elevacion_m: float
    humedad_pct: float = Field(ge=0, le=100)
    departamento: str
    tipo_suelo: str
    geologia: str


NIVELES = [
    (25, "Bajo", "#4caf50", "Sin alerta", ["Monitoreo rutinario."]),
    (50, "Moderado", "#ff9800", "Vigilancia", ["Aumentar la frecuencia de monitoreo.", "Informar a las autoridades locales."]),
    (75, "Alto", "#f44336", "Alerta preventiva", ["Activar el protocolo de emergencia.", "Preparar la evacuación preventiva."]),
    (101, "Muy Alto", "#b71c1c", "Evacuación", ["Evacuación inmediata de la zona.", "Notificar a Defensa Civil."]),
]


def clasificar(porcentaje: float) -> dict:
    for limite, nivel, color, alerta, acciones in NIVELES:
        if porcentaje < limite:
            return {
                "nivel_riesgo": nivel,
                "porcentaje": porcentaje,
                "color": color,
                "acciones": acciones,
                # Campos anteriores, usados por static/script.js
                "indice": porcentaje,
                "nivel": nivel.upper(),
                "alerta": alerta,
                "accion": " ".join(acciones),
            }




@app.post("/predecir")
def predecir_riesgo(datos: DatosIngreso):
    # Mismo orden/columnas que train_model.py: 0 para todas y se rellenan las presentes
    fila = {col: 0 for col in columnas}
    fila["precipitacion_24h_mm"] = datos.precipitacion_24h_mm
    fila["precipitacion_3d_mm"] = datos.precipitacion_3d_mm
    fila["precipitacion_7d_mm"] = datos.precipitacion_7d_mm
    fila["pendiente_grados"] = datos.pendiente_grados
    fila["elevacion_m"] = datos.elevacion_m
    fila["humedad_pct"] = datos.humedad_pct

    for prefijo in ("departamento", "tipo_suelo", "geologia"):
        valor = _norm(getattr(datos, prefijo))
        if valor in CATEGORIAS[prefijo]:
            fila[CATEGORIAS[prefijo][valor]] = 1
        elif valor != CATEGORIA_BASE[prefijo]:
            validos = sorted(set(CATEGORIAS[prefijo]) | {CATEGORIA_BASE[prefijo]})
            raise HTTPException(
                status_code=422,
                detail=f"{prefijo} '{getattr(datos, prefijo)}' no es válido para el modelo. Valores: {validos}",
            )

    X = pd.DataFrame([fila])[list(columnas)]
    probabilidad = float(modelo.predict_proba(scaler.transform(X))[0][1])
    return clasificar(round(probabilidad * 100, 2))


VARS_CAT = ["departamento", "tipo_suelo", "geologia"]
VARS_NUM = ["precipitacion_24h_mm", "precipitacion_3d_mm", "precipitacion_7d_mm",
            "pendiente_grados", "elevacion_m", "humedad_pct"]


@lru_cache(maxsize=1)
def _metricas_modelo() -> dict:
    """Accuracy/F1 sobre el conjunto de prueba (mismo split 80/20 de train_model.py). Se calcula una vez."""
    df = pd.read_csv(BASE_DIR / "Dataset_10000_Bolivia_Sintetico.csv")
    X = pd.concat([df[VARS_NUM], pd.get_dummies(df[VARS_CAT], prefix=VARS_CAT, drop_first=True, dtype=int)], axis=1)[list(columnas)]
    y = df["deslizamiento_ocurrido"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred = modelo.predict(scaler.transform(X_test))
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "dataset_size": int(len(df)),
        "test_size": int(len(y_test)),
    }


@app.get("/info")
def info_modelo():
    m = _metricas_modelo()
    return {
        "modelo": "Random Forest Classifier",
        "n_estimators": int(modelo.n_estimators),
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "version": "1.0.0",
        "dataset_size": m["dataset_size"],
        "evaluado_en": f"conjunto de prueba ({m['test_size']} registros, 20 %)",
    }
