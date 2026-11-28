"""Evaluación comparativa y reproducible de algoritmos de GeoRisk AI.

Responsable: Integrante 4 (rama feat/comparacion-algoritmos).

Qué hace:
  1. Carga el dataset sintético y aplica el mismo preprocesamiento de train_model.py.
  2. Compara los algoritmos supervisados del proyecto bajo EXACTAMENTE las mismas condiciones:
     mismos datos, mismo preprocesamiento, mismos pliegues de validación y misma semilla.
  3. Valida con validación cruzada estratificada repetida (5 pliegues x 3 repeticiones) sobre el
     80 % de entrenamiento y mide una sola vez sobre el 20 % de prueba que nunca se usó.
  4. Genera tablas (CSV) y gráficos (PNG) en la carpeta resultados_evaluacion/.

Importante: este módulo EVALÚA los modelos tal como están definidos en el repositorio.
No ajusta ni modifica hiperparámetros (eso corresponde a otro integrante). Las
configuraciones están juntas en `construir_modelos()` para poder auditarlas.

Uso (desde la carpeta del proyecto):
    python Dataset_10000_Bolivia_Sintetico.py     # solo si falta el CSV
    python evaluacion_algoritmos.py               # genera resultados_evaluacion/
    python evaluacion_algoritmos.py --verificar   # además comprueba la reproducibilidad
"""
import argparse
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42
DATASET = "Dataset_10000_Bolivia_Sintetico.csv"
SALIDA = "resultados_evaluacion"
TARGET = "deslizamiento_ocurrido"
NUMERICAS = ["precipitacion_24h_mm", "precipitacion_3d_mm", "precipitacion_7d_mm",
             "pendiente_grados", "elevacion_m", "humedad_pct"]
CATEGORICAS = ["departamento", "tipo_suelo", "geologia"]

# Métricas: F1-macro y recall de "Ocurre" son las del proyecto (sección 2 del notebook CRISP-DM)
METRICAS = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "roc_auc": "roc_auc",
}
METRICA_PRINCIPAL = "f1_macro"
ORDEN_METRICAS = list(METRICAS)

NOMBRES = {
    "referencia": "Referencia (siempre 'No ocurre')",
    "logistica": "Regresión logística",
    "arbol": "Árbol de decisión",
    "rf_develop": "Random Forest (develop)",
    "rf_gini5": "Random Forest Gini, prof. 5",
}


def construir_modelos():
    """Algoritmos con los hiperparámetros ya definidos en el repositorio (sin modificarlos).

    - referencia: no aprende; sirve de piso para saber si los modelos aportan algo.
    - logistica: Alerta_Deslizamientos_CRISP_DM.ipynb, sección 8 (modelo base).
    - arbol: "Modelo de Arboles de Decision.md" (max_depth=5, min_samples_leaf=15).
    - rf_develop: train_model.py de la rama develop (modelo del sistema).
    - rf_gini5: train_model.py de la rama feat/flujo-profesional-georisk (Gini, prof. 5).
    """
    return {
        "referencia": DummyClassifier(strategy="constant", constant=0),
        "logistica": LogisticRegression(random_state=SEMILLA, max_iter=1000),
        "arbol": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=SEMILLA),
        "rf_develop": RandomForestClassifier(n_estimators=100, max_depth=10,
                                             random_state=SEMILLA, class_weight="balanced"),
        "rf_gini5": RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5,
                                           min_samples_split=10, random_state=SEMILLA,
                                           class_weight="balanced"),
    }


def cargar_datos(ruta=DATASET):
    """Carga el CSV (lo genera con el script del repositorio si falta) y arma X, y."""
    ruta = Path(ruta)
    if not ruta.exists():
        runpy.run_path("Dataset_10000_Bolivia_Sintetico.py")
    df = pd.read_csv(ruta)
    return preparar(df)


def preparar(df):
    """Misma matriz de variables que train_model.py. Excluye id, score e índice de riesgo
    (se usaron para generar el objetivo: data leakage)."""
    dummies = pd.get_dummies(df[CATEGORICAS], prefix=CATEGORICAS, drop_first=True, dtype=int)
    X = pd.concat([df[NUMERICAS], dummies], axis=1)
    return X, df[TARGET]


def crear_pipeline(estimador):
    """Escalado y SMOTE se ajustan SOLO con los datos de entrenamiento de cada pliegue."""
    return Pipeline([
        ("escalado", StandardScaler()),
        ("smote", SMOTE(random_state=SEMILLA, k_neighbors=5)),
        ("modelo", estimador),
    ])


def prueba_t_corregida(dif, n_train, n_test):
    """Prueba t pareada corregida para validación cruzada repetida (Nadeau y Bengio, 2003).

    Corrige que los pliegues comparten datos; sin la corrección, los p-valores salen
    demasiado optimistas. `dif` = diferencia por pliegue entre dos modelos."""
    dif = np.asarray(dif, dtype=float)
    k = len(dif)
    varianza = dif.var(ddof=1)
    if varianza == 0:
        return float("nan"), float("nan")
    t = dif.mean() / np.sqrt((1.0 / k + n_test / n_train) * varianza)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def metricas_prueba(y_real, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "f1_macro": f1_score(y_real, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_real, y_prob),
        "VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp),
    }


def evaluar(X, y, n_splits=5, n_repeats=3, test_size=0.2):
    """Evalúa todos los modelos bajo las mismas condiciones.

    Devuelve un diccionario con: cv_folds, cv_resumen, test, comparacion y (n_train, n_test).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEMILLA, stratify=y)
    # Un único objeto de validación: con semilla fija genera los MISMOS pliegues para cada modelo
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEMILLA)

    filas_folds, filas_test = [], []
    for clave, estimador in construir_modelos().items():
        nombre = NOMBRES[clave]
        pipe = crear_pipeline(estimador)

        res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=METRICAS, n_jobs=1)
        for i in range(n_splits * n_repeats):
            fila = {"modelo": nombre, "pliegue": i + 1}
            fila.update({m: float(res[f"test_{m}"][i]) for m in ORDEN_METRICAS})
            filas_folds.append(fila)

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        prob = pipe.predict_proba(X_te)[:, 1]
        fila = {"modelo": nombre}
        fila.update(metricas_prueba(y_te, pred, prob))
        filas_test.append(fila)

    cv_folds = pd.DataFrame(filas_folds)
    agrupado = cv_folds.groupby("modelo", sort=False)[ORDEN_METRICAS]
    resumen = agrupado.mean().add_suffix("_media").join(agrupado.std(ddof=1).add_suffix("_desv"))
    resumen = resumen.reset_index()
    test = pd.DataFrame(filas_test)

    # Comparación estadística del mejor modelo contra los demás (métrica principal)
    medias = cv_folds.groupby("modelo", sort=False)[METRICA_PRINCIPAL].mean()
    mejor = medias.idxmax()
    base = cv_folds[cv_folds["modelo"] == mejor][METRICA_PRINCIPAL].to_numpy()
    filas_cmp = []
    for nombre in medias.index:
        if nombre == mejor:
            continue
        otro = cv_folds[cv_folds["modelo"] == nombre][METRICA_PRINCIPAL].to_numpy()
        t, p = prueba_t_corregida(base - otro, n_train=len(X_tr), n_test=len(X_te))
        filas_cmp.append({"mejor_modelo": mejor, "comparado_con": nombre,
                          "diferencia_f1_macro": float((base - otro).mean()),
                          "t": t, "p_valor": p,
                          "diferencia_significativa_p<0.05": bool(p < 0.05)})
    comparacion = pd.DataFrame(filas_cmp)

    return {"cv_folds": cv_folds, "cv_resumen": resumen, "test": test,
            "comparacion": comparacion, "mejor": mejor,
            "n_train": len(X_tr), "n_test": len(X_te)}


def guardar_tablas(res, carpeta=SALIDA):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    res["cv_folds"].round(6).to_csv(carpeta / "cv_por_pliegue.csv", index=False)
    res["cv_resumen"].round(4).to_csv(carpeta / "cv_resumen.csv", index=False)
    res["test"].round(4).to_csv(carpeta / "test_resultados.csv", index=False)
    res["comparacion"].round(4).to_csv(carpeta / "comparacion_estadistica.csv", index=False)


def graficar(res, carpeta=SALIDA):
    """Tres gráficos: métricas medias con desviación, dispersión por pliegue y matrices de confusión."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = res["cv_resumen"].set_index("modelo")
    modelos = list(resumen.index)
    resalta = res["mejor"]
    gris, dorado, oscuro = "#b9b0c8", "#e0a42e", "#232133"

    # 1) Métricas principales con barras de error (media +/- desviación entre pliegues)
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metrica, titulo in zip(ejes, ["f1_macro", "recall"],
                                   ["F1-macro (desempeño general)", "Recall de 'Ocurre' (detección)"]):
        medias = resumen[f"{metrica}_media"].to_numpy()
        desv = resumen[f"{metrica}_desv"].to_numpy()
        colores = [dorado if m == resalta else gris for m in modelos]
        pos = np.arange(len(modelos))
        ax.barh(pos, medias, xerr=desv, color=colores, ecolor=oscuro, capsize=4)
        ax.set_yticks(pos)
        ax.set_yticklabels(modelos)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(titulo, fontsize=13, fontweight="bold")
        ax.set_xlabel("Media de la validación cruzada (línea = desviación)")
        for p, m, d in zip(pos, medias, desv):
            ax.text(m + d + 0.02, p, f"{m:.3f}", va="center", fontsize=11)
    fig.suptitle("Comparación de algoritmos: validación cruzada 5x3 (mismas condiciones)", fontsize=14)
    fig.text(0.5, 0.005, "Dorado = mejor F1-macro de la validación cruzada. Las diferencias entre modelos "
             "reales son pequeñas (ver comparacion_estadistica.csv).", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(carpeta / "grafico_1_metricas_principales.png", dpi=150)
    plt.close(fig)

    # 2) Dispersión de F1-macro por pliegue (qué tan estable es cada modelo)
    fig, ax = plt.subplots(figsize=(10, 5))
    reales = [m for m in modelos if not m.startswith("Referencia")]
    datos = [res["cv_folds"].loc[res["cv_folds"]["modelo"] == m, METRICA_PRINCIPAL] for m in reales]
    caja = ax.boxplot(datos, patch_artist=True, widths=0.55)
    for parte, m in zip(caja["boxes"], reales):
        parte.set_facecolor(dorado if m == resalta else gris)
    for mediana in caja["medians"]:
        mediana.set_color(oscuro)
    ax.set_xticks(range(1, len(reales) + 1))
    ax.set_xticklabels([m.replace(" (", "\n(").replace(", ", ",\n") for m in reales], fontsize=9)
    ax.set_ylabel("F1-macro por pliegue")
    ax.set_title("Estabilidad de cada modelo en los 15 pliegues (sin la referencia, que da 0,36)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(carpeta / "grafico_2_dispersion_pliegues.png", dpi=150)
    plt.close(fig)

    # 3) Matrices de confusión sobre el 20 % de prueba
    test = res["test"].set_index("modelo")
    fig, ejes = plt.subplots(1, len(modelos), figsize=(3.6 * len(modelos), 3.9))
    for ax, m in zip(np.atleast_1d(ejes), modelos):
        f = test.loc[m]
        matriz = np.array([[f["VN"], f["FP"]], [f["FN"], f["VP"]]])
        ax.imshow(matriz, cmap="Purples", vmin=0, vmax=1124)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matriz[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if matriz[i, j] > 560 else oscuro)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No ocurre", "Ocurre"], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No ocurre", "Ocurre"], fontsize=9)
        ax.set_xlabel("Predicho")
        ax.set_title(m.replace(" (", "\n("), fontsize=10)
    np.atleast_1d(ejes)[0].set_ylabel("Real")
    fig.suptitle("Matrices de confusión en los 2.000 registros de prueba", fontsize=13)
    fig.tight_layout()
    fig.savefig(carpeta / "grafico_3_matrices_confusion.png", dpi=150)
    plt.close(fig)


def verificar_reproducibilidad(X, y, **kwargs):
    """Ejecuta la evaluación dos veces y comprueba que los resultados son idénticos."""
    a = evaluar(X, y, **kwargs)
    b = evaluar(X, y, **kwargs)
    iguales = (a["cv_folds"].equals(b["cv_folds"]) and a["test"].equals(b["test"])
               and a["comparacion"].equals(b["comparacion"]))
    return iguales


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--salida", default=SALIDA)
    parser.add_argument("--verificar", action="store_true",
                        help="repite toda la evaluación y comprueba que da lo mismo")
    args = parser.parse_args()

    X, y = cargar_datos(args.dataset)
    res = evaluar(X, y)
    guardar_tablas(res, args.salida)
    graficar(res, args.salida)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\nValidación cruzada 5x3 (media):")
    print(res["cv_resumen"][["modelo"] + [f"{m}_media" for m in ORDEN_METRICAS]].round(4).to_string(index=False))
    print("\nPrueba (20 % no usado en el entrenamiento):")
    print(res["test"].round(4).to_string(index=False))
    print(f"\nMejor modelo por {METRICA_PRINCIPAL} (validación cruzada): {res['mejor']}")
    print(res["comparacion"].round(4).to_string(index=False))

    if args.verificar:
        ok = verificar_reproducibilidad(X, y)
        print("\nReproducibilidad:", "OK, dos ejecuciones dan resultados idénticos" if ok else "FALLÓ")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
