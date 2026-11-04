# Integración K-Means en GeoRisk AI

K-Means complementa al clasificador de deslizamientos: no sustituye su probabilidad. Agrupa observaciones geoclimáticas semejantes y devuelve el perfil asociado en cada evaluación.

## Arquitectura y variables

`train_kmeans.py` lee el CSV, valida los datos, normaliza con `StandardScaler`, evalúa K=2..6 mediante silhouette y WCSS, persiste el mejor modelo y genera perfiles. Se usan precipitación de 24 horas, 3 y 7 días, pendiente, elevación y humedad. Nulos e infinitos se excluyen, se eliminan duplicados, se rechazan variables constantes y se limitan atípicos a los percentiles 1 y 99. La etiqueta de ocurrencia no participa del entrenamiento.

Los nombres se ordenan usando el índice histórico medio de riesgo únicamente para interpretar los clusters. El número de cluster no representa un nivel fijo.

## Entrenamiento y ejecución en Windows

```powershell
cd C:\Users\acer\Downloads\caso8\caso8
.\.venv\Scripts\Activate.ps1
python Dataset_10000_Bolivia_Sintetico.py
python train_model.py
python train_kmeans.py
uvicorn app:app --reload
```

Se generan tres archivos en `artifacts/`. `GET /segmentacion` muestra métricas y perfiles; `POST /predecir` incluye `segmentacion`. Si el artefacto falta, la predicción existente sigue disponible y devuelve segmentación nula.

## Pruebas

```powershell
python -m pytest -q
cd frontend
npm run lint
npm run build
```

## Limitaciones

El dataset es sintético y no contiene fecha ni coordenadas: los clusters describen condiciones, no zonas geográficas ni evolución temporal. Como no existe autenticación ni base de datos, el reentrenamiento es local y requiere reiniciar FastAPI.
