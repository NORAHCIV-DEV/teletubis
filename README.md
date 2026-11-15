# GeoRisk AI - Sistema de Alerta de Deslizamientos

Proyecto de la materia **Ingeniería de Software 2** en UNIFRANZ. Es un sistema web que predice el riesgo de deslizamientos de tierra en Bolivia usando datos climáticos y geológicos.

## Integrantes
- Clara Machicado
- Juan [apellido]

## Qué hace el sistema

El usuario ingresa datos como lluvia acumulada, pendiente del terreno, tipo de suelo, etc. El sistema procesa eso con un modelo de Machine Learning (Random Forest) y devuelve un nivel de riesgo: Bajo, Moderado, Alto o Muy Alto. En la misma evaluación, K-Means asigna un perfil geoclimático y permite comparar el registro con observaciones de características similares.

La interfaz muestra el cluster asignado en el resultado principal, conserva el segmento en el historial y ofrece una vista de análisis con la distribución de observaciones y cuatro perfiles aprendidos. La comparación Silhouette/WCSS se conserva como diagnóstico del modelo y la vista aplica una narrativa H4 basada en Data Storytelling y principios Gestalt.

Los cuatro perfiles tienen nombres basados en su rasgo dominante: **Saturación por lluvia acumulada**, **Lluvia reciente de alta intensidad**, **Valles de menor elevación** y **Altiplano de gran elevación**. El módulo K-Means explica además cómo se generaron las 10.000 observaciones y qué representa cada variable.

## Tecnologías

- **Frontend:** React + Vite
- **Backend:** FastAPI (Python)
- **ML:** Random Forest + segmentación K-Means con scikit-learn
- **Metodología:** CRISP-DM

## Cómo ejecutar

### 1. Backend (FastAPI)
```bash
cd caso8
pip install -r requirements.txt
python Dataset_10000_Bolivia_Sintetico.py
python train_model.py      # genera el clasificador y sus artefactos
python train_kmeans.py     # genera la segmentación geoclimática
uvicorn app:app --port 8000
```

### 2. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

El frontend corre en `http://localhost:5173` y el backend en `http://localhost:8000`.

## Estructura del proyecto
```
caso8/
├── app.py                    # API FastAPI
├── train_model.py            # Entrenamiento del modelo
├── train_kmeans.py           # Entrenamiento de segmentos K-Means
├── kmeans_service.py         # Pipeline y clasificación de segmentos
├── docs/KMEANS_INTEGRACION.md
├── Dataset_10000_Bolivia_Sintetico.py   # Generador del dataset
├── Alerta_Deslizamientos_CRISP_DM.ipynb # Notebook con el análisis
├── H4_Entrega_KMeans_GeoRisk_AI.ipynb    # Entrega H4 ejecutada de clusterización K-Means
├── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Componente principal
│   │   └── components/
│   │       ├── MapaZonas.jsx
│   │       ├── Historial.jsx
│   │       ├── Configuracion.jsx
│   │       └── Segmentacion.jsx
│   └── package.json
└── .gitignore
```

## Notas
- Los archivos `.joblib` (modelo entrenado) no se suben a git porque pesan mucho. Hay que generarlos corriendo `train_model.py`.
- El dataset `.csv` tampoco se sube por el mismo motivo. Se genera con `Dataset_10000_Bolivia_Sintetico.py`.
- Los artefactos de K-Means se generan dentro de `artifacts/` y tampoco se versionan.

## Entrega H4: Notebook de K-Means y perfiles geoclimáticos

### Documentación y resultados

- [Notebook reproducible (K=4)](notebooks/H4_Entrega_KMeans_GeoRisk_AI.ipynb): descripción de variables, método del codo, Silhouette, visualizaciones, perfiles y nombres descriptivos de los clusters. Si falta el CSV sintético, el notebook lo genera con el script del repositorio.
- [Informe de K-Means](docs/INFORME_KMEANS_PARA_GITHUB.md): tabla de variables, características de cada cluster, resultados y limitaciones.
- El historial del frontend incluye el cluster y el nombre del segmento, y permite exportar los resultados a CSV.
- El archivo `H4_Entrega_KMeans_GeoRisk_AI.ipynb` contiene la preparación de datos, selección de K, visualizaciones, interpretación y conclusión de la entrega académica H4.

### Cómo ejecutar el notebook

1. Instalar las dependencias:

   `python -m pip install -r requirements.txt notebook`

2. Iniciar Jupyter Notebook:

   `python -m notebook`

3. Abrir `notebooks/H4_Entrega_KMeans_GeoRisk_AI.ipynb`.
4. Seleccionar **Run All** para ejecutar todas las celdas.

### Interpretación de los resultados

Se utilizan cuatro clusters (K=4) para la práctica académica y la interpretación de perfiles geoclimáticos.

El criterio Silhouette puede favorecer otra cantidad de clusters. Los datos utilizados son sintéticos y los grupos obtenidos no representan alertas reales de deslizamiento.

## Reproducir el analisis H4 de K-Means

Requisitos: Python 3.11 y las dependencias del proyecto.

1. Instalar dependencias:
   python -m pip install -r requirements.txt
   python -m pip install notebook ipykernel

2. Generar los 10000 registros sinteticos:
   python Dataset_10000_Bolivia_Sintetico.py

3. Iniciar Jupyter:
   python -m notebook

4. Abrir notebooks/H4_Entrega_KMeans_GeoRisk_AI.ipynb y seleccionar Run All.

El notebook utiliza K=4 para describir cuatro perfiles geoclimaticos.
El coeficiente Silhouette puede favorecer K=2.
Los datos son sinteticos y los clusters no representan alertas reales.

