# GeoRisk AI - Sistema de Alerta de Deslizamientos

Proyecto de la materia **Ingeniería de Software 2** en UNIFRANZ. Es un sistema web que predice el riesgo de deslizamientos de tierra en Bolivia usando datos climáticos y geológicos.

## Integrantes
- Clara Machicado
- Juan [apellido]

## Qué hace el sistema

El usuario ingresa datos como lluvia acumulada, pendiente del terreno, tipo de suelo, etc. El sistema procesa eso con un modelo de Machine Learning (Random Forest) y devuelve un nivel de riesgo: Bajo, Moderado, Alto o Muy Alto.

## Tecnologías

- **Frontend:** React + Vite
- **Backend:** FastAPI (Python)
- **ML:** Random Forest con scikit-learn
- **Metodología:** CRISP-DM

## Cómo ejecutar

### 1. Backend (FastAPI)
```bash
cd caso8
pip install -r requirements.txt
python train_model.py      # genera los archivos .joblib
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
├── Dataset_10000_Bolivia_Sintetico.py   # Generador del dataset
├── Alerta_Deslizamientos_CRISP_DM.ipynb # Notebook con el análisis
├── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Componente principal
│   │   └── components/
│   │       ├── MapaZonas.jsx
│   │       ├── Historial.jsx
│   │       └── Configuracion.jsx
│   └── package.json
└── .gitignore
```

## Notas
- Los archivos `.joblib` (modelo entrenado) no se suben a git porque pesan mucho. Hay que generarlos corriendo `train_model.py`.
- El dataset `.csv` tampoco se sube por el mismo motivo. Se genera con `Dataset_10000_Bolivia_Sintetico.py`.
