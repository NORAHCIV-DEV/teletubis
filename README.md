# Proyecto de Minería de Datos y Machine Learning: Alerta Temprana de Deslizamientos en Bolivia

Repositorio de modelos predictivos y analítica avanzada para la prevención y mitigación de riesgos de deslizamientos de ladera en Bolivia (VIDECI / SENAMHI / SERGEOMIN).

---

## 📂 Contenido del Repositorio

### 1. Modelos Principales de Clasificación (Metodología de 9 Pasos)
* **Regresión Logística (`statsmodels.api` & `scikit-learn`):**
  * Notebook Interactivo: [`Modelo de Regresión Logística para problemas de clasificación.ipynb`](./Modelo%20de%20Regresi%C3%B3n%20Log%C3%ADstica%20para%20problemas%20de%20clasificaci%C3%B3n.ipynb)
  * Documentación Técnica: [`Modelo de Regresión Logística para problemas de clasificación.md`](./Modelo%20de%20Regresi%C3%B3n%20Log%C3%ADstica%20para%20problemas%20de%20clasificaci%C3%B3n.md)
  * **Resultados Clave:** Exactitud en Test = 82.95%, Recall = 82.35%, AUC-ROC = 0.884, Inferencia estadística completa con p-valores.

* **Árboles de Decisión (`scikit-learn CART`):**
  * Notebook Interactivo: [`Modelo de árboles de decisión para problemas de clasificación.ipynb`](./Modelo%20de%20%C3%A1rboles%20de%20decisi%C3%B3n%20para%20problemas%20de%20clasificaci%C3%B3n.ipynb)
  * Documentación Técnica: [`Modelo de árboles de decisión para problemas de clasificación.md`](./Modelo%20de%20%C3%A1rboles%20de%20decisi%C3%B3n%20para%20problemas%20de%20clasificaci%C3%B3n.md)
  * **Resultados Clave:** Exactitud en Test = 83.35%, Recall = 83.90%, AUC-ROC = 0.885, Poda óptima con `max_depth=5` descartando sobreajuste.

---

### 2. Entregas y Evaluaciones Grupales
* **Evaluación Grupal CRISP-DM (Fases 1 a 4):**
  * Notebook: [`h3_03_evaluacion1_grupal_proyecto_crisp.ipynb`](./h3_03_evaluacion1_grupal_proyecto_crisp.ipynb)
  * Informe Técnico: [`h3_03_evaluacion1_grupal_proyecto_crisp.md`](./h3_03_evaluacion1_grupal_proyecto_crisp.md)

---

### 3. Datos y Scripts del Sistema
* Dataset Sintetizado de 10,000 registros: `Dataset_10000_Bolivia_Sintetico.csv`
* Sistema de Alerta Temprana en Python: `sistema_alerta_deslizamientos.py`
* Aplicación Web y API: `app.py`, `backend/`, `frontend/`