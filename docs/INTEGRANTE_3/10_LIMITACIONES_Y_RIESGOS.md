# Limitaciones y riesgos

1. **Datos sintéticos:** los 10.000 registros no provienen de sensores, inventarios de eventos ni levantamientos geotécnicos; sus métricas no estiman rendimiento ante eventos reales.
2. **Etiqueta derivada de una regla:** `deslizamiento_ocurrido` se muestrea con probabilidad basada en un índice que se generó a partir de precipitación 7d, pendiente y humedad, con ruido. La aleatoriedad de la etiqueta limita la predictibilidad; al mismo tiempo, el proceso no simula toda la complejidad física de un deslizamiento.
3. **Incoherencia temporal:** 2.532 observaciones tienen precipitación 24h superior a 3d y 2.449 tienen precipitación 3d superior a 7d. El generador produce cada ventana independientemente.
4. **Rendimiento modesto:** ROC-AUC en test fue 0,5400 para Árbol y 0,6091 para Regresión Logística. Recall fue 0,4292 y 0,3265, respectivamente. No sustentan operación crítica.
5. **Errores asimétricos no definidos:** el análisis no cuenta con costos de falsos negativos y falsas alarmas acordados por responsables del sistema. El umbral 0,5 es convencional, no una decisión de política pública.
6. **Validación única:** se reporta un split estratificado reproducible y CV de cinco folds sobre train; no hay validación temporal, espacial ni externa.
7. **Supuesto de independencia:** la validación aleatoria presupone observaciones intercambiables. No se han definido coordenadas o dependencia temporal para evaluar generalización geográfica.
8. **Sensibilidad no causal:** las perturbaciones individuales pueden generar combinaciones climáticas físicamente inviables; no describen efectos en el terreno.
9. **Importancia limitada:** la importancia de impureza del Árbol tiene sesgos conocidos; coeficientes de Regresión Logística dependen de la codificación y categorías de referencia.
10. **Sin calibración operativa:** no se evaluó calibración de probabilidades, costo-beneficio, explicabilidad con usuarios finales ni umbrales de alerta aprobados.

Hasta cerrar estas limitaciones, el uso apropiado es educativo y de exploración metodológica. No debe utilizarse el resultado para alertas ni decisiones de evacuación.
