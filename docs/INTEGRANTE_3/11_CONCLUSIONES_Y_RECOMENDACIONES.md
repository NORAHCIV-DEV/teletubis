# Conclusiones y recomendaciones

## Conclusiones

- La inspección reproducible encontró 10.000 filas, 13 columnas, ningún nulo ni duplicado y un objetivo con 43,82 % de ocurrencia sintética.
- La partición 80/20 estratificada se hizo antes de ajustar imputación, escalado o codificación. Se excluyeron identificador, score e índice generadores del objetivo.
- Las clases difieren descriptivamente más en precipitación 7d (medias 155,43 frente a 210,88 mm), pendiente (28,73° frente a 32,45°) y humedad (58,28 % frente a 61,39 %). Se observa amplio solapamiento.
- Regresión Logística obtuvo la mayor ROC-AUC de test (0,6091); el Árbol obtuvo mayor recall (0,4292) y F1 (0,4541). Ambos tienen desempeño limitado y no existe selección de despliegue respaldada.
- La sensibilidad a +1 desviación estándar indicó aumentos medios de probabilidad en ambos modelos al perturbar precipitación 7d, pendiente y humedad. Estas medidas son específicas al modelo y dataset sintético.

## Recomendaciones

1. Corregir el generador para que acumulados por ventana cumplan coherencia temporal y documentar distribuciones y supuestos.
2. Reemplazar o complementar la simulación con observaciones geográficas, temporales y geotécnicas reales, auditadas y con etiquetas verificables.
3. Definir junto a los responsables del proyecto los costos de falsos negativos y falsas alarmas antes de proponer umbrales; elegirlos con datos de validación, no con el test final.
4. Evaluar división temporal y/o geográfica y realizar validación externa independiente.
5. Medir calibración y estabilidad de probabilidades y documentar resultados por departamento y condiciones relevantes cuando existan datos suficientes.
6. Mantener el notebook como evidencia reproducible del análisis académico, sin convertir sus resultados en alertas reales.

## Reproducibilidad

En la raíz del repositorio:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Analisis_Sensibilidad_GeoRisk_AI.ipynb
```

La corrida ejecutada guardó salidas tabulares y figuras en el notebook.
