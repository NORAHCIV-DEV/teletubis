# Comprensión del negocio — CRISP-DM

## Problema y objetivo

GeoRisk AI es un proyecto académico de alerta de deslizamientos en Bolivia. Esta contribución estudia si las variables climáticas, topográficas y geológicas disponibles permiten discriminar la variable binaria `deslizamiento_ocurrido` del generador sintético, y compara dos clasificadores para apoyar la comprensión del problema.

## Criterio de éxito y alcance de la decisión

La comparación utiliza Accuracy, Precision, Recall, F1-Score y ROC-AUC en el mismo conjunto de prueba, junto con validación cruzada estratificada de cinco folds dentro de train. No se fija un umbral operativo ni se afirma que una métrica equivalga a seguridad en campo: el repositorio no define costos de errores ni aporta observaciones reales etiquetadas.

En test, Regresión Logística obtuvo ROC-AUC 0,6091 y el Árbol 0,5400. El Árbol obtuvo F1 0,4541 frente a 0,4175 de Regresión Logística. Esta diferencia de criterios requiere acordar prioridades del negocio antes de seleccionar un modelo.

## Supuestos y restricciones

- Cada fila representa una observación geoclimática sintética.
- La etiqueta es binaria: 0 = No ocurre, 1 = Ocurre.
- El objetivo es estudiar y documentar el proceso, no emitir recomendaciones de emergencia.
- Los datos simulados no sustituyen lluvia medida, cartografía, geotecnia, coordenadas ni validación de especialistas.

## Fase CRISP-DM

Este documento cubre comprensión del negocio. La comprensión de datos, preparación, modelado, evaluación y recomendaciones se desarrollan en los demás documentos de `docs/INTEGRANTE_3/`.
