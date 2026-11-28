# Análisis exploratorio de datos (EDA)

El EDA se ejecutó sobre los 10.000 registros antes de ajustar modelos. Las distribuciones de lluvia presentan asimetría y colas derechas; precipitación 7d tiene media 179,73 mm, mediana 98,12 mm y máximo 2.066,28 mm. Pendiente promedia 30,36° (mediana 30,17°), humedad 59,64 % (mediana 59,50 %) y elevación 2.761,38 m (mediana 2.744,14 m).

## Diferencias descriptivas por clase

| Variable | Clase 0: media | Clase 0: mediana | Clase 1: media | Clase 1: mediana |
|---|---:|---:|---:|---:|
| Precipitación 24 h (mm) | 14,98 | 10,39 | 15,00 | 10,47 |
| Precipitación 3 días (mm) | 45,15 | 31,56 | 44,70 | 30,27 |
| Precipitación 7 días (mm) | 155,43 | 88,58 | 210,88 | 114,94 |
| Pendiente (°) | 28,73 | 28,71 | 32,45 | 32,06 |
| Elevación (m) | 2.750,35 | 2.726,58 | 2.775,53 | 2.769,89 |
| Humedad (%) | 58,28 | 58,39 | 61,39 | 61,04 |

En esta muestra la clase 1 presenta mayores medias de lluvia de 7 días, pendiente y humedad. Las distribuciones se solapan y estas diferencias son descriptivas; no establecen causalidad.

## Correlación y categorías

La correlación lineal con el objetivo es 0,128 para precipitación 7d, 0,124 para pendiente y 0,079 para humedad. Precipitación 24h (0,001), precipitación 3d (-0,005) y elevación (0,010) presentan correlaciones cercanas a cero. Score e índice correlacionan 0,240 cada uno; están incluidos en el mapa solo como explicación exploratoria y no participan en el entrenamiento.

Las tasas del objetivo por categoría variaron entre 43,51 % y 44,92 % en departamentos; entre 42,98 % y 45,40 % en tipos de suelo; y entre 43,42 % y 44,41 % en geologías. La categoría Arcilloso presentó 45,40 %, y la categoría Arenoso 42,98 %. Diferencias pequeñas dentro de datos simulados no deben convertirse en afirmaciones territoriales.

## Visualizaciones guardadas

El notebook contiene histogramas por clase, boxplots, tasas de clase por categoría y una matriz de correlación. Las escalas simétrico-logarítmicas en acumulados de lluvia ayudan a visualizar la asimetría y los valores extremos.
