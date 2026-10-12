import pandas as pd
import numpy as np

# 1. Configuración inicial
n_registros = 10000
np.random.seed(42) # Para reproducibilidad

# 2. Generación de variables base (Topografía, Clima, Geografía)
datos = {
    'id_registro': range(1, n_registros + 1),
    'departamento': np.random.choice(['La Paz', 'Cochabamba', 'Chuquisaca', 'Tarija', 'Oruro', 'Potosi'], n_registros),
    'precipitacion_24h_mm': np.random.exponential(scale=15, size=n_registros), 
    'precipitacion_3d_mm': np.random.exponential(scale=45, size=n_registros),
    # Precipitaciones de 7 días con algunos valores extremos superando los 900mm
    'precipitacion_7d_mm': np.random.exponential(scale=120, size=n_registros) + np.random.choice([0, 500, 800], n_registros, p=[0.9, 0.08, 0.02]),
    'pendiente_grados': np.random.normal(loc=30, scale=15, size=n_registros).clip(0, 90),
    'elevacion_m': np.random.uniform(500, 5000, n_registros),
    'humedad_pct': np.random.normal(loc=60, scale=20, size=n_registros).clip(0, 100),
    'tipo_suelo': np.random.choice(['Franco', 'Arcilloso', 'Arenoso', 'Limoso'], n_registros),
    'geologia': np.random.choice(['Sedimentaria', 'Metamórfica', 'Ígnea'], n_registros)
}

df = pd.DataFrame(datos)

# 3. Creación de reglas de negocio para "probabilidad teórica" y "deslizamiento_ocurrido"
# Pesos teóricos: ~35% precipitación, ~25% pendiente, ~15% humedad (saturación)
df['score_riesgo_bruto'] = (
    (df['precipitacion_7d_mm'] / df['precipitacion_7d_mm'].max() * 35) +
    (df['pendiente_grados'] / 90 * 25) +
    (df['humedad_pct'] / 100 * 15) + 
    np.random.normal(0, 5, n_registros) # Ruido aleatorio
)

# Escalar el score a un índice de 0 a 100
min_score = df['score_riesgo_bruto'].min()
max_score = df['score_riesgo_bruto'].max()
df['indice_riesgo_deslizamiento'] = ((df['score_riesgo_bruto'] - min_score) / (max_score - min_score)) * 100

# 4. Generación de la Variable Target (Balanceo aproximado)
# Asumimos que si el índice supera 50, hay alta probabilidad de deslizamiento
probabilidad_falla = df['indice_riesgo_deslizamiento'] / 100
df['deslizamiento_ocurrido'] = np.random.binomial(1, probabilidad_falla)

# 5. Exportar a CSV
nombre_archivo = 'Dataset_10000_Bolivia_Sintetico.csv'
df.to_csv(nombre_archivo, index=False)

print(f"¡Archivo '{nombre_archivo}' generado exitosamente con {len(df)} registros!")
print(df['deslizamiento_ocurrido'].value_counts(normalize=True)) # Para verificar el balance