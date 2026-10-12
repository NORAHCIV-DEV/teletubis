import sys
import json
import joblib
import pandas as pd
import warnings
import base64
import os

warnings.filterwarnings('ignore')

try:
    # Leer el argumento en base64 pasado por PHP
    decoded_bytes = base64.b64decode(sys.argv[1])
    input_data = json.loads(decoded_bytes.decode('utf-8'))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo = joblib.load(os.path.join(base_dir, "modelo_deslizamientos.joblib"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.joblib"))
    columnas = joblib.load(os.path.join(base_dir, "columnas_modelo.joblib"))
    
    data_dict = {col: 0 for col in columnas}
    data_dict['precipitacion_24h_mm'] = float(input_data['precipitacion_24h_mm'])
    data_dict['precipitacion_3d_mm'] = float(input_data['precipitacion_3d_mm'])
    data_dict['precipitacion_7d_mm'] = float(input_data['precipitacion_7d_mm'])
    data_dict['pendiente_grados'] = float(input_data['pendiente_grados'])
    data_dict['elevacion_m'] = float(input_data['elevacion_m'])
    data_dict['humedad_pct'] = float(input_data['humedad_pct'])
    
    dep_col = f"departamento_{input_data['departamento']}"
    if dep_col in data_dict: data_dict[dep_col] = 1
        
    suelo_col = f"tipo_suelo_{input_data['tipo_suelo']}"
    if suelo_col in data_dict: data_dict[suelo_col] = 1
        
    geo_col = f"geologia_{input_data['geologia']}"
    if geo_col in data_dict: data_dict[geo_col] = 1

    df_input = pd.DataFrame([data_dict])
    X_scaled = scaler.transform(df_input)
    prob = float(modelo.predict_proba(X_scaled)[0][1])
    
    indice = round(prob * 100, 2)
    if indice < 25:
        res = {'indice': indice, 'nivel': 'BAJO', 'color': '#4CAF50', 'alerta': 'Sin alerta', 'accion': 'Monitoreo rutinario.'}
    elif indice < 50:
        res = {'indice': indice, 'nivel': 'MODERADO', 'color': '#FFEB3B', 'alerta': 'Vigilancia', 'accion': 'Aumentar frecuencia de monitoreo.'}
    elif indice < 75:
        res = {'indice': indice, 'nivel': 'ALTO', 'color': '#FF9800', 'alerta': 'Alerta preventiva', 'accion': 'ACTIVAR PROTOCOLO. Preparar evacuación.'}
    else:
        res = {'indice': indice, 'nivel': 'MUY ALTO', 'color': '#F44336', 'alerta': 'Evacuación', 'accion': 'EVACUACIÓN INMEDIATA.'}
        
    print(json.dumps(res))
except Exception as e:
    print(json.dumps({"error": str(e)}))
