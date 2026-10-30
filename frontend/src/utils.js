export const API_BASE = 'http://localhost:8000';

export const KEY_HISTORIAL = 'georisk_historial';
export const KEY_CONFIG = 'georisk_config';

// Departamentos con los que se entrenó el modelo (Santa Cruz, Beni y Pando no están en el dataset)
export const DEPARTAMENTOS_MODELO = ['La Paz', 'Cochabamba', 'Potosi', 'Oruro', 'Chuquisaca', 'Tarija'];

export const CONFIG_DEFECTO = {
  umbrales: { lluvia24h: 80, lluvia7d: 300, pendiente: 35, humedad: 85 },
  notificaciones: { alto: true, muyAlto: true, sonido: false, navegador: false },
};

export const COLOR_NIVEL = {
  Bajo: '#4caf50',
  Moderado: '#ff9800',
  Alto: '#f44336',
  'Muy Alto': '#b71c1c',
};

export function leerLocal(clave, defecto) {
  try {
    const raw = localStorage.getItem(clave);
    return raw ? JSON.parse(raw) : defecto;
  } catch {
    return defecto;
  }
}

export function guardarLocal(clave, valor) {
  try {
    localStorage.setItem(clave, JSON.stringify(valor));
  } catch {
    /* localStorage no disponible: se ignora */
  }
}

export function formatearFecha(iso) {
  return new Date(iso).toLocaleString('es-BO', { dateStyle: 'short', timeStyle: 'medium' });
}
