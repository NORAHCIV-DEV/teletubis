import { useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import {
  LayoutDashboard, Activity, History, Settings, LogOut, Map as MapIcon
} from 'lucide-react';
import './App.css';
import MapaZonas from './components/MapaZonas';
import Historial from './components/Historial';
import Configuracion from './components/Configuracion';

const API_URL = 'http://localhost:8000/predecir';

function App() {
  const [moduloActivo, setModuloActivo] = useState('dashboard');

  const [formData, setFormData] = useState({
    precipitacion_24h_mm: 50,
    precipitacion_3d_mm: 120,
    precipitacion_7d_mm: 200,
    pendiente_grados: 25,
    elevacion_m: 2500,
    humedad_pct: 80,
    departamento: 'La Paz',
    tipo_suelo: 'Arcilloso',
    geologia: 'Sedimentaria'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [chartHistory, setChartHistory] = useState([]);     // para la gráfica del dashboard (máx 10)
  const [historialCompleto, setHistorialCompleto] = useState([]); // para el módulo Historial

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const camposNumericos = [
        'precipitacion_24h_mm', 'precipitacion_3d_mm', 'precipitacion_7d_mm',
        'pendiente_grados', 'elevacion_m', 'humedad_pct'
      ];
      const payload = { ...formData };
      camposNumericos.forEach((c) => { payload[c] = Number(payload[c]); });

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) {
        const detalle = Array.isArray(data.detail)
          ? data.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join('; ')
          : data.detail;
        throw new Error(detalle || `HTTP ${response.status}`);
      }

      setResult(data);

      const ahora = new Date();
      const hora = ahora.toLocaleTimeString('es-BO', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const fecha = ahora.toLocaleDateString('es-BO');

      // Registro para gráfica del dashboard (últimos 10)
      setChartHistory(prev => [...prev.slice(-9), {
        time: hora,
        riesgo: data.porcentaje,
        departamento: formData.departamento
      }]);

      // Registro completo para módulo Historial
      const registro = {
        id: Date.now(),
        fecha,
        hora,
        departamento: formData.departamento,
        pendiente: formData.pendiente_grados,
        lluvia24h: formData.precipitacion_24h_mm,
        humedad: formData.humedad_pct,
        nivel_riesgo: data.nivel_riesgo,
        porcentaje: data.porcentaje,
      };
      setHistorialCompleto(prev => [...prev, registro]);

    } catch (error) {
      alert('Error con el backend (FastAPI): ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Llamado desde MapaZonas → pre-selecciona el departamento y va al Dashboard
  const irAEvaluarDepartamento = (nombreDepto) => {
    setFormData(prev => ({ ...prev, departamento: nombreDepto }));
    setModuloActivo('dashboard');
  };

  // Limpiar historial
  const limpiarHistorial = () => {
    setHistorialCompleto([]);
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard',      icon: LayoutDashboard },
    { id: 'mapa',      label: 'Mapa de Zonas',  icon: MapIcon },
    { id: 'historial', label: 'Historial',       icon: History, badge: historialCompleto.length || null },
    { id: 'config',    label: 'Configuración',   icon: Settings },
  ];

  return (
    <div className="app-container">
      {/* ── Sidebar ── */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <div className="logo-icon"><Activity size={28} color="#3b82f6" /></div>
          <h2>GeoRisk AI</h2>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(({ id, label, icon: Icon, badge }) => (
            <a key={id} href="#" className={`nav-item ${moduloActivo === id ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); setModuloActivo(id); }}>
              <Icon size={20} />
              <span>{label}</span>
              {badge ? (
                <span style={{
                  marginLeft: 'auto', background: '#3b82f6', color: '#fff',
                  fontSize: '0.7rem', fontWeight: 700, padding: '0.15rem 0.5rem',
                  borderRadius: '999px', minWidth: '20px', textAlign: 'center'
                }}>{badge > 99 ? '99+' : badge}</span>
              ) : null}
            </a>
          ))}
        </nav>

        <div className="sidebar-footer">
          <a href="#" className="nav-item"
            onClick={(e) => { e.preventDefault(); if (window.confirm('¿Cerrar sesión?')) window.location.reload(); }}>
            <LogOut size={20} />
            <span>Salir</span>
          </a>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">

        {/* ── DASHBOARD ── */}
        {moduloActivo === 'dashboard' && (
          <>
            <header className="top-header">
              <div>
                <h1>Monitoreo de Deslizamientos</h1>
                <p>Sistema predictivo basado en Machine Learning (React + FastAPI)</p>
              </div>
            </header>

            <div className="dashboard-grid">
              {/* Formulario */}
              <div className="glass-card form-section">
                <h2 className="section-title">Parámetros Geológicos y Climáticos</h2>
                <form onSubmit={handleSubmit}>
                  <div className="input-grid">
                    <div className="input-group">
                      <label>Lluvia 24h (mm)</label>
                      <input type="number" step="0.1" name="precipitacion_24h_mm" value={formData.precipitacion_24h_mm} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Lluvia 3 días (mm)</label>
                      <input type="number" step="0.1" name="precipitacion_3d_mm" value={formData.precipitacion_3d_mm} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Lluvia 7 días (mm)</label>
                      <input type="number" step="0.1" name="precipitacion_7d_mm" value={formData.precipitacion_7d_mm} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Pendiente (°)</label>
                      <input type="number" step="0.1" name="pendiente_grados" value={formData.pendiente_grados} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Elevación (m)</label>
                      <input type="number" step="0.1" name="elevacion_m" value={formData.elevacion_m} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Humedad Suelo (%)</label>
                      <input type="number" step="0.1" name="humedad_pct" value={formData.humedad_pct} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                      <label>Departamento</label>
                      <select name="departamento" value={formData.departamento} onChange={handleChange}>
                        <option>La Paz</option><option>Cochabamba</option><option>Potosi</option>
                        <option>Oruro</option><option>Chuquisaca</option><option>Tarija</option>
                        <option>Santa Cruz</option><option>Beni</option><option>Pando</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label>Tipo de Suelo</label>
                      <select name="tipo_suelo" value={formData.tipo_suelo} onChange={handleChange}>
                        <option>Arcilloso</option><option>Arenoso</option><option>Limoso</option><option>Franco</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label>Geología</label>
                      <select name="geologia" value={formData.geologia} onChange={handleChange}>
                        <option>Sedimentaria</option><option>Ígnea</option><option>Metamórfica</option>
                      </select>
                    </div>
                  </div>
                  <button type="submit" className="btn-predict" disabled={loading}>
                    {loading ? 'Procesando modelo AI...' : 'Evaluar Nivel de Riesgo'}
                  </button>
                </form>
              </div>

              {/* Resultados */}
              <div className="results-section">
                <div className="glass-card text-center">
                  <h2 className="section-title">Predicción Actual</h2>
                  <div className="gauge-container" style={{ '--score': result ? result.porcentaje : 0, '--color': result ? result.color : '#334155' }}>
                    <div className="gauge">
                      <div className="gauge-fill"></div>
                      <div className="gauge-mask"></div>
                    </div>
                    <div className="gauge-value">
                      {result ? result.porcentaje : 0}<span className="pct">%</span>
                    </div>
                  </div>
                  {result ? (
                    <div className="alert-box" style={{ borderColor: result.color, boxShadow: `0 0 15px ${result.color}30` }}>
                      <h3 style={{ color: result.color }}>{result.nivel_riesgo}</h3>
                      <p><strong>{result.alerta}</strong></p>
                      {result.acciones.map((a) => (
                        <p key={a} className="action-text">{a}</p>
                      ))}
                    </div>
                  ) : (
                    <div className="alert-box empty-state">
                      <p>Esperando parámetros para evaluación...</p>
                    </div>
                  )}
                </div>

                <div className="glass-card chart-card">
                  <h2 className="section-title">Tendencia de Evaluaciones</h2>
                  {chartHistory.length > 0 ? (
                    <div style={{ width: '100%', height: 250 }}>
                      <ResponsiveContainer>
                        <AreaChart data={chartHistory} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="colorRiesgo" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                          <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                          <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'rgba(15,23,42,0.9)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                            itemStyle={{ color: '#60a5fa' }}
                          />
                          <Area type="monotone" dataKey="riesgo" stroke="#3b82f6" fillOpacity={1} fill="url(#colorRiesgo)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="empty-chart">
                      <Activity size={40} color="rgba(255,255,255,0.1)" />
                      <p>Realiza evaluaciones para ver la gráfica</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}

        {/* ── MAPA DE ZONAS ── */}
        {moduloActivo === 'mapa' && (
          <MapaZonas onEvaluarDepartamento={irAEvaluarDepartamento} />
        )}

        {/* ── HISTORIAL ── */}
        {moduloActivo === 'historial' && (
          <Historial historial={historialCompleto} onLimpiar={limpiarHistorial} />
        )}

        {/* ── CONFIGURACIÓN ── */}
        {moduloActivo === 'config' && (
          <Configuracion />
        )}
      </main>
    </div>
  );
}

export default App;
