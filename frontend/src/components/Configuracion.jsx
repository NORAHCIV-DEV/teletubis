import { useState, useEffect } from 'react';
import { Settings, Bell, Info, Save, RotateCcw, CheckCircle, AlertTriangle } from 'lucide-react';

const DEFAULTS = {
  lluvia_critica_24h: 80,
  lluvia_critica_7d: 300,
  pendiente_critica: 35,
  humedad_critica: 85,
};

const DEFAULT_NOTIF = {
  alerta_alto: true,
  alerta_muy_alto: true,
  sonido: false,
  notif_navegador: false,
};

function Toggle({ activo, onChange }) {
  return (
    <div onClick={onChange} style={{
      width: '44px', height: '24px', borderRadius: '999px', cursor: 'pointer',
      background: activo ? '#3b82f6' : 'rgba(255,255,255,0.1)',
      position: 'relative', transition: 'background 0.25s', flexShrink: 0
    }}>
      <div style={{
        width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
        position: 'absolute', top: '3px',
        left: activo ? '23px' : '3px', transition: 'left 0.25s'
      }} />
    </div>
  );
}

function Slider({ label, value, min, max, step = 1, unit, onChange, color = '#3b82f6' }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <span style={{ fontSize: '0.85rem', color: '#cbd5e1', fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: '0.9rem', fontWeight: 800, color }}>{value} {unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          width: '100%', appearance: 'none', height: '4px', borderRadius: '999px',
          background: `linear-gradient(to right, ${color} ${pct}%, rgba(255,255,255,0.1) ${pct}%)`,
          outline: 'none', cursor: 'pointer'
        }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.2rem' }}>
        <span style={{ fontSize: '0.7rem', color: '#475569' }}>{min} {unit}</span>
        <span style={{ fontSize: '0.7rem', color: '#475569' }}>{max} {unit}</span>
      </div>
    </div>
  );
}

export default function Configuracion() {
  const [tab, setTab] = useState('umbrales');
  const [umbrales, setUmbrales] = useState(() => {
    try { return JSON.parse(localStorage.getItem('georisk_umbrales')) || DEFAULTS; }
    catch { return DEFAULTS; }
  });
  const [notif, setNotif] = useState(() => {
    try { return JSON.parse(localStorage.getItem('georisk_notif')) || DEFAULT_NOTIF; }
    catch { return DEFAULT_NOTIF; }
  });
  const [guardado, setGuardado] = useState(false);
  const [infoModelo, setInfoModelo] = useState(null);
  const [loadingInfo, setLoadingInfo] = useState(false);

  useEffect(() => {
    if (tab === 'info' && !infoModelo) {
      setLoadingInfo(true);
      fetch('http://localhost:8000/info')
        .then(r => r.json())
        .then(d => setInfoModelo(d))
        .catch(() => setInfoModelo({ error: true }))
        .finally(() => setLoadingInfo(false));
    }
  }, [tab, infoModelo]);

  const guardarUmbrales = () => {
    localStorage.setItem('georisk_umbrales', JSON.stringify(umbrales));
    localStorage.setItem('georisk_notif', JSON.stringify(notif));
    setGuardado(true);
    setTimeout(() => setGuardado(false), 2500);
  };

  const restaurarDefaults = () => {
    setUmbrales(DEFAULTS);
    setNotif(DEFAULT_NOTIF);
  };

  const toggleNotif = (key) => {
    if (key === 'notif_navegador' && !notif.notif_navegador) {
      Notification.requestPermission().then(perm => {
        if (perm === 'granted') setNotif(p => ({ ...p, notif_navegador: true }));
      });
    } else {
      setNotif(p => ({ ...p, [key]: !p[key] }));
    }
  };

  const TABS = [
    { id: 'umbrales',  label: 'Umbrales de Alerta', icon: Settings },
    { id: 'notif',     label: 'Notificaciones',      icon: Bell },
    { id: 'info',      label: 'Sobre el Sistema',    icon: Info },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', maxWidth: '800px' }}>
      <header className="top-header">
        <div>
          <h1>Configuración</h1>
          <p>Personaliza los parámetros y notificaciones del sistema GeoRisk AI</p>
        </div>
      </header>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0' }}>
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.7rem 1.2rem', border: 'none', cursor: 'pointer',
              background: 'transparent', fontWeight: 600, fontSize: '0.88rem',
              color: tab === id ? '#3b82f6' : '#64748b',
              borderBottom: tab === id ? '2px solid #3b82f6' : '2px solid transparent',
              transition: 'all 0.2s', marginBottom: '-1px'
            }}>
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* Tab: Umbrales */}
      {tab === 'umbrales' && (
        <div className="glass-card">
          <h2 className="section-title" style={{ fontSize: '1rem' }}>
            Umbrales críticos — valores que activan las alertas
          </h2>
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '2rem' }}>
            Ajusta los límites según las condiciones específicas de la zona que estás monitoreando.
          </p>

          <Slider label="Lluvia crítica en 24 horas" value={umbrales.lluvia_critica_24h}
            min={20} max={200} unit="mm" color="#3b82f6"
            onChange={v => setUmbrales(p => ({ ...p, lluvia_critica_24h: v }))} />
          <Slider label="Lluvia crítica en 7 días" value={umbrales.lluvia_critica_7d}
            min={50} max={600} unit="mm" color="#6366f1"
            onChange={v => setUmbrales(p => ({ ...p, lluvia_critica_7d: v }))} />
          <Slider label="Pendiente crítica" value={umbrales.pendiente_critica}
            min={10} max={80} unit="°" color="#f44336"
            onChange={v => setUmbrales(p => ({ ...p, pendiente_critica: v }))} />
          <Slider label="Humedad de suelo crítica" value={umbrales.humedad_critica}
            min={40} max={100} unit="%" color="#ff9800"
            onChange={v => setUmbrales(p => ({ ...p, humedad_critica: v }))} />

          {/* Previsualización de umbrales */}
          <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '0.75rem', padding: '1rem', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
            <p style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '0.75rem', fontWeight: 600 }}>RESUMEN DE UMBRALES CONFIGURADOS</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              {[
                ['Lluvia 24h', `> ${umbrales.lluvia_critica_24h} mm`, '#3b82f6'],
                ['Lluvia 7 días', `> ${umbrales.lluvia_critica_7d} mm`, '#6366f1'],
                ['Pendiente', `> ${umbrales.pendiente_critica}°`, '#f44336'],
                ['Humedad', `> ${umbrales.humedad_critica}%`, '#ff9800'],
              ].map(([k, v, c]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
                  <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>{k}</span>
                  <span style={{ fontSize: '0.82rem', fontWeight: 700, color: c }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button onClick={guardarUmbrales}
              style={{
                flex: 1, padding: '0.85rem', borderRadius: '0.6rem', border: 'none',
                background: guardado ? 'rgba(76,175,80,0.3)' : 'linear-gradient(135deg, #3b82f6, #6366f1)',
                color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: '0.9rem',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                transition: 'all 0.3s'
              }}>
              {guardado ? <><CheckCircle size={18} /> ¡Guardado!</> : <><Save size={18} /> Guardar configuración</>}
            </button>
            <button onClick={restaurarDefaults}
              style={{
                padding: '0.85rem 1.2rem', borderRadius: '0.6rem',
                border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)',
                color: '#94a3b8', fontWeight: 600, cursor: 'pointer', fontSize: '0.9rem',
                display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}>
              <RotateCcw size={16} /> Restaurar
            </button>
          </div>
        </div>
      )}

      {/* Tab: Notificaciones */}
      {tab === 'notif' && (
        <div className="glass-card">
          <h2 className="section-title" style={{ fontSize: '1rem' }}>Preferencias de Notificación</h2>
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '2rem' }}>
            Configura cuándo y cómo GeoRisk AI te notificará sobre alertas.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {[
              { key: 'alerta_alto',       label: 'Alerta cuando el riesgo sea "Alto"',       desc: 'Notifica cuando el porcentaje de riesgo supere 50%',  color: '#f44336' },
              { key: 'alerta_muy_alto',   label: 'Alerta cuando el riesgo sea "Muy Alto"',   desc: 'Notifica en casos de riesgo máximo — evacuación',      color: '#b71c1c' },
              { key: 'sonido',            label: 'Sonido de alerta',                          desc: 'Reproduce un tono de alerta al detectar riesgo alto', color: '#ff9800' },
              { key: 'notif_navegador',   label: 'Notificaciones del navegador',              desc: 'Muestra notificaciones nativas del sistema operativo', color: '#3b82f6' },
            ].map(({ key, label, desc, color }, i, arr) => (
              <div key={key} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '1.1rem 0',
                borderBottom: i < arr.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none'
              }}>
                <div style={{ flex: 1, marginRight: '1rem' }}>
                  <p style={{ fontWeight: 600, marginBottom: '0.2rem', fontSize: '0.9rem' }}>{label}</p>
                  <p style={{ fontSize: '0.78rem', color: '#64748b' }}>{desc}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ fontSize: '0.78rem', color: notif[key] ? color : '#475569', fontWeight: 600 }}>
                    {notif[key] ? 'Activado' : 'Desactivado'}
                  </span>
                  <Toggle activo={notif[key]} onChange={() => toggleNotif(key)} />
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(59,130,246,0.08)', borderRadius: '0.75rem', border: '1px solid rgba(59,130,246,0.2)' }}>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              <Info size={16} color="#3b82f6" style={{ marginTop: '2px', flexShrink: 0 }} />
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', lineHeight: 1.6 }}>
                Las notificaciones del navegador requieren permiso explícito del usuario y solo funcionan mientras la aplicación esté abierta.
              </p>
            </div>
          </div>

          <button onClick={guardarUmbrales} style={{
            marginTop: '1.5rem', width: '100%', padding: '0.85rem', borderRadius: '0.6rem', border: 'none',
            background: guardado ? 'rgba(76,175,80,0.3)' : 'linear-gradient(135deg, #3b82f6, #6366f1)',
            color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: '0.9rem',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', transition: 'all 0.3s'
          }}>
            {guardado ? <><CheckCircle size={18} /> ¡Guardado!</> : <><Save size={18} /> Guardar preferencias</>}
          </button>
        </div>
      )}

      {/* Tab: Info del sistema */}
      {tab === 'info' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-card">
            <h2 className="section-title" style={{ fontSize: '1rem' }}>Sobre GeoRisk AI</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
              {[
                ['Sistema', 'GeoRisk AI'],
                ['Versión', '1.0.0'],
                ['Institución', 'UNIFRANZ'],
                ['Materia', 'Ingeniería de Software 2'],
                ['Metodología', 'CRISP-DM'],
                ['Frontend', 'React + Vite'],
                ['Backend', 'FastAPI (Python)'],
                ['Dataset', '10,000 registros sintéticos'],
              ].map(([k, v]) => (
                <div key={k} style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '0.5rem' }}>
                  <p style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600, marginBottom: '0.2rem' }}>{k.toUpperCase()}</p>
                  <p style={{ fontSize: '0.9rem', fontWeight: 600 }}>{v}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Métricas del modelo */}
          <div className="glass-card">
            <h2 className="section-title" style={{ fontSize: '1rem' }}>Métricas del Modelo (en vivo)</h2>
            {loadingInfo && (
              <div style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>⏳</div>
                Calculando métricas del modelo...
              </div>
            )}
            {infoModelo && !infoModelo.error && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Exactitud (Accuracy)', value: `${(infoModelo.accuracy * 100).toFixed(1)}%`, color: '#4caf50', desc: 'Proporción de predicciones correctas' },
                  { label: 'F1-Score Macro', value: `${(infoModelo.f1_macro * 100).toFixed(1)}%`, color: '#3b82f6', desc: 'Promedio ponderado entre precisión y recall' },
                  { label: 'Estimadores (Árboles)', value: infoModelo.n_estimators, color: '#ff9800', desc: 'Cantidad de árboles en el Random Forest' },
                ].map(({ label, value, color, desc }) => (
                  <div key={label} style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.03)', borderRadius: '0.75rem', textAlign: 'center', border: `1px solid ${color}33` }}>
                    <p style={{ fontSize: '2rem', fontWeight: 800, color, marginBottom: '0.25rem' }}>{value}</p>
                    <p style={{ fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.25rem' }}>{label}</p>
                    <p style={{ fontSize: '0.72rem', color: '#64748b' }}>{desc}</p>
                  </div>
                ))}
              </div>
            )}
            {infoModelo?.error && (
              <div style={{ padding: '1rem', background: 'rgba(244,67,54,0.08)', borderRadius: '0.5rem', border: '1px solid rgba(244,67,54,0.2)' }}>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <AlertTriangle size={16} color="#f44336" />
                  <p style={{ fontSize: '0.85rem', color: '#f44336' }}>
                    No se pudo conectar con el backend. Asegúrate de que FastAPI esté corriendo en <strong>localhost:8000</strong>.
                  </p>
                </div>
              </div>
            )}
            {infoModelo && !infoModelo.error && (
              <p style={{ fontSize: '0.75rem', color: '#475569', marginTop: '1rem', textAlign: 'center' }}>
                Evaluado en conjunto de prueba · {infoModelo.evaluado_en}
              </p>
            )}
          </div>

          {/* Descripción del proyecto */}
          <div className="glass-card">
            <h2 className="section-title" style={{ fontSize: '1rem' }}>Descripción del Proyecto</h2>
            <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.8, marginBottom: '1rem' }}>
              <strong>GeoRisk AI</strong> es un sistema de alerta temprana para deslizamientos de tierra en Bolivia,
              desarrollado con Machine Learning. Predice el nivel de riesgo en 4 categorías
              (Bajo, Moderado, Alto, Muy Alto) basándose en condiciones climáticas y geológicas en tiempo real.
            </p>
            <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.8, marginBottom: '1rem' }}>
              El modelo fue entrenado con un dataset sintético de <strong>10,000 registros</strong> que simulan
              condiciones topográficas y climáticas de Bolivia, usando la técnica <strong>SMOTE</strong> para
              balancear clases y <strong>Random Forest</strong> como algoritmo principal.
            </p>
            <div style={{ padding: '0.75rem 1rem', background: 'rgba(255,152,0,0.08)', borderRadius: '0.5rem', border: '1px solid rgba(255,152,0,0.2)' }}>
              <p style={{ fontSize: '0.78rem', color: '#ff9800' }}>
                ⚠️ <strong>Limitación:</strong> El modelo fue entrenado con datos sintéticos. Para uso en producción,
                se requiere validación con datos reales de campo y sensores IoT.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
