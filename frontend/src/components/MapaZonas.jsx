import { useState } from 'react';
import { MapPin, AlertTriangle, CheckCircle, Navigation } from 'lucide-react';

const DEPARTAMENTOS = [
  {
    id: 'la_paz', nombre: 'La Paz', capital: 'La Paz',
    riesgo: 'Muy Alto', porcentaje: 88, color: '#b71c1c',
    precipitacion: 320, pendiente: 42, humedad: 92,
    descripcion: 'Zona de alta sismicidad y lluvias intensas. Riesgo máximo en valles.',
    municipios: ['La Paz', 'El Alto', 'Coroico', 'Caranavi']
  },
  {
    id: 'cochabamba', nombre: 'Cochabamba', capital: 'Cochabamba',
    riesgo: 'Alto', porcentaje: 72, color: '#f44336',
    precipitacion: 240, pendiente: 35, humedad: 78,
    descripcion: 'Terrenos inestables en zonas montañosas. Temporada crítica dic-feb.',
    municipios: ['Cochabamba', 'Sacaba', 'Quillacollo', 'Colomi']
  },
  {
    id: 'santa_cruz', nombre: 'Santa Cruz', capital: 'Santa Cruz',
    riesgo: 'Moderado', porcentaje: 45, color: '#ff9800',
    precipitacion: 180, pendiente: 18, humedad: 65,
    descripcion: 'Riesgo moderado en zonas de transición. Monitoreo preventivo activo.',
    municipios: ['Santa Cruz', 'Montero', 'Warnes', 'Portachuelo']
  },
  {
    id: 'oruro', nombre: 'Oruro', capital: 'Oruro',
    riesgo: 'Alto', porcentaje: 68, color: '#f44336',
    precipitacion: 210, pendiente: 30, humedad: 74,
    descripcion: 'Altiplano con suelos vulnerables a saturación hídrica.',
    municipios: ['Oruro', 'Huanuni', 'Machacamarca', 'Caracollo']
  },
  {
    id: 'potosi', nombre: 'Potosí', capital: 'Potosí',
    riesgo: 'Muy Alto', porcentaje: 82, color: '#b71c1c',
    precipitacion: 290, pendiente: 38, humedad: 85,
    descripcion: 'Terrenos mineros y pendientes pronunciadas. Riesgo muy elevado.',
    municipios: ['Potosí', 'Llallagua', 'Uyuni', 'Tupiza']
  },
  {
    id: 'chuquisaca', nombre: 'Chuquisaca', capital: 'Sucre',
    riesgo: 'Moderado', porcentaje: 48, color: '#ff9800',
    precipitacion: 195, pendiente: 25, humedad: 70,
    descripcion: 'Zona de valles con riesgo moderado. Vigilancia en temporada de lluvias.',
    municipios: ['Sucre', 'Camargo', 'Villa Serrano', 'Monteagudo']
  },
  {
    id: 'tarija', nombre: 'Tarija', capital: 'Tarija',
    riesgo: 'Bajo', porcentaje: 22, color: '#4caf50',
    precipitacion: 120, pendiente: 15, humedad: 55,
    descripcion: 'Condiciones climáticas estables. Riesgo bajo en condiciones normales.',
    municipios: ['Tarija', 'Bermejo', 'Yacuiba', 'Entre Ríos']
  },
  {
    id: 'beni', nombre: 'Beni', capital: 'Trinidad',
    riesgo: 'Bajo', porcentaje: 18, color: '#4caf50',
    precipitacion: 95, pendiente: 8, humedad: 48,
    descripcion: 'Llanuras amazónicas con bajo riesgo de deslizamientos.',
    municipios: ['Trinidad', 'Riberalta', 'Guayaramerín', 'San Borja']
  },
  {
    id: 'pando', nombre: 'Pando', capital: 'Cobija',
    riesgo: 'Bajo', porcentaje: 15, color: '#4caf50',
    precipitacion: 85, pendiente: 5, humedad: 42,
    descripcion: 'Terreno plano amazónico. Riesgo mínimo de deslizamientos.',
    municipios: ['Cobija', 'Porvenir', 'Filadelfia', 'Puerto Rico']
  }
];

const RIESGO_BADGE = {
  'Bajo':     { bg: 'rgba(76,175,80,0.15)',   border: '#4caf50',  icon: CheckCircle },
  'Moderado': { bg: 'rgba(255,152,0,0.15)',   border: '#ff9800',  icon: AlertTriangle },
  'Alto':     { bg: 'rgba(244,67,54,0.15)',   border: '#f44336',  icon: AlertTriangle },
  'Muy Alto': { bg: 'rgba(183,28,28,0.15)',   border: '#b71c1c',  icon: AlertTriangle },
};

// Paths SVG aproximados de los departamentos de Bolivia
const SVG_PATHS = {
  pando:      "M 60 30 L 160 30 L 160 100 L 80 100 L 60 80 Z",
  beni:       "M 80 100 L 160 100 L 180 200 L 140 220 L 80 200 Z",
  la_paz:     "M 60 80 L 80 100 L 80 200 L 50 210 L 30 170 L 30 110 Z",
  oruro:      "M 30 170 L 50 210 L 80 200 L 90 240 L 50 260 L 20 230 Z",
  cochabamba: "M 80 200 L 140 220 L 150 260 L 120 280 L 90 270 L 90 240 Z",
  santa_cruz: "M 140 220 L 180 200 L 230 220 L 240 320 L 160 340 L 130 300 L 150 260 Z",
  potosi:     "M 50 260 L 90 240 L 90 270 L 120 280 L 130 320 L 90 340 L 50 320 Z",
  chuquisaca: "M 90 270 L 120 280 L 130 300 L 130 340 L 100 360 L 80 340 L 90 310 Z",
  tarija:     "M 90 310 L 130 300 L 160 340 L 150 390 L 110 400 L 80 370 Z",
};

export default function MapaZonas({ onEvaluarDepartamento }) {
  const [seleccionado, setSeleccionado] = useState(null);
  const [filtroRiesgo, setFiltroRiesgo] = useState('Todos');

  const depto = seleccionado ? DEPARTAMENTOS.find(d => d.id === seleccionado) : null;
  const lista = filtroRiesgo === 'Todos'
    ? [...DEPARTAMENTOS].sort((a, b) => b.porcentaje - a.porcentaje)
    : [...DEPARTAMENTOS].filter(d => d.riesgo === filtroRiesgo).sort((a, b) => b.porcentaje - a.porcentaje);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header className="top-header">
        <div>
          <h1>Mapa de Zonas de Riesgo</h1>
          <p>Visualización interactiva de niveles de riesgo por departamento — Bolivia</p>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: '2rem' }}>

        {/* Panel izquierdo: lista + filtro */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Filtro */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem', fontWeight: 600 }}>FILTRAR POR NIVEL</p>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {['Todos', 'Muy Alto', 'Alto', 'Moderado', 'Bajo'].map(n => (
                <button key={n} onClick={() => setFiltroRiesgo(n)}
                  style={{
                    padding: '0.35rem 0.85rem', borderRadius: '999px', border: '1px solid',
                    borderColor: filtroRiesgo === n ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                    background: filtroRiesgo === n ? 'rgba(59,130,246,0.2)' : 'transparent',
                    color: filtroRiesgo === n ? '#3b82f6' : 'var(--text-muted)',
                    cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600
                  }}>
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Lista de departamentos */}
          <div className="glass-card" style={{ padding: '1.25rem', maxHeight: '520px', overflowY: 'auto' }}>
            <h3 className="section-title" style={{ fontSize: '1rem' }}>Departamentos ({lista.length})</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {lista.map(d => {
                const badge = RIESGO_BADGE[d.riesgo];
                return (
                  <div key={d.id} onClick={() => setSeleccionado(d.id === seleccionado ? null : d.id)}
                    style={{
                      padding: '0.85rem 1rem', borderRadius: '0.75rem', cursor: 'pointer',
                      background: seleccionado === d.id ? 'rgba(59,130,246,0.15)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${seleccionado === d.id ? '#3b82f6' : 'rgba(255,255,255,0.07)'}`,
                      transition: 'all 0.2s'
                    }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <MapPin size={14} color={d.color} />
                        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{d.nombre}</span>
                      </div>
                      <span style={{
                        fontSize: '0.75rem', fontWeight: 700, padding: '0.2rem 0.6rem',
                        borderRadius: '999px', background: badge.bg, border: `1px solid ${badge.border}`,
                        color: d.color
                      }}>{d.riesgo}</span>
                    </div>
                    {/* Barra de progreso */}
                    <div style={{ marginTop: '0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '999px', height: '4px' }}>
                      <div style={{ width: `${d.porcentaje}%`, height: '100%', borderRadius: '999px', background: d.color, transition: 'width 0.6s' }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.3rem' }}>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{d.capital}</span>
                      <span style={{ fontSize: '0.72rem', color: d.color, fontWeight: 700 }}>{d.porcentaje}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Panel derecho: mapa + detalle */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Mapa SVG Bolivia */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 className="section-title" style={{ fontSize: '1rem' }}>Bolivia — Mapa de Riesgo</h3>
            <svg viewBox="0 0 270 430" style={{ width: '100%', maxHeight: '320px' }}>
              {DEPARTAMENTOS.map(d => (
                <g key={d.id} onClick={() => setSeleccionado(d.id === seleccionado ? null : d.id)}
                  style={{ cursor: 'pointer' }}>
                  <path d={SVG_PATHS[d.id]}
                    fill={seleccionado === d.id ? d.color : `${d.color}88`}
                    stroke={seleccionado === d.id ? '#fff' : `${d.color}cc`}
                    strokeWidth={seleccionado === d.id ? 2.5 : 1.5}
                    style={{ transition: 'all 0.2s' }}
                  />
                </g>
              ))}
              {/* Leyenda interna */}
              {[['#b71c1c','Muy Alto'],['#f44336','Alto'],['#ff9800','Moderado'],['#4caf50','Bajo']].map(([c,l], i) => (
                <g key={l}>
                  <rect x={10} y={370 - i * 20} width={12} height={12} fill={c} rx={2} />
                  <text x={26} y={381 - i * 20} fill="#94a3b8" fontSize={10}>{l}</text>
                </g>
              ))}
            </svg>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.5rem' }}>
              Haz clic en un departamento para ver detalles
            </p>
          </div>

          {/* Panel de detalle */}
          {depto ? (
            <div className="glass-card" style={{ padding: '1.5rem', borderColor: depto.color, borderWidth: '1px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#fff' }}>{depto.nombre}</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Capital: {depto.capital}</p>
                </div>
                <span style={{
                  padding: '0.4rem 1rem', borderRadius: '999px', fontWeight: 800,
                  background: RIESGO_BADGE[depto.riesgo].bg,
                  border: `1px solid ${depto.color}`, color: depto.color, fontSize: '0.9rem'
                }}>{depto.riesgo}</span>
              </div>

              <p style={{ fontSize: '0.88rem', color: '#cbd5e1', marginBottom: '1rem', lineHeight: 1.6 }}>{depto.descripcion}</p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
                {[
                  ['Lluvia 24h', `${depto.precipitacion} mm`],
                  ['Pendiente', `${depto.pendiente}°`],
                  ['Humedad', `${depto.humedad}%`]
                ].map(([k, v]) => (
                  <div key={k} style={{ background: 'rgba(255,255,255,0.04)', borderRadius: '0.5rem', padding: '0.6rem', textAlign: 'center' }}>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>{k}</p>
                    <p style={{ fontWeight: 700, color: depto.color }}>{v}</p>
                  </div>
                ))}
              </div>

              <div style={{ marginBottom: '1rem' }}>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.4rem', fontWeight: 600 }}>MUNICIPIOS EN VIGILANCIA</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                  {depto.municipios.map(m => (
                    <span key={m} style={{
                      padding: '0.2rem 0.6rem', borderRadius: '999px', fontSize: '0.75rem',
                      background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)'
                    }}>{m}</span>
                  ))}
                </div>
              </div>

              <button onClick={() => onEvaluarDepartamento && onEvaluarDepartamento(depto.nombre)}
                style={{
                  width: '100%', padding: '0.75rem', borderRadius: '0.5rem', border: 'none',
                  background: `linear-gradient(135deg, ${depto.color}, ${depto.color}88)`,
                  color: '#fff', fontWeight: 700, cursor: 'pointer', display: 'flex',
                  alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontSize: '0.9rem'
                }}>
                <Navigation size={16} />
                Evaluar {depto.nombre} en el Dashboard
              </button>
            </div>
          ) : (
            <div className="glass-card" style={{ padding: '2rem', textAlign: 'center' }}>
              <MapPin size={40} color="rgba(255,255,255,0.1)" style={{ margin: '0 auto 1rem' }} />
              <p style={{ color: 'var(--text-muted)' }}>Selecciona un departamento para ver el detalle de riesgo</p>
            </div>
          )}
        </div>
      </div>

      {/* Resumen estadístico */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        {[
          { nivel: 'Muy Alto', color: '#b71c1c', count: DEPARTAMENTOS.filter(d => d.riesgo === 'Muy Alto').length },
          { nivel: 'Alto',     color: '#f44336', count: DEPARTAMENTOS.filter(d => d.riesgo === 'Alto').length },
          { nivel: 'Moderado', color: '#ff9800', count: DEPARTAMENTOS.filter(d => d.riesgo === 'Moderado').length },
          { nivel: 'Bajo',     color: '#4caf50', count: DEPARTAMENTOS.filter(d => d.riesgo === 'Bajo').length },
        ].map(({ nivel, color, count }) => (
          <div key={nivel} className="glass-card" style={{ padding: '1.25rem', textAlign: 'center', borderColor: color }}>
            <p style={{ fontSize: '2rem', fontWeight: 800, color }}>{count}</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>{nivel}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
