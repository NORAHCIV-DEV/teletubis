import { useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { Trash2, Download, History, TrendingUp, AlertTriangle } from 'lucide-react';

const BADGE = {
  'Bajo':     { bg: 'rgba(76,175,80,0.15)',  color: '#4caf50' },
  'Moderado': { bg: 'rgba(255,152,0,0.15)',  color: '#ff9800' },
  'Alto':     { bg: 'rgba(244,67,54,0.15)',  color: '#f44336' },
  'Muy Alto': { bg: 'rgba(183,28,28,0.15)',  color: '#b71c1c' },
};

function exportarCSV(historial) {
  const cabecera = ['#', 'Fecha', 'Hora', 'Departamento', 'Pendiente (°)',
                    'Lluvia 24h (mm)', 'Humedad (%)', 'Nivel de Riesgo', 'Porcentaje (%)'];
  const filas = historial.map((r, i) => [
    i + 1, r.fecha, r.hora, r.departamento,
    r.pendiente, r.lluvia24h, r.humedad, r.nivel_riesgo, r.porcentaje
  ]);
  const csv = [cabecera, ...filas].map(f => f.join(',')).join('\n');
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `GeoRiskAI_Historial_${new Date().toLocaleDateString('es-BO').replace(/\//g, '-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '0.75rem', padding: '0.75rem 1rem', fontSize: '0.82rem'
    }}>
      <p style={{ color: '#94a3b8', marginBottom: '0.25rem' }}>{d.departamento} · {d.hora}</p>
      <p style={{ color: BADGE[d.nivel_riesgo]?.color || '#fff', fontWeight: 700 }}>
        {d.nivel_riesgo} — {d.porcentaje}%
      </p>
    </div>
  );
};

export default function Historial({ historial, onLimpiar }) {
  const [confirmando, setConfirmando] = useState(false);
  const [paginaActual, setPaginaActual] = useState(1);
  const porPagina = 8;

  const total = historial.length;
  const totalPaginas = Math.ceil(total / porPagina);
  const inicio = (paginaActual - 1) * porPagina;
  const registrosPagina = [...historial].reverse().slice(inicio, inicio + porPagina);

  const promedio = total > 0
    ? Math.round(historial.reduce((s, r) => s + r.porcentaje, 0) / total)
    : 0;
  const maximo = total > 0 ? Math.max(...historial.map(r => r.porcentaje)) : 0;
  const criticos = historial.filter(r => r.nivel_riesgo === 'Muy Alto' || r.nivel_riesgo === 'Alto').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header className="top-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', width: '100%' }}>
          <div>
            <h1>Historial de Evaluaciones</h1>
            <p>Registro completo de predicciones realizadas en esta sesión</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {total > 0 && (
              <>
                <button onClick={() => exportarCSV(historial)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    padding: '0.6rem 1.1rem', borderRadius: '0.5rem', border: '1px solid rgba(59,130,246,0.4)',
                    background: 'rgba(59,130,246,0.1)', color: '#3b82f6',
                    cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem'
                  }}>
                  <Download size={16} /> Exportar CSV
                </button>
                {confirmando ? (
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.8rem', color: '#f44336' }}>¿Confirmar?</span>
                    <button onClick={() => { onLimpiar(); setConfirmando(false); setPaginaActual(1); }}
                      style={{ padding: '0.6rem 0.9rem', borderRadius: '0.5rem', border: '1px solid #f44336', background: 'rgba(244,67,54,0.15)', color: '#f44336', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem' }}>
                      Sí, limpiar
                    </button>
                    <button onClick={() => setConfirmando(false)}
                      style={{ padding: '0.6rem 0.9rem', borderRadius: '0.5rem', border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.05)', color: '#94a3b8', cursor: 'pointer', fontSize: '0.85rem' }}>
                      Cancelar
                    </button>
                  </div>
                ) : (
                  <button onClick={() => setConfirmando(true)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '0.5rem',
                      padding: '0.6rem 1.1rem', borderRadius: '0.5rem', border: '1px solid rgba(244,67,54,0.4)',
                      background: 'rgba(244,67,54,0.1)', color: '#f44336',
                      cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem'
                    }}>
                    <Trash2 size={16} /> Limpiar
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      {/* Tarjetas resumen */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        {[
          { label: 'Total evaluaciones', value: total, color: '#3b82f6', icon: History },
          { label: 'Promedio de riesgo', value: `${promedio}%`, color: '#ff9800', icon: TrendingUp },
          { label: 'Riesgo máximo', value: `${maximo}%`, color: '#f44336', icon: AlertTriangle },
          { label: 'Alertas críticas', value: criticos, color: '#b71c1c', icon: AlertTriangle },
        ].map(({ label, value, color, icon: Icon }) => (
          <div key={label} className="glass-card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <Icon size={22} color={color} style={{ margin: '0 auto 0.5rem' }} />
            <p style={{ fontSize: '1.8rem', fontWeight: 800, color }}>{value}</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>{label}</p>
          </div>
        ))}
      </div>

      {total === 0 ? (
        <div className="glass-card" style={{ padding: '4rem', textAlign: 'center' }}>
          <History size={60} color="rgba(255,255,255,0.07)" style={{ margin: '0 auto 1.5rem' }} />
          <h3 style={{ color: '#64748b', marginBottom: '0.5rem' }}>No hay evaluaciones registradas aún</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Ve al <strong style={{ color: '#3b82f6' }}>Dashboard</strong> para realizar tu primera evaluación.
          </p>
        </div>
      ) : (
        <>
          {/* Gráfico de tendencia */}
          <div className="glass-card chart-card">
            <h2 className="section-title">Tendencia de Riesgo</h2>
            <div style={{ width: '100%', height: 220 }}>
              <ResponsiveContainer>
                <AreaChart data={historial} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gradHist" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#f44336" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#f44336" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                  <XAxis dataKey="hora" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={75} stroke="#f44336" strokeDasharray="4 4" label={{ value: 'Alto', fill: '#f44336', fontSize: 11 }} />
                  <ReferenceLine y={50} stroke="#ff9800" strokeDasharray="4 4" label={{ value: 'Moderado', fill: '#ff9800', fontSize: 11 }} />
                  <Area type="monotone" dataKey="porcentaje" stroke="#f44336" fill="url(#gradHist)" strokeWidth={2} dot={{ fill: '#f44336', r: 3 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Tabla */}
          <div className="glass-card" style={{ padding: '1.5rem', overflowX: 'auto' }}>
            <h2 className="section-title">Registros ({total})</h2>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  {['#', 'Fecha', 'Hora', 'Departamento', 'Pendiente', 'Lluvia 24h', 'Humedad', 'Nivel de Riesgo', '%'].map(h => (
                    <th key={h} style={{ padding: '0.75rem', textAlign: 'left', color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {registrosPagina.map((r, i) => {
                  const n = total - inicio - i;
                  const badge = BADGE[r.nivel_riesgo] || {};
                  return (
                    <tr key={r.id || i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', transition: 'background 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                      <td style={{ padding: '0.85rem 0.75rem', color: '#64748b' }}>{n}</td>
                      <td style={{ padding: '0.85rem 0.75rem', whiteSpace: 'nowrap' }}>{r.fecha}</td>
                      <td style={{ padding: '0.85rem 0.75rem', color: '#94a3b8' }}>{r.hora}</td>
                      <td style={{ padding: '0.85rem 0.75rem', fontWeight: 600 }}>{r.departamento}</td>
                      <td style={{ padding: '0.85rem 0.75rem', color: '#94a3b8' }}>{r.pendiente}°</td>
                      <td style={{ padding: '0.85rem 0.75rem', color: '#94a3b8' }}>{r.lluvia24h} mm</td>
                      <td style={{ padding: '0.85rem 0.75rem', color: '#94a3b8' }}>{r.humedad}%</td>
                      <td style={{ padding: '0.85rem 0.75rem' }}>
                        <span style={{
                          padding: '0.25rem 0.7rem', borderRadius: '999px', fontWeight: 700, fontSize: '0.78rem',
                          background: badge.bg, color: badge.color, border: `1px solid ${badge.color}44`
                        }}>{r.nivel_riesgo}</span>
                      </td>
                      <td style={{ padding: '0.85rem 0.75rem', fontWeight: 800, color: badge.color }}>{r.porcentaje}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {/* Paginación */}
            {totalPaginas > 1 && (
              <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', marginTop: '1.25rem' }}>
                <button onClick={() => setPaginaActual(p => Math.max(1, p - 1))} disabled={paginaActual === 1}
                  style={{ padding: '0.4rem 0.9rem', borderRadius: '0.4rem', border: '1px solid rgba(255,255,255,0.1)', background: 'transparent', color: paginaActual === 1 ? '#334155' : '#94a3b8', cursor: paginaActual === 1 ? 'default' : 'pointer' }}>
                  ← Anterior
                </button>
                {Array.from({ length: totalPaginas }, (_, i) => i + 1).map(p => (
                  <button key={p} onClick={() => setPaginaActual(p)}
                    style={{ padding: '0.4rem 0.75rem', borderRadius: '0.4rem', border: '1px solid', borderColor: p === paginaActual ? '#3b82f6' : 'rgba(255,255,255,0.1)', background: p === paginaActual ? 'rgba(59,130,246,0.2)' : 'transparent', color: p === paginaActual ? '#3b82f6' : '#94a3b8', cursor: 'pointer', fontWeight: p === paginaActual ? 700 : 400 }}>
                    {p}
                  </button>
                ))}
                <button onClick={() => setPaginaActual(p => Math.min(totalPaginas, p + 1))} disabled={paginaActual === totalPaginas}
                  style={{ padding: '0.4rem 0.9rem', borderRadius: '0.4rem', border: '1px solid rgba(255,255,255,0.1)', background: 'transparent', color: paginaActual === totalPaginas ? '#334155' : '#94a3b8', cursor: paginaActual === totalPaginas ? 'default' : 'pointer' }}>
                  Siguiente →
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
