import { useEffect, useState } from 'react';
import { BarChart, Bar, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const API = 'http://localhost:8000/segmentacion';

export default function Segmentacion() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(API).then(async (response) => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
      setData(body);
    }).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="glass-card"><p className="error-texto">{error}</p></div>;
  if (!data) return <div className="glass-card"><p>Cargando modelo K-Means...</p></div>;

  return <>
    <header className="top-header"><div><h1>Segmentación K-Means</h1>
      <p>Perfiles geoclimáticos aprendidos del conjunto de observaciones</p></div></header>
    <div className="metric-grid">
      <div className="glass-card metric"><span>Registros</span><strong>{data.records.toLocaleString('es-BO')}</strong></div>
      <div className="glass-card metric"><span>Clusters</span><strong>{data.selected_k}</strong></div>
      <div className="glass-card metric"><span>Silhouette</span><strong>{data.silhouette.toFixed(4)}</strong></div>
      <div className="glass-card metric"><span>WCSS</span><strong>{data.wcss.toFixed(1)}</strong></div>
    </div>
    <div className="glass-card chart-card">
      <h2 className="section-title">Distribución de observaciones</h2>
      <div style={{ width: '100%', height: 300 }}><ResponsiveContainer>
        <BarChart data={data.segments}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.1)" />
          <XAxis dataKey="nombre" stroke="#94a3b8" /><YAxis stroke="#94a3b8" />
          <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#334155' }} />
          <Bar dataKey="cantidad" fill="#3b82f6" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer></div>
    </div>
    <div className="segment-grid">{data.segments.map((segment) =>
      <article className="glass-card" key={segment.cluster}>
        <span className="cluster-tag">Cluster {segment.cluster}</span><h2>{segment.nombre}</h2>
        <p className="muted-small">{segment.descripcion}</p>
        <div className="segment-values"><span>{segment.cantidad} registros</span>
          <span>Índice histórico: {segment.indice_riesgo_deslizamiento?.toFixed(1) ?? 'N/D'}</span>
          <span>Ocurrencia: {segment.deslizamiento_ocurrido != null ? `${(segment.deslizamiento_ocurrido * 100).toFixed(1)}%` : 'N/D'}</span></div>
      </article>)}</div>
    <p className="muted-small model-foot">Entrenado: {new Date(data.trained_at).toLocaleString('es-BO')} · Versión {data.version}</p>
  </>;
}
