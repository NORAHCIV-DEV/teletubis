import { useEffect, useState } from 'react';
import { BarChart, Bar, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
  LineChart, Line, Legend, ReferenceLine } from 'recharts';
import { BrainCircuit, CheckCircle2, Database, Layers3, Target, BookOpen, Info } from 'lucide-react';

const API = 'http://localhost:8000/segmentacion';

export default function Segmentacion({ segmentoActual }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(API).then(async (response) => {
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
      setData(body);
    }).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="glass-card kmeans-state"><BrainCircuit size={34} /><p className="error-texto">{error}</p></div>;
  if (!data) return <div className="glass-card kmeans-state"><BrainCircuit className="kmeans-pulse" size={34} /><p>Cargando modelo K-Means...</p></div>;

  return <>
    <header className="top-header"><div><h1>Segmentación K-Means</h1>
      <p>H4 · Cuatro perfiles convierten 10.000 observaciones en una historia geoclimática comprensible</p></div></header>
    {segmentoActual && <section className="current-segment-banner" aria-label="Segmento de la última predicción">
      <CheckCircle2 size={24} />
      <div><span>Tu última evaluación pertenece al</span><strong>Cluster {segmentoActual.cluster} · {segmentoActual.segmento}</strong></div>
    </section>}
    <div className="metric-grid">
      <div className="glass-card metric"><Database size={19} /><span>Registros analizados</span><strong>{data.records.toLocaleString('es-BO')}</strong></div>
      <div className="glass-card metric"><Layers3 size={19} /><span>Clusters seleccionados</span><strong>{data.selected_k}</strong></div>
      <div className="glass-card metric"><Target size={19} /><span>Coeficiente Silhouette</span><strong>{data.silhouette.toFixed(4)}</strong></div>
      <div className="glass-card metric"><BrainCircuit size={19} /><span>Algoritmo</span><strong className="metric-text">{data.algorithm}</strong></div>
    </div>
    <div className="kmeans-info-strip">
      <strong>¿Por qué k={data.selected_k}?</strong>
      <span>{data.selection_reason || 'Se definieron cuatro perfiles para comparar condiciones geoclimáticas con una narrativa visual común.'} Silhouette y WCSS se conservan como indicadores de calidad, no como una selección automática.</span>
    </div>
    <section className="glass-card dataset-story" aria-labelledby="dataset-origin-title">
      <div className="dataset-story-heading">
        <div className="dataset-story-icon"><BookOpen size={22} /></div>
        <div>
          <span className="cluster-tag">Origen de los datos</span>
          <h2 id="dataset-origin-title">¿De dónde salen las 10.000 observaciones?</h2>
        </div>
      </div>
      <p>{data.dataset_origin?.descripcion}</p>
      <div className="dataset-scope"><Info size={16} /><span>{data.dataset_origin?.alcance}</span></div>
      <div className="variables-grid">
        {data.feature_details?.map((variable) => <article className="variable-card" key={variable.key}>
          <div className="variable-card-title"><strong>{variable.nombre}</strong><span>{variable.unidad}</span></div>
          <p>{variable.uso}</p>
          <small><b>Generación:</b> {variable.origen}</small>
        </article>)}
      </div>
    </section>
    {data.candidates?.length > 0 && <div className="glass-card chart-card">
      <h2 className="section-title">Justificación del número de clusters</h2>
      <div style={{ width: '100%', height: 280 }}><ResponsiveContainer>
        <LineChart data={data.candidates}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.1)" />
          <XAxis dataKey="k" stroke="#94a3b8" label={{ value: 'Número de clusters (k)', position: 'insideBottom', offset: -4, fill: '#94a3b8' }} />
          <YAxis yAxisId="left" stroke="#a78bfa" domain={['auto', 'auto']} />
          <YAxis yAxisId="right" orientation="right" stroke="#60a5fa" domain={['auto', 'auto']} />
          <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#334155' }} />
          <Legend verticalAlign="top" height={34} />
          <ReferenceLine x={data.selected_k} yAxisId="left" stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'k elegido', fill: '#fbbf24', position: 'insideTopRight' }} />
          <Line yAxisId="left" type="monotone" dataKey="silhouette" name="Silhouette" stroke="#a78bfa" strokeWidth={3} dot={{ r: 5 }} />
          <Line yAxisId="right" type="monotone" dataKey="wcss" name="WCSS" stroke="#60a5fa" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer></div>
    </div>}
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
      <article className={`glass-card segment-card ${segmentoActual?.cluster === segment.cluster ? 'segment-card-current' : ''}`} key={segment.cluster}>
        <span className="cluster-tag">Cluster {segment.cluster}</span><h2>{segment.nombre}</h2>
        {segmentoActual?.cluster === segment.cluster && <span className="current-chip"><CheckCircle2 size={14} /> Tu evaluación actual</span>}
        <p className="segment-difference"><strong>Qué lo diferencia:</strong> {segment.diferenciador}</p>
        <p className="muted-small">{segment.descripcion}</p>
        <div className="segment-values"><span>{segment.cantidad} registros</span>
          <span>Lluvia 3 días: {segment.precipitacion_3d_mm.toFixed(1)} mm</span>
          <span>Lluvia 7 días: {segment.precipitacion_7d_mm.toFixed(1)} mm</span>
          <span>Elevación: {segment.elevacion_m.toFixed(0)} m</span>
          <span>Índice histórico: {segment.indice_riesgo_deslizamiento?.toFixed(1) ?? 'N/D'}</span>
          <span>Ocurrencia: {segment.deslizamiento_ocurrido != null ? `${(segment.deslizamiento_ocurrido * 100).toFixed(1)}%` : 'N/D'}</span></div>
      </article>)}</div>
    <p className="muted-small model-foot">Entrenado: {new Date(data.trained_at).toLocaleString('es-BO')} · Versión {data.version}</p>
  </>;
}
