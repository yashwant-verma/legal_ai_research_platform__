import React from 'react'
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

export function Card({ children, className = '' }) {
  return <section className={`card ${className}`}>{children}</section>
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return <div className="page-header">
    <div>
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      <h1>{title}</h1>
      {description && <p>{description}</p>}
    </div>
    {actions && <div className="page-actions">{actions}</div>}
  </div>
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function Spinner({ label = 'Loading…' }) {
  return <div className="inline-status"><Loader2 className="spin" size={17}/><span>{label}</span></div>
}

export function Notice({ type = 'info', children }) {
  return <div className={`notice notice-${type}`}>
    {type === 'error' ? <AlertCircle size={18}/> : <CheckCircle2 size={18}/>}<div>{children}</div>
  </div>
}

export function EmptyState({ title, description }) {
  return <div className="empty-state"><h3>{title}</h3><p>{description}</p></div>
}

export function Stat({ label, value, helper }) {
  return <div className="stat-card"><div className="stat-label">{label}</div><div className="stat-value">{value ?? '—'}</div>{helper && <div className="stat-helper">{helper}</div>}</div>
}

export function Metric({ label, value }) {
  const shown = value === null || value === undefined ? '—' : `${Math.round(Number(value) * 100)}%`
  return <div className="metric-row"><span>{label}</span><strong>{shown}</strong></div>
}

export function formatDate(value) {
  if (!value) return '—'
  try { return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)) }
  catch { return String(value) }
}
