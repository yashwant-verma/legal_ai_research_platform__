import React, { useState } from 'react'
import { ChevronDown, ChevronUp, ExternalLink, FileText } from 'lucide-react'
import { Badge, EmptyState } from './UI'

export default function EvidenceList({ evidence = [] }) {
  if (!evidence.length) return <EmptyState title="No evidence returned" description="The RAG branch did not receive any accepted legal evidence." />
  return <div className="evidence-list">{evidence.map((item) => <EvidenceCard key={item.chunk_id} item={item}/>)}</div>
}

function EvidenceCard({ item }) {
  const [open, setOpen] = useState(false)
  return <div className="evidence-card">
    <button className="evidence-head" onClick={() => setOpen(!open)}>
      <div className="evidence-rank">S{item.rank}</div>
      <div className="evidence-title"><strong>{item.title || item.document_id}</strong><span>{[item.part, item.schedule, item.section && `Section ${item.section}`, item.article && `Article ${item.article}`, `Pages ${item.page_start}-${item.page_end}`].filter(Boolean).join(' · ')}</span></div>
      <div className="evidence-score"><Badge tone={item.accepted_as_evidence ? 'success' : 'warning'}>{item.accepted_as_evidence ? 'accepted' : 'candidate'}</Badge><span>{Number(item.rerank_score ?? item.dense_score).toFixed(3)}</span>{open ? <ChevronUp size={18}/> : <ChevronDown size={18}/>}</div>
    </button>
    {open && <div className="evidence-body">
      <div className="evidence-meta">
        <span><FileText size={15}/>{item.document_type || 'legal document'}</span>
        {item.jurisdiction && <span>{item.jurisdiction}</span>}
        {item.status && <span>{item.status}</span>}
        {item.paragraph_ids?.length > 0 && <span>Paras {item.paragraph_ids.join(', ')}</span>}
        {item.source_url && <a href={item.source_url} target="_blank" rel="noreferrer"><ExternalLink size={14}/>Source</a>}
      </div>
      <p className="evidence-text">{item.text}</p>
      <div className="id-line">{item.document_id} · {item.chunk_id}</div>
    </div>}
  </div>
}
