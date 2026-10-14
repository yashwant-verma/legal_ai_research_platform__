import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { MessageSquareText, BookOpenCheck, GitCompareArrows, Database, Upload, FlaskConical, ArrowRight, ShieldCheck } from 'lucide-react'
import { api } from '../lib/api'
import { Card, PageHeader, Stat } from '../components/UI'

const cards = [
  ['/chat/llm','LLM Only Chat','Baseline answers without project knowledge-base retrieval.',MessageSquareText],
  ['/chat/rag','RAG Chat','Evidence-gated answers with legal source traceability.',BookOpenCheck],
  ['/compare','Compare Both','Run the same legal query through both experimental branches.',GitCompareArrows],
  ['/datasets','Testing Datasets','Review the benchmark or import your own CSV/JSON/XLSX dataset.',Upload],
  ['/experiments','Experiments','Run datasets and inspect retrieval plus answer-level metrics.',FlaskConical],
  ['/knowledge-base','Knowledge Base','Ingest PDFs and monitor indexed, OCR and review status.',Database],
]

export default function Dashboard(){
  const [stats,setStats]=useState({})
  const [config,setConfig]=useState({})
  useEffect(()=>{ Promise.all([api.get('/system/stats'),api.get('/system/config')]).then(([s,c])=>{setStats(s.data);setConfig(c.data)}).catch(()=>{}) },[])
  return <>
    <PageHeader eyebrow="Research workspace" title="Multilingual Indian Legal QA" description="Controlled LLM-only vs evidence-gated RAG experimentation with traceable legal sources, datasets and reproducible metrics."/>
    <div className="hero-strip"><ShieldCheck size={22}/><div><strong>Evidence-first behavior is active</strong><span>RAG must abstain when sufficiently relevant legal text is not available.</span></div></div>
    <div className="stats-grid four"><Stat label="Legal documents" value={stats.documents}/><Stat label="Retrievable chunks" value={stats.chunks}/><Stat label="Datasets" value={stats.datasets}/><Stat label="Experiments" value={stats.experiments}/></div>
    <div className="feature-grid">{cards.map(([to,title,desc,Icon])=><Link to={to} key={to} className="feature-link"><Card className="feature-card"><div className="feature-icon"><Icon size={22}/></div><h3>{title}</h3><p>{desc}</p><span className="feature-go">Open <ArrowRight size={16}/></span></Card></Link>)}</div>
    <Card><div className="section-title"><div><span className="section-kicker">Frozen configuration</span><h2>Current research pipeline</h2></div></div><div className="config-grid"><div><span>LLM</span><strong>{config.model || '—'}</strong></div><div><span>Embedding</span><strong>{config.embedding_model || '—'}</strong></div><div><span>Reranker</span><strong>{config.reranker_model || 'Disabled'}</strong></div><div><span>Evidence threshold</span><strong>{config.dense_min_score ?? '—'}</strong></div><div><span>Top-K</span><strong>{config.final_top_k ?? '—'}</strong></div><div><span>Chunk version</span><strong>{config.chunk_version || '—'}</strong></div></div></Card>
  </>
}
