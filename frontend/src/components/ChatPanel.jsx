import React, { useEffect, useRef, useState } from 'react'
import { Send, ShieldAlert } from 'lucide-react'
import { streamChat, errorMessage } from '../lib/api'
import { Badge, Card, Notice, Spinner } from './UI'
import EvidenceList from './Evidence'

export default function ChatPanel({ mode }) {
  const isRag = mode === 'rag'
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState({ jurisdiction: 'India', official_only: false, include_historical: false })
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [partial, setPartial] = useState('')
  const requestRef = useRef(null)

  useEffect(() => () => {
    requestRef.current?.abort()
    requestRef.current = null
  }, [])

  async function submit(e) {
    e.preventDefault()
    if (query.trim().length < 2 || requestRef.current) return
    const controller = new AbortController()
    requestRef.current = controller
    setLoading(true); setError(''); setAnswer(null); setPartial('')
    try {
      const body = { query: query.trim(), language: 'auto' }
      if (isRag) body.filters = { ...filters, jurisdiction: filters.jurisdiction || null }
      await streamChat(mode, body, {
        signal: controller.signal,
        onEvent(event) {
          if (requestRef.current !== controller || controller.signal.aborted) return
          if (event.type === 'delta') setPartial(text => text + event.text)
          if (event.type === 'answer') { setAnswer(event.answer); setPartial('') }
        },
      })
    } catch (err) {
      if (requestRef.current === controller && !controller.signal.aborted) setError(errorMessage(err))
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null
        setLoading(false)
      }
    }
  }

  return <div className="chat-layout">
    <Card>
      <form onSubmit={submit} className="query-form">
        <label>Legal question</label>
        <textarea value={query} onChange={e=>setQuery(e.target.value)} placeholder={isRag ? 'Ask a legal question. The answer will use accepted knowledge-base evidence only.' : 'Ask a legal question for the LLM-only baseline.'} rows={5}/>
        {isRag && <div className="filter-grid">
          <label>Jurisdiction<input value={filters.jurisdiction} onChange={e=>setFilters({...filters, jurisdiction:e.target.value})} placeholder="India"/></label>
          <label className="check"><input type="checkbox" checked={filters.official_only} onChange={e=>setFilters({...filters, official_only:e.target.checked})}/>Official sources only</label>
          <label className="check"><input type="checkbox" checked={filters.include_historical} onChange={e=>setFilters({...filters, include_historical:e.target.checked})}/>Include historical law</label>
        </div>}
        <button className="primary-btn" disabled={loading || query.trim().length < 2}>{loading ? <Spinner label="Running…"/> : <><Send size={17}/>Submit</>}</button>
      </form>
    </Card>

    {error && <Notice type="error">{error}</Notice>}
    {partial && !answer && <Card>
      <div className="answer-head"><h2>{isRag ? 'Evidence-grounded answer' : 'LLM-only answer'}</h2><Badge tone={error ? 'warning' : 'success'}>{error ? 'incomplete' : 'generating'}</Badge></div>
      <div className="answer-text">{partial}</div>
      {error && <div className="answer-foot">Generation did not complete. Please retry.</div>}
    </Card>}
    {answer && <>
      <Card className={answer.abstained ? 'abstain-card' : ''}>
        <div className="answer-head"><div><span className="section-kicker">Response</span><h2>{isRag ? 'Evidence-grounded answer' : 'LLM-only answer'}</h2></div><div className="answer-badges"><Badge tone={answer.abstained ? 'warning' : 'success'}>{answer.abstained ? 'abstained' : 'generated'}</Badge><Badge>{answer.model}</Badge></div></div>
        {answer.abstained && <div className="abstain-banner"><ShieldAlert size={20}/><span>The evidence gate rejected the available retrieval candidates, so no unsupported legal conclusion was generated.</span></div>}
        {(answer.notices||[]).map((notice,i)=><Notice key={`notice-${i}`} type="info">{notice}</Notice>)}
        {(answer.warnings||[]).map((w,i)=><Notice key={i} type="error">{w}</Notice>)}
        <div className="answer-text">{answer.answer}</div>
        <div className="answer-foot">Prompt: {answer.prompt_version} · System: {answer.system_type}</div>
      </Card>
      {isRag && <Card><div className="section-title"><div><span className="section-kicker">Traceability</span><h2>Accepted legal evidence</h2></div><Badge>{answer.evidence?.length || 0} sources</Badge></div><EvidenceList evidence={answer.evidence}/></Card>}
    </>}
  </div>
}
