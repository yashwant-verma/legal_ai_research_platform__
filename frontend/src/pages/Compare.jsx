import React, { useState } from 'react'
import { GitCompareArrows, Send, ShieldAlert } from 'lucide-react'
import { api, chatTimeout, errorMessage } from '../lib/api'
import { Badge, Card, Notice, PageHeader, Spinner } from '../components/UI'
import EvidenceList from '../components/Evidence'

export default function Compare(){
  const [query,setQuery]=useState(''); const [data,setData]=useState(null); const [loading,setLoading]=useState(false); const [error,setError]=useState('')
  async function run(e){e.preventDefault();setLoading(true);setData(null);setError('');try{const r=await api.post('/chat/compare',{query,language:'auto',filters:{jurisdiction:'India',official_only:false,include_historical:false}}, { timeout: chatTimeout * 2 });setData(r.data)}catch(err){setError(errorMessage(err))}finally{setLoading(false)}}
  return <><PageHeader eyebrow="Controlled comparison" title="Compare both systems" description="The same query is sent to the configured LLM-only branch and the evidence-gated RAG branch."/>
    <Card><form onSubmit={run} className="query-form"><label>Same legal question for both branches</label><textarea rows={4} value={query} onChange={e=>setQuery(e.target.value)} placeholder="Enter one legal question…"/><button className="primary-btn" disabled={loading||query.trim().length<2}>{loading?<Spinner label="Comparing…"/>:<><GitCompareArrows size={17}/>Compare</>}</button></form></Card>
    {error&&<Notice type="error">{error}</Notice>}
    {data&&<><div className="compare-grid"><AnswerBox title="LLM Only" answer={data.llm_only}/><AnswerBox title="RAG" answer={data.rag}/></div><Card><div className="section-title"><div><span className="section-kicker">RAG evidence</span><h2>What the grounded branch actually used</h2></div><Badge>{data.rag.evidence?.length||0} sources</Badge></div><EvidenceList evidence={data.rag.evidence}/></Card></>}
  </>
}
function AnswerBox({title,answer}){return <Card className={answer.abstained?'abstain-card':''}><div className="answer-head"><h2>{title}</h2><Badge tone={answer.abstained?'warning':'success'}>{answer.abstained?'abstained':'generated'}</Badge></div>{answer.abstained&&<div className="abstain-banner"><ShieldAlert size={18}/>Insufficient legal evidence for a grounded answer.</div>}<div className="answer-text">{answer.answer}</div><div className="answer-foot">{answer.model} · {answer.prompt_version}</div></Card>}
