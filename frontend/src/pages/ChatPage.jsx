import React from 'react'
import ChatPanel from '../components/ChatPanel'
import { PageHeader } from '../components/UI'
export default function ChatPage({mode}) {
  const rag=mode==='rag'
  return <><PageHeader eyebrow={rag?'RAG branch':'Baseline branch'} title={rag?'RAG-only legal chat':'LLM-only legal chat'} description={rag?'Uses accepted retrieved legal evidence. If the evidence gate fails, the system abstains instead of using unrelated text.':'Uses the configured LLM directly and does not consult the project knowledge base.'}/><ChatPanel key={mode} mode={mode}/></>
}
