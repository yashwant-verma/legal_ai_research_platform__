import React, { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Scale, LayoutDashboard, MessageSquareText, BookOpenCheck, GitCompareArrows, Database, FlaskConical, Upload, Menu, X, ShieldCheck } from 'lucide-react'

const items = [
  ['/', 'Dashboard', LayoutDashboard],
  ['/chat/llm', 'LLM Only', MessageSquareText],
  ['/chat/rag', 'RAG Chat', BookOpenCheck],
  ['/compare', 'Compare Both', GitCompareArrows],
  ['/knowledge-base', 'Knowledge Base', Database],
  ['/datasets', 'Datasets', Upload],
  ['/experiments', 'Experiments', FlaskConical],
]

export default function Layout() {
  const [open, setOpen] = useState(false)
  return <div className="app-shell">
    <button className="mobile-menu" onClick={() => setOpen(!open)}>{open ? <X/> : <Menu/>}</button>
    <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Scale size={23}/></div><div><strong>Legal AI</strong><span>Research Platform</span></div></div>
      <div className="research-pill"><ShieldCheck size={16}/><span>Evidence-gated legal RAG</span></div>
      <nav>{items.map(([to,label,Icon]) => <NavLink key={to} to={to} end={to === '/'} onClick={() => setOpen(false)} className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}><Icon size={19}/><span>{label}</span></NavLink>)}</nav>
      <div className="sidebar-note"><strong>Research rule</strong><span>No sufficient evidence → no legal conclusion.</span></div>
    </aside>
    <main className="main-content"><Outlet/></main>
  </div>
}
