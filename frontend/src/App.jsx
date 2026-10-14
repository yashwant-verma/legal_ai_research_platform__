import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ChatPage from './pages/ChatPage'
import Compare from './pages/Compare'
import KnowledgeBase from './pages/KnowledgeBase'
import Datasets from './pages/Datasets'
import Experiments from './pages/Experiments'
import NotFound from './pages/NotFound'

export default function App(){return <Routes><Route element={<Layout/>}><Route path="/" element={<Dashboard/>}/><Route path="/chat/llm" element={<ChatPage mode="llm"/>}/><Route path="/chat/rag" element={<ChatPage mode="rag"/>}/><Route path="/compare" element={<Compare/>}/><Route path="/knowledge-base" element={<KnowledgeBase/>}/><Route path="/datasets" element={<Datasets/>}/><Route path="/experiments" element={<Experiments/>}/><Route path="*" element={<NotFound/>}/></Route></Routes>}
