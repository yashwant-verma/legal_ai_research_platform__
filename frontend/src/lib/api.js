import axios from 'axios'
import { readChatStream } from './chatStream'

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
  timeout: 300000,
})

export function errorMessage(error) {
  const detail = error?.response?.data?.detail
  if (Array.isArray(detail)) return detail.map(x => x.msg || JSON.stringify(x)).join(', ')
  return detail || error?.message || 'Something went wrong.'
}

export async function downloadExperimentCsv(experimentId) {
  const response = await api.get(`/experiments/${experimentId}/export.csv`, { responseType: 'blob' })
  const url = URL.createObjectURL(response.data)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${experimentId}.csv`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}


// A total UI deadline also covers cold model loading and retrieval.
const configuredChatTimeout = Number(import.meta.env.VITE_CHAT_TIMEOUT_MS || 600000)
export const chatTimeout = Number.isFinite(configuredChatTimeout) && configuredChatTimeout > 0 ? configuredChatTimeout : 600000

export async function streamChat(mode, body, { signal, onEvent }) {
  if (mode !== 'llm' && mode !== 'rag') throw new Error('Unknown chat mode.')
  const controller = new AbortController()
  const abort = () => controller.abort()
  let timedOut = false
  signal?.addEventListener('abort', abort, { once: true })
  if (signal?.aborted) abort()
  const timer = setTimeout(() => { timedOut = true; controller.abort() }, chatTimeout)
  try {
    const base = api.defaults.baseURL.replace(/\/$/, '')
    const response = await fetch(`${base}/chat/${mode}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    await readChatStream(response, onEvent, mode === 'llm' ? 'llm_only' : 'rag')
  } catch (error) {
    if (timedOut) throw new Error('The chat request timed out. Check Ollama and the backend terminal, then retry.')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

export async function downloadExperimentJson(id){
 const response=await api.get(`/experiments/${id}/export.json`,{responseType:'blob'});
 const url=URL.createObjectURL(response.data);const a=document.createElement('a');a.href=url;a.download=`${id}.json`;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
}
