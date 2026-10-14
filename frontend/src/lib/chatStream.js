// NDJSON frames can be split anywhere, including within a UTF-8 character.
export async function readChatStream(response, onEvent, expectedSystem) {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    const detail = data.detail
    throw new Error(Array.isArray(detail) ? detail.map(x => x.msg).join(', ') : detail || `Request failed (${response.status})`)
  }
  if (!response.body) throw new Error('Streaming is unavailable in this browser.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let complete = false
  function accept(line) {
    if (!line.trim()) return
    const event = JSON.parse(line)
    if (event.type === 'error') throw new Error(event.detail || 'Generation failed.')
    if (event.type === 'answer') {
      if (event.answer?.system_type !== expectedSystem) throw new Error('Response mode did not match the request. Please retry.')
      complete = true
    }
    if (event.type === 'answer' || event.type === 'delta') onEvent(event)
  }
  try {
    while (!complete) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      let end
      while ((end = buffer.indexOf('\n')) >= 0 && !complete) {
        const line = buffer.slice(0, end)
        buffer = buffer.slice(end + 1)
        accept(line)
      }
      if (done) {
        if (!complete) accept(buffer)
        break
      }
    }
    if (!complete) throw new Error('The answer stream ended early. Please retry.')
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
