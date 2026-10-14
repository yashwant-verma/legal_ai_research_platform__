import test from 'node:test'
import assert from 'node:assert/strict'
import { readChatStream } from '../src/lib/chatStream.js'

function response(text) {
  const bytes = new TextEncoder().encode(text)
  return new Response(new ReadableStream({start(controller) {
    // Exercise splits inside JSON fields and Hindi UTF-8 characters.
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
    controller.close()
  }}))
}

test('Hindi deltas arrive before the validated final answer', async () => {
  const events = []
  await readChatStream(response('{"type":"delta","text":"नमस्ते"}\n{"type":"answer","answer":{"system_type":"llm_only","answer":"नमस्ते"}}'), event => events.push(event), 'llm_only')
  assert.equal(events[0].text, 'नमस्ते')
  assert.equal(events[1].type, 'answer')
})

test('mismatched RAG response cannot enter LLM-only chat', async () => {
  await assert.rejects(readChatStream(response('{"type":"answer","answer":{"system_type":"rag"}}\n'), () => assert.fail('should not render'), 'llm_only'), /mode did not match/)
})

test('partial output with no final event is an error', async () => {
  await assert.rejects(readChatStream(response('{"type":"delta","text":"partial"}\n'), () => {}, 'rag'), /ended early/)
})

test('backend error ends generation with its actual message', async () => {
  await assert.rejects(readChatStream(response('{"type":"error","detail":"Ollama timed out"}\n'), () => {}, 'llm_only'), /Ollama timed out/)
})

test('validation errors are readable', async () => {
  await assert.rejects(readChatStream(new Response('{"detail":[{"msg":"Query too short"}]}', {status: 422}), () => {}, 'llm_only'), /Query too short/)
})
