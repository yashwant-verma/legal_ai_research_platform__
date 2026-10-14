import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ChatPage from '../src/pages/ChatPage'
import { streamChat } from '../src/lib/api'

vi.mock('../src/lib/api', () => ({ streamChat: vi.fn(), errorMessage: error => error.message }))

beforeEach(() => vi.clearAllMocks())
afterEach(cleanup)
const finalAnswer = {system_type:'llm_only', answer:'Complete response', evidence:[], abstained:false, model:'qwen3:8b', prompt_version:'legal_qa_v1'}

it('switching mode resets typed question and never starts a request by itself', () => {
  const view = render(<ChatPage mode="llm" />)
  fireEvent.change(screen.getByRole('textbox'), {target:{value:'LLM question'}})
  view.rerender(<ChatPage mode="rag" />)
  expect(screen.getByRole('textbox', {name:''}).value).toBe('')
  expect(streamChat).not.toHaveBeenCalled()
  fireEvent.change(screen.getByRole('textbox', {name:''}), {target:{value:'RAG question'}})
  view.rerender(<ChatPage mode="llm" />)
  expect(screen.getByRole('textbox').value).toBe('')
})

it('LLM mode sends only llm with no retrieval filters and displays tokens before completion', async () => {
  let handlers, finish
  streamChat.mockImplementation((mode, body, options) => {
    handlers = options
    return new Promise(resolve => {finish = resolve})
  })
  render(<ChatPage mode="llm" />)
  fireEvent.change(screen.getByRole('textbox'), {target:{value:'test question'}})
  fireEvent.submit(screen.getByRole('button', {name:'Submit'}).closest('form'))
  expect(streamChat.mock.calls[0][0]).toBe('llm')
  expect(streamChat.mock.calls[0][1]).toEqual({query:'test question', language:'auto'})
  act(() => handlers.onEvent({type:'delta', text:'नमस्ते partial'}))
  expect(screen.getByText('नमस्ते partial')).toBeTruthy()
  expect(screen.queryByText('generated')).toBeNull()
  await act(async () => { handlers.onEvent({type:'answer', answer:finalAnswer}); finish() })
  expect(screen.getByText('Complete response')).toBeTruthy()
  expect(screen.getByText('generated')).toBeTruthy()
})

it('pending requests are aborted on mode switch and late events cannot fill the new mode', async () => {
  let handlers, finish
  streamChat.mockImplementation((mode, body, options) => {
    handlers = options
    return new Promise(resolve => {finish = resolve})
  })
  const view = render(<ChatPage mode="llm" />)
  fireEvent.change(screen.getByRole('textbox'), {target:{value:'slow question'}})
  fireEvent.submit(screen.getByRole('button', {name:'Submit'}).closest('form'))
  view.rerender(<ChatPage mode="rag" />)
  expect(handlers.signal.aborted).toBe(true)
  await act(async () => {
    handlers.onEvent({type:'delta', text:'stale LLM output'})
    handlers.onEvent({type:'answer', answer:finalAnswer})
    finish()
  })
  expect(screen.queryByText('stale LLM output')).toBeNull()
  expect(screen.queryByText('Complete response')).toBeNull()
  expect(screen.getByRole('textbox', {name:''}).value).toBe('')
})

it('RAG keeps filters and evidence panel', async () => {
  streamChat.mockImplementation(async (mode, body, options) => {
    options.onEvent({type:'answer', answer:{...finalAnswer, system_type:'rag', abstained:true, answer:'No supporting text'}})
  })
  render(<ChatPage mode="rag" />)
  fireEvent.change(screen.getByRole('textbox', {name:''}), {target:{value:'rag question'}})
  fireEvent.click(screen.getByLabelText('Official sources only'))
  fireEvent.submit(screen.getByRole('button', {name:'Submit'}).closest('form'))
  await screen.findByText('No supporting text')
  expect(streamChat.mock.calls[0][0]).toBe('rag')
  expect(streamChat.mock.calls[0][1].filters).toEqual({jurisdiction:'India', official_only:true, include_historical:false})
  expect(screen.getByText('Accepted legal evidence')).toBeTruthy()
  expect(screen.getByText('abstained')).toBeTruthy()
})

it('errors stop loading and allow a retry', async () => {
  streamChat.mockRejectedValue(new Error('Ollama timed out'))
  render(<ChatPage mode="llm" />)
  fireEvent.change(screen.getByRole('textbox'), {target:{value:'test question'}})
  fireEvent.submit(screen.getByRole('button', {name:'Submit'}).closest('form'))
  await screen.findByText('Ollama timed out')
  expect(screen.getByRole('button', {name:'Submit'}).disabled).toBe(false)
})

it('duplicate submits during an active generation do not queue a second Ollama request', async () => {
  let finish
  streamChat.mockImplementation(() => new Promise(resolve => {finish=resolve}))
  render(<ChatPage mode="llm" />)
  fireEvent.change(screen.getByRole('textbox'), {target:{value:'test question'}})
  const form = screen.getByRole('button', {name:'Submit'}).closest('form')
  fireEvent.submit(form)
  fireEvent.submit(form)
  expect(streamChat).toHaveBeenCalledTimes(1)
  await act(async () => finish())
})
