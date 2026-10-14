"""Suppress tagged reasoning, including delimiters split across transport chunks."""
import re
from app.services.llm.base import LLMError

def visible_tokens(tokens):
    buf = ""
    hidden = False
    initial = True
    markers = ("<think>", "okay, let me", "let me think", "i need to", "first, i need")
    for token in tokens:
        buf += token
        if initial:
            start = buf.lstrip().lower()
            if not start or any(m.startswith(start) and m != start for m in markers):
                continue
            if any(start.startswith(m) for m in markers[1:]):
                hidden = True
            initial = False
        while buf:
            match = re.search(r"</?think>", buf, re.I)
            if match:
                if not hidden and match.start():
                    if match.group().lower() == '</think>':
                        raise LLMError('Unexpected reasoning delimiter; answer rejected. Retry with thinking disabled.')
                    yield buf[:match.start()]
                hidden = match.group().lower() == '<think>'
                buf = buf[match.end():]
                continue
            # Retain a partial delimiter across tokens.
            tail = buf.rfind('<')
            if tail >= 0 and any(m.startswith(buf[tail:].lower()) for m in ('<think>', '</think>')):
                if not hidden and tail:
                    yield buf[:tail]
                buf = buf[tail:]
            else:
                if not hidden:
                    yield buf
                buf = ''
            break
    if hidden:
        raise LLMError('Model returned unfinished reasoning instead of a final answer.')
    if buf:
        yield buf
