"""Completion-gated Ollama output shared by chat and experiment paths.

The HTTP stream is consumed internally; UI deltas are released only after
completion checks. There are deliberately no hidden retries or model changes.
"""
from functools import lru_cache
import json
import logging
import re
from time import monotonic
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.llm.base import LLMClient, LLMError

settings = get_settings()
logger = logging.getLogger("uvicorn.error.ollama")
PATCH_VERSION = "completion_gate_v3"

CHAT_FORMAT = {
    "type": "object",
    "properties": {"final_answer": {"type": "string", "minLength": 1}},
    "required": ["final_answer"],
    "additionalProperties": False,
}

# Conservative output-format check, NOT a semantic correctness evaluator.
# Reject detected drafting instead of trying to recover a correct-looking answer.
_DRAFTING = re.compile(
    r"(?:^|\n)\s*(?:we are given (?:the user|a question)|"
    r"the user (?:is asking|asks|wants|wrote)|"
    r"(?:okay[,.:]?\s*)?let(?:'s| us| me) (?:think|check|analy[sz]e)|"
    r"i need to (?:answer|make sure|return|think)|"
    r"(?:therefore,?\s*)?the answer should be|"
    r"so (?:we|i) (?:output|should output)|according to the rules)",
    re.I,
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate output key")
        result[key] = value
    return result


def _chat_answer(raw: str) -> str:
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object)
    except ValueError as exc:
        raise LLMError("Ollama did not return the required final-answer JSON. No partial text was displayed.") from exc
    if (not isinstance(result, dict) or set(result) != {"final_answer"}
            or not isinstance(result["final_answer"], str)
            or not result["final_answer"].strip()):
        raise LLMError("Invalid final-answer envelope. No answer was displayed.")
    answer = result["final_answer"].strip()
    if re.search(r"</?think\b", answer, re.I) or _DRAFTING.search(answer):
        raise LLMError(
            "Model returned drafting instead of a final answer; rejected. "
            "Check the installed Ollama version and Qwen3 model/template. "
            "Increasing tokens alone will not fix this."
        )
    return answer


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def generate(self, system_prompt: str, user_prompt: str,
                 response_format: str | dict[str, Any] | None = None) -> str:
        return "".join(self.stream(system_prompt, user_prompt, response_format)).strip()

    def stream(self, system_prompt: str, user_prompt: str,
               response_format: str | dict[str, Any] | None = None):
        if settings.ollama_think:
            raise LLMError("This final-answer pipeline requires OLLAMA_THINK=false. Restart the backend after changing it.")
        ordinary_chat = response_format is None
        output_format = CHAT_FORMAT if ordinary_chat else response_format
        if ordinary_chat:
            system_prompt += (
                '\nOUTPUT CONTRACT: Return exactly one JSON object with only '
                '"final_answer": a concise user-facing answer in the requested language. '
                'Keep source citations such as [S1] inside that answer. '
                'Do not describe the user, instructions, evidence-checking process, '
                'analysis or drafting. If evidence is insufficient, say so directly. '
                'The JSON envelope is transport formatting, not part of the answer.'
            )
        # Put Qwen's soft switch AFTER evidence, not before a large quoted passage.
        # The request-level boolean remains the primary control.
        if self.model.lower().split(":", 1)[0].split("/")[-1].startswith("qwen3"):
            user_prompt = user_prompt.rstrip() + "\n\n/no_think"

        payload = {
            "model": self.model, "stream": True, "think": False,
            "keep_alive": settings.ollama_keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": output_format, "truncate": False,
            "options": {
                "num_ctx": settings.ollama_num_ctx,
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_output_tokens,
            },
        }
        if settings.ollama_num_gpu is not None:
            payload["options"]["num_gpu"] = settings.ollama_num_gpu
        logger.info("patch=%s model=%s think=false mode=%s num_ctx=%s num_predict=%s",
                    PATCH_VERSION, self.model, "chat_envelope" if ordinary_chat else "factual_schema",
                    settings.ollama_num_ctx, settings.llm_max_output_tokens)
        started = monotonic()
        parts = []
        done = False
        thinking_chars = 0
        try:
            timeout = httpx.Timeout(settings.ollama_timeout_seconds, connect=5.0)
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code >= 400:
                        response.read()
                        try:
                            body = response.json()
                            detail = body.get("error", response.reason_phrase) if isinstance(body, dict) else response.reason_phrase
                        except ValueError:
                            detail = response.reason_phrase
                        raise LLMError(f"Ollama HTTP {response.status_code}: {detail}",
                                       503 if response.status_code == 404 else 502)
                    for line in response.iter_lines():
                        if monotonic() - started > settings.ollama_max_duration_seconds:
                            raise LLMError("Ollama exceeded its generation time limit. No partial answer was displayed.", 504)
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if not isinstance(data, dict):
                                raise ValueError("Expected object")
                            if data.get("error"):
                                raise LLMError(f"Ollama generation failed: {data['error']}")
                            message = data.get("message") or {}
                            if not isinstance(message, dict):
                                raise ValueError("Invalid message")
                            content = message.get("content", "")
                            thinking = message.get("thinking", "")
                            if not isinstance(content, str) or not isinstance(thinking, str):
                                raise ValueError("Invalid text fields")
                        except (ValueError, TypeError) as exc:
                            raise LLMError("Ollama returned an invalid response.") from exc
                        # Never send thinking or incomplete content to the UI.
                        thinking_chars += len(thinking)
                        if content:
                            parts.append(content)
                        if data.get("done"):
                            reason = data.get("done_reason")
                            logger.info(
                                "patch=%s model=%s elapsed=%.2fs prompt_tokens=%s output_tokens=%s "
                                "done_reason=%s thinking_chars=%s load_seconds=%.2f",
                                PATCH_VERSION, self.model, monotonic() - started,
                                data.get("prompt_eval_count"), data.get("eval_count"), reason,
                                thinking_chars, (data.get("load_duration") or 0) / 1e9,
                            )
                            if reason == "length":
                                raise LLMError(
                                    "Output token limit reached before completion; no partial answer displayed. "
                                    f"model={self.model}, num_predict={settings.llm_max_output_tokens}, "
                                    f"prompt_tokens={data.get('prompt_eval_count')}, "
                                    f"output_tokens={data.get('eval_count')}, thinking_chars={thinking_chars}. "
                                    "Check runtime/template if thinking persists with think=false."
                                )
                            if reason != "stop":
                                raise LLMError(f"Ollama did not report normal completion (done_reason={reason!r}).")
                            done = True
                            break
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama timed out; no partial answer displayed. Check model load and CPU/GPU memory.", 504) from exc
        except httpx.ConnectError as exc:
            raise LLMError(f"Cannot connect to Ollama at {self.base_url}.", 503) from exc
        except httpx.HTTPError as exc:
            raise LLMError("Ollama connection interrupted; no partial answer displayed.") from exc
        if not done:
            raise LLMError("Ollama stream ended without completion; no partial answer displayed.")
        raw = "".join(parts).strip()
        if not raw:
            raise LLMError(f"Ollama returned no final content (thinking_chars={thinking_chars}); check model/template.")
        if thinking_chars:
            logger.warning("Ollama emitted thinking despite think=false; ignored %s characters. Verify runtime/model template.", thinking_chars)
        # Keep factual output raw for the existing scorer: do not recover numbers
        # from explanations or convert invalid outputs into successful answers.
        yield _chat_answer(raw) if ordinary_chat else raw


@lru_cache
def get_ollama_client() -> OllamaClient:
    return OllamaClient()
