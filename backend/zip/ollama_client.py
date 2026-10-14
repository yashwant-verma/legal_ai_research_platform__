from functools import lru_cache
import json
import logging
from time import monotonic
from typing import Any

import httpx

from app.core.config import get_settings
from app.services.llm.base import LLMClient, LLMError

settings = get_settings()
logger = logging.getLogger("uvicorn.error.ollama")


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "".join(self.stream(system_prompt, user_prompt)).strip()

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | dict[str, Any] | None = None,
    ):
        factual_json_mode = response_format is not None

        payload = {
            "model": self.model,

            # Factual JSON requests do not need token streaming.
            # Interactive chat remains streaming.
            "stream": not factual_json_mode,

            "think": settings.ollama_think,
            "keep_alive": settings.ollama_keep_alive,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            "truncate": False,

            "options": {
                "num_ctx": settings.ollama_num_ctx,
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_output_tokens,
            },
        }

        # Only factual benchmark calls use JSON mode.
        # Normal legal chat remains free-form text.
        if response_format is not None:
            payload["format"] = response_format

        if settings.ollama_num_gpu is not None:
            payload["options"]["num_gpu"] = settings.ollama_num_gpu

        logger.info(
            "Ollama request model=%s think=%s format=%s max_output_tokens=%s",
            self.model,
            settings.ollama_think,
            response_format,
            settings.llm_max_output_tokens,
        )

        started = monotonic()
        has_content = False
        done = False

        try:
            timeout = httpx.Timeout(
                settings.ollama_timeout_seconds,
                connect=5.0,
            )

            # trust_env=False prevents system proxy interference with localhost.
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:

                    if response.status_code == 404:
                        raise LLMError(
                            f"Ollama model {self.model!r} was not found. "
                            f"Run: ollama pull {self.model}",
                            503,
                        )

                    if response.status_code >= 400:
                        response.read()

                        try:
                            detail = response.json().get(
                                "error",
                                response.reason_phrase,
                            )
                        except (ValueError, AttributeError):
                            detail = response.reason_phrase

                        raise LLMError(
                            f"Ollama request failed "
                            f"({response.status_code}): {detail}"
                        )

                    for line in response.iter_lines():
                        if (
                            monotonic() - started
                            > settings.ollama_max_duration_seconds
                        ):
                            raise LLMError(
                                "Ollama generation exceeded its time limit. "
                                "Check ollama ps and local GPU load.",
                                504,
                            )

                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)

                            if not isinstance(data, dict):
                                raise ValueError("Expected an object")

                            if data.get("error"):
                                raise LLMError(
                                    f"Ollama generation failed: {data['error']}"
                                )

                            message = data.get("message") or {}

                            # Qwen thinking output is intentionally ignored.
                            # Only message.content is used as final answer.
                            content = message.get("content", "")

                            if not isinstance(content, str):
                                raise ValueError("Invalid content")

                        except (ValueError, AttributeError, TypeError) as exc:
                            raise LLMError(
                                "Ollama returned an invalid response."
                            ) from exc

                        if content:
                            has_content = has_content or bool(content.strip())
                            yield content

                        if data.get("done"):
                            done_reason = data.get("done_reason")

                            if done_reason == "length":
                                raise LLMError(
                                    "Answer is incomplete: output token limit "
                                    "reached. Factual JSON generation was "
                                    "truncated before completion."
                                )

                            done = True

                            logger.info(
                                (
                                    "Ollama model=%s elapsed=%.2fs "
                                    "load=%.2fs prompt_tokens=%s "
                                    "output_tokens=%s done_reason=%s"
                                ),
                                self.model,
                                monotonic() - started,
                                (data.get("load_duration") or 0) / 1e9,
                                data.get("prompt_eval_count"),
                                data.get("eval_count"),
                                done_reason,
                            )
                            break

        except httpx.TimeoutException as exc:
            raise LLMError(
                "Ollama timed out while loading or generating. "
                "Check ollama ps and local GPU load.",
                504,
            ) from exc

        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start Ollama and check OLLAMA_BASE_URL.",
                503,
            ) from exc

        except httpx.HTTPError as exc:
            raise LLMError(
                "The connection to Ollama was interrupted. Please retry."
            ) from exc

        if not done:
            raise LLMError(
                "Ollama stopped before completing the answer. Please retry."
            )

        if not has_content:
            raise LLMError(
                "Ollama returned no answer text. Check "
                "OLLAMA_THINK=false and the output-token limit."
            )


@lru_cache
def get_ollama_client() -> OllamaClient:
    return OllamaClient()