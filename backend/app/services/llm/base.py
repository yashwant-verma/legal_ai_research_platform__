from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str | dict[str, Any] | None = None,
    ):
        # Keep providers implementing only generate() compatible.
        if response_format is None:
            yield self.generate(system_prompt, user_prompt)
        else:
            yield self.generate(
                system_prompt,
                user_prompt,
                response_format=response_format,
            )


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
