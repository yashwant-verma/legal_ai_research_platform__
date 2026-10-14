from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def stream(self, system_prompt: str, user_prompt: str):
        # Keep providers implementing only generate() compatible.
        yield self.generate(system_prompt, user_prompt)


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code
