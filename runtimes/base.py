
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class GenerationRequest:
    """Configuration for a single model inference request."""

    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass
class GenerationResult:
    """Generated output and associated inference metrics."""

    output: str
    total_duration_seconds: float
    time_to_first_token_seconds: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None


class InferenceRuntime(ABC):
    """Common interface implemented by inference runtimes."""

    @property
    @abstractmethod
    def runtime_name(self) -> str:
        """Name identifying the inference runtime."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a response using the configured inference runtime."""
        pass
