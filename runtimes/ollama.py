import json
import time
import requests

from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
)

NANOSECONDS_PER_SECOND = 1_000_000_000

def nanoseconds_to_seconds(value: int | None) -> float | None:
    """Convert nanoseconds to seconds, preserving missing values."""

    if value is None:
        return None

    return value / NANOSECONDS_PER_SECOND


class OllamaRuntime(InferenceRuntime):
    """Inference runtime adapter for Ollama."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    @property
    def runtime_name(self) -> str:
        return "ollama"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        output_parts = []
        first_token_time = None
        prompt_tokens = None
        generated_tokens = None

        model_load_duration = None
        prompt_eval_duration = None
        generation_duration = None

        start_time = time.perf_counter()

        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(5, 600),
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                chunk = json.loads(line)

                generated_text = chunk.get("response", "")

                if generated_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    output_parts.append(generated_text)

                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count")
                    generated_tokens = chunk.get("eval_count")
                    model_load_duration = chunk.get("load_duration")
                    prompt_eval_duration = chunk.get("prompt_eval_duration")
                    generation_duration = chunk.get("eval_duration")

        end_time = time.perf_counter()

        total_duration = end_time - start_time

        time_to_first_token = (
            first_token_time - start_time
            if first_token_time is not None
            else None
        )

        return GenerationResult(
            output="".join(output_parts),
            total_duration_seconds=total_duration,
            time_to_first_token_seconds=time_to_first_token,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            model_load_duration_seconds=nanoseconds_to_seconds(model_load_duration),
            prompt_eval_duration_seconds=nanoseconds_to_seconds(prompt_eval_duration),
            generation_duration_seconds=nanoseconds_to_seconds(generation_duration),
        )
