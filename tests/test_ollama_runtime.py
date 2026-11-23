from runtimes.base import GenerationRequest
from runtimes.ollama import OllamaRuntime


def test_ollama_generates_response():
    runtime = OllamaRuntime()

    request = GenerationRequest(
        model="qwen2.5-coder:7b",
        prompt="Write a Python function that adds two numbers.",
        temperature=0.0,
        max_tokens=150,
    )

    result = runtime.generate(request)

    assert result.output
    assert result.total_duration_seconds > 0
    assert result.time_to_first_token_seconds is not None
    assert result.prompt_tokens is not None
    assert result.generated_tokens is not None
    assert result.model_load_duration_seconds is not None
    assert result.prompt_eval_duration_seconds is not None
    assert result.generation_duration_seconds is not None
    assert result.runtime_total_duration_seconds is not None

    print("\nGeneration result:")
    print(result)
