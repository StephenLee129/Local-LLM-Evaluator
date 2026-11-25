import json
import pytest
import requests

from runtimes.base import GenerationRequest, InferenceRuntimeError
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


class MockResponse:
    """Mock HTTP response for Ollama runtime tests."""

    def __init__(self, lines=None, error=None):
        self.lines = lines or []
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_lines(self, chunk_size=1, decode_unicode=True):
        yield from self.lines


def create_request():
    return GenerationRequest(
        model="qwen2.5-coder:7b",
        prompt="Test prompt",
    )


@pytest.mark.parametrize(
    ("exception", "expected_category"),
    [
        (requests.ConnectionError("Connection refused"), "connection"),
        (requests.Timeout("Request timed out"), "timeout"),
        (requests.HTTPError("HTTP 500"), "http"),
    ],
)
def test_ollama_translates_request_errors(
    monkeypatch,
    exception,
    expected_category,
):
    runtime = OllamaRuntime()

    def mock_post(*args, **kwargs):
        if isinstance(exception, requests.HTTPError):
            return MockResponse(error=exception)

        raise exception

    monkeypatch.setattr("runtimes.ollama.requests.post", mock_post)

    with pytest.raises(InferenceRuntimeError) as exc_info:
        runtime.generate(create_request())

    assert exc_info.value.category == expected_category
    assert exc_info.value.original_type == type(exception).__name__


def test_ollama_translates_invalid_json(monkeypatch):
    runtime = OllamaRuntime()

    response = MockResponse(lines=["not valid JSON"])

    monkeypatch.setattr(
        "runtimes.ollama.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(InferenceRuntimeError) as exc_info:
        runtime.generate(create_request())

    assert exc_info.value.category == "invalid_response"
    assert exc_info.value.original_type == "JSONDecodeError"


def test_ollama_translates_inference_error(monkeypatch):
    runtime = OllamaRuntime()

    response = MockResponse(
        lines=[json.dumps({"error": "model not found"})]
    )

    monkeypatch.setattr(
        "runtimes.ollama.requests.post",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(InferenceRuntimeError) as exc_info:
        runtime.generate(create_request())

    assert exc_info.value.category == "inference"
    assert exc_info.value.original_type == "OllamaError"
    assert "model not found" in str(exc_info.value)
