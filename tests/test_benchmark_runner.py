import pytest

from benchmark.runner import BenchmarkRunner
from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
)


class FakeRuntime(InferenceRuntime):
    """Predictable runtime for testing the benchmark runner."""

    def __init__(self):
        self.call_count = 0

    @property
    def runtime_name(self) -> str:
        return "fake"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1

        return GenerationResult(
            output=f"response-{self.call_count}",
            total_duration_seconds=1.0,
        )


def test_runner_executes_warmups_and_measured_iterations():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    request = GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )

    results = runner.run(
        request=request,
        iterations=3,
        warmup_runs=2,
    )

    assert runtime.call_count == 5
    assert len(results) == 3

    assert [result.output for result in results] == [
        "response-3",
        "response-4",
        "response-5",
    ]


def test_runner_supports_zero_warmup_runs():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    request = GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )

    results = runner.run(
        request=request,
        iterations=2,
        warmup_runs=0,
    )

    assert runtime.call_count == 2
    assert len(results) == 2


def test_runner_rejects_zero_iterations():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    request = GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )

    with pytest.raises(ValueError):
        runner.run(request=request, iterations=0)


def test_runner_rejects_negative_warmup_runs():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    request = GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )

    with pytest.raises(ValueError):
        runner.run(
            request=request,
            iterations=1,
            warmup_runs=-1,
        )
