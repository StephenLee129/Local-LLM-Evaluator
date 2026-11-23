import pytest

from benchmark.runner import BenchmarkRunner
from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
    InferenceRuntimeError,
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


class FailingFakeRuntime(FakeRuntime):
    """Fake runtime that fails on selected calls."""

    def __init__(self, failing_calls: set[int]):
        super().__init__()
        self.failing_calls = failing_calls

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1

        if self.call_count in self.failing_calls:
            raise InferenceRuntimeError(
                category="timeout",
                message="Test inference timed out",
                original_type="Timeout",
            )

        return GenerationResult(
            output=f"response-{self.call_count}",
            total_duration_seconds=1.0,
        )


def create_request():
    return GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )


def test_runner_executes_warmups_and_measured_iterations():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    warmup_summary, results = runner.run(
        request=create_request(),
        iterations=3,
        warmup_runs=2,
    )

    assert runtime.call_count == 5

    assert warmup_summary.runs_requested == 2
    assert warmup_summary.runs_completed == 2
    assert warmup_summary.runs_failed == 0

    assert len(results) == 3
    assert all(result.status == "success" for result in results)

    assert [result.generation.output for result in results] == [
        "response-3",
        "response-4",
        "response-5",
    ]


def test_runner_supports_zero_warmup_runs():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    warmup_summary, results = runner.run(
        request=create_request(),
        iterations=2,
        warmup_runs=0,
    )

    assert runtime.call_count == 2

    assert warmup_summary.runs_requested == 0
    assert warmup_summary.runs_completed == 0
    assert warmup_summary.runs_failed == 0

    assert len(results) == 2


def test_runner_records_failed_iteration_and_continues():
    # Call 3 is the first measured iteration because calls 1 and 2 are warmups.
    runtime = FailingFakeRuntime(failing_calls={3})
    runner = BenchmarkRunner(runtime)

    warmup_summary, results = runner.run(
        request=create_request(),
        iterations=3,
        warmup_runs=2,
    )

    assert runtime.call_count == 5

    assert warmup_summary.runs_completed == 2
    assert warmup_summary.runs_failed == 0

    assert len(results) == 3

    assert results[0].status == "failed"
    assert results[0].generation is None
    assert results[0].error.category == "timeout"
    assert results[0].error.type == "Timeout"

    assert results[1].status == "success"
    assert results[1].generation.output == "response-4"

    assert results[2].status == "success"
    assert results[2].generation.output == "response-5"


def test_runner_records_failed_warmup():
    runtime = FailingFakeRuntime(failing_calls={1})
    runner = BenchmarkRunner(runtime)

    warmup_summary, results = runner.run(
        request=create_request(),
        iterations=2,
        warmup_runs=2,
    )

    assert runtime.call_count == 4

    assert warmup_summary.runs_requested == 2
    assert warmup_summary.runs_completed == 1
    assert warmup_summary.runs_failed == 1

    assert len(results) == 2
    assert all(result.status == "success" for result in results)


def test_runner_rejects_zero_iterations():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    with pytest.raises(ValueError):
        runner.run(request=create_request(), iterations=0)


def test_runner_rejects_negative_warmup_runs():
    runtime = FakeRuntime()
    runner = BenchmarkRunner(runtime)

    with pytest.raises(ValueError):
        runner.run(
            request=create_request(),
            iterations=1,
            warmup_runs=-1,
        )
