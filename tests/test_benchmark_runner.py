import pytest

from benchmark.models import (
    ResourceProfile,
    ResourceSample,
    ResourceStatistics,
)
from benchmark.runner import BenchmarkRunner
from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
    InferenceRuntimeError,
)


class FakeRuntime(InferenceRuntime):
    """Predictable runtime for testing the benchmark runner."""

    def __init__(self, events=None):
        self.call_count = 0
        self.events = events if events is not None else []

    @property
    def runtime_name(self) -> str:
        return "fake"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        self.events.append(f"generate_{self.call_count}")

        return GenerationResult(
            output=f"response-{self.call_count}",
            total_duration_seconds=1.0,
        )


class FailingFakeRuntime(FakeRuntime):
    """Fake runtime that fails on selected calls."""

    def __init__(self, failing_calls: set[int], events=None):
        super().__init__(events)
        self.failing_calls = failing_calls

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        self.events.append(f"generate_{self.call_count}")

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


class FakeResourceMonitor:
    """Predictable resource monitor for testing benchmark execution."""

    def __init__(self, events=None, failing_snapshots=None):
        self.events = events if events is not None else []
        self.failing_snapshots = failing_snapshots or set()

        self.snapshot_count = 0
        self.errors = []
        self.samples = []
        self.running = False

    def capture_snapshot(self):
        self.snapshot_count += 1
        self.events.append(f"snapshot_{self.snapshot_count}")

        if self.snapshot_count in self.failing_snapshots:
            raise RuntimeError("Test snapshot failed")

        return ResourceSample(
            timestamp_seconds=0.0,
            gpu_memory_used_mb=1000.0 * self.snapshot_count,
            system_ram_used_mb=8000.0 * self.snapshot_count,
        )

    def start(self):
        self.running = True
        self.samples.clear()
        self.errors.clear()
        self.events.append("monitor_start")

    def stop(self):
        self.events.append("monitor_stop")
        self.running = False
        return self.samples

    def _calculate_statistics(self):
        return ResourceStatistics(sample_count=0)


def create_request():
    return GenerationRequest(
        model="test-model",
        prompt="Test prompt",
    )


def create_runner(runtime, resource_monitor=None):
    return BenchmarkRunner(
        runtime=runtime,
        resource_monitor=resource_monitor or FakeResourceMonitor(),
    )


def test_runner_executes_warmups_and_measured_iterations():
    runtime = FakeRuntime()
    resource_monitor = FakeResourceMonitor()
    runner = create_runner(runtime, resource_monitor)

    warmup_summary, results, resource_profile = runner.run(
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

    assert isinstance(resource_profile, ResourceProfile)
    assert resource_profile.pre_warmup.gpu_memory_used_mb == 1000.0
    assert resource_profile.post_warmup.gpu_memory_used_mb == 2000.0


def test_runner_captures_snapshots_before_and_after_warmups():
    events = []
    runtime = FakeRuntime(events)
    resource_monitor = FakeResourceMonitor(events)
    runner = create_runner(runtime, resource_monitor)

    runner.run(
        request=create_request(),
        iterations=1,
        warmup_runs=2,
    )

    # Initial snapshot, two warmups, then post-warmup snapshot.
    assert events[:4] == [
        "snapshot_1",
        "generate_1",
        "generate_2",
        "snapshot_2",
    ]


def test_runner_supports_zero_warmup_runs():
    runtime = FakeRuntime()
    resource_monitor = FakeResourceMonitor()
    runner = create_runner(runtime, resource_monitor)

    warmup_summary, results, resource_profile = runner.run(
        request=create_request(),
        iterations=2,
        warmup_runs=0,
    )

    assert runtime.call_count == 2

    assert warmup_summary.runs_requested == 0
    assert warmup_summary.runs_completed == 0
    assert warmup_summary.runs_failed == 0

    assert len(results) == 2
    assert isinstance(resource_profile, ResourceProfile)

    assert resource_monitor.snapshot_count == 2


def test_runner_records_failed_iteration_and_continues():
    # Call 3 is the first measured iteration because calls 1 and 2 are warmups.
    runtime = FailingFakeRuntime(failing_calls={3})
    resource_monitor = FakeResourceMonitor()
    runner = create_runner(runtime, resource_monitor)

    warmup_summary, results, resource_profile = runner.run(
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

    assert isinstance(resource_profile, ResourceProfile)


def test_runner_records_failed_warmup():
    runtime = FailingFakeRuntime(failing_calls={1})
    resource_monitor = FakeResourceMonitor()
    runner = create_runner(runtime, resource_monitor)

    warmup_summary, results, resource_profile = runner.run(
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

    assert isinstance(resource_profile, ResourceProfile)


def test_runner_continues_when_resource_snapshot_fails():
    runtime = FakeRuntime()
    resource_monitor = FakeResourceMonitor(failing_snapshots={1})
    runner = create_runner(runtime, resource_monitor)

    warmup_summary, results, resource_profile = runner.run(
        request=create_request(),
        iterations=2,
        warmup_runs=1,
    )

    assert runtime.call_count == 3

    assert warmup_summary.runs_completed == 1
    assert len(results) == 2
    assert all(result.status == "success" for result in results)

    # Both snapshots are required to construct a complete profile.
    assert resource_profile is None

    assert any(
        warning.message == "Resource snapshot failed."
        for warning in runner.warnings
    )


def test_runner_rejects_zero_iterations():
    runtime = FakeRuntime()
    runner = create_runner(runtime)

    with pytest.raises(ValueError):
        runner.run(request=create_request(), iterations=0)


def test_runner_rejects_negative_warmup_runs():
    runtime = FakeRuntime()
    runner = create_runner(runtime)

    with pytest.raises(ValueError):
        runner.run(
            request=create_request(),
            iterations=1,
            warmup_runs=-1,
        )
