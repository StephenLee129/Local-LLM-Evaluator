from datetime import timezone
from uuid import UUID

import experiments.benchmark as benchmark_module

from benchmark.models import (
    BenchmarkError,
    BenchmarkRunResult,
    BenchmarkWarning,
    EnvironmentInfo,
    ResourceProfile,
    ResourceSample,
    WarmupSummary,
)
from experiments.benchmark import BenchmarkExperiment
from experiments.benchmark_config import BenchmarkExperimentConfig
from runtimes.base import GenerationResult


class MockRuntime:
    @property
    def runtime_name(self) -> str:
        return "mock-runtime"


class MockResourceMonitor:
    pass


class MockBenchmarkRunner:
    """Provides controlled benchmark results without running inference."""

    def __init__(self, runtime, resource_monitor=None):
        self.runtime = runtime
        self.resource_monitor = resource_monitor
        self.warnings = [
            BenchmarkWarning(
                component="resource_monitor",
                message="Mock monitoring warning",
                details="Simulated warning",
            )
        ]
        self.request = None
        self.iterations = None
        self.warmup_runs = None

    def run(self, request, iterations, warmup_runs):
        self.request = request
        self.iterations = iterations
        self.warmup_runs = warmup_runs

        generation = GenerationResult(
            output="Generated response",
            total_duration_seconds=0.5,
            time_to_first_token_seconds=0.1,
            generated_tokens=20,
            generation_duration_seconds=0.4,
        )

        results = [
            BenchmarkRunResult(
                iteration=1,
                status="success",
                duration_seconds=0.5,
                generation=generation,
            )
        ]

        resource_profile = ResourceProfile(
            pre_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=1000.0,
            ),
            post_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=2000.0,
            ),
        )

        return (
            WarmupSummary(
                runs_requested=warmup_runs,
                runs_completed=warmup_runs,
                runs_failed=0,
            ),
            results,
            resource_profile,
        )


def test_benchmark_experiment_returns_complete_execution(monkeypatch):
    environment_warnings = [
        BenchmarkWarning(
            component="environment",
            message="Mock environment warning",
        )
    ]

    monkeypatch.setattr(
        benchmark_module,
        "collect_environment",
        lambda: (
            EnvironmentInfo(
                operating_system="Test OS",
                python_version="3.12.4",
            ),
            environment_warnings,
        ),
    )

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        MockBenchmarkRunner,
    )

    config = BenchmarkExperimentConfig(
        prompt="Write a Python function.",
        task_id="code-generation",
    )

    experiment = BenchmarkExperiment(
        model="test-model",
        config=config,
        runtime=MockRuntime(),
    )

    execution = experiment.run()

    assert isinstance(execution.metadata.execution_id, UUID)
    assert execution.metadata.timestamp_utc.tzinfo == timezone.utc
    assert execution.metadata.schema_version == "1.0"
    assert execution.metadata.execution_duration_seconds >= 0

    assert execution.configuration.model == "test-model"
    assert execution.configuration.runtime == "mock-runtime"
    assert execution.configuration.prompt == config.prompt
    assert execution.configuration.task_id == config.task_id
    assert execution.configuration.warmup_runs == config.warmup_runs
    assert execution.configuration.measured_iterations == config.iterations

    assert execution.environment.operating_system == "Test OS"
    assert execution.warmup_summary.runs_completed == config.warmup_runs
    assert len(execution.measured_runs) == 1

    assert execution.analysis.reliability.total_iterations == 1
    assert execution.analysis.reliability.successful_iterations == 1

    assert execution.resource_profile is not None
    assert (
        execution.resource_profile.pre_warmup.gpu_memory_used_mb
        == 1000.0
    )

    assert len(execution.warnings) == 2
    assert execution.warnings[0].component == "environment"
    assert execution.warnings[1].component == "resource_monitor"


def test_benchmark_experiment_passes_configuration_to_runner(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "collect_environment",
        lambda: (EnvironmentInfo(), []),
    )

    runner_instances = []

    class CapturingRunner(MockBenchmarkRunner):
        def __init__(self, runtime, resource_monitor=None):
            super().__init__(runtime, resource_monitor)
            runner_instances.append(self)

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        CapturingRunner,
    )

    config = BenchmarkExperimentConfig(
        prompt="Explain binary search.",
        task_id="code-understanding",
        warmup_runs=3,
        iterations=8,
        temperature=0.5,
        max_tokens=300,
    )

    experiment = BenchmarkExperiment(
        model="test-model",
        config=config,
        runtime=MockRuntime(),
    )

    experiment.run()

    runner = runner_instances[0]

    assert runner.request.model == "test-model"
    assert runner.request.prompt == config.prompt
    assert runner.request.temperature == config.temperature
    assert runner.request.max_tokens == config.max_tokens

    assert runner.iterations == 8
    assert runner.warmup_runs == 3


def test_benchmark_experiment_passes_resource_monitor_to_runner(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "collect_environment",
        lambda: (EnvironmentInfo(), []),
    )

    runner_instances = []

    class CapturingRunner(MockBenchmarkRunner):
        def __init__(self, runtime, resource_monitor=None):
            super().__init__(runtime, resource_monitor)
            runner_instances.append(self)

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        CapturingRunner,
    )

    resource_monitor = MockResourceMonitor()

    experiment = BenchmarkExperiment(
        model="test-model",
        config=BenchmarkExperimentConfig(
            prompt="Test prompt",
            task_id="test-task",
        ),
        runtime=MockRuntime(),
        resource_monitor=resource_monitor,
    )

    experiment.run()

    assert runner_instances[0].resource_monitor is resource_monitor


def test_benchmark_experiment_preserves_failed_iterations(monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "collect_environment",
        lambda: (EnvironmentInfo(), []),
    )

    class FailingRunner(MockBenchmarkRunner):
        def run(self, request, iterations, warmup_runs):
            return (
                WarmupSummary(
                    runs_requested=warmup_runs,
                    runs_completed=warmup_runs,
                    runs_failed=0,
                ),
                [
                    BenchmarkRunResult(
                        iteration=1,
                        status="failed",
                        duration_seconds=0.2,
                        error=BenchmarkError(
                            category="connection",
                            type="ConnectionError",
                            message="Runtime unavailable",
                        ),
                    )
                ],
                None,
            )

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        FailingRunner,
    )

    experiment = BenchmarkExperiment(
        model="test-model",
        config=BenchmarkExperimentConfig(
            prompt="Test prompt",
            task_id="test-task",
        ),
        runtime=MockRuntime(),
    )

    execution = experiment.run()

    assert execution.measured_runs[0].status == "failed"
    assert execution.measured_runs[0].error.category == "connection"
    assert execution.analysis.reliability.failed_iterations == 1
    assert execution.analysis.reliability.successful_iterations == 0
