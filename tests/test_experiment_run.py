import json

from experiments.benchmark import BenchmarkExperiment
from experiments.factory import ExperimentFactory
from experiments.models import ExperimentStatus
from experiments.registry import ExperimentRegistry
from experiments.run import create_registry, run_experiment_suite
from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
    InferenceRuntimeError,
)


class FakeRuntime(InferenceRuntime):
    @property
    def runtime_name(self) -> str:
        return "fake"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            output="def example():\n    return 42",
            total_duration_seconds=0.1,
            prompt_tokens=10,
            generated_tokens=10,
        )


def create_test_yaml(tmp_path):
    config_path = tmp_path / "suite.yaml"

    config_path.write_text(
        """
name: test-suite
models:
  - model-a
  - model-b
experiments:
  - type: benchmark
    config:
      prompt: Write a Python function.
      task_id: code-generation-basic
      warmup_runs: 0
      iterations: 1
""",
        encoding="utf-8",
    )

    return config_path


def test_create_registry_registers_benchmark():
    registry = create_registry()

    assert registry.get("benchmark") is BenchmarkExperiment


def test_run_experiment_suite_executes_and_persists_results(tmp_path):
    config_path = create_test_yaml(tmp_path)
    output_directory = tmp_path / "results"

    registry = create_registry()
    factory = ExperimentFactory(runtime_factory=FakeRuntime)

    result = run_experiment_suite(
        config_path=config_path,
        output_directory=output_directory,
        registry=registry,
        factory=factory,
    )

    assert result.name == "test-suite"
    assert len(result.experiments) == 2

    assert all(
        experiment.status == ExperimentStatus.COMPLETED
        for experiment in result.experiments
    )

    assert all(
        experiment.results.configuration.runtime == "fake"
        for experiment in result.experiments
    )

    saved_files = list(output_directory.glob("*.json"))
    assert len(saved_files) == 1

    saved_result = json.loads(
        saved_files[0].read_text(encoding="utf-8")
    )

    assert saved_result["name"] == "test-suite"
    assert len(saved_result["experiments"]) == 2


def test_run_experiment_suite_preserves_failed_experiments(tmp_path):
    config_path = create_test_yaml(tmp_path)
    output_directory = tmp_path / "results"

    class FailingRuntime(FakeRuntime):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            raise InferenceRuntimeError(
                category="inference",
                message="Simulated inference failure",
                original_type="RuntimeError",
            )

    registry = create_registry()
    factory = ExperimentFactory(runtime_factory=FailingRuntime)

    result = run_experiment_suite(
        config_path=config_path,
        output_directory=output_directory,
        registry=registry,
        factory=factory,
    )

    assert len(result.experiments) == 2

    assert all(
        experiment.status == ExperimentStatus.COMPLETED
        for experiment in result.experiments
    )

    # BenchmarkExperiment records failed inference iterations in its
    # BenchmarkExecution. The experiment itself can still complete.
    assert all(
        experiment.results.analysis.reliability.failed_iterations == 1
        for experiment in result.experiments
    )

    saved_files = list(output_directory.glob("*.json"))
    assert len(saved_files) == 1


def test_run_experiment_suite_uses_default_dependencies(tmp_path):
    config_path = create_test_yaml(tmp_path)
    output_directory = tmp_path / "results"

    result = run_experiment_suite(
        config_path=config_path,
        output_directory=output_directory,
        factory=ExperimentFactory(runtime_factory=FakeRuntime),
    )

    assert len(result.experiments) == 2
    assert all(
        experiment.status == ExperimentStatus.COMPLETED
        for experiment in result.experiments
    )
