import pytest

from experiments.base import Experiment
from experiments.benchmark import BenchmarkExperiment
from experiments.benchmark_config import BenchmarkExperimentConfig
from experiments.factory import ExperimentFactory
from experiments.models import ExperimentDefinition
from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
)
from runtimes.ollama import OllamaRuntime


class FakeRuntime(InferenceRuntime):
    """Test runtime that does not perform actual inference."""

    @property
    def runtime_name(self) -> str:
        return "fake"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


class OtherExperiment(Experiment[str]):
    """Experiment used to test unsupported experiment classes."""

    @property
    def experiment_type(self) -> str:
        return "other"

    def run(self) -> str:
        return "test"


def create_benchmark_definition(
    config: dict | None = None,
) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_type="benchmark",
        config=config
        or {
            "prompt": "Write a Python function.",
            "task_id": "code-generation-basic",
        },
    )


def test_creates_benchmark_experiment_with_injected_runtime():
    runtime = FakeRuntime()
    factory = ExperimentFactory(runtime_factory=lambda: runtime)

    definition = create_benchmark_definition(
        {
            "prompt": "Write a Python function.",
            "task_id": "code-generation-basic",
            "warmup_runs": 3,
            "iterations": 10,
            "temperature": 0.2,
            "max_tokens": 200,
        }
    )

    experiment = factory(
        BenchmarkExperiment,
        definition,
        "qwen2.5-coder:7b",
    )

    assert isinstance(experiment, BenchmarkExperiment)
    assert experiment._model == "qwen2.5-coder:7b"
    assert experiment._runtime is runtime

    assert experiment._config == BenchmarkExperimentConfig(
        prompt="Write a Python function.",
        task_id="code-generation-basic",
        warmup_runs=3,
        iterations=10,
        temperature=0.2,
        max_tokens=200,
    )


def test_uses_default_ollama_runtime_when_none_is_injected():
    factory = ExperimentFactory()

    experiment = factory(
        BenchmarkExperiment,
        create_benchmark_definition(),
        "qwen2.5-coder:7b",
    )

    assert isinstance(experiment._runtime, OllamaRuntime)


def test_creates_new_runtime_for_each_experiment():
    created_runtimes = []

    def create_runtime():
        runtime = FakeRuntime()
        created_runtimes.append(runtime)
        return runtime

    factory = ExperimentFactory(runtime_factory=create_runtime)

    definition = create_benchmark_definition()

    first_experiment = factory(
        BenchmarkExperiment,
        definition,
        "qwen2.5-coder:7b",
    )

    second_experiment = factory(
        BenchmarkExperiment,
        definition,
        "llama3.1:8b",
    )

    assert len(created_runtimes) == 2
    assert first_experiment._runtime is created_runtimes[0]
    assert second_experiment._runtime is created_runtimes[1]
    assert first_experiment._runtime is not second_experiment._runtime


def test_rejects_incompatible_experiment_class():
    factory = ExperimentFactory()

    definition = create_benchmark_definition()

    with pytest.raises(ValueError, match="must inherit from BenchmarkExperiment"):
        factory(
            OtherExperiment,
            definition,
            "qwen2.5-coder:7b",
        )


def test_rejects_unsupported_experiment_type():
    factory = ExperimentFactory()

    definition = ExperimentDefinition(
        experiment_type="unknown",
    )

    with pytest.raises(ValueError, match="Unsupported experiment type"):
        factory(
            BenchmarkExperiment,
            definition,
            "qwen2.5-coder:7b",
        )


def test_rejects_invalid_benchmark_configuration():
    factory = ExperimentFactory(
        runtime_factory=FakeRuntime,
    )

    definition = ExperimentDefinition(
        experiment_type="benchmark",
        config={
            "prompt": "Write a Python function.",
            "task_id": "code-generation-basic",
            "unexpected_option": True,
        },
    )

    with pytest.raises(TypeError):
        factory(
            BenchmarkExperiment,
            definition,
            "qwen2.5-coder:7b",
        )
