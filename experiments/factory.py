from collections.abc import Callable
from typing import Any

from experiments.base import Experiment
from experiments.benchmark import BenchmarkExperiment
from experiments.benchmark_config import BenchmarkExperimentConfig
from experiments.models import ExperimentDefinition
from runtimes.base import InferenceRuntime
from runtimes.ollama import OllamaRuntime


RuntimeFactory = Callable[[], InferenceRuntime]


class ExperimentFactory:
    """Constructs experiment instances and their dependencies."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or OllamaRuntime

    def __call__(
        self,
        experiment_class: type[Experiment[Any]],
        definition: ExperimentDefinition,
        model: str,
    ) -> Experiment[Any]:
        """Create an experiment instance from its definition."""

        if definition.experiment_type == "benchmark":
            if not issubclass(experiment_class, BenchmarkExperiment):
                raise ValueError(
                    "The registered experiment class for 'benchmark' "
                    "must inherit from BenchmarkExperiment."
                )

            config = BenchmarkExperimentConfig(**definition.config)
            runtime = self._runtime_factory()

            return experiment_class(
                model=model,
                config=config,
                runtime=runtime,
            )

        raise ValueError(
            f"Unsupported experiment type: '{definition.experiment_type}'."
        )
