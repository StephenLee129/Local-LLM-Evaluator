from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from experiments.base import Experiment
from experiments.models import (
    ExperimentDefinition,
    ExperimentResult,
    ExperimentStatus,
    ExperimentSuiteDefinition,
    ExperimentSuiteResult,
)
from experiments.registry import ExperimentRegistry


ExperimentFactory = Callable[
    [type[Experiment[Any]], ExperimentDefinition, str],
    Experiment[Any],
]


class ExperimentOrchestrator:
    """Coordinates sequential execution of experiment suites."""

    def __init__(
        self,
        registry: ExperimentRegistry,
        experiment_factory: ExperimentFactory,
    ) -> None:
        self._registry = registry
        self._experiment_factory = experiment_factory

    def run(
        self,
        suite: ExperimentSuiteDefinition,
    ) -> ExperimentSuiteResult:
        """Execute every experiment against every model sequentially."""

        suite_id = uuid4()
        suite_started_at_utc = datetime.now(timezone.utc)
        suite_start_time = perf_counter()

        results: list[ExperimentResult[Any]] = []

        for model in suite.models:
            for definition in suite.experiments:
                experiment_id = uuid4()
                started_at_utc = datetime.now(timezone.utc)
                start_time = perf_counter()

                try:
                    experiment_class = self._registry.get(
                        definition.experiment_type
                    )

                    experiment = self._experiment_factory(
                        experiment_class,
                        definition,
                        model,
                    )

                    if experiment.experiment_type != definition.experiment_type:
                        raise ValueError(
                            f"Registered experiment type "
                            f"'{definition.experiment_type}' does not match "
                            f"implementation type '{experiment.experiment_type}'."
                        )

                    experiment_results = experiment.run()

                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        suite_id=suite_id,
                        experiment_type=definition.experiment_type,
                        model=model,
                        status=ExperimentStatus.COMPLETED,
                        started_at_utc=started_at_utc,
                        duration_seconds=perf_counter() - start_time,
                        results=experiment_results,
                        config=definition.config.copy(),
                    )

                except Exception as exc:
                    result = ExperimentResult(
                        experiment_id=experiment_id,
                        suite_id=suite_id,
                        experiment_type=definition.experiment_type,
                        model=model,
                        status=ExperimentStatus.FAILED,
                        started_at_utc=started_at_utc,
                        duration_seconds=perf_counter() - start_time,
                        error=f"{type(exc).__name__}: {exc}",
                        config=definition.config.copy(),
                    )

                results.append(result)

        return ExperimentSuiteResult(
            suite_id=suite_id,
            name=suite.name,
            started_at_utc=suite_started_at_utc,
            duration_seconds=perf_counter() - suite_start_time,
            experiments=results,
        )
