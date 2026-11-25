from dataclasses import dataclass
from uuid import UUID

from experiments.base import Experiment
from experiments.models import (
    ExperimentDefinition,
    ExperimentStatus,
    ExperimentSuiteDefinition,
)
from experiments.orchestrator import ExperimentOrchestrator
from experiments.registry import ExperimentRegistry


@dataclass
class MockResults:
    message: str


class SuccessfulExperiment(Experiment[MockResults]):
    @property
    def experiment_type(self) -> str:
        return "successful"

    def run(self) -> MockResults:
        return MockResults(message="Experiment completed")


class FailingExperiment(Experiment[MockResults]):
    @property
    def experiment_type(self) -> str:
        return "failing"

    def run(self) -> MockResults:
        raise RuntimeError("Simulated experiment failure")


def create_factory():
    def factory(experiment_class, definition, model):
        return experiment_class()

    return factory


def create_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.register("successful", SuccessfulExperiment)
    registry.register("failing", FailingExperiment)
    return registry


def create_suite(
    experiments: list[ExperimentDefinition],
    models: list[str] | None = None,
) -> ExperimentSuiteDefinition:
    return ExperimentSuiteDefinition(
        name="test-suite",
        models=models or ["qwen2.5-coder:7b"],
        experiments=experiments,
    )


def test_successful_experiment_returns_completed_result():
    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(experiment_type="successful")
    ]))

    assert len(suite.experiments) == 1

    result = suite.experiments[0]

    assert result.status == ExperimentStatus.COMPLETED
    assert result.results.message == "Experiment completed"
    assert result.error is None
    assert result.model == "qwen2.5-coder:7b"


def test_failed_experiment_does_not_abort_suite():
    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(experiment_type="failing"),
        ExperimentDefinition(experiment_type="successful"),
    ]))

    assert len(suite.experiments) == 2

    assert suite.experiments[0].status == ExperimentStatus.FAILED
    assert "Simulated experiment failure" in suite.experiments[0].error

    assert suite.experiments[1].status == ExperimentStatus.COMPLETED


def test_experiments_receive_unique_ids_and_shared_suite_id():
    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(experiment_type="successful"),
        ExperimentDefinition(experiment_type="successful"),
    ]))

    first, second = suite.experiments

    assert isinstance(suite.suite_id, UUID)
    assert first.suite_id == suite.suite_id
    assert second.suite_id == suite.suite_id

    assert first.experiment_id != second.experiment_id


def test_unknown_experiment_type_is_recorded_as_failure():
    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(experiment_type="unknown")
    ]))

    result = suite.experiments[0]

    assert result.status == ExperimentStatus.FAILED
    assert "Unknown experiment type" in result.error


def test_empty_suite_returns_valid_result():
    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([]))

    assert isinstance(suite.suite_id, UUID)
    assert suite.experiments == []
    assert suite.duration_seconds >= 0


def test_experiment_result_preserves_configuration():
    config = {
        "warmup_runs": 2,
        "iterations": 5,
    }

    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(
            experiment_type="successful",
            config=config,
        )
    ]))

    assert suite.experiments[0].config == config


def test_mismatched_experiment_type_is_recorded_as_failure():
    class MismatchedExperiment(SuccessfulExperiment):
        @property
        def experiment_type(self) -> str:
            return "different_type"

    registry = ExperimentRegistry()
    registry.register("successful", MismatchedExperiment)

    orchestrator = ExperimentOrchestrator(
        registry=registry,
        experiment_factory=create_factory(),
    )

    suite = orchestrator.run(create_suite([
        ExperimentDefinition(experiment_type="successful")
    ]))

    result = suite.experiments[0]

    assert result.status == ExperimentStatus.FAILED
    assert "does not match implementation type" in result.error


def test_orchestrator_runs_every_experiment_for_every_model():
    executed_combinations = []

    def factory(experiment_class, definition, model):
        executed_combinations.append(
            (model, definition.experiment_type)
        )
        return experiment_class()

    orchestrator = ExperimentOrchestrator(
        registry=create_registry(),
        experiment_factory=factory,
    )

    suite = orchestrator.run(
        create_suite(
            experiments=[
                ExperimentDefinition(experiment_type="successful"),
                ExperimentDefinition(experiment_type="failing"),
            ],
            models=[
                "qwen2.5-coder:7b",
                "llama3.1:8b",
            ],
        )
    )

    assert executed_combinations == [
        ("qwen2.5-coder:7b", "successful"),
        ("qwen2.5-coder:7b", "failing"),
        ("llama3.1:8b", "successful"),
        ("llama3.1:8b", "failing"),
    ]

    assert len(suite.experiments) == 4

    assert [result.model for result in suite.experiments] == [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:7b",
        "llama3.1:8b",
        "llama3.1:8b",
    ]

    assert [
        result.status for result in suite.experiments
    ] == [
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
    ]
