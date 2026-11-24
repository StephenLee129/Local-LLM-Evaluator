import argparse
from pathlib import Path

from experiments.benchmark import BenchmarkExperiment
from experiments.config_loader import load_experiment_suite
from experiments.factory import ExperimentFactory
from experiments.models import ExperimentSuiteResult
from experiments.orchestrator import ExperimentOrchestrator
from experiments.persistence import save_suite_result
from experiments.registry import ExperimentRegistry


def create_registry() -> ExperimentRegistry:
    """Create and populate the experiment registry."""

    registry = ExperimentRegistry()
    registry.register("benchmark", BenchmarkExperiment)

    return registry


def run_experiment_suite(
    config_path: str | Path,
    output_directory: str | Path | None = None,
    registry: ExperimentRegistry | None = None,
    factory: ExperimentFactory | None = None,
) -> ExperimentSuiteResult:
    """Load, execute, and persist an experiment suite."""

    suite = load_experiment_suite(config_path)

    registry = registry if registry is not None else create_registry()
    factory = factory if factory is not None else ExperimentFactory()

    orchestrator = ExperimentOrchestrator(
        registry=registry,
        experiment_factory=factory,
    )

    result = orchestrator.run(suite)

    if output_directory is None:
        result_path = save_suite_result(result)
    else:
        result_path = save_suite_result(result, output_directory)

    successful = sum(
        experiment.status.value == "completed"
        for experiment in result.experiments
    )
    failed = len(result.experiments) - successful

    print(f"Experiment suite: {result.name}")
    print(f"Suite ID: {result.suite_id}")
    print(f"Experiments completed: {successful}")
    print(f"Experiments failed: {failed}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Results saved to: {result_path}")

    return result


def main() -> None:
    """Run an experiment suite from the command line."""

    parser = argparse.ArgumentParser(
        description="Run an experiment suite from a YAML configuration."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML experiment suite configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where experiment results will be saved.",
    )

    args = parser.parse_args()

    run_experiment_suite(
        config_path=args.config,
        output_directory=args.output_dir,
    )


if __name__ == "__main__":
    main()
