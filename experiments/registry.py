from typing import Any

from experiments.base import Experiment


class ExperimentRegistry:
    """Registry mapping experiment types to their implementations."""

    def __init__(self) -> None:
        self._experiments: dict[
            str, type[Experiment[Any]]
        ] = {}

    def register(
        self,
        experiment_type: str,
        experiment_class: type[Experiment[Any]],
    ) -> None:
        """Register an experiment implementation."""

        experiment_type = experiment_type.strip()

        if not experiment_type:
            raise ValueError("Experiment type cannot be empty.")

        if experiment_type in self._experiments:
            raise ValueError(
                f"Experiment type '{experiment_type}' is already registered."
            )

        self._experiments[experiment_type] = experiment_class

    def get(
        self,
        experiment_type: str,
    ) -> type[Experiment[Any]]:
        """Retrieve an experiment implementation by type."""

        experiment_type = experiment_type.strip()

        if experiment_type not in self._experiments:
            raise ValueError(
                f"Unknown experiment type: '{experiment_type}'."
            )

        return self._experiments[experiment_type]
