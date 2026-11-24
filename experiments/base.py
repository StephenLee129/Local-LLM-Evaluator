from abc import ABC, abstractmethod

from experiments.models import ExperimentResult


class Experiment[T](ABC):
    """Common interface implemented by all experiment types."""

    @property
    @abstractmethod
    def experiment_type(self) -> str:
        """Unique identifier for this experiment type."""
        pass

    @abstractmethod
    def run(self) -> T:
        """Execute the experiment and return its type-specific results."""
        pass
