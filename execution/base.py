from abc import ABC, abstractmethod

from execution.models import ExecutionRequest, ExecutionResult


class ExecutionBackend(ABC):
    """Abstract interface for executing generated code."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of the execution backend."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code and return its result."""
        raise NotImplementedError


class ExecutionBackendError(Exception):
    """Raised when code execution infrastructure fails."""