from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class ExperimentStatus(str, Enum):
    """Execution status of an experiment."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentDefinition:
    """Configuration describing an experiment to execute."""

    experiment_type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSuiteDefinition:
    """Configuration describing a suite of experiments."""

    name: str
    models: list[str]
    experiments: list[ExperimentDefinition]


@dataclass
class ExperimentResult[T]:
    """Shared execution metadata and type-specific experiment results."""

    experiment_id: UUID
    suite_id: UUID
    experiment_type: str
    model: str
    status: ExperimentStatus
    started_at_utc: datetime
    duration_seconds: float
    results: T | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSuiteResult:
    """Results and execution metadata for an entire experiment suite."""
    suite_id: UUID
    name: str
    started_at_utc: datetime
    duration_seconds: float
    experiments: list[ExperimentResult[Any]] = field(default_factory=list)