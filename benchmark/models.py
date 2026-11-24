from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from runtimes.base import GenerationResult


@dataclass
class ResourceSample:
    """A snapshot of system and GPU resource usage."""

    timestamp_seconds: float
    gpu_memory_used_mb: float | None = None
    gpu_utilization_percent: float | None = None
    system_ram_used_mb: float | None = None
    system_ram_available_mb: float | None = None
    captured_at_utc: datetime | None = None


@dataclass
class ResourceProfile:
    """Resource measurements captured before and after model warmup."""

    pre_warmup: ResourceSample
    post_warmup: ResourceSample


@dataclass
class ResourceStatistics:
    sample_count: int
    gpu_memory_peak_mb: float | None = None
    gpu_memory_baseline_mb: float | None = None
    gpu_memory_average_mb: float | None = None
    gpu_memory_peak_delta_mb: float | None = None
    gpu_utilization_average_percent: float | None = None
    gpu_utilization_peak_percent: float | None = None
    system_ram_peak_mb: float | None = None
    system_ram_baseline_mb: float | None = None
    system_ram_peak_delta_mb: float | None = None


@dataclass
class BenchmarkMetadata:
    execution_id: UUID
    timestamp_utc: datetime
    schema_version: str
    execution_duration_seconds: float | None = None


@dataclass
class BenchmarkConfiguration:
    model: str
    runtime: str
    temperature: float
    max_tokens: int | None
    task_id: str
    prompt: str
    warmup_runs: int
    measured_iterations: int


@dataclass
class GPUInfo:
    model: str | None = None
    total_vram_gb: float | None = None
    driver_version: str | None = None


@dataclass
class EnvironmentInfo:
    operating_system: str | None = None
    python_version: str | None = None
    cpu: str | None = None
    ram_gb: float | None = None
    gpu: GPUInfo | None = None


@dataclass
class BenchmarkError:
    category: str
    type: str
    message: str


@dataclass
class BenchmarkRunResult:
    iteration: int
    status: Literal["success", "failed"]
    duration_seconds: float
    generation: GenerationResult | None = None
    error: BenchmarkError | None = None
    resources: ResourceStatistics | None = None

    def __post_init__(self):
        """Validate the consistency of an individual benchmark result."""

        if self.iteration < 1:
            raise ValueError("iteration must be at least 1")

        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")

        if self.status == "success":
            if self.generation is None or self.error is not None:
                raise ValueError(
                    "Successful results must have a generation "
                    "and cannot have an error."
                )

        elif self.status == "failed":
            if self.generation is not None or self.error is None:
                raise ValueError(
                    "Failed results must have an error "
                    "and cannot have a generation."
                )

        else:
            raise ValueError(
                f"Unsupported benchmark result status: {self.status}"
            )


@dataclass
class WarmupSummary:
    runs_requested: int
    runs_completed: int
    runs_failed: int


@dataclass
class MetricStatistics:
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    standard_deviation: float | None


@dataclass
class ReliabilityStatistics:
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    success_rate: float
    failure_rate: float
    failed_duration: MetricStatistics | None
    errors_by_category: dict[str, int]

    longest_consecutive_failures: int = 0
    failures_followed_by_success: int = 0
    failures_followed_by_failure: int = 0
    recovery_rate: float | None = None


@dataclass
class BenchmarkAnalysis:
    metrics: dict[str, MetricStatistics | None]
    reliability: ReliabilityStatistics


@dataclass
class BenchmarkWarning:
    component: str
    message: str
    details: str | None = None


@dataclass
class BenchmarkExecution:
    metadata: BenchmarkMetadata
    configuration: BenchmarkConfiguration
    environment: EnvironmentInfo
    warmup_summary: WarmupSummary
    measured_runs: list[BenchmarkRunResult]
    analysis: BenchmarkAnalysis
    warnings: list[BenchmarkWarning]
    resource_profile: ResourceProfile | None = None
