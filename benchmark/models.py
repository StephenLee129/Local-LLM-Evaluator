from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from runtimes.base import GenerationResult


@dataclass
class ResourceSample:
    timestamp_seconds: float
    gpu_memory_used_mb: float | None = None
    gpu_utilization_percent: float | None = None
    system_ram_used_mb: float | None = None
    system_ram_available_mb: float | None = None


@dataclass
class ResourceStatistics:
    sample_count: int
    gpu_memory_peak_mb: float | None = None
    gpu_memory_baseline_mb: float | None = None
    gpu_memory_average_mb: float | None = None
    gpu_utilization_average_percent: float | None = None
    gpu_utilization_peak_percent: float | None = None
    system_ram_peak_mb: float | None = None
    system_ram_baseline_mb: float | None = None

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
