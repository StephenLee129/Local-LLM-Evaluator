from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from benchmark.analysis import analyze_results
from benchmark.environment import collect_environment
from benchmark.models import (
    BenchmarkConfiguration,
    BenchmarkExecution,
    BenchmarkMetadata,
)
from benchmark.monitoring import ResourceMonitor
from benchmark.runner import BenchmarkRunner
from experiments.base import Experiment
from experiments.benchmark_config import BenchmarkExperimentConfig
from runtimes.base import GenerationRequest, InferenceRuntime


SCHEMA_VERSION = "1.0"


class BenchmarkExperiment(Experiment[BenchmarkExecution]):
    """Runs a benchmark against a model using an inference runtime."""

    def __init__(
        self,
        model: str,
        config: BenchmarkExperimentConfig,
        runtime: InferenceRuntime,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._runtime = runtime
        self._resource_monitor = resource_monitor

    @property
    def experiment_type(self) -> str:
        return "benchmark"

    def run(self) -> BenchmarkExecution:
        """Execute the benchmark and assemble its results."""

        execution_id = uuid4()
        timestamp_utc = datetime.now(timezone.utc)
        execution_start = perf_counter()

        request = GenerationRequest(
            model=self._model,
            prompt=self._config.prompt,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        environment, warnings = collect_environment()

        runner = BenchmarkRunner(
            runtime=self._runtime,
            resource_monitor=self._resource_monitor,
        )

        warmup_summary, results, resource_profile = runner.run(
            request=request,
            iterations=self._config.iterations,
            warmup_runs=self._config.warmup_runs,
        )

        warnings.extend(runner.warnings)

        analysis = analyze_results(results)

        execution_duration = perf_counter() - execution_start

        return BenchmarkExecution(
            metadata=BenchmarkMetadata(
                execution_id=execution_id,
                timestamp_utc=timestamp_utc,
                schema_version=SCHEMA_VERSION,
                execution_duration_seconds=execution_duration,
            ),
            configuration=BenchmarkConfiguration(
                model=self._model,
                runtime=self._runtime.runtime_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                task_id=self._config.task_id,
                prompt=self._config.prompt,
                warmup_runs=self._config.warmup_runs,
                measured_iterations=self._config.iterations,
            ),
            environment=environment,
            warmup_summary=warmup_summary,
            measured_runs=results,
            analysis=analysis,
            warnings=warnings,
            resource_profile=resource_profile,
        )
