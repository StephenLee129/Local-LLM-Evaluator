import time

from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
    InferenceRuntimeError,
)

from benchmark.models import (
    BenchmarkError,
    BenchmarkRunResult,
    BenchmarkWarning,
    ResourceStatistics,
    WarmupSummary,
)

from benchmark.monitoring import ResourceMonitor


DEFAULT_WARMUP_RUNS = 1


class BenchmarkRunner:
    """Executes repeated inference requests against a runtime."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        resource_monitor: ResourceMonitor | None = None,
    ):
        self.runtime = runtime
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.warnings: list[BenchmarkWarning] = []

    def _execute_request(
        self,
        request: GenerationRequest,
        monitor_resources: bool = False,
    ) -> tuple[
        GenerationResult | None,
        BenchmarkError | None,
        float,
        ResourceStatistics | None,
    ]:
        """Execute one inference request and capture its outcome and metrics."""

        monitor_started = False

        if monitor_resources:
            try:
                self.resource_monitor.start()
                monitor_started = True

            except Exception as exc:
                self.warnings.append(
                    BenchmarkWarning(
                        component="resource_monitor",
                        message="Resource monitoring failed to start.",
                        details=f"{type(exc).__name__}: {exc}",
                    )
                )

        start_time = time.perf_counter()

        try:
            try:
                generation = self.runtime.generate(request)
                error = None

            except InferenceRuntimeError as exc:
                generation = None
                error = BenchmarkError(
                    category=exc.category,
                    type=exc.original_type,
                    message=str(exc),
                )

            duration = time.perf_counter() - start_time

        finally:
            if monitor_started:
                try:
                    self.resource_monitor.stop()

                except Exception as exc:
                    self.warnings.append(
                        BenchmarkWarning(
                            component="resource_monitor",
                            message="Resource monitoring failed to stop.",
                            details=f"{type(exc).__name__}: {exc}",
                        )
                    )

        resources = None

        if monitor_started:
            resources = self.resource_monitor._calculate_statistics()

            for error_message in self.resource_monitor.errors:
                self.warnings.append(
                    BenchmarkWarning(
                        component="resource_monitor",
                        message="Resource monitoring encountered an error.",
                        details=error_message,
                    )
                )

        return generation, error, duration, resources

    def run(
        self,
        request: GenerationRequest,
        iterations: int,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
    ) -> tuple[WarmupSummary, list[BenchmarkRunResult]]:
        """Run warmups followed by measured inference iterations."""

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")

        self.warnings.clear()

        warmup_completed = 0
        warmup_failed = 0

        # Warmup runs are intentionally not monitored.
        for _ in range(warmup_runs):
            _, error, _, _ = self._execute_request(request)

            if error is None:
                warmup_completed += 1
            else:
                warmup_failed += 1

        warmup_summary = WarmupSummary(
            runs_requested=warmup_runs,
            runs_completed=warmup_completed,
            runs_failed=warmup_failed,
        )

        results = []

        # Collect resource measurements for each measured iteration.
        for iteration in range(1, iterations + 1):
            generation, error, duration, resources = self._execute_request(
                request,
                monitor_resources=True,
            )

            result = BenchmarkRunResult(
                iteration=iteration,
                status="failed" if error is not None else "success",
                duration_seconds=duration,
                generation=generation,
                error=error,
                resources=resources,
            )

            results.append(result)

        return warmup_summary, results
