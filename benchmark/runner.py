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
    WarmupSummary,
)


DEFAULT_WARMUP_RUNS = 1


class BenchmarkRunner:
    """Executes repeated inference requests against a runtime."""

    def __init__(self, runtime: InferenceRuntime):
        self.runtime = runtime

    def _execute_request(
        self,
        request: GenerationRequest,
    ) -> tuple[GenerationResult | None, BenchmarkError | None, float]:
        """Execute one inference request and capture its outcome and duration."""

        start_time = time.perf_counter()

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

        return generation, error, duration

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

        warmup_completed = 0
        warmup_failed = 0

        for _ in range(warmup_runs):
            _, error, _ = self._execute_request(request)

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

        for iteration in range(1, iterations + 1):
            generation, error, duration = self._execute_request(request)

            result = BenchmarkRunResult(
                iteration=iteration,
                status="failed" if error is not None else "success",
                duration_seconds=duration,
                generation=generation,
                error=error,
            )

            results.append(result)

        return warmup_summary, results
