from runtimes.base import (
    GenerationRequest,
    GenerationResult,
    InferenceRuntime,
)


DEFAULT_WARMUP_RUNS = 1


class BenchmarkRunner:
    """Executes repeated inference requests against a runtime."""

    def __init__(self, runtime: InferenceRuntime):
        self.runtime = runtime

    def run(
        self,
        request: GenerationRequest,
        iterations: int,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
    ) -> list[GenerationResult]:
        """Run warmup requests followed by measured inference requests."""

        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")

        for _ in range(warmup_runs):
            self.runtime.generate(request)

        results = []

        for _ in range(iterations):
            result = self.runtime.generate(request)
            results.append(result)

        return results
