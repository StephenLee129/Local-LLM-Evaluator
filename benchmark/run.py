from runtimes.base import GenerationRequest
from runtimes.ollama import OllamaRuntime

from benchmark.runner import BenchmarkRunner
from benchmark.analysis import analyze_results


# Benchmark configuration
MODEL_NAME = "qwen2.5-coder:7b"
PROMPT = "Write a Python function that adds two numbers."

WARMUP_RUNS = 2
BENCHMARK_ITERATIONS = 5

TEMPERATURE = 0.0
MAX_TOKENS = 150


def main():
    runtime = OllamaRuntime()
    runner = BenchmarkRunner(runtime)

    request = GenerationRequest(
        model=MODEL_NAME,
        prompt=PROMPT,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    print("Starting benchmark...")
    print(f"Runtime: {runtime.runtime_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Measured iterations: {BENCHMARK_ITERATIONS}")
    print()

    results = runner.run(
        request=request,
        iterations=BENCHMARK_ITERATIONS,
        warmup_runs=WARMUP_RUNS,
    )

    analysis = analyze_results(results)

    print("Benchmark results:")
    print("-" * 60)

    for metric_name, statistics in analysis.metrics.items():
        print(f"\n{metric_name}")

        if statistics is None:
            print("  No measurements available")
            continue

        print(f"  Count: {statistics.count}")
        print(f"  Mean: {statistics.mean:.4f}")
        print(f"  Median: {statistics.median:.4f}")
        print(f"  Min: {statistics.minimum:.4f}")
        print(f"  Max: {statistics.maximum:.4f}")

        if statistics.standard_deviation is not None:
            print(f"  Std dev: {statistics.standard_deviation:.4f}")

    print("\nIndividual run results:")
    print("-" * 60)

    for index, result in enumerate(results, start=1):
        print(
            f"Run {index}: "
            f"{result.total_duration_seconds:.3f}s total, "
            f"{result.generated_tokens} generated tokens"
        )


if __name__ == "__main__":
    main()
