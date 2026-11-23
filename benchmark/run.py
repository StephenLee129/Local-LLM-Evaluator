from runtimes.base import GenerationRequest
from runtimes.ollama import OllamaRuntime
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from benchmark.models import (
    BenchmarkConfiguration,
    BenchmarkExecution,
    BenchmarkMetadata,
)
from benchmark.environment import collect_environment
from benchmark.persistence import save_execution
from benchmark.runner import BenchmarkRunner
from benchmark.analysis import analyze_results


# Benchmark configuration
MODEL_NAME = "qwen2.5-coder:7b"
PROMPT = "Write a Python function that adds two numbers."

WARMUP_RUNS = 2
BENCHMARK_ITERATIONS = 5

TEMPERATURE = 0.0
MAX_TOKENS = 150

TASK_ID = "code-generation-basic"
SCHEMA_VERSION = "1.0"


def print_metric_statistics(metric_name, statistics):
    print(f"\n{metric_name}")

    if statistics is None:
        print("  No measurements available")
        return

    print(f"  Count: {statistics.count}")
    print(f"  Mean: {statistics.mean:.4f}")
    print(f"  Median: {statistics.median:.4f}")
    print(f"  Min: {statistics.minimum:.4f}")
    print(f"  Max: {statistics.maximum:.4f}")

    if statistics.standard_deviation is not None:
        print(f"  Std dev: {statistics.standard_deviation:.4f}")


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

    # Collect static environment metadata before benchmarking.
    environment, warnings = collect_environment()

    # Capture execution metadata.
    execution_id = uuid4()
    timestamp_utc = datetime.now(timezone.utc)
    execution_start = perf_counter()

    warmup_summary, results = runner.run(
        request=request,
        iterations=BENCHMARK_ITERATIONS,
        warmup_runs=WARMUP_RUNS,
    )

    # Include resource monitoring warnings in the execution.
    warnings.extend(runner.warnings)

    execution_duration = perf_counter() - execution_start

    analysis = analyze_results(results)

    # Assemble the complete benchmark execution.
    execution = BenchmarkExecution(
        metadata=BenchmarkMetadata(
            execution_id=execution_id,
            timestamp_utc=timestamp_utc,
            schema_version=SCHEMA_VERSION,
            execution_duration_seconds=execution_duration,
        ),
        configuration=BenchmarkConfiguration(
            model=MODEL_NAME,
            runtime=runtime.runtime_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            task_id=TASK_ID,
            prompt=PROMPT,
            warmup_runs=WARMUP_RUNS,
            measured_iterations=BENCHMARK_ITERATIONS,
        ),
        environment=environment,
        warmup_summary=warmup_summary,
        measured_runs=results,
        analysis=analysis,
        warnings=warnings,
    )

    # Persist the completed execution.
    output_path = save_execution(execution)

    print(f"\nBenchmark result saved to: {output_path}")

    # Warmup summary
    print("Warmup summary:")
    print("-" * 60)
    print(f"  Requested: {warmup_summary.runs_requested}")
    print(f"  Completed: {warmup_summary.runs_completed}")
    print(f"  Failed: {warmup_summary.runs_failed}")

    # Performance statistics
    print("\nPerformance results:")
    print("-" * 60)

    for metric_name, statistics in analysis.metrics.items():
        print_metric_statistics(metric_name, statistics)

    # Reliability statistics
    reliability = analysis.reliability

    print("\nReliability results:")
    print("-" * 60)
    print(f"  Total iterations: {reliability.total_iterations}")
    print(f"  Successful: {reliability.successful_iterations}")
    print(f"  Failed: {reliability.failed_iterations}")
    print(f"  Success rate: {reliability.success_rate:.1%}")
    print(f"  Failure rate: {reliability.failure_rate:.1%}")

    print("\nFailed request duration:")
    print_metric_statistics(
        "Failed request duration",
        reliability.failed_duration,
    )

    if reliability.errors_by_category:
        print("\nErrors by category:")

        for category, count in reliability.errors_by_category.items():
            print(f"  {category}: {count}")

    # Individual iteration results
    print("\nIndividual run results:")
    print("-" * 60)

    for result in results:
        print(
            f"Run {result.iteration}: "
            f"{result.status.upper()} - "
            f"{result.duration_seconds:.3f}s"
        )

        if result.status == "success":
            generation = result.generation

            print(f"  Generated tokens: {generation.generated_tokens}")
            print(f"  Output: {generation.output}")

        else:
            error = result.error

            print(f"  Error category: {error.category}")
            print(f"  Error type: {error.type}")
            print(f"  Error message: {error.message}")

        # Display resource statistics for this iteration.
        if result.resources is not None:
            resources = result.resources

            print(f"  Resource samples: {resources.sample_count}")

            if resources.gpu_memory_peak_mb is not None:
                print(
                    f"  GPU memory peak: "
                    f"{resources.gpu_memory_peak_mb:.2f} MB"
                )

            if resources.gpu_utilization_average_percent is not None:
                print(
                    f"  GPU utilization average: "
                    f"{resources.gpu_utilization_average_percent:.2f}%"
                )

            if resources.system_ram_peak_mb is not None:
                print(
                    f"  System RAM peak: "
                    f"{resources.system_ram_peak_mb:.2f} MB"
                )

    # Monitoring and environment warnings
    if warnings:
        print("\nBenchmark warnings:")
        print("-" * 60)

        for warning in warnings:
            print(f"  Component: {warning.component}")
            print(f"  Message: {warning.message}")

            if warning.details:
                print(f"  Details: {warning.details}")

if __name__ == "__main__":
    main()
    