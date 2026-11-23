from collections import Counter
from statistics import mean, median, stdev

from runtimes.base import GenerationResult
from benchmark.models import (
    BenchmarkRunResult,
    BenchmarkAnalysis,
    MetricStatistics,
    ReliabilityStatistics,
)


def summarize_metric(values: list[float]) -> MetricStatistics | None:
    """Calculate descriptive statistics for available measurements."""

    if not values:
        return None

    return MetricStatistics(
        count=len(values),
        mean=mean(values),
        median=median(values),
        minimum=min(values),
        maximum=max(values),
        standard_deviation=stdev(values) if len(values) > 1 else None,
    )


def analyze_results(
    results: list[BenchmarkRunResult],
) -> BenchmarkAnalysis:
    """Analyze performance and reliability across measured iterations."""

    if not results:
        raise ValueError("Cannot analyze an empty list of results")

    successful_runs = [
        result for result in results
        if result.status == "success" and result.generation is not None
    ]

    failed_runs = [
        result for result in results
        if result.status == "failed" and result.error is not None
    ]

    metric_values: dict[str, list[float]] = {
        "total_duration_seconds": [],
        "time_to_first_token_seconds": [],
        "model_load_duration_seconds": [],
        "prompt_eval_duration_seconds": [],
        "generation_duration_seconds": [],
        "tokens_per_second": [],
        "runtime_total_duration_seconds": [],
    }

    for result in successful_runs:
        generation = result.generation

        metric_values["total_duration_seconds"].append(
            result.duration_seconds
        )

        optional_metrics = {
            "time_to_first_token_seconds": generation.time_to_first_token_seconds,
            "model_load_duration_seconds": generation.model_load_duration_seconds,
            "prompt_eval_duration_seconds": generation.prompt_eval_duration_seconds,
            "generation_duration_seconds": generation.generation_duration_seconds,
            "runtime_total_duration_seconds": generation.runtime_total_duration_seconds,
        }

        for metric_name, value in optional_metrics.items():
            if value is not None:
                metric_values[metric_name].append(value)

        if (
            generation.generated_tokens is not None
            and generation.generation_duration_seconds is not None
            and generation.generation_duration_seconds > 0
        ):
            tokens_per_second = (
                generation.generated_tokens
                / generation.generation_duration_seconds
            )
            metric_values["tokens_per_second"].append(tokens_per_second)

    performance_summaries = {
        metric_name: summarize_metric(values)
        for metric_name, values in metric_values.items()
    }

    total_iterations = len(results)
    successful_iterations = len(successful_runs)
    failed_iterations = len(failed_runs)

    success_rate = successful_iterations / total_iterations
    failure_rate = failed_iterations / total_iterations

    failed_durations = [
        result.duration_seconds
        for result in failed_runs
    ]

    errors_by_category = dict(
        Counter(result.error.category for result in failed_runs)
    )

    reliability = ReliabilityStatistics(
        total_iterations=total_iterations,
        successful_iterations=successful_iterations,
        failed_iterations=failed_iterations,
        success_rate=success_rate,
        failure_rate=failure_rate,
        failed_duration=summarize_metric(failed_durations),
        errors_by_category=errors_by_category,
    )

    return BenchmarkAnalysis(
        metrics=performance_summaries,
        reliability=reliability,
    )
