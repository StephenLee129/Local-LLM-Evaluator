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


def calculate_longest_consecutive_failures(
    results: list[BenchmarkRunResult],
) -> int:
    """Calculate the longest uninterrupted sequence of failed requests."""

    longest_streak = 0
    current_streak = 0

    for result in results:
        if result.status == "failed":
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    return longest_streak


def analyze_failure_recovery(
    results: list[BenchmarkRunResult],
) -> tuple[int, int, float | None]:
    """Analyze whether failed requests are followed by successful requests."""

    failures_followed_by_success = 0
    failures_followed_by_failure = 0

    # The final request cannot be evaluated for recovery because
    # there is no subsequent request to observe.
    for current, following in zip(results, results[1:]):
        if current.status != "failed":
            continue

        if following.status == "success":
            failures_followed_by_success += 1
        else:
            failures_followed_by_failure += 1

    recoverable_failures = (
        failures_followed_by_success + failures_followed_by_failure
    )

    recovery_rate = (
        failures_followed_by_success / recoverable_failures
        if recoverable_failures > 0
        else None
    )

    return (
        failures_followed_by_success,
        failures_followed_by_failure,
        recovery_rate,
    )


def analyze_results(
    results: list[BenchmarkRunResult],
) -> BenchmarkAnalysis:
    """Analyze performance and reliability across measured iterations."""

    if not results:
        raise ValueError("Cannot analyze an empty list of results")

    # Ensure reliability calculations follow execution order.
    ordered_results = sorted(results, key=lambda result: result.iteration)

    successful_runs = [
        result for result in ordered_results
        if result.status == "success"
    ]

    failed_runs = [
        result for result in ordered_results
        if result.status == "failed"
    ]

    metric_values: dict[str, list[float]] = {
        "client_duration_seconds": [],
        "time_to_first_token_seconds": [],
        "model_load_duration_seconds": [],
        "prompt_eval_duration_seconds": [],
        "generation_duration_seconds": [],
        "tokens_per_second": [],
        "runtime_total_duration_seconds": [],
    }

    for result in successful_runs:
        generation = result.generation

        # BenchmarkRunResult guarantees successful results have a generation.
        assert generation is not None

        metric_values["client_duration_seconds"].append(
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

    total_iterations = len(ordered_results)
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

    longest_consecutive_failures = (
        calculate_longest_consecutive_failures(ordered_results)
    )

    (
        failures_followed_by_success,
        failures_followed_by_failure,
        recovery_rate,
    ) = analyze_failure_recovery(ordered_results)

    reliability = ReliabilityStatistics(
        total_iterations=total_iterations,
        successful_iterations=successful_iterations,
        failed_iterations=failed_iterations,
        success_rate=success_rate,
        failure_rate=failure_rate,
        failed_duration=summarize_metric(failed_durations),
        errors_by_category=errors_by_category,
        longest_consecutive_failures=longest_consecutive_failures,
        failures_followed_by_success=failures_followed_by_success,
        failures_followed_by_failure=failures_followed_by_failure,
        recovery_rate=recovery_rate,
    )

    return BenchmarkAnalysis(
        metrics=performance_summaries,
        reliability=reliability,
    )
