import pytest

from benchmark.analysis import analyze_results, summarize_metric
from benchmark.models import (
    BenchmarkError,
    BenchmarkRunResult,
)
from runtimes.base import GenerationResult


def create_successful_run(
    iteration: int,
    duration: float,
    generation: GenerationResult,
) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        iteration=iteration,
        status="success",
        duration_seconds=duration,
        generation=generation,
    )


def create_failed_run(
    iteration: int,
    duration: float,
    category: str = "timeout",
) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        iteration=iteration,
        status="failed",
        duration_seconds=duration,
        error=BenchmarkError(
            category=category,
            type="TestError",
            message="Test failure",
        ),
    )


def test_summarize_metric_calculates_statistics():
    summary = summarize_metric([1.0, 2.0, 3.0])

    assert summary is not None
    assert summary.count == 3
    assert summary.mean == 2.0
    assert summary.median == 2.0
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0
    assert summary.standard_deviation == 1.0


def test_summarize_metric_returns_none_for_empty_values():
    assert summarize_metric([]) is None


def test_summarize_metric_handles_single_value():
    summary = summarize_metric([2.5])

    assert summary is not None
    assert summary.count == 1
    assert summary.mean == 2.5
    assert summary.standard_deviation is None


def test_analyze_results_calculates_performance_metrics():
    results = [
        create_successful_run(
            iteration=1,
            duration=2.0,
            generation=GenerationResult(
                output="response 1",
                total_duration_seconds=1.8,
                time_to_first_token_seconds=0.5,
                generated_tokens=40,
                generation_duration_seconds=1.0,
            ),
        ),
        create_successful_run(
            iteration=2,
            duration=4.0,
            generation=GenerationResult(
                output="response 2",
                total_duration_seconds=3.8,
                time_to_first_token_seconds=1.0,
                generated_tokens=60,
                generation_duration_seconds=2.0,
            ),
        ),
    ]

    analysis = analyze_results(results)

    # Client-observed durations come from BenchmarkRunResult.
    assert analysis.metrics["total_duration_seconds"].mean == 3.0

    # Throughput uses runtime generation measurements.
    assert analysis.metrics["tokens_per_second"].mean == 35.0
    assert analysis.metrics["tokens_per_second"].count == 2

    assert analysis.reliability.total_iterations == 2
    assert analysis.reliability.successful_iterations == 2
    assert analysis.reliability.failed_iterations == 0
    assert analysis.reliability.success_rate == 1.0
    assert analysis.reliability.failure_rate == 0.0


def test_analyze_results_handles_missing_metrics():
    results = [
        create_successful_run(
            iteration=1,
            duration=2.0,
            generation=GenerationResult(
                output="response",
                total_duration_seconds=1.8,
            ),
        )
    ]

    analysis = analyze_results(results)

    assert analysis.metrics["total_duration_seconds"].mean == 2.0
    assert analysis.metrics["time_to_first_token_seconds"] is None
    assert analysis.metrics["tokens_per_second"] is None

    assert analysis.reliability.failed_iterations == 0
    assert analysis.reliability.failed_duration is None
    assert analysis.reliability.errors_by_category == {}


def test_analyze_results_excludes_failed_runs_from_performance_metrics():
    results = [
        create_successful_run(
            iteration=1,
            duration=2.0,
            generation=GenerationResult(
                output="response",
                total_duration_seconds=1.8,
                generated_tokens=40,
                generation_duration_seconds=1.0,
            ),
        ),
        create_failed_run(
            iteration=2,
            duration=10.0,
            category="timeout",
        ),
    ]

    analysis = analyze_results(results)

    # Failed request duration must not inflate successful latency.
    assert analysis.metrics["total_duration_seconds"].count == 1
    assert analysis.metrics["total_duration_seconds"].mean == 2.0

    # The failed request is still included in reliability statistics.
    assert analysis.reliability.total_iterations == 2
    assert analysis.reliability.successful_iterations == 1
    assert analysis.reliability.failed_iterations == 1
    assert analysis.reliability.success_rate == 0.5
    assert analysis.reliability.failure_rate == 0.5

    assert analysis.reliability.failed_duration.count == 1
    assert analysis.reliability.failed_duration.mean == 10.0
    assert analysis.reliability.errors_by_category == {"timeout": 1}


def test_analyze_results_handles_multiple_error_categories():
    results = [
        create_failed_run(1, 5.0, "timeout"),
        create_failed_run(2, 2.0, "connection"),
        create_failed_run(3, 4.0, "timeout"),
    ]

    analysis = analyze_results(results)

    assert analysis.reliability.total_iterations == 3
    assert analysis.reliability.successful_iterations == 0
    assert analysis.reliability.failed_iterations == 3
    assert analysis.reliability.success_rate == 0.0
    assert analysis.reliability.failure_rate == 1.0

    assert analysis.reliability.errors_by_category == {
        "timeout": 2,
        "connection": 1,
    }

    assert analysis.reliability.failed_duration.count == 3
    assert analysis.reliability.failed_duration.mean == pytest.approx(11.0 / 3.0)

    # There are no successful measurements to analyze.
    assert analysis.metrics["total_duration_seconds"] is None
    assert analysis.metrics["tokens_per_second"] is None


def test_analyze_results_rejects_empty_results():
    with pytest.raises(ValueError):
        analyze_results([])
