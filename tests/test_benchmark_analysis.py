import pytest

from benchmark.analysis import analyze_results, summarize_metric
from runtimes.base import GenerationResult


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


def test_analyze_results_calculates_tokens_per_second():
    results = [
        GenerationResult(
            output="response 1",
            total_duration_seconds=2.0,
            time_to_first_token_seconds=0.5,
            generated_tokens=40,
            generation_duration_seconds=1.0,
        ),
        GenerationResult(
            output="response 2",
            total_duration_seconds=4.0,
            time_to_first_token_seconds=1.0,
            generated_tokens=60,
            generation_duration_seconds=2.0,
        ),
    ]

    analysis = analyze_results(results)

    assert analysis.metrics["total_duration_seconds"].mean == 3.0
    assert analysis.metrics["tokens_per_second"].mean == 35.0
    assert analysis.metrics["tokens_per_second"].count == 2


def test_analyze_results_handles_missing_metrics():
    results = [
        GenerationResult(
            output="response",
            total_duration_seconds=2.0,
        )
    ]

    analysis = analyze_results(results)

    assert analysis.metrics["total_duration_seconds"].mean == 2.0
    assert analysis.metrics["time_to_first_token_seconds"] is None
    assert analysis.metrics["tokens_per_second"] is None


def test_analyze_results_rejects_empty_results():
    with pytest.raises(ValueError):
        analyze_results([])
