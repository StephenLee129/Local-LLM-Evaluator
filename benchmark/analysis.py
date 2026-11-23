from statistics import mean, median, stdev

from runtimes.base import GenerationResult
from benchmark.models import MetricStatistics, BenchmarkAnalysis


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


def analyze_results(results: list[GenerationResult]) -> BenchmarkAnalysis:
    """Analyze collected inference results."""

    if not results:
        raise ValueError("Cannot analyze an empty list of results")

    metric_values: dict[str, list[float]] = {
        "total_duration_seconds": [],
        "time_to_first_token_seconds": [],
        "model_load_duration_seconds": [],
        "prompt_eval_duration_seconds": [],
        "generation_duration_seconds": [],
        "tokens_per_second": [],
        "runtime_total_duration_seconds": [],
    }

    for result in results:
        metric_values["total_duration_seconds"].append(
            result.total_duration_seconds
        )

        optional_metrics = {
            "time_to_first_token_seconds": result.time_to_first_token_seconds,
            "model_load_duration_seconds": result.model_load_duration_seconds,
            "prompt_eval_duration_seconds": result.prompt_eval_duration_seconds,
            "generation_duration_seconds": result.generation_duration_seconds,
            "runtime_total_duration_seconds": result.runtime_total_duration_seconds,
        }

        for metric_name, value in optional_metrics.items():
            if value is not None:
                metric_values[metric_name].append(value)

        if (
            result.generated_tokens is not None
            and result.generation_duration_seconds is not None
            and result.generation_duration_seconds > 0
        ):
            tokens_per_second = (
                result.generated_tokens / result.generation_duration_seconds
            )
            metric_values["tokens_per_second"].append(tokens_per_second)

    summaries = {
        metric_name: summarize_metric(values)
        for metric_name, values in metric_values.items()
    }

    return BenchmarkAnalysis(metrics=summaries)
