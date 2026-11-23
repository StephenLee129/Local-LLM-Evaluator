import json
import re

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import UUID

from benchmark.models import BenchmarkExecution


DEFAULT_RESULTS_DIRECTORY = Path("results/local")


def execution_to_dict(execution: BenchmarkExecution) -> dict:
    """Convert a benchmark execution into the persisted JSON structure."""

    return {
        "metadata": asdict(execution.metadata),
        "configuration": asdict(execution.configuration),
        "environment": asdict(execution.environment),
        "results": {
            "warmup_summary": asdict(execution.warmup_summary),
            "measured_runs": [
                asdict(run) for run in execution.measured_runs
            ],
        },
        "analysis": asdict(execution.analysis),
        "warnings": [
            asdict(warning) for warning in execution.warnings
        ],
    }


def json_serializer(value):
    """Convert non-native JSON types into serializable values."""

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def serialize_execution(execution: BenchmarkExecution) -> str:
    """Serialize a benchmark execution into formatted JSON."""

    execution_data = execution_to_dict(execution)

    return json.dumps(
        execution_data,
        default=json_serializer,
        indent=2,
        ensure_ascii=False,
    )


def sanitize_filename_component(value: str) -> str:
    """Replace filesystem-unfriendly characters with hyphens."""

    sanitized = re.sub(r"[^a-zA-Z0-9.-]+", "-", value)

    return sanitized.strip("-.")


def generate_result_filename(
    execution: BenchmarkExecution,
) -> str:
    """Generate a descriptive, sortable benchmark result filename."""

    timestamp = execution.metadata.timestamp_utc
    model = sanitize_filename_component(
        execution.configuration.model
    )
    task_id = sanitize_filename_component(
        execution.configuration.task_id
    )
    short_id = str(execution.metadata.execution_id)[:8]

    date = timestamp.strftime("%Y-%m-%d")

    return f"{date}_{model}_{task_id}_{short_id}.json"


def save_execution(
    execution: BenchmarkExecution,
    output_directory: Path = DEFAULT_RESULTS_DIRECTORY,
) -> Path:
    """Serialize and save a benchmark execution to disk."""

    output_directory.mkdir(parents=True, exist_ok=True)

    filename = generate_result_filename(execution)
    output_path = output_directory / filename

    serialized = serialize_execution(execution)

    output_path.write_text(
        serialized,
        encoding="utf-8",
    )

    return output_path
