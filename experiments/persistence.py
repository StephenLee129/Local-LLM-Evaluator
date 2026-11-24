import json
import re
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from experiments.models import ExperimentSuiteResult


DEFAULT_RESULTS_DIRECTORY = Path("results/local")


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert supported Python objects into JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_compatible(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {
            str(key): _to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported type for JSON serialization: {type(value).__name__}"
    )


def suite_result_to_dict(result: ExperimentSuiteResult) -> dict[str, Any]:
    """Convert an experiment suite result into a dictionary."""

    return _to_json_compatible(result)


def serialize_suite_result(result: ExperimentSuiteResult) -> str:
    """Serialize an experiment suite result as formatted JSON."""

    return json.dumps(
        suite_result_to_dict(result),
        indent=2,
        ensure_ascii=False,
    )


def generate_suite_result_filename(result: ExperimentSuiteResult) -> str:
    """Generate a filename identifying an experiment suite execution."""

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        result.name.strip(),
    ).strip("-._")

    if not safe_name:
        safe_name = "experiment-suite"

    date_prefix = result.started_at_utc.strftime("%Y-%m-%d")

    return f"{date_prefix}_{safe_name}_{result.suite_id}.json"


def save_suite_result(
    result: ExperimentSuiteResult,
    output_directory: str | Path = DEFAULT_RESULTS_DIRECTORY,
) -> Path:
    """Serialize and save an experiment suite result to disk."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = generate_suite_result_filename(result)
    file_path = output_path / filename

    file_path.write_text(
        serialize_suite_result(result),
        encoding="utf-8",
    )

    return file_path
