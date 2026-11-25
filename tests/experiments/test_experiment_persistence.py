import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

import pytest

from benchmark.models import (
    BenchmarkAnalysis,
    BenchmarkConfiguration,
    BenchmarkExecution,
    BenchmarkMetadata,
    EnvironmentInfo,
    ReliabilityStatistics,
    ResourceProfile,
    ResourceSample,
    WarmupSummary,
)
from experiments.models import (
    ExperimentResult,
    ExperimentStatus,
    ExperimentSuiteResult,
)
from experiments.persistence import (
    _to_json_compatible,
    generate_suite_result_filename,
    save_suite_result,
    serialize_suite_result,
    suite_result_to_dict,
)


@dataclass
class ExamplePayload:
    """Generic experiment payload used to test recursive serialization."""

    score: float
    labels: list[str]
    details: dict[str, int]


class ExampleStatus(str, Enum):
    READY = "ready"


def create_benchmark_execution() -> BenchmarkExecution:
    """Create a minimal benchmark execution with nested resource data."""

    timestamp = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    return BenchmarkExecution(
        metadata=BenchmarkMetadata(
            execution_id=uuid4(),
            timestamp_utc=timestamp,
            schema_version="1.0",
            execution_duration_seconds=2.5,
        ),
        configuration=BenchmarkConfiguration(
            model="qwen2.5-coder:7b",
            runtime="ollama",
            temperature=0.0,
            max_tokens=150,
            task_id="code-generation-basic",
            prompt="Write a Python function.",
            warmup_runs=2,
            measured_iterations=5,
        ),
        environment=EnvironmentInfo(),
        warmup_summary=WarmupSummary(
            runs_requested=2,
            runs_completed=2,
            runs_failed=0,
        ),
        measured_runs=[],
        analysis=BenchmarkAnalysis(
            metrics={},
            reliability=ReliabilityStatistics(
                total_iterations=0,
                successful_iterations=0,
                failed_iterations=0,
                success_rate=1.0,
                failure_rate=0.0,
                failed_duration=None,
                errors_by_category={},
            ),
        ),
        warnings=[],
        resource_profile=ResourceProfile(
            pre_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=1000.0,
                captured_at_utc=timestamp,
            ),
            post_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=4500.0,
                captured_at_utc=timestamp,
            ),
        ),
    )


def create_suite_result() -> ExperimentSuiteResult:
    """Create a suite containing successful and failed experiments."""

    suite_id = uuid4()
    started_at = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)

    successful_experiment = ExperimentResult(
        experiment_id=uuid4(),
        suite_id=suite_id,
        experiment_type="example",
        model="example-model",
        status=ExperimentStatus.COMPLETED,
        started_at_utc=started_at,
        duration_seconds=1.25,
        results=ExamplePayload(
            score=0.95,
            labels=["quality", "reliability"],
            details={"cases": 10},
        ),
        warnings=["Example warning"],
        config={"temperature": 0.2},
    )

    failed_experiment = ExperimentResult(
        experiment_id=uuid4(),
        suite_id=suite_id,
        experiment_type="unknown",
        model="another-model",
        status=ExperimentStatus.FAILED,
        started_at_utc=started_at,
        duration_seconds=0.1,
        error="ValueError: Unknown experiment type",
        config={},
    )

    return ExperimentSuiteResult(
        suite_id=suite_id,
        name="test-suite",
        started_at_utc=started_at,
        duration_seconds=1.5,
        experiments=[
            successful_experiment,
            failed_experiment,
        ],
    )


def test_suite_result_to_dict_preserves_suite_metadata():
    result = create_suite_result()

    converted = suite_result_to_dict(result)

    assert converted["suite_id"] == str(result.suite_id)
    assert converted["name"] == "test-suite"
    assert converted["started_at_utc"] == result.started_at_utc.isoformat()
    assert converted["duration_seconds"] == 1.5
    assert len(converted["experiments"]) == 2


def test_suite_result_to_dict_preserves_success_and_failure_details():
    result = create_suite_result()

    converted = suite_result_to_dict(result)

    successful = converted["experiments"][0]
    failed = converted["experiments"][1]

    assert successful["status"] == "completed"
    assert successful["model"] == "example-model"
    assert successful["results"] == {
        "score": 0.95,
        "labels": ["quality", "reliability"],
        "details": {"cases": 10},
    }
    assert successful["warnings"] == ["Example warning"]
    assert successful["config"] == {"temperature": 0.2}

    assert failed["status"] == "failed"
    assert failed["model"] == "another-model"
    assert failed["results"] is None
    assert failed["error"] == "ValueError: Unknown experiment type"


def test_suite_result_to_dict_preserves_nested_benchmark_results():
    result = create_suite_result()
    benchmark_execution = create_benchmark_execution()

    result.experiments[0].experiment_type = "benchmark"
    result.experiments[0].results = benchmark_execution

    converted = suite_result_to_dict(result)
    benchmark = converted["experiments"][0]["results"]

    assert benchmark["configuration"]["model"] == "qwen2.5-coder:7b"
    assert benchmark["configuration"]["runtime"] == "ollama"
    assert benchmark["metadata"]["schema_version"] == "1.0"

    assert benchmark["resource_profile"]["pre_warmup"]["gpu_memory_used_mb"] == 1000.0
    assert benchmark["resource_profile"]["post_warmup"]["gpu_memory_used_mb"] == 4500.0

    assert (
        benchmark["resource_profile"]["pre_warmup"]["captured_at_utc"]
        == benchmark_execution.resource_profile.pre_warmup.captured_at_utc.isoformat()
    )


def test_to_json_compatible_converts_enum_and_tuple_values():
    converted = _to_json_compatible({
        "status": ExampleStatus.READY,
        "values": (1, 2, 3),
    })

    assert converted == {
        "status": "ready",
        "values": [1, 2, 3],
    }


def test_to_json_compatible_rejects_unsupported_types():
    with pytest.raises(
        TypeError,
        match="Unsupported type for JSON serialization: object",
    ):
        _to_json_compatible(object())


def test_serialize_suite_result_produces_valid_json_and_preserves_nulls():
    result = create_suite_result()

    serialized = serialize_suite_result(result)
    parsed = json.loads(serialized)

    assert parsed["name"] == "test-suite"
    assert parsed["experiments"][0]["status"] == "completed"
    assert parsed["experiments"][1]["results"] is None
    assert parsed["experiments"][1]["error"] is not None

    assert serialized.startswith("{")
    assert "\n" in serialized


def test_generate_suite_result_filename_sanitizes_suite_name():
    result = create_suite_result()
    result.name = " Local Code / Model Comparison! "

    filename = generate_suite_result_filename(result)

    assert filename == (
        f"2026-11-23_Local-Code-Model-Comparison_{result.suite_id}.json"
    )


def test_generate_suite_result_filename_uses_fallback_for_empty_name():
    result = create_suite_result()
    result.name = " ! "

    filename = generate_suite_result_filename(result)

    assert filename == f"2026-11-23_experiment-suite_{result.suite_id}.json"


def test_save_suite_result_creates_directory_and_file(tmp_path):
    result = create_suite_result()
    output_directory = tmp_path / "nested" / "results"

    file_path = save_suite_result(result, output_directory)

    assert file_path.exists()
    assert file_path.parent == output_directory
    assert file_path.name == generate_suite_result_filename(result)

    saved_data = json.loads(file_path.read_text(encoding="utf-8"))

    assert saved_data["suite_id"] == str(result.suite_id)
    assert saved_data["name"] == result.name
    assert len(saved_data["experiments"]) == 2
