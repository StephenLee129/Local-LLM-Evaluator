import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from benchmark.models import (
    BenchmarkAnalysis,
    BenchmarkConfiguration,
    BenchmarkError,
    BenchmarkExecution,
    BenchmarkMetadata,
    BenchmarkRunResult,
    BenchmarkWarning,
    EnvironmentInfo,
    GPUInfo,
    MetricStatistics,
    ReliabilityStatistics,
    ResourceProfile,
    ResourceSample,
    WarmupSummary,
)
from benchmark.persistence import (
    execution_to_dict,
    generate_result_filename,
    save_execution,
    serialize_execution,
)
from runtimes.base import GenerationResult


def create_execution() -> BenchmarkExecution:
    """Create predictable benchmark data for persistence tests."""

    return BenchmarkExecution(
        metadata=BenchmarkMetadata(
            execution_id=uuid4(),
            timestamp_utc=datetime(2026, 11, 23, 22, 0, tzinfo=timezone.utc),
            schema_version="1.0",
            execution_duration_seconds=3.42,
        ),
        configuration=BenchmarkConfiguration(
            model="qwen2.5-coder:7b",
            runtime="ollama",
            temperature=0.0,
            max_tokens=150,
            task_id="code-generation-basic",
            prompt="Write a Python function that adds two numbers.",
            warmup_runs=2,
            measured_iterations=2,
        ),
        environment=EnvironmentInfo(
            operating_system="Windows 11",
            python_version="3.12.4",
            cpu="Test CPU",
            ram_gb=32.0,
            gpu=GPUInfo(
                model="RTX 3080",
                total_vram_gb=10.0,
                driver_version="Test driver",
            ),
        ),
        warmup_summary=WarmupSummary(
            runs_requested=2,
            runs_completed=2,
            runs_failed=0,
        ),
        measured_runs=[
            BenchmarkRunResult(
                iteration=1,
                status="success",
                duration_seconds=0.429,
                generation=GenerationResult(
                    output="def add(a, b): return a + b",
                    total_duration_seconds=0.425,
                    generated_tokens=10,
                    generation_duration_seconds=0.1,
                ),
            ),
            BenchmarkRunResult(
                iteration=2,
                status="failed",
                duration_seconds=5.0,
                error=BenchmarkError(
                    category="timeout",
                    type="Timeout",
                    message="Inference timed out",
                ),
            ),
        ],
        analysis=BenchmarkAnalysis(
            metrics={
                "client_duration_seconds": MetricStatistics(
                    count=1,
                    mean=0.429,
                    median=0.429,
                    minimum=0.429,
                    maximum=0.429,
                    standard_deviation=None,
                ),
            },
            reliability=ReliabilityStatistics(
                total_iterations=2,
                successful_iterations=1,
                failed_iterations=1,
                success_rate=0.5,
                failure_rate=0.5,
                failed_duration=MetricStatistics(
                    count=1,
                    mean=5.0,
                    median=5.0,
                    minimum=5.0,
                    maximum=5.0,
                    standard_deviation=None,
                ),
                errors_by_category={"timeout": 1},
            ),
        ),
        warnings=[
            BenchmarkWarning(
                component="environment",
                message="Test warning",
            )
        ],
        resource_profile=ResourceProfile(
            pre_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=1000.0,
                gpu_utilization_percent=5.0,
                system_ram_used_mb=8000.0,
                system_ram_available_mb=24000.0,
                captured_at_utc=datetime(
                    2026, 11, 23, 21, 59, 50, tzinfo=timezone.utc
                ),
            ),
            post_warmup=ResourceSample(
                timestamp_seconds=0.0,
                gpu_memory_used_mb=7000.0,
                gpu_utilization_percent=95.0,
                system_ram_used_mb=8500.0,
                system_ram_available_mb=23500.0,
                captured_at_utc=datetime(
                    2026, 11, 23, 21, 59, 55, tzinfo=timezone.utc
                ),
            ),
        ),
    )


def test_execution_to_dict_preserves_top_level_structure():
    execution = create_execution()

    result = execution_to_dict(execution)

    assert set(result.keys()) == {
        "metadata",
        "configuration",
        "environment",
        "results",
        "analysis",
        "warnings",
    }

    assert result["results"]["warmup_summary"]["runs_completed"] == 2
    assert len(result["results"]["measured_runs"]) == 2
    assert result["results"]["resource_profile"] is not None


def test_execution_to_dict_preserves_success_and_failure_details():
    execution = create_execution()

    result = execution_to_dict(execution)

    successful_run = result["results"]["measured_runs"][0]
    failed_run = result["results"]["measured_runs"][1]

    assert successful_run["status"] == "success"
    assert successful_run["generation"]["generated_tokens"] == 10
    assert successful_run["error"] is None

    assert failed_run["status"] == "failed"
    assert failed_run["generation"] is None
    assert failed_run["error"]["category"] == "timeout"


def test_execution_to_dict_preserves_resource_profile():
    execution = create_execution()

    result = execution_to_dict(execution)

    resource_profile = result["results"]["resource_profile"]

    assert resource_profile["pre_warmup"]["gpu_memory_used_mb"] == 1000.0
    assert resource_profile["post_warmup"]["gpu_memory_used_mb"] == 7000.0


def test_execution_to_dict_handles_missing_resource_profile():
    execution = create_execution()
    execution.resource_profile = None

    result = execution_to_dict(execution)

    assert result["results"]["resource_profile"] is None


def test_serialize_execution_produces_valid_json():
    execution = create_execution()

    serialized = serialize_execution(execution)
    result = json.loads(serialized)

    assert result["metadata"]["execution_id"] == str(
        execution.metadata.execution_id
    )
    assert result["metadata"]["timestamp_utc"] == (
        "2026-11-23T22:00:00+00:00"
    )


def test_serialize_execution_preserves_null_values():
    execution = create_execution()

    serialized = serialize_execution(execution)
    result = json.loads(serialized)

    assert result["results"]["measured_runs"][0]["generation"][
        "time_to_first_token_seconds"
    ] is None


def test_serialize_execution_preserves_resource_profile_timestamps():
    execution = create_execution()

    serialized = serialize_execution(execution)
    result = json.loads(serialized)

    resource_profile = result["results"]["resource_profile"]

    assert resource_profile["pre_warmup"]["captured_at_utc"] == (
        "2026-11-23T21:59:50+00:00"
    )

    assert resource_profile["post_warmup"]["captured_at_utc"] == (
        "2026-11-23T21:59:55+00:00"
    )


def test_generate_result_filename():
    execution = create_execution()

    filename = generate_result_filename(execution)

    assert filename == (
        f"2026-11-23_qwen2.5-coder-7b_"
        f"code-generation-basic_"
        f"{str(execution.metadata.execution_id)[:8]}.json"
    )


def test_save_execution_creates_directory_and_file(tmp_path: Path):
    execution = create_execution()

    output_directory = tmp_path / "results" / "local"

    output_path = save_execution(
        execution,
        output_directory=output_directory,
    )

    assert output_path.exists()
    assert output_path.parent == output_directory

    saved_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_data["configuration"]["model"] == "qwen2.5-coder:7b"
    assert saved_data["analysis"]["reliability"]["failure_rate"] == 0.5
