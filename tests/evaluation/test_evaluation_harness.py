import json
import subprocess
import sys

import pytest

from evaluation.harness import build_test_harness
from evaluation.models import EvaluationTask, TestCase


@pytest.fixture
def addition_task():
    return EvaluationTask(
        task_id="addition",
        description="Add two numbers.",
        instructions="Implement a function that adds two numbers.",
        language="python",
        entry_point="add",
        test_cases=[
            TestCase(input=[1, 2], expected_output=3),
            TestCase(input=[5, 7], expected_output=12),
            TestCase(input=[-1, 1], expected_output=0),
        ],
    )


def run_harness(task, generated_code):
    """Execute a generated harness locally for unit testing."""

    harness = build_test_harness(task, generated_code)

    process = subprocess.run(
        [sys.executable, "-I", "-B", "-c", harness],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert process.returncode == 0, process.stderr

    return json.loads(process.stdout)


def test_harness_evaluates_multiple_test_cases(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    return a + b
""",
    )

    assert len(results) == 3
    assert all(result["passed"] for result in results)
    assert [result["actual_output"] for result in results] == [
        3,
        12,
        0,
    ]


def test_harness_records_failed_test_cases(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    return a - b
""",
    )

    assert len(results) == 3
    assert all(not result["passed"] for result in results)


def test_harness_continues_after_test_case_exception(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    if a == 5:
        raise ValueError("Intentional test failure")

    return a + b
""",
    )

    assert len(results) == 3
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False
    assert "Intentional test failure" in results[1]["error"]
    assert results[2]["passed"] is True


def test_harness_handles_generated_code_syntax_error(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b)
    return a + b
""",
    )

    assert len(results) == 3
    assert all(not result["passed"] for result in results)
    assert all(result["error"] for result in results)


def test_harness_handles_missing_entry_point(addition_task):
    results = run_harness(
        addition_task,
        """
def subtract(a, b):
    return a - b
""",
    )

    assert len(results) == 3
    assert all(not result["passed"] for result in results)
    assert all("add" in result["error"] for result in results)


def test_harness_keeps_stdout_json_parseable(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    print("Debug output")
    return a + b
""",
    )

    assert len(results) == 3
    assert all(result["passed"] for result in results)


def test_harness_handles_non_json_return_values(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    return {1, 2, 3}
""",
    )

    assert len(results) == 3
    assert all(not result["passed"] for result in results)
    assert all(result["error"] for result in results)


def test_harness_includes_expected_and_actual_outputs(addition_task):
    results = run_harness(
        addition_task,
        """
def add(a, b):
    return a + b
""",
    )

    assert results[0]["input"] == [1, 2]
    assert results[0]["expected_output"] == 3
    assert results[0]["actual_output"] == 3
    assert results[0]["error"] is None
