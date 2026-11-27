import pytest

from evaluation.results import (
    EvaluationResult,
    TaskEvaluationResult,
    TestCaseResult,
)
from execution.models import ExecutionResult


@pytest.fixture
def successful_execution_result():
    return ExecutionResult(
        stdout='[{"actual_output": true}]',
        stderr="",
        exit_code=0,
        duration_seconds=0.5,
        timed_out=False,
    )


@pytest.fixture
def successful_test_case_result():
    return TestCaseResult(
        input=[1, 2],
        expected_output=3,
        actual_output=3,
        passed=True,
    )


@pytest.fixture
def failed_test_case_result():
    return TestCaseResult(
        input=[1, 2],
        expected_output=3,
        actual_output=4,
        passed=False,
    )


@pytest.fixture
def successful_task_result(
    successful_execution_result,
    successful_test_case_result,
):
    return TaskEvaluationResult(
        task_id="addition",
        generated_code="def add(a, b): return a + b",
        execution_result=successful_execution_result,
        test_case_results=[
            successful_test_case_result,
        ],
    )


@pytest.fixture
def partially_successful_task_result(
    successful_execution_result,
    successful_test_case_result,
    failed_test_case_result,
):
    return TaskEvaluationResult(
        task_id="addition",
        generated_code="def add(a, b): return a + b",
        execution_result=successful_execution_result,
        test_case_results=[
            successful_test_case_result,
            failed_test_case_result,
        ],
    )


# TestCaseResult tests


def test_test_case_result_stores_values():
    result = TestCaseResult(
        input=[2, 3],
        expected_output=5,
        actual_output=5,
        passed=True,
    )

    assert result.input == [2, 3]
    assert result.expected_output == 5
    assert result.actual_output == 5
    assert result.passed is True
    assert result.error is None


def test_test_case_result_stores_error():
    result = TestCaseResult(
        input=[2, 3],
        expected_output=5,
        actual_output=None,
        passed=False,
        error="Execution failed.",
    )

    assert result.actual_output is None
    assert result.passed is False
    assert result.error == "Execution failed."


def test_test_case_result_supports_none_as_actual_output():
    result = TestCaseResult(
        input=[],
        expected_output=None,
        actual_output=None,
        passed=True,
    )

    assert result.actual_output is None
    assert result.expected_output is None
    assert result.passed is True


# TaskEvaluationResult tests


def test_task_evaluation_result_stores_values(
    successful_task_result,
    successful_execution_result,
):
    assert successful_task_result.task_id == "addition"
    assert successful_task_result.generated_code == (
        "def add(a, b): return a + b"
    )
    assert successful_task_result.execution_result is (
        successful_execution_result
    )
    assert successful_task_result.total_test_cases == 1
    assert successful_task_result.passed_test_cases == 1
    assert successful_task_result.success is True


def test_task_evaluation_result_counts_partial_success(
    partially_successful_task_result,
):
    assert partially_successful_task_result.total_test_cases == 2
    assert partially_successful_task_result.passed_test_cases == 1
    assert partially_successful_task_result.success is False


def test_task_evaluation_result_fails_when_no_test_cases():
    result = TaskEvaluationResult(
        task_id="empty",
        generated_code="",
        execution_result=None,
        test_case_results=[],
    )

    assert result.total_test_cases == 0
    assert result.passed_test_cases == 0
    assert result.success is False


def test_task_evaluation_result_fails_when_execution_is_missing(
    successful_test_case_result,
):
    result = TaskEvaluationResult(
        task_id="addition",
        generated_code="def add(a, b): return a + b",
        execution_result=None,
        test_case_results=[successful_test_case_result],
    )

    assert result.success is False


def test_task_evaluation_result_fails_when_execution_exits_with_error(
    successful_test_case_result,
):
    execution_result = ExecutionResult(
        stdout="",
        stderr="Runtime error",
        exit_code=1,
        duration_seconds=0.5,
    )

    result = TaskEvaluationResult(
        task_id="addition",
        generated_code="def add(a, b): return a + b",
        execution_result=execution_result,
        test_case_results=[successful_test_case_result],
    )

    assert result.success is False


def test_task_evaluation_result_fails_when_execution_times_out(
    successful_test_case_result,
):
    execution_result = ExecutionResult(
        stdout="",
        stderr="",
        exit_code=-1,
        duration_seconds=10.0,
        timed_out=True,
    )

    result = TaskEvaluationResult(
        task_id="addition",
        generated_code="def add(a, b): return a + b",
        execution_result=execution_result,
        test_case_results=[successful_test_case_result],
    )

    assert result.success is False


# EvaluationResult tests


def test_evaluation_result_stores_dataset_metadata(
    successful_task_result,
):
    result = EvaluationResult(
        dataset_name="Python Basics",
        dataset_version="1.0.0",
        task_results=[successful_task_result],
        total_duration_seconds=1.5,
    )

    assert result.dataset_name == "Python Basics"
    assert result.dataset_version == "1.0.0"
    assert result.total_duration_seconds == 1.5


def test_evaluation_result_calculates_task_metrics(
    successful_task_result,
    partially_successful_task_result,
):
    result = EvaluationResult(
        dataset_name="Python Basics",
        dataset_version="1.0.0",
        task_results=[
            successful_task_result,
            partially_successful_task_result,
        ],
        total_duration_seconds=2.0,
    )

    assert result.total_tasks == 2
    assert result.passed_tasks == 1
    assert result.task_accuracy == 0.5


def test_evaluation_result_calculates_test_case_metrics(
    successful_task_result,
    partially_successful_task_result,
):
    result = EvaluationResult(
        dataset_name="Python Basics",
        dataset_version="1.0.0",
        task_results=[
            successful_task_result,
            partially_successful_task_result,
        ],
        total_duration_seconds=2.0,
    )

    assert result.total_test_cases == 3
    assert result.passed_test_cases == 2
    assert result.test_case_accuracy == pytest.approx(2 / 3)


def test_evaluation_result_handles_zero_tasks():
    result = EvaluationResult(
        dataset_name="Empty Dataset",
        dataset_version="1.0.0",
        task_results=[],
        total_duration_seconds=0.0,
    )

    assert result.total_tasks == 0
    assert result.passed_tasks == 0
    assert result.task_accuracy == 0.0
    assert result.total_test_cases == 0
    assert result.passed_test_cases == 0
    assert result.test_case_accuracy == 0.0


def test_evaluation_result_handles_tasks_without_test_cases():
    task_result = TaskEvaluationResult(
        task_id="empty",
        generated_code="",
        execution_result=None,
        test_case_results=[],
    )

    result = EvaluationResult(
        dataset_name="Python Basics",
        dataset_version="1.0.0",
        task_results=[task_result],
        total_duration_seconds=0.0,
    )

    assert result.total_tasks == 1
    assert result.passed_tasks == 0
    assert result.task_accuracy == 0.0
    assert result.total_test_cases == 0
    assert result.test_case_accuracy == 0.0
