from dataclasses import dataclass

from evaluation.models import JSONValue
from execution.models import ExecutionResult


@dataclass
class TestCaseResult:
    """The outcome of evaluating one test case."""

    __test__ = False

    input: list[JSONValue]
    expected_output: JSONValue
    actual_output: JSONValue | None
    passed: bool
    error: str | None = None


@dataclass
class TaskEvaluationResult:
    """The evaluation outcome for one generated task."""

    task_id: str
    generated_code: str
    execution_result: ExecutionResult | None
    test_case_results: list[TestCaseResult]

    @property
    def passed_test_cases(self) -> int:
        """Return the number of passing test cases."""
        return sum(
            result.passed for result in self.test_case_results
        )

    @property
    def total_test_cases(self) -> int:
        """Return the number of evaluated test cases."""
        return len(self.test_case_results)

    @property
    def success(self) -> bool:
        """Return whether execution succeeded and every test case passed."""
        return (
            self.execution_result is not None
            and self.execution_result.exit_code == 0
            and not self.execution_result.timed_out
            and self.total_test_cases > 0
            and self.passed_test_cases == self.total_test_cases
        )


@dataclass
class EvaluationResult:
    """Aggregated results for an entire evaluation dataset."""

    dataset_name: str
    dataset_version: str
    task_results: list[TaskEvaluationResult]
    total_duration_seconds: float

    @property
    def total_tasks(self) -> int:
        """Return the number of evaluated tasks."""
        return len(self.task_results)

    @property
    def passed_tasks(self) -> int:
        """Return the number of tasks that passed all test cases."""
        return sum(
            result.success for result in self.task_results
        )

    @property
    def total_test_cases(self) -> int:
        """Return the total number of evaluated test cases."""
        return sum(
            result.total_test_cases for result in self.task_results
        )

    @property
    def passed_test_cases(self) -> int:
        """Return the total number of passing test cases."""
        return sum(
            result.passed_test_cases for result in self.task_results
        )

    @property
    def task_accuracy(self) -> float:
        """Return the proportion of tasks that passed."""
        if self.total_tasks == 0:
            return 0.0

        return self.passed_tasks / self.total_tasks

    @property
    def test_case_accuracy(self) -> float:
        """Return the proportion of test cases that passed."""
        if self.total_test_cases == 0:
            return 0.0

        return self.passed_test_cases / self.total_test_cases
