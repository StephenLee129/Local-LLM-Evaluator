import math
import re
from dataclasses import dataclass


def _validate_json_value(value: object, path: str = "value") -> None:
    """Validate that a value can be represented as standard JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} dictionary keys must be strings.")

            _validate_json_value(item, f"{path}.{key}")
        return

    raise TypeError(
        f"{path} contains unsupported JSON value type: "
        f"{type(value).__name__}."
    )


type JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list[JSONValue]
    | dict[str, JSONValue]
)


@dataclass
class TestCase:
    """A single test case containing function arguments and expected output."""

    __test__ = False

    input: list[JSONValue]
    expected_output: JSONValue

    def __post_init__(self) -> None:
        if not isinstance(self.input, list):
            raise TypeError("Test case input must be a list.")

        _validate_json_value(self.input, "input")
        _validate_json_value(self.expected_output, "expected_output")


@dataclass
class EvaluationTask:
    """A code-generation task and its associated test cases."""

    task_id: str
    description: str
    instructions: str
    language: str
    entry_point: str
    test_cases: list[TestCase]

    def __post_init__(self) -> None:
        required_fields = {
            "task_id": self.task_id,
            "description": self.description,
            "instructions": self.instructions,
            "entry_point": self.entry_point,
        }

        for field_name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

        if self.language != "python":
            raise ValueError(
                f"Unsupported language '{self.language}'. "
                "Currently, only Python is supported."
            )

        if not isinstance(self.test_cases, list) or not self.test_cases:
            raise ValueError("An evaluation task must have at least one test case.")

        if not all(isinstance(test_case, TestCase) for test_case in self.test_cases):
            raise TypeError("All test cases must be TestCase instances.")

@dataclass
class DatasetMetadata:
    """Metadata describing an evaluation dataset."""

    name: str
    description: str
    version: str

    def __post_init__(self) -> None:
        required_fields = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }

        for field_name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if not re.fullmatch(
            r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)",
            self.version,
        ):
            raise ValueError(
                "Dataset version must follow semantic versioning (MAJOR.MINOR.PATCH)."
            )

@dataclass
class EvaluationDataset:
    """A collection of evaluation tasks and its metadata."""

    metadata: DatasetMetadata
    tasks: list[EvaluationTask]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, DatasetMetadata):
            raise TypeError("Dataset metadata must be DatasetMetadata.")

        if not isinstance(self.tasks, list):
            raise TypeError("Dataset tasks must be a list.")

        if not self.tasks:
            raise ValueError("An evaluation dataset must contain at least one task.")

        if not all(isinstance(task, EvaluationTask) for task in self.tasks):
            raise TypeError("All dataset tasks must be EvaluationTask instances.")

        task_ids = [task.task_id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Dataset task IDs must be unique.")

    @property
    def task_count(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.tasks)
