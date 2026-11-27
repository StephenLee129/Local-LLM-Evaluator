import json
from pathlib import Path

import yaml

from evaluation.models import (
    DatasetMetadata,
    EvaluationDataset,
    EvaluationTask,
    TestCase,
)


def load_dataset(dataset_path: Path) -> EvaluationDataset:
    """Load an evaluation dataset from a directory."""

    dataset_path = Path(dataset_path)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    metadata_path = dataset_path / "metadata.yaml"
    tasks_path = dataset_path / "tasks.jsonl"

    metadata = _load_metadata(metadata_path)
    tasks = _load_tasks(tasks_path)

    try:
        return EvaluationDataset(
            metadata=metadata,
            tasks=tasks,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid evaluation dataset in {dataset_path}: {error}"
        ) from error


def _load_metadata(metadata_path: Path) -> DatasetMetadata:
    """Load and validate dataset metadata from YAML."""

    with metadata_path.open("r", encoding="utf-8") as file:
        try:
            metadata_data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML in {metadata_path}: {error}"
            ) from error

    if not isinstance(metadata_data, dict):
        raise ValueError(
            f"Dataset metadata must be a YAML mapping: {metadata_path}"
        )

    try:
        return DatasetMetadata(**metadata_data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid dataset metadata in {metadata_path}: {error}"
        ) from error


def _load_tasks(tasks_path: Path) -> list[EvaluationTask]:
    """Load evaluation tasks from a JSONL file."""

    tasks = []

    with tasks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                task_data = json.loads(line)

                if not isinstance(task_data, dict):
                    raise ValueError("Task must be a JSON object.")

                test_cases_data = task_data.get("test_cases")

                if not isinstance(test_cases_data, list):
                    raise ValueError("Task test_cases must be a JSON array.")

                test_cases = [
                    TestCase(**test_case_data)
                    for test_case_data in test_cases_data
                    if isinstance(test_case_data, dict)
                ]

                if len(test_cases) != len(test_cases_data):
                    raise ValueError(
                        "Every test case must be a JSON object."
                    )

                task_data["test_cases"] = test_cases

                tasks.append(EvaluationTask(**task_data))

            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid task in {tasks_path}, "
                    f"line {line_number}: {error}"
                ) from error

    return tasks
