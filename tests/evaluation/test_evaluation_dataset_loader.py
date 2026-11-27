import json
import pytest
import yaml

from evaluation.dataset_loader import load_dataset
from evaluation.models import EvaluationDataset
from pathlib import Path


def test_loads_python_basics_dataset():
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = (
        project_root / "evaluation" / "datasets" / "python-basics"
    )

    dataset = load_dataset(dataset_path)

    assert dataset.metadata.name == "python-basics"
    assert dataset.metadata.version == "1.0.0"
    assert dataset.task_count == 10

    assert sum(
        len(task.test_cases) for task in dataset.tasks
    ) == 60

    assert {task.task_id for task in dataset.tasks} == {
        "is_palindrome",
        "reverse_string",
        "count_vowels",
        "find_max",
        "remove_duplicates",
        "word_frequency",
        "two_sum",
        "fibonacci",
        "flatten_list",
        "is_prime",
    }

@pytest.fixture
def dataset_directory(tmp_path):
    """Create a temporary valid evaluation dataset."""

    metadata = {
        "name": "python-basics",
        "description": "Basic Python evaluation tasks.",
        "version": "1.0.0",
    }

    task = {
        "task_id": "palindrome",
        "description": "Check whether a string is a palindrome.",
        "instructions": "Implement a function that checks for palindromes.",
        "language": "python",
        "entry_point": "is_palindrome",
        "test_cases": [
            {
                "input": ["racecar"],
                "expected_output": True,
            },
            {
                "input": ["hello"],
                "expected_output": False,
            },
        ],
    }

    metadata_path = tmp_path / "metadata.yaml"
    tasks_path = tmp_path / "tasks.jsonl"

    metadata_path.write_text(
        yaml.safe_dump(metadata),
        encoding="utf-8",
    )

    tasks_path.write_text(
        json.dumps(task) + "\n",
        encoding="utf-8",
    )

    return tmp_path


class TestLoadDataset:

    def test_loads_valid_dataset(self, dataset_directory):
        dataset = load_dataset(dataset_directory)

        assert isinstance(dataset, EvaluationDataset)
        assert dataset.metadata.name == "python-basics"
        assert dataset.metadata.version == "1.0.0"
        assert dataset.task_count == 1

        task = dataset.tasks[0]
        assert task.task_id == "palindrome"
        assert task.entry_point == "is_palindrome"
        assert len(task.test_cases) == 2

    def test_ignores_blank_lines(self, dataset_directory):
        tasks_path = dataset_directory / "tasks.jsonl"
        task_line = tasks_path.read_text(encoding="utf-8")

        tasks_path.write_text(
            "\n" + task_line + "\n\n",
            encoding="utf-8",
        )

        dataset = load_dataset(dataset_directory)

        assert dataset.task_count == 1

    def test_rejects_missing_dataset_directory(self, tmp_path):
        missing_path = tmp_path / "missing"

        with pytest.raises(FileNotFoundError, match="Dataset directory"):
            load_dataset(missing_path)

    def test_rejects_missing_metadata_file(self, dataset_directory):
        (dataset_directory / "metadata.yaml").unlink()

        with pytest.raises(FileNotFoundError):
            load_dataset(dataset_directory)

    def test_rejects_missing_tasks_file(self, dataset_directory):
        (dataset_directory / "tasks.jsonl").unlink()

        with pytest.raises(FileNotFoundError):
            load_dataset(dataset_directory)

    def test_rejects_invalid_yaml(self, dataset_directory):
        metadata_path = dataset_directory / "metadata.yaml"
        metadata_path.write_text(
            "name: [invalid yaml",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_dataset(dataset_directory)

    def test_rejects_non_mapping_metadata(self, dataset_directory):
        metadata_path = dataset_directory / "metadata.yaml"
        metadata_path.write_text(
            "- item1\n- item2",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="YAML mapping"):
            load_dataset(dataset_directory)

    def test_rejects_invalid_json(self, dataset_directory):
        tasks_path = dataset_directory / "tasks.jsonl"
        tasks_path.write_text(
            '{"task_id": invalid}\n',
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="line 1"):
            load_dataset(dataset_directory)

    def test_rejects_invalid_test_case(self, dataset_directory):
        tasks_path = dataset_directory / "tasks.jsonl"

        task = {
            "task_id": "invalid-task",
            "description": "Invalid test case.",
            "instructions": "Implement a function.",
            "language": "python",
            "entry_point": "example",
            "test_cases": ["not a test case"],
        }

        tasks_path.write_text(
            json.dumps(task) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="line 1"):
            load_dataset(dataset_directory)

    def test_rejects_duplicate_task_ids(self, dataset_directory):
        tasks_path = dataset_directory / "tasks.jsonl"
        task = json.loads(tasks_path.read_text(encoding="utf-8"))

        tasks_path.write_text(
            json.dumps(task) + "\n" + json.dumps(task) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="must be unique"):
            load_dataset(dataset_directory)
