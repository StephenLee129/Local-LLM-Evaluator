import pytest
import math

from evaluation.models import (
    DatasetMetadata,
    EvaluationDataset,
    EvaluationTask,
    TestCase as EvaluationTestCase,
)


class TestTestCase:
    def test_accepts_string_input(self):
        test_case = EvaluationTestCase(
            input=["Race car"],
            expected_output=True,
        )

        assert test_case.input == ["Race car"]
        assert test_case.expected_output is True

    def test_accepts_structured_input(self):
        test_case = EvaluationTestCase(
            input=[{"numbers": [1, 2, 3]}],
            expected_output={"result": True},
        )

        assert test_case.input == [{"numbers": [1, 2, 3]}]
        assert test_case.expected_output == {"result": True}

    def test_accepts_empty_input(self):
        test_case = EvaluationTestCase(
            input=[],
            expected_output=None,
        )

        assert test_case.input == []
        assert test_case.expected_output is None

    def test_rejects_non_list_input(self):
        with pytest.raises(TypeError, match="input must be a list"):
            EvaluationTestCase(
                input="Race car",
                expected_output=True,
            )

    def test_accepts_nested_json_values(self):
        test_case = EvaluationTestCase(
            input=[
                {"user": {"name": "Stephen", "active": True}},
                [1, 2, None, "hello"],
            ],
            expected_output={
                "result": [True, False],
                "metadata": {"count": 2},
            },
        )

        assert isinstance(test_case, EvaluationTestCase)

    @pytest.mark.parametrize(
        "invalid_value",
        [
            {1, 2, 3},
            object(),
        ],
    )
    def test_rejects_unsupported_json_types(self, invalid_value):
        with pytest.raises(TypeError):
            EvaluationTestCase(
                input=[invalid_value],
                expected_output=True,
            )

    def test_rejects_non_string_dictionary_keys(self):
        with pytest.raises(TypeError, match="dictionary keys must be strings"):
            EvaluationTestCase(
                input=[{1: "invalid key"}],
                expected_output=True,
            )

    @pytest.mark.parametrize(
        "invalid_number",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_rejects_non_finite_numbers(self, invalid_number):
        with pytest.raises(ValueError, match="NaN or infinity"):
            EvaluationTestCase(
                input=[invalid_number],
                expected_output=True,
            )

    def test_rejects_invalid_nested_expected_output(self):
        with pytest.raises(TypeError, match="unsupported JSON value type"):
            EvaluationTestCase(
                input=["hello"],
                expected_output={"result": {1, 2, 3}},
            )


class TestEvaluationTask:
    def test_accepts_valid_python_task(self):
        test_case = EvaluationTestCase(
            input=["Race car"],
            expected_output=True,
        )

        task = EvaluationTask(
            task_id="string-palindrome",
            description="Check whether a string is a palindrome.",
            instructions="Ignore capitalization and spaces.",
            language="python",
            entry_point="is_palindrome",
            test_cases=[test_case],
        )

        assert task.task_id == "string-palindrome"
        assert task.language == "python"
        assert task.entry_point == "is_palindrome"
        assert task.test_cases == [test_case]

    @pytest.mark.parametrize(
        "field_name",
        ["task_id", "description", "instructions", "entry_point"],
    )
    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_empty_required_string_fields(self, field_name, value):
        task_fields = {
            "task_id": "string-palindrome",
            "description": "Check whether a string is a palindrome.",
            "instructions": "Ignore capitalization and spaces.",
            "language": "python",
            "entry_point": "is_palindrome",
            "test_cases": [
                EvaluationTestCase(input=["Race car"], expected_output=True)
            ],
        }
        task_fields[field_name] = value

        with pytest.raises(ValueError, match=field_name):
            EvaluationTask(**task_fields)

    def test_rejects_unsupported_language(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            EvaluationTask(
                task_id="string-palindrome",
                description="Check whether a string is a palindrome.",
                instructions="Ignore capitalization and spaces.",
                language="javascript",
                entry_point="is_palindrome",
                test_cases=[
                    EvaluationTestCase(input=["Race car"], expected_output=True)
                ],
            )

    def test_rejects_empty_test_cases(self):
        with pytest.raises(ValueError, match="at least one test case"):
            EvaluationTask(
                task_id="string-palindrome",
                description="Check whether a string is a palindrome.",
                instructions="Ignore capitalization and spaces.",
                language="python",
                entry_point="is_palindrome",
                test_cases=[],
            )

    def test_rejects_non_test_case_items(self):
        with pytest.raises(TypeError, match="TestCase instances"):
            EvaluationTask(
                task_id="string-palindrome",
                description="Check whether a string is a palindrome.",
                instructions="Ignore capitalization and spaces.",
                language="python",
                entry_point="is_palindrome",
                test_cases=["not a test case"],
            )

class TestDatasetMetadata:

    def test_accepts_valid_metadata(self):
        metadata = DatasetMetadata(
            name="python-basics",
            description="Basic Python evaluation tasks.",
            version="1.0.0",
        )

        assert metadata.name == "python-basics"
        assert metadata.description == "Basic Python evaluation tasks."
        assert metadata.version == "1.0.0"

    @pytest.mark.parametrize(
        "version",
        [
            "1",
            "1.0",
            "1.0.0.0",
            "v1.0.0",
            "1.0.x",
            "01.0.0",
            "-1.0.0",
        ],
    )
    def test_rejects_invalid_versions(self, version):
        with pytest.raises(ValueError, match="semantic versioning"):
            DatasetMetadata(
                name="python-basics",
                description="Basic Python evaluation tasks.",
                version=version,
            )

    @pytest.mark.parametrize(
        "field_name",
        ["name", "description", "version"],
    )
    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_empty_required_fields(self, field_name, value):
        metadata = {
            "name": "python-basics",
            "description": "Basic Python evaluation tasks.",
            "version": "1.0.0",
        }
        metadata[field_name] = value

        with pytest.raises(ValueError, match=field_name):
            DatasetMetadata(**metadata)

class TestEvaluationDataset:

    @pytest.fixture
    def metadata(self):
        return DatasetMetadata(
            name="python-basics",
            description="Basic Python evaluation tasks.",
            version="1.0.0",
        )

    @pytest.fixture
    def task(self):
        return EvaluationTask(
            task_id="palindrome",
            description="Check whether a string is a palindrome.",
            instructions="Implement a function that checks for palindromes.",
            language="python",
            entry_point="is_palindrome",
            test_cases=[
                EvaluationTestCase(
                    input=["racecar"],
                    expected_output=True,
                ),
            ],
        )

    def test_accepts_valid_dataset(self, metadata, task):
        dataset = EvaluationDataset(
            metadata=metadata,
            tasks=[task],
        )

        assert dataset.metadata == metadata
        assert dataset.tasks == [task]

    def test_task_count_returns_number_of_tasks(self, metadata, task):
        second_task = EvaluationTask(
            task_id="reverse-string",
            description="Reverse a string.",
            instructions="Implement a function that reverses a string.",
            language="python",
            entry_point="reverse_string",
            test_cases=[
                EvaluationTestCase(
                    input=["hello"],
                    expected_output="olleh",
                ),
            ],
        )

        dataset = EvaluationDataset(
            metadata=metadata,
            tasks=[task, second_task],
        )

        assert dataset.task_count == 2

    def test_rejects_invalid_metadata(self, task):
        with pytest.raises(TypeError, match="DatasetMetadata"):
            EvaluationDataset(
                metadata="invalid",
                tasks=[task],
            )

    @pytest.mark.parametrize("tasks", [None, "invalid", {}])
    def test_rejects_non_list_tasks(self, metadata, tasks):
        with pytest.raises(TypeError, match="must be a list"):
            EvaluationDataset(
                metadata=metadata,
                tasks=tasks,
            )

    def test_rejects_empty_tasks(self, metadata):
        with pytest.raises(ValueError, match="at least one task"):
            EvaluationDataset(
                metadata=metadata,
                tasks=[],
            )

    def test_rejects_non_evaluation_task_items(self, metadata):
        with pytest.raises(TypeError, match="EvaluationTask instances"):
            EvaluationDataset(
                metadata=metadata,
                tasks=["invalid"],
            )

    def test_rejects_duplicate_task_ids(self, metadata, task):
        duplicate_task = EvaluationTask(
            task_id="palindrome",
            description="Another palindrome task.",
            instructions="Implement another palindrome function.",
            language="python",
            entry_point="is_palindrome",
            test_cases=[
                EvaluationTestCase(
                    input=["hello"],
                    expected_output=False,
                ),
            ],
        )

        with pytest.raises(ValueError, match="must be unique"):
            EvaluationDataset(
                metadata=metadata,
                tasks=[task, duplicate_task],
            )
