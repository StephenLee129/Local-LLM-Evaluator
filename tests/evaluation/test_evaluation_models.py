import pytest
import math

from evaluation.models import EvaluationTask, TestCase as EvaluationTestCase


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
