import json

from evaluation.models import EvaluationTask


def build_test_harness(
    task: EvaluationTask,
    generated_code: str,
) -> str:
    """Build a standalone Python test harness for a generated implementation."""

    if not isinstance(generated_code, str) or not generated_code.strip():
        raise ValueError("Generated code cannot be empty.")

    payload = {
        "entry_point": task.entry_point,
        "test_cases": [
            {
                "input": test_case.input,
                "expected_output": test_case.expected_output,
            }
            for test_case in task.test_cases
        ],
    }

    serialized_payload = json.dumps(payload, allow_nan=False)
    serialized_code = json.dumps(generated_code)

    return f"""
import contextlib
import io
import json
import math
import traceback


GENERATED_CODE = json.loads({json.dumps(serialized_code)})
TEST_DATA = json.loads({json.dumps(serialized_payload)})


def is_json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return True

    if isinstance(value, float):
        return math.isfinite(value)

    if isinstance(value, list):
        return all(is_json_value(item) for item in value)

    if isinstance(value, dict):
        return (
            all(isinstance(key, str) for key in value)
            and all(is_json_value(item) for item in value.values())
        )

    return False


def run_tests():
    results = []

    try:
        namespace = {{}}

        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(
                    compile(GENERATED_CODE, "<generated>", "exec"),
                    namespace,
                )

        entry_point = TEST_DATA["entry_point"]
        function = namespace.get(entry_point)

        if not callable(function):
            raise ValueError(
                f"Entry point '{{entry_point}}' is missing or is not callable."
            )

    except Exception as exc:
        error = f"{{type(exc).__name__}}: {{exc}}"

        return [
            {{
                "input": test_case["input"],
                "expected_output": test_case["expected_output"],
                "actual_output": None,
                "passed": False,
                "error": error,
            }}
            for test_case in TEST_DATA["test_cases"]
        ]

    for test_case in TEST_DATA["test_cases"]:
        actual_output = None
        error = None
        passed = False

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    actual_output = function(*test_case["input"])

            if not is_json_value(actual_output):
                raise TypeError(
                    "Function returned a value that cannot be represented as JSON."
                )

            if not is_json_value(test_case["expected_output"]):
                raise TypeError(
                    "Expected output cannot be represented as JSON."
                )

            passed = actual_output == test_case["expected_output"]

        except Exception as exc:
            error = f"{{type(exc).__name__}}: {{exc}}"

        results.append(
            {{
                "input": test_case["input"],
                "expected_output": test_case["expected_output"],
                "actual_output": actual_output if error is None else None,
                "passed": passed,
                "error": error,
            }}
        )

    return results


print(json.dumps(run_tests(), allow_nan=False))
"""
