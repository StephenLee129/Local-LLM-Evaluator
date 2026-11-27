import subprocess
from unittest.mock import patch

import pytest

from execution.base import ExecutionBackendError
from execution.docker_runner import DockerExecutionBackend
from execution.models import ExecutionRequest


def test_execute_returns_successful_result():
    backend = DockerExecutionBackend()

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="2\n",
        stderr="",
    )

    with patch("execution.docker_runner.subprocess.run", return_value=completed) as mock_run:
        result = backend.execute(
            ExecutionRequest(code="print(1 + 1)")
        )

    assert result.stdout == "2\n"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.timed_out is False

    command = mock_run.call_args.args[0]

    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--pull=never" in command


def test_execute_rejects_empty_code():
    backend = DockerExecutionBackend()

    with pytest.raises(ValueError, match="cannot be empty"):
        backend.execute(ExecutionRequest(code="   "))


def test_execute_rejects_oversized_code():
    backend = DockerExecutionBackend()
    oversized_code = "x" * (backend.MAX_CODE_BYTES + 1)

    with pytest.raises(ValueError, match="exceeds"):
        backend.execute(ExecutionRequest(code=oversized_code))


def test_execute_returns_timeout_result():
    backend = DockerExecutionBackend()

    timeout_error = subprocess.TimeoutExpired(
        cmd="docker",
        timeout=10,
        output="partial output",
        stderr="",
    )

    with patch(
        "execution.docker_runner.subprocess.run",
        side_effect=timeout_error,
    ):
        with patch.object(backend, "_remove_container") as mock_cleanup:
            result = backend.execute(
                ExecutionRequest(code="while True: pass")
            )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.stdout == "partial output"
    mock_cleanup.assert_called_once()


def test_execute_raises_backend_error_when_docker_unavailable():
    backend = DockerExecutionBackend()

    with patch(
        "execution.docker_runner.subprocess.run",
        side_effect=FileNotFoundError("docker not found"),
    ):
        with patch.object(backend, "_remove_container"):
            with pytest.raises(ExecutionBackendError):
                backend.execute(ExecutionRequest(code="print(1)"))

def test_execute_normalizes_stdin_and_decodes_output():
    backend = DockerExecutionBackend()

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"'worker'\n",
        stderr=b"",
    )

    with patch(
        "execution.docker_runner.subprocess.run",
        return_value=completed,
    ) as mock_run:
        result = backend.execute(
            ExecutionRequest(
                code="value = input()\nprint(repr(value))",
                stdin="worker\r\n",
            )
        )

    # Verify Windows line endings are normalized before execution.
    assert mock_run.call_args.kwargs["input"] == b"worker\n"

    # Verify subprocess byte output is decoded into strings.
    assert result.stdout == "'worker'\n"
    assert result.stderr == ""
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)
