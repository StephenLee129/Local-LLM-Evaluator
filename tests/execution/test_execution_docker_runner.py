import io
import subprocess
from unittest.mock import patch

import pytest

from execution.base import ExecutionBackendError
from execution.docker_runner import DockerExecutionBackend
from execution.models import ExecutionRequest


class FakeInput:
    """Capture data written to a fake subprocess stdin stream."""

    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> int:
        self.data += data
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeOutput(io.BytesIO):
    """Track how much output has been consumed."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


class FakeProcess:
    """Minimal Popen substitute for execution backend tests."""

    def __init__(
        self,
        stdout: bytes = b"2\n",
        stderr: bytes = b"",
        returncode: int = 0,
        timeout_on_wait: bool = False,
    ):
        self.stdin = FakeInput()
        self.stdout = FakeOutput(stdout)
        self.stderr = FakeOutput(stderr)
        self.returncode = returncode
        self.timeout_on_wait = timeout_on_wait
        self.killed = False

    def wait(self, timeout=None):
        if self.timeout_on_wait and not self.killed:
            raise subprocess.TimeoutExpired(
                cmd="docker",
                timeout=timeout,
            )

        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_execute_returns_successful_result():
    backend = DockerExecutionBackend()
    process = FakeProcess()

    with patch(
        "execution.docker_runner.subprocess.Popen",
        return_value=process,
    ) as mock_popen:
        result = backend.execute(
            ExecutionRequest(code="print(1 + 1)")
        )

    assert result.stdout == "2\n"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.timed_out is False

    command = mock_popen.call_args.args[0]

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
    process = FakeProcess(
        stdout=b"partial output",
        timeout_on_wait=True,
    )

    with patch(
        "execution.docker_runner.subprocess.Popen",
        return_value=process,
    ):
        with patch.object(backend, "_remove_container") as mock_cleanup:
            result = backend.execute(
                ExecutionRequest(code="while True: pass")
            )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.stdout == "partial output"
    assert process.killed is True
    mock_cleanup.assert_called_once()


def test_execute_raises_backend_error_when_docker_unavailable():
    backend = DockerExecutionBackend()

    with patch(
        "execution.docker_runner.subprocess.Popen",
        side_effect=FileNotFoundError("docker not found"),
    ):
        with patch.object(backend, "_remove_container"):
            with pytest.raises(ExecutionBackendError):
                backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_normalizes_stdin_and_decodes_output():
    backend = DockerExecutionBackend()
    process = FakeProcess(
        stdout=b"'worker'\n",
    )

    with patch(
        "execution.docker_runner.subprocess.Popen",
        return_value=process,
    ):
        result = backend.execute(
            ExecutionRequest(
                code="value = input()\nprint(repr(value))",
                stdin="worker\r\n",
            )
        )

    assert process.stdin.data == b"worker\n"
    assert result.stdout == "'worker'\n"
    assert result.stderr == ""
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)


def test_execute_truncates_output_but_continues_reading():
    backend = DockerExecutionBackend()
    backend.MAX_OUTPUT_CHARS = 5

    large_output = b"x" * 100
    process = FakeProcess(stdout=large_output)

    with patch(
        "execution.docker_runner.subprocess.Popen",
        return_value=process,
    ):
        result = backend.execute(
            ExecutionRequest(code="print('x')")
        )

    assert result.stdout == "xxxxx"

    # The reader must consume all output, even after reaching its limit.
    assert process.stdout.bytes_read == len(large_output)


def test_execute_truncates_stdout_and_stderr_independently():
    backend = DockerExecutionBackend()
    backend.MAX_OUTPUT_CHARS = 5

    process = FakeProcess(
        stdout=b"abcdefgh",
        stderr=b"12345678",
    )

    with patch(
        "execution.docker_runner.subprocess.Popen",
        return_value=process,
    ):
        result = backend.execute(
            ExecutionRequest(code="print('test')")
        )

    assert result.stdout == "abcde"
    assert result.stderr == "12345"
