from unittest.mock import Mock, patch

import pytest
import requests

from execution.base import ExecutionBackend, ExecutionBackendError
from execution.models import ExecutionRequest, ExecutionResult
from execution.remote_runner import RemoteExecutionBackend


WORKER_TOKEN = "test-worker-token"

SUCCESS_RESPONSE = {
    "stdout": "42\n",
    "stderr": "",
    "exit_code": 0,
    "duration_seconds": 0.125,
    "timed_out": False,
}


def create_backend(**kwargs) -> RemoteExecutionBackend:
    """Create a backend with test configuration."""

    return RemoteExecutionBackend(
        base_url="http://192.168.1.94:8765",
        token=WORKER_TOKEN,
        **kwargs,
    )


def create_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> Mock:
    """Create a mock HTTP response."""

    response = Mock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = (
        SUCCESS_RESPONSE if json_data is None else json_data
    )

    return response


def test_backend_implements_execution_backend():
    backend = create_backend()

    assert isinstance(backend, ExecutionBackend)
    assert backend.backend_name == "remote"


def test_execute_sends_request_and_returns_result():
    backend = create_backend()
    response = create_response()

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ) as mock_post:
        result = backend.execute(
            ExecutionRequest(
                code="print(42)",
                stdin="input data",
            )
        )

    assert result == ExecutionResult(
        stdout="42\n",
        stderr="",
        exit_code=0,
        duration_seconds=0.125,
        timed_out=False,
    )

    mock_post.assert_called_once_with(
        "http://192.168.1.94:8765/api/v1/execute",
        json={
            "code": "print(42)",
            "stdin": "input data",
        },
        headers={
            "X-Worker-Token": WORKER_TOKEN,
        },
        timeout=15,
    )


def test_execute_preserves_failed_program_exit_code():
    backend = create_backend()
    response = create_response(
        json_data={
            "stdout": "",
            "stderr": "Something went wrong",
            "exit_code": 7,
            "duration_seconds": 0.25,
            "timed_out": False,
        }
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        result = backend.execute(
            ExecutionRequest(code="raise SystemExit(7)")
        )

    assert result.exit_code == 7
    assert result.stderr == "Something went wrong"
    assert result.timed_out is False


def test_execute_preserves_timeout_result():
    backend = create_backend()
    response = create_response(
        json_data={
            "stdout": "partial output",
            "stderr": "",
            "exit_code": -1,
            "duration_seconds": 10.0,
            "timed_out": True,
        }
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        result = backend.execute(
            ExecutionRequest(code="while True: pass")
        )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.stdout == "partial output"


def test_execute_raises_error_when_token_is_missing():
    with patch.dict(
        "os.environ",
        {"LLM_EVAL_WORKER_TOKEN": ""},
    ):
        backend = RemoteExecutionBackend(
            base_url="http://localhost:8765",
        )

        with pytest.raises(
            ExecutionBackendError,
            match="authentication token is not configured",
        ):
            backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_maps_bad_request_to_value_error():
    backend = create_backend()
    response = create_response(
        status_code=400,
        json_data={"detail": "Code cannot be empty."},
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        with pytest.raises(ValueError, match="Code cannot be empty"):
            backend.execute(ExecutionRequest(code=" "))


@pytest.mark.parametrize("status_code", [401, 429, 500, 503])
def test_execute_maps_worker_errors_to_backend_error(status_code):
    backend = create_backend()
    response = create_response(
        status_code=status_code,
        json_data={"detail": "Worker unavailable."},
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        with pytest.raises(
            ExecutionBackendError,
            match=f"HTTP {status_code}",
        ):
            backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_handles_connection_error():
    backend = create_backend()

    with patch(
        "execution.remote_runner.requests.post",
        side_effect=requests.ConnectionError("Connection refused"),
    ):
        with pytest.raises(
            ExecutionBackendError,
            match="Could not connect",
        ):
            backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_handles_http_timeout():
    backend = create_backend()

    with patch(
        "execution.remote_runner.requests.post",
        side_effect=requests.Timeout("Request timed out"),
    ):
        with pytest.raises(
            ExecutionBackendError,
            match="timed out",
        ):
            backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_handles_invalid_json_response():
    backend = create_backend()
    response = Mock()
    response.status_code = 200
    response.json.side_effect = ValueError("Invalid JSON")

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        with pytest.raises(
            ExecutionBackendError,
            match="invalid JSON",
        ):
            backend.execute(ExecutionRequest(code="print(1)"))


def test_execute_handles_missing_response_field():
    backend = create_backend()
    response = create_response(
        json_data={
            "stdout": "42\n",
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 0.125,
        }
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        with pytest.raises(
            ExecutionBackendError,
            match="missing 'timed_out'",
        ):
            backend.execute(ExecutionRequest(code="print(42)"))


def test_execute_handles_invalid_response_field_type():
    backend = create_backend()
    response = create_response(
        json_data={
            **SUCCESS_RESPONSE,
            "exit_code": "0",
        }
    )

    with patch(
        "execution.remote_runner.requests.post",
        return_value=response,
    ):
        with pytest.raises(
            ExecutionBackendError,
            match="invalid 'exit_code'",
        ):
            backend.execute(ExecutionRequest(code="print(42)"))


def test_constructor_reads_worker_configuration_from_environment():
    with patch.dict(
        "os.environ",
        {
            "LLM_EVAL_WORKER_URL": "http://192.168.1.94:8765/",
            "LLM_EVAL_WORKER_TOKEN": WORKER_TOKEN,
        },
    ):
        backend = RemoteExecutionBackend()

    assert backend.base_url == "http://192.168.1.94:8765"
    assert backend.token == WORKER_TOKEN
