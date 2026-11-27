import os

import requests

from execution.base import ExecutionBackend, ExecutionBackendError
from execution.models import ExecutionRequest, ExecutionResult


class RemoteExecutionBackend(ExecutionBackend):
    """Execution backend that submits Python code to a remote worker."""

    DEFAULT_BASE_URL = "http://127.0.0.1:8765"
    DEFAULT_TIMEOUT_SECONDS = 15

    WORKER_URL_ENV = "LLM_EVAL_WORKER_URL"
    WORKER_TOKEN_ENV = "LLM_EVAL_WORKER_TOKEN"

    EXECUTE_ENDPOINT = "/api/v1/execute"
    TOKEN_HEADER = "X-Worker-Token"

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.base_url = (
            base_url
            or os.environ.get(self.WORKER_URL_ENV)
            or self.DEFAULT_BASE_URL
        ).rstrip("/")

        self.token = (
            token
            if token is not None
            else os.environ.get(self.WORKER_TOKEN_ENV)
        )

        if not self.base_url:
            raise ValueError("Worker base URL cannot be empty.")

        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be greater than zero.")

        self.timeout_seconds = timeout_seconds

    @property
    def backend_name(self) -> str:
        return "remote"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Submit code to the remote worker and return its execution result."""

        if not self.token:
            raise ExecutionBackendError(
                f"Worker authentication token is not configured. "
                f"Set the {self.WORKER_TOKEN_ENV} environment variable."
            )

        url = f"{self.base_url}{self.EXECUTE_ENDPOINT}"

        headers = {
            self.TOKEN_HEADER: self.token,
        }

        payload = {
            "code": request.code,
            "stdin": request.stdin,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )

        except requests.Timeout as exc:
            raise ExecutionBackendError(
                f"Remote execution request timed out: {exc}"
            ) from exc

        except requests.ConnectionError as exc:
            raise ExecutionBackendError(
                f"Could not connect to the execution worker: {exc}"
            ) from exc

        except requests.RequestException as exc:
            raise ExecutionBackendError(
                f"Remote execution request failed: {exc}"
            ) from exc

        if not 200 <= response.status_code < 300:
            message = self._get_error_detail(response)
            error_message = (
                f"Execution worker returned HTTP {response.status_code}: "
                f"{message}"
            )

            if response.status_code in (400, 413, 422):
                raise ValueError(error_message)

            raise ExecutionBackendError(error_message)

        try:
            data = response.json()

        except ValueError as exc:
            raise ExecutionBackendError(
                "Execution worker returned invalid JSON."
            ) from exc

        return self._parse_result(data)

    @staticmethod
    def _get_error_detail(response: requests.Response) -> str:
        """Extract a useful error message from a worker response."""

        try:
            data = response.json()

            if isinstance(data, dict) and "detail" in data:
                return str(data["detail"])

        except ValueError:
            pass

        return response.text or "No error details provided."

    @staticmethod
    def _parse_result(data: object) -> ExecutionResult:
        """Validate and convert the worker response into an execution result."""

        if not isinstance(data, dict):
            raise ExecutionBackendError(
                "Execution worker returned an unexpected response format."
            )

        required_fields = {
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "duration_seconds": (int, float),
            "timed_out": bool,
        }

        for field, expected_type in required_fields.items():
            if field not in data:
                raise ExecutionBackendError(
                    f"Execution worker response is missing '{field}'."
                )

            value = data[field]

            # bool is a subclass of int in Python, but is not a valid
            # exit code or duration.
            if field in ("exit_code", "duration_seconds") and isinstance(
                value, bool
            ):
                raise ExecutionBackendError(
                    f"Execution worker returned an invalid '{field}' value."
                )

            if not isinstance(value, expected_type):
                raise ExecutionBackendError(
                    f"Execution worker returned an invalid '{field}' value."
                )

        return ExecutionResult(
            stdout=data["stdout"],
            stderr=data["stderr"],
            exit_code=data["exit_code"],
            duration_seconds=float(data["duration_seconds"]),
            timed_out=data["timed_out"],
        )
