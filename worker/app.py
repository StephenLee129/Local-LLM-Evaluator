import hmac
import os
import threading

from fastapi import FastAPI, Header, HTTPException, status

from execution.base import ExecutionBackendError
from execution.docker_runner import DockerExecutionBackend
from execution.models import ExecutionRequest
from worker.models import ExecuteRequest, ExecuteResponse
from worker.middleware import RequestBodyLimitMiddleware


app = FastAPI(
    title="Local LLM Evaluator Execution Worker",
    description="Remote Docker-based code execution service.",
    version="1.0.0",
)

MAX_REQUEST_BODY_BYTES = 256 * 1024

app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
)

backend = DockerExecutionBackend()

WORKER_TOKEN_ENV = "LLM_EVAL_WORKER_TOKEN"

MAX_CONCURRENT_EXECUTIONS = 2
execution_slots = threading.BoundedSemaphore(
    MAX_CONCURRENT_EXECUTIONS
)


def verify_worker_token(token: str | None) -> None:
    """Validate the shared secret supplied by the desktop evaluator."""

    expected_token = os.environ.get(WORKER_TOKEN_ENV)

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Worker authentication is not configured.",
        )

    if token is None or not hmac.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing worker token.",
        )


@app.get("/api/v1/health")
def health_check():
    """Report whether the execution worker is available."""

    return {
        "status": "healthy",
        "backend": backend.backend_name,
    }


@app.post(
    "/api/v1/execute",
    response_model=ExecuteResponse,
)
def execute_code(
    request: ExecuteRequest,
    worker_token: str | None = Header(default=None, alias="X-Worker-Token"),
):
    """Execute submitted Python code through the Docker backend."""

    verify_worker_token(worker_token)

    if not execution_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum concurrent executions reached.",
            headers={"Retry-After": "1"},
        )

    try:
        try:
            result = backend.execute(
                ExecutionRequest(
                    code=request.code,
                    stdin=request.stdin,
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        except ExecutionBackendError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        return ExecuteResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
        )

    finally:
        execution_slots.release()
