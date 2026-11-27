import base64
import subprocess
import time
import uuid

from execution.base import ExecutionBackend, ExecutionBackendError
from execution.models import ExecutionRequest, ExecutionResult


class DockerExecutionBackend(ExecutionBackend):
    """Execute untrusted Python code in restricted Docker containers."""

    IMAGE = "python:3.12-slim"

    EXECUTION_TIMEOUT_SECONDS = 10
    CLEANUP_TIMEOUT_SECONDS = 5
    MAX_CODE_BYTES = 16_000

    MEMORY_LIMIT = "128m"
    CPU_LIMIT = "0.5"
    PID_LIMIT = 32
    TEMP_STORAGE_LIMIT = "16m"

    MAX_OUTPUT_CHARS = 100_000

    def __init__(self, docker_executable: str = "docker"):
        self.docker_executable = docker_executable

    @property
    def backend_name(self) -> str:
        return "docker"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute Python code in an ephemeral restricted container."""

        if not request.code.strip():
            raise ValueError("Execution code cannot be empty.")

        code_bytes = request.code.encode("utf-8")

        if len(code_bytes) > self.MAX_CODE_BYTES:
            raise ValueError(
                f"Code exceeds the {self.MAX_CODE_BYTES}-byte limit."
            )

        container_name = f"llm-eval-{uuid.uuid4().hex[:12]}"
        encoded_code = base64.b64encode(code_bytes).decode("ascii")

        wrapper = (
            "import base64; "
            "exec(compile(base64.b64decode("
            "'''{}'''"
            "), '<submitted>', 'exec'))"
        ).format(encoded_code)

        command = [
            self.docker_executable,
            "run",
            "--rm",
            "--pull=never",
            f"--name={container_name}",
            "--network=none",
            "--memory={}".format(self.MEMORY_LIMIT),
            "--memory-swap={}".format(self.MEMORY_LIMIT),
            "--cpus={}".format(self.CPU_LIMIT),
            "--pids-limit={}".format(self.PID_LIMIT),
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--ipc=private",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={},uid=65534,gid=65534".format(
                self.TEMP_STORAGE_LIMIT
            ),
            "--tmpfs=/workspace:rw,noexec,nosuid,nodev,size={},uid=65534,gid=65534".format(
                self.TEMP_STORAGE_LIMIT
            ),
            "--workdir=/workspace",
            "--init",
            "-i",
            self.IMAGE,
            "python",
            "-I",
            "-B",
            "-c",
            wrapper,
        ]

        started_at = time.monotonic()

        try:
            completed = subprocess.run(
                command,
                input=request.stdin,
                capture_output=True,
                text=True,
                timeout=self.EXECUTION_TIMEOUT_SECONDS,
                check=False,
            )

            duration = time.monotonic() - started_at

            return ExecutionResult(
                stdout=completed.stdout[: self.MAX_OUTPUT_CHARS],
                stderr=completed.stderr[: self.MAX_OUTPUT_CHARS],
                exit_code=completed.returncode,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started_at
            self._remove_container(container_name)

            return ExecutionResult(
                stdout=self._decode_output(exc.stdout)[: self.MAX_OUTPUT_CHARS],
                stderr=self._decode_output(exc.stderr)[: self.MAX_OUTPUT_CHARS],
                exit_code=-1,
                duration_seconds=duration,
                timed_out=True,
            )

        except OSError as exc:
            self._remove_container(container_name)

            raise ExecutionBackendError(
                f"Unable to execute Docker command: {exc}"
            ) from exc

    def _remove_container(self, container_name: str) -> None:
        """Force-remove a container, including one still running."""

        try:
            subprocess.run(
                [
                    self.docker_executable,
                    "rm",
                    "--force",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=self.CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _decode_output(output: str | bytes | None) -> str:
        """Normalize subprocess output into a string."""

        if output is None:
            return ""

        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")

        return output
