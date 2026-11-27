import base64
import subprocess
import threading
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
    OUTPUT_READ_CHUNK_BYTES = 4096

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
            f"--memory={self.MEMORY_LIMIT}",
            f"--memory-swap={self.MEMORY_LIMIT}",
            f"--cpus={self.CPU_LIMIT}",
            f"--pids-limit={self.PID_LIMIT}",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--ipc=private",
            (
                f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,"
                f"size={self.TEMP_STORAGE_LIMIT},uid=65534,gid=65534"
            ),
            (
                f"--tmpfs=/workspace:rw,noexec,nosuid,nodev,"
                f"size={self.TEMP_STORAGE_LIMIT},uid=65534,gid=65534"
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

        normalized_stdin = (
            request.stdin.replace("\r\n", "\n").replace("\r", "\n")
        ).encode("utf-8")

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        started_at = time.monotonic()

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            stdout_thread = threading.Thread(
                target=self._read_output,
                args=(process.stdout, stdout_chunks),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._read_output,
                args=(process.stderr, stderr_chunks),
                daemon=True,
            )
            stdin_thread = threading.Thread(
                target=self._write_input,
                args=(process.stdin, normalized_stdin),
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()
            stdin_thread.start()

            try:
                exit_code = process.wait(
                    timeout=self.EXECUTION_TIMEOUT_SECONDS
                )
                timed_out = False

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exit_code = -1
                timed_out = True
                self._remove_container(container_name)

            # Wait for all streams to finish after the process exits.
            stdin_thread.join()
            stdout_thread.join()
            stderr_thread.join()

            duration = time.monotonic() - started_at

            return ExecutionResult(
                stdout=self._decode_output(b"".join(stdout_chunks)),
                stderr=self._decode_output(b"".join(stderr_chunks)),
                exit_code=exit_code,
                duration_seconds=duration,
                timed_out=timed_out,
            )

        except OSError as exc:
            self._remove_container(container_name)

            raise ExecutionBackendError(
                f"Unable to execute Docker command: {exc}"
            ) from exc

    def _read_output(self, stream, chunks: list[bytes]) -> None:
        """Drain a process output stream while retaining bounded output."""

        max_bytes = self.MAX_OUTPUT_CHARS * 4
        retained_bytes = 0

        try:
            while True:
                chunk = stream.read(self.OUTPUT_READ_CHUNK_BYTES)

                if not chunk:
                    break

                remaining_bytes = max_bytes - retained_bytes

                if remaining_bytes > 0:
                    retained_chunk = chunk[:remaining_bytes]
                    chunks.append(retained_chunk)
                    retained_bytes += len(retained_chunk)

        except (OSError, ValueError):
            pass

        finally:
            stream.close()

    @staticmethod
    def _write_input(stream, input_bytes: bytes) -> None:
        """Write stdin independently so it cannot block timeout handling."""

        try:
            stream.write(input_bytes)
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            stream.close()

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

    def _decode_output(self, output: bytes) -> str:
        """Decode captured output, replacing invalid UTF-8 sequences."""

        return output.decode(
            "utf-8",
            errors="replace",
        )[: self.MAX_OUTPUT_CHARS]
