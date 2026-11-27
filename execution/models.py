from dataclasses import dataclass


@dataclass
class ExecutionRequest:
    """Python code and input submitted for remote execution."""

    code: str
    stdin: str = ""


@dataclass
class ExecutionResult:
    """Output and execution metadata returned by an execution worker."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False
