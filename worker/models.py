from pydantic import BaseModel, Field, field_validator


MAX_CODE_BYTES = 16_000
MAX_STDIN_BYTES = 16_000


class ExecuteRequest(BaseModel):
    """HTTP request containing code and optional standard input."""

    code: str = Field(min_length=1, max_length=MAX_CODE_BYTES)
    stdin: str = Field(default="", max_length=MAX_STDIN_BYTES)

    @field_validator("code")
    @classmethod
    def validate_code_bytes(cls, value: str) -> str:
        """Enforce the backend's UTF-8 byte limit."""

        if len(value.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError(
                f"Code exceeds the {MAX_CODE_BYTES}-byte limit."
            )

        return value

    @field_validator("stdin")
    @classmethod
    def validate_stdin_bytes(cls, value: str) -> str:
        """Prevent excessively large standard input."""

        if len(value.encode("utf-8")) > MAX_STDIN_BYTES:
            raise ValueError(
                f"Standard input exceeds the {MAX_STDIN_BYTES}-byte limit."
            )

        return value


class ExecuteResponse(BaseModel):
    """HTTP response containing execution output and metadata."""

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool
