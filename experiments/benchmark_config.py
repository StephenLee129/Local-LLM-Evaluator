from dataclasses import dataclass


DEFAULT_WARMUP_RUNS = 2
DEFAULT_ITERATIONS = 5
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 150


@dataclass
class BenchmarkExperimentConfig:
    prompt: str
    task_id: str
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    iterations: int = DEFAULT_ITERATIONS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")

        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a nonempty string")

        if (
            not isinstance(self.warmup_runs, int)
            or isinstance(self.warmup_runs, bool)
            or self.warmup_runs < 0
        ):
            raise ValueError("warmup_runs must be a nonnegative integer")

        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")

        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a nonnegative number")

        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer or None")
