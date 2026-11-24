import pytest

from experiments.benchmark_config import BenchmarkExperimentConfig


def test_config_uses_defaults():
    config = BenchmarkExperimentConfig(
        prompt="Write a Python function.",
        task_id="code-generation",
    )

    assert config.warmup_runs == 2
    assert config.iterations == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 150


def test_config_accepts_custom_values():
    config = BenchmarkExperimentConfig(
        prompt="Explain binary search.",
        task_id="code-understanding",
        warmup_runs=3,
        iterations=10,
        temperature=0.5,
        max_tokens=300,
    )

    assert config.warmup_runs == 3
    assert config.iterations == 10
    assert config.temperature == 0.5
    assert config.max_tokens == 300


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", ""),
        ("prompt", "   "),
        ("task_id", ""),
        ("task_id", "   "),
    ],
)
def test_config_rejects_empty_required_strings(field, value):
    values = {
        "prompt": "Valid prompt",
        "task_id": "valid-task",
    }
    values[field] = value

    with pytest.raises(ValueError):
        BenchmarkExperimentConfig(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_runs", -1),
        ("warmup_runs", 1.5),
        ("warmup_runs", True),
        ("iterations", 0),
        ("iterations", -1),
        ("iterations", 1.5),
        ("iterations", True),
        ("temperature", -0.1),
        ("temperature", True),
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("max_tokens", 1.5),
        ("max_tokens", True),
    ],
)
def test_config_rejects_invalid_values(field, value):
    values = {
        "prompt": "Valid prompt",
        "task_id": "valid-task",
    }
    values[field] = value

    with pytest.raises(ValueError):
        BenchmarkExperimentConfig(**values)


def test_config_allows_none_max_tokens():
    config = BenchmarkExperimentConfig(
        prompt="Write a Python function.",
        task_id="code-generation",
        max_tokens=None,
    )

    assert config.max_tokens is None
