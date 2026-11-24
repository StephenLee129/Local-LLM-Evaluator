import pytest

from experiments.config_loader import (
    ExperimentConfigurationError,
    load_experiment_suite,
)
from experiments.models import ExperimentSuiteDefinition


def test_load_experiment_suite_returns_valid_definition(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: basic-model-evaluation

models:
  - qwen2.5-coder:7b
  - llama3.1:8b

experiments:
  - type: benchmark
    config:
      warmup_runs: 2
      iterations: 5

  - type: cold_start
    config:
      unload_before_run: true
""",
        encoding="utf-8",
    )

    suite = load_experiment_suite(config_file)

    assert isinstance(suite, ExperimentSuiteDefinition)
    assert suite.name == "basic-model-evaluation"
    assert suite.models == ["qwen2.5-coder:7b", "llama3.1:8b"]
    assert len(suite.experiments) == 2

    assert suite.experiments[0].experiment_type == "benchmark"
    assert suite.experiments[0].config == {
        "warmup_runs": 2,
        "iterations": 5,
    }

    assert suite.experiments[1].experiment_type == "cold_start"
    assert suite.experiments[1].config == {
        "unload_before_run": True,
    }


def test_load_experiment_suite_defaults_missing_experiment_config(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: basic-model-evaluation
models:
  - qwen2.5-coder:7b
experiments:
  - type: benchmark
""",
        encoding="utf-8",
    )

    suite = load_experiment_suite(config_file)

    assert suite.experiments[0].config == {}


def test_load_experiment_suite_rejects_missing_file(tmp_path):
    config_file = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_experiment_suite(config_file)


def test_load_experiment_suite_rejects_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("name: [invalid", encoding="utf-8")

    with pytest.raises(ExperimentConfigurationError, match="Invalid YAML syntax"):
        load_experiment_suite(config_file)


def test_load_experiment_suite_rejects_duplicate_models(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: duplicate-models
models:
  - qwen2.5-coder:7b
  - qwen2.5-coder:7b
experiments:
  - type: benchmark
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigurationError, match="cannot contain duplicates"):
        load_experiment_suite(config_file)


def test_load_experiment_suite_rejects_empty_models(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: missing-models
models: []
experiments:
  - type: benchmark
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigurationError, match="models"):
        load_experiment_suite(config_file)


def test_load_experiment_suite_rejects_missing_experiments(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: missing-experiments
models:
  - qwen2.5-coder:7b
experiments: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigurationError, match="experiments"):
        load_experiment_suite(config_file)


def test_load_experiment_suite_rejects_non_mapping_config(tmp_path):
    config_file = tmp_path / "suite.yaml"
    config_file.write_text(
        """
name: invalid-config
models:
  - qwen2.5-coder:7b
experiments:
  - type: benchmark
    config: invalid
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigurationError, match="config must be a mapping"):
        load_experiment_suite(config_file)


@pytest.mark.parametrize(
    "yaml_content",
    [
        """
models:
  - qwen2.5-coder:7b
experiments:
  - type: benchmark
""",
        """
name: ""
models:
  - qwen2.5-coder:7b
experiments:
  - type: benchmark
""",
        """
name: invalid-models
models: qwen2.5-coder:7b
experiments:
  - type: benchmark
""",
        """
name: invalid-experiments
models:
  - qwen2.5-coder:7b
experiments: benchmark
""",
    ],
)
def test_load_experiment_suite_rejects_invalid_suite_fields(
    tmp_path,
    yaml_content,
):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ExperimentConfigurationError):
        load_experiment_suite(config_file)


@pytest.mark.parametrize(
    "experiment_yaml",
    [
        "null",
        "- benchmark",
        "type: ''",
        "type: 123",
    ],
)
def test_load_experiment_suite_rejects_invalid_experiment_entries(
    tmp_path,
    experiment_yaml,
):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        f"""
name: invalid-experiment
models:
  - qwen2.5-coder:7b
experiments:
  - {experiment_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentConfigurationError):
        load_experiment_suite(config_file)

