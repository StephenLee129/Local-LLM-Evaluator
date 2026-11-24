from pathlib import Path
from typing import Any

import yaml

from experiments.models import (
    ExperimentDefinition,
    ExperimentSuiteDefinition,
)


class ExperimentConfigurationError(Exception):
    """Raised when an experiment suite configuration is invalid."""


def load_experiment_suite(
    file_path: str | Path,
) -> ExperimentSuiteDefinition:
    """Load and validate an experiment suite from a YAML file."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ExperimentConfigurationError(
            f"Invalid YAML syntax in '{path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ExperimentConfigurationError(
            "Experiment suite configuration must be a YAML mapping."
        )

    name = data.get("name")
    models = data.get("models")
    experiment_data = data.get("experiments")

    if not isinstance(name, str) or not name.strip():
        raise ExperimentConfigurationError(
            "Experiment suite 'name' must be a nonempty string."
        )

    if not isinstance(models, list) or not models:
        raise ExperimentConfigurationError(
            "Experiment suite 'models' must be a nonempty list."
        )

    for index, model in enumerate(models):
        if not isinstance(model, str) or not model.strip():
            raise ExperimentConfigurationError(
                f"Model at index {index} must be a nonempty string."
            )

    if len(models) != len(set(models)):
        raise ExperimentConfigurationError(
            "Experiment suite 'models' cannot contain duplicates."
        )

    if not isinstance(experiment_data, list) or not experiment_data:
        raise ExperimentConfigurationError(
            "Experiment suite 'experiments' must be a nonempty list."
        )

    experiments: list[ExperimentDefinition] = []

    for index, experiment in enumerate(experiment_data):
        if not isinstance(experiment, dict):
            raise ExperimentConfigurationError(
                f"Experiment at index {index} must be a YAML mapping."
            )

        experiment_type = experiment.get("type")
        config: Any = experiment.get("config", {})

        if (
            not isinstance(experiment_type, str)
            or not experiment_type.strip()
        ):
            raise ExperimentConfigurationError(
                f"Experiment at index {index} must have a nonempty 'type'."
            )

        if not isinstance(config, dict):
            raise ExperimentConfigurationError(
                f"Experiment '{experiment_type}' config must be a mapping."
            )

        experiments.append(
            ExperimentDefinition(
                experiment_type=experiment_type.strip(),
                config=config,
            )
        )

    return ExperimentSuiteDefinition(
        name=name.strip(),
        models=models,
        experiments=experiments,
    )
