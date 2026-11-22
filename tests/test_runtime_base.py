import pytest

from runtimes.base import InferenceRuntime


def test_inference_runtime_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        InferenceRuntime()
