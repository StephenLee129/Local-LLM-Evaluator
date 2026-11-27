import pytest

from execution.base import ExecutionBackend


def test_execution_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ExecutionBackend()