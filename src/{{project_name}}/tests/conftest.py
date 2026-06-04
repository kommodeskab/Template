import pytest
import torch
from {{project_name}}.datasets import DummyDataset
from {{project_name}} import Batch, ModelOutput


@pytest.fixture
def dummy_dataset():
    return DummyDataset(size=16)


@pytest.fixture
def dummy_batch():
    return Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))


@pytest.fixture
def dummy_model_output():
    return ModelOutput(output=torch.randn(16, 1))
