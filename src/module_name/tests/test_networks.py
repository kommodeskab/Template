from unittest.mock import patch
import torch
import torch.nn as nn
from module_name.networks import DummyNetwork, PretrainedModel


def test_dummy_network():
    model = DummyNetwork(input_size=10, output_size=2)
    sample_input = torch.randn(4, 10)
    output = model(sample_input)
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"


@patch("module_name.networks.pretrained.model_from_id")
def test_pretrained_model(mock_model_from_id):
    dummy_model = nn.Linear(5, 5)
    mock_model_from_id.return_value = dummy_model

    model = PretrainedModel(id="test_run_id", model_keyword="network_kw")
    assert model is dummy_model
    mock_model_from_id.assert_called_once_with(
        "test_run_id",
        "network_kw",
        "last",
    )
