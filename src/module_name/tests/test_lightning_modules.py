import torch
import torch.nn as nn
from module_name.lightning_modules import DummyModule
from module_name import Batch
from module_name.losses import MSELoss


def test_dummy_module():
    network = nn.Linear(10, 1)
    loss_fn = MSELoss()
    module = DummyModule(network=network, loss_fn=loss_fn)

    batch = Batch(input=torch.randn(4, 10), target=torch.randn(4, 1))

    # Test forward
    output = module(batch)
    assert isinstance(output, dict)
    assert "output" in output
    assert output["output"].shape == (4, 1)

    # Test common_step
    step_output = module.common_step(batch, 0)
    assert isinstance(step_output, dict)
    assert "loss" in step_output
    assert "model_output" in step_output
    assert step_output["loss"].shape == ()
