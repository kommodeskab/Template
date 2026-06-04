from module_name.losses import MSELoss
from module_name import ModelOutput, Batch
import torch


def test_mse_loss():
    loss_fn = MSELoss()
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn.forward(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"


def test_weighted_loss():
    from module_name.losses import WeightedLoss, SmoothL1Loss

    loss_fn = WeightedLoss(
        losses=[MSELoss(), SmoothL1Loss(), MSELoss()], weights=[0.3, 0.3, 0.4]
    )
    batch = Batch(input=torch.randn(16, 10), target=torch.randn(16, 1))
    model_output = ModelOutput(output=torch.randn(16, 1))
    loss = loss_fn(model_output, batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert "MSELoss_loss" in loss, "Loss output should contain MSELoss sub-loss"
    assert (
        "SmoothL1Loss_loss" in loss
    ), "Loss output should contain SmoothL1Loss sub-loss"
    assert (
        "MSELoss_2_loss" in loss
    ), "Loss output should contain second MSELoss sub-loss"
