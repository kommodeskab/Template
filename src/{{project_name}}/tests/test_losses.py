from {{project_name}}.losses import MSELoss


def test_mse_loss(dummy_batch, dummy_model_output):
    loss_fn = MSELoss()
    loss = loss_fn.forward(dummy_model_output, dummy_batch)
    assert "loss" in loss, "Loss output should contain 'loss' key"
    assert loss["loss"].item() >= 0, "Loss value should be non-negative"
