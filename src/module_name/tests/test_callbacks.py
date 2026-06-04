import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import time
from module_name.callbacks import (
    StopTrainingCallback,
    BatchesPerSecondCallback,
    LogLossCallback,
)


@patch("module_name.callbacks.stop_training.wandb")
def test_stop_training_callback_max_params(mock_wandb):
    model = nn.Linear(5, 2)  # 5 * 2 + 2 = 12 parameters

    callback = StopTrainingCallback(
        key="stop_reason",
        value=1.0,
        max_num_params=10,
    )

    trainer = MagicMock()

    with pytest.raises(KeyboardInterrupt) as excinfo:
        callback.on_train_start(trainer, model)

    assert (
        "number of parameters in the model (12) exceeds the maximum allowed (10)"
        in str(excinfo.value)
    )
    mock_wandb.log.assert_called_once_with({"stop_reason": 1.0})
    mock_wandb.finish.assert_called_once()


def test_batches_per_second_callback():
    callback = BatchesPerSecondCallback()
    pl_module = MagicMock()

    # First batch (only initializes last_time tracker)
    callback.on_train_batch_end(None, pl_module, None, None, 0)
    pl_module.log.assert_not_called()

    # Sleep briefly to ensure time differences are measurable
    time.sleep(0.001)

    # Second batch (should log BPS metric)
    callback.on_train_batch_end(None, pl_module, None, None, 1)

    pl_module.log.assert_called_once()
    args, kwargs = pl_module.log.call_args
    assert args[0] == "train_batches_per_second"
    assert isinstance(args[1], float)
    assert args[1] > 0


def test_log_loss_callback():
    callback = LogLossCallback()
    pl_module = MagicMock()

    # Mock StepOutput structure with custom losses to log
    outputs = {
        "loss": torch.tensor(0.5),
        "loss_output": {
            "loss": torch.tensor(0.5),
            "reconstruction_loss": torch.tensor(0.3),
        },
    }

    callback.on_train_batch_end(None, pl_module, outputs, None, 0)

    pl_module.log_dict.assert_called_once()
    args, kwargs = pl_module.log_dict.call_args
    logged_dict = args[0]

    assert "train_loss" in logged_dict
    assert "train_reconstruction_loss" in logged_dict
    assert kwargs.get("prog_bar") is True
