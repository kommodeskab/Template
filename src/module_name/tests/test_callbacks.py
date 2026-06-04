import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import time
from module_name.callbacks import (
    StopTrainingCallback,
    BatchesPerSecondCallback,
    LogLossCallback,
    EMACallback,
    LogGradsCallback,
    WandbWatchCallback,
    LogGraphCallback,
    ParameterCountCallback,
    MetricsCallback,
)
from module_name.callbacks.metrics.base import BaseMetric
from module_name.callbacks.extras.base import ExtraMetricOutput


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


def test_ema_callback():
    callback = EMACallback(decay=0.9)
    assert callback.decay == 0.9

    trainer = MagicMock()
    pl_module = MagicMock()

    linear = nn.Linear(3, 2)
    pl_module.parameters.return_value = linear.parameters()
    pl_module.device = "cpu"

    callback.setup(trainer, pl_module, "fit")
    assert hasattr(callback, "ema")

    callback.on_fit_start(trainer, pl_module)

    # Test state dict serialization/deserialization
    state = callback.state_dict()
    assert isinstance(state, dict)

    callback.load_state_dict(state)


@patch("module_name.callbacks.log_gradients.grad_norm")
def test_log_grads_callback(mock_grad_norm):
    callback = LogGradsCallback()
    trainer = MagicMock()
    trainer.global_step = 100
    trainer.log_every_n_steps = 100

    pl_module = MagicMock()
    mock_grad_norm.return_value = {"grad_norm_total": 0.5}

    callback.on_before_optimizer_step(trainer, pl_module, MagicMock())
    mock_grad_norm.assert_called_once_with(pl_module, norm_type=2)
    pl_module.log_dict.assert_called_once_with({"grad_norm_total": 0.5})


def test_wandb_watch_callback():
    callback = WandbWatchCallback(log="gradients", log_frequency=500, log_graph=False)
    trainer = MagicMock()
    pl_module = MagicMock()

    callback.on_fit_start(trainer, pl_module)
    pl_module.logger.watch.assert_called_once_with(
        pl_module,
        log="gradients",
        log_graph=False,
        log_freq=500,
    )


@patch("module_name.callbacks.log_graph.draw_graph")
@patch("module_name.callbacks.log_graph.Image")
def test_log_graph_callback(mock_image, mock_draw_graph):
    callback = LogGraphCallback(keyword="my_network", input_shape=[(1, 10)])

    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.my_network = MagicMock()
    pl_module.device = "cpu"

    mock_graph = MagicMock()
    mock_graph.visual_graph.pipe.return_value = b"png_data"
    mock_draw_graph.return_value = mock_graph

    mock_pil_img = MagicMock()
    mock_image.open.return_value = mock_pil_img

    callback.on_train_start(trainer, pl_module)

    mock_draw_graph.assert_called_once()
    args, kwargs = mock_draw_graph.call_args
    assert args[0] is pl_module.my_network
    assert len(args[1]) == 1
    assert isinstance(args[1][0], torch.Tensor)
    assert args[1][0].shape == (1, 10)

    mock_image.open.assert_called_once()
    pl_module.logger.log_image.assert_called_once_with(
        key="model_graph",
        images=[mock_pil_img],
    )


@patch("module_name.callbacks.parametercount.wandb")
def test_parameter_count_callback(mock_wandb):
    callback = ParameterCountCallback(depth=0)

    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.logger = MagicMock()

    callback.on_train_start(trainer, pl_module)

    mock_wandb.Table.assert_called_once()
    mock_wandb.plot.bar.assert_called_once()
    pl_module.logger.log_metrics.assert_called_once()


def test_metrics_callback():
    metric = MagicMock(spec=BaseMetric)
    metric.name.return_value = "dummy_metric"
    metric.compute.return_value = {"val": 0.95}

    extra = MagicMock(spec=ExtraMetricOutput)
    extra.return_value = {"extra_key": torch.tensor(1.0)}

    callback = MetricsCallback(metrics=[metric], extras=[extra])

    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.device = "cpu"
    pl_module.logger = MagicMock()

    callback.on_fit_start(trainer, pl_module)
    metric.to.assert_called_once_with("cpu")
    extra.to.assert_called_once_with("cpu")

    # Test duplicate check on init
    metric2 = MagicMock(spec=BaseMetric)
    metric2.name.return_value = "dummy_metric"
    with pytest.raises(ValueError, match="Duplicate metric names"):
        MetricsCallback(metrics=[metric, metric2])
