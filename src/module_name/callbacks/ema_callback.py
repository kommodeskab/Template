from torch_ema import ExponentialMovingAverage
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
import logging
from module_name.lightning_modules import BaseLightningModule

logger = logging.getLogger(__name__)


class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) Callback for PyTorch Lightning.
    This callback maintains an exponential moving average of the model parameters during training.
    The EMA weights are used during validation to improve performance and stability.
    The EMA state is saved and loaded with the model checkpoint.

    Args:
        decay (float): The decay rate for the EMA. Default is 0.999.
    """

    def __init__(
        self,
        decay: float = 0.999,
    ):
        super().__init__()
        self.decay = decay

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        logger.info("Initializing EMA...")
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)
        if hasattr(self, "_loaded_state_dict"):
            self.ema.load_state_dict(self._loaded_state_dict)
            del self._loaded_state_dict

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ema.to(pl_module.device)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ema.to(pl_module.device)

    def on_before_zero_grad(
        self, trainer: pl.Trainer, pl_module: BaseLightningModule, optimizer: Optimizer
    ):
        self.ema.update()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule):
        logger.info("Applying EMA weights for validation.")
        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule):
        logger.info("Restoring original weights after validation.")
        self.ema.restore()

    def state_dict(self) -> dict:
        return self.ema.state_dict() if hasattr(self, "ema") else {}

    def load_state_dict(self, state_dict: dict) -> None:
        if hasattr(self, "ema"):
            self.ema.load_state_dict(state_dict)
        else:
            self._loaded_state_dict = state_dict
