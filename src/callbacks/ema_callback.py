from torch_ema import ExponentialMovingAverage
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer
import logging
from src.lightning_modules import BaseLightningModule

logger = logging.getLogger(__name__)

class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
    ):
        super().__init__()
        self.decay = decay
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("Initializing EMA...")
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)
        self.ema.to(pl_module.device)
        
    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: BaseLightningModule, optimizer: Optimizer) -> None:
        self.ema.update()
        
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: BaseLightningModule, checkpoint: dict) -> None:
        logger.info("Saving EMA state dict to checkpoint.")
        checkpoint['ema_state_dict'] = self.ema.state_dict()
        
    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: BaseLightningModule, checkpoint: dict) -> None:
        if 'ema_state_dict' in checkpoint:
            logger.info("Loading EMA state dict from checkpoint.")
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
                                                
    def on_validation_start(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        logger.info("Applying EMA weights for validation.")
        self.ema.store()
        self.ema.copy_to()
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: BaseLightningModule) -> None:
        logger.info("Restoring original weights after validation.")
        self.ema.restore()