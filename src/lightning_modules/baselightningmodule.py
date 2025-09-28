from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from pytorch_lightning.utilities import grad_norm
from torch.utils.data import Dataset
from losses import BaseLossFunction
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler    
from ..utils import Data, temporary_seed

class BaseLightningModule(pl.LightningModule):
    loss_fn: BaseLossFunction
    
    def __init__(
        self,
        optimizer : partial[Optimizer] | None = None,
        lr_scheduler : dict[str, partial[LRScheduler] | str] | None = None,
        ):
        super().__init__()
        
        self.partial_optimizer = optimizer
        self.partial_lr_scheduler = lr_scheduler
        
    def forward(self, batch : Data) -> Data: ...

    def common_step(self, batch : Data, batch_idx : int) -> Data:
        model_output = self.forward(batch)
        loss = self.loss_fn(model_output, batch)
        return loss

    def training_step(self, batch : Data, batch_idx : int) -> Tensor:
        loss = self.common_step(batch, batch_idx)
        loss = {f'train_{k}': v for k, v in loss.items()}
        self.log_dict(loss)
        return loss['train_loss']

    def validation_step(self, batch : Data, batch_idx : int) -> Tensor:
        with temporary_seed(0):
            loss = self.common_step(batch, batch_idx)
        loss = {f'val_{k}': v for k, v in loss.items()}
        self.log_dict(loss)
        return loss['val_loss']

    def on_before_optimizer_step(self, optimizer : Optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    @property
    def logger(self) -> WandbLogger:
        return self.trainer.logger
    
    @property
    def global_step(self) -> int:
        return self.trainer.global_step
    
    @property
    def train_dataset(self) -> Dataset:
        return self.trainer.datamodule.train_dataset
    
    @property
    def val_dataset(self) -> Dataset:
        return self.trainer.datamodule.val_dataset
    
    def configure_optimizers(self):
        assert self.partial_optimizer is not None, "Optimizer must be provided during training."
        assert self.partial_lr_scheduler is not None, "Learning rate scheduler must be provided during training."
        
        optim = self.partial_optimizer(self.parameters())
        scheduler = self.partial_lr_scheduler.pop('scheduler')(optim)
        return {
            'optimizer': optim,
            'lr_scheduler':  {
                'scheduler': scheduler,
                **self.partial_lr_scheduler
            }
        }