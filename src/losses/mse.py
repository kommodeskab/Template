from losses import BaseLossFunction
from utils import Data
import torch

class MSELoss(BaseLossFunction):
    """
    Simple example of how to implement Mean Squared Error Loss
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, model_output: Data, batch: Data) -> Data:
        loss = torch.nn.functional.mse_loss(model_output.out, batch.targets)
        return {'loss': loss}