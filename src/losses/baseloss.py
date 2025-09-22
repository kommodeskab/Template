import torch.nn as nn
from utils import Data

class BaseLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_output: Data, batch: Data) -> Data:
        raise NotImplementedError("Loss function not implemented")

    def __call__(self, model_output: Data, batch: Data) -> Data:
        return self.forward(model_output, batch)