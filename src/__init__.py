from typing import TypedDict, Dict, Optional
from functools import partial
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from matplotlib.figure import Figure
import numpy as np

TensorDict = Dict[str, Tensor | float | int]
OptimizerType = Optional[partial[Optimizer]]
LRSchedulerType = Optional[dict[str, partial[LRScheduler] | str]]
ImageType = list[Tensor | Figure | np.ndarray]

class Batch(TypedDict):
    input: Tensor
    target: Tensor

class ModelOutput(TypedDict):
    output: Tensor
    
class LossOutput(TypedDict):
    loss: Tensor