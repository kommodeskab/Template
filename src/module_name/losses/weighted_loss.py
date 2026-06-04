from .baseloss import BaseLossFunction
from module_name import Batch, ModelOutput, LossOutput
import torch.nn as nn


class WeightedLoss(BaseLossFunction):
    def __init__(
        self,
        losses: list[BaseLossFunction],
        weights: list[float],
    ):
        super().__init__()
        assert len(losses) == len(
            weights
        ), "Losses and weights must have the same length"
        self.losses: list[BaseLossFunction] = nn.ModuleList(losses)
        self.weights = weights

        self.loss_names = []
        for loss_fn in losses:
            base_name = loss_fn.__class__.__name__
            name = base_name
            counter = 2
            while name in self.loss_names:
                name = f"{base_name}_{counter}"
                counter += 1
            self.loss_names.append(name)

    def forward(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        loss = {"loss": 0.0}

        for loss_fn, weight, loss_name in zip(
            self.losses, self.weights, self.loss_names
        ):
            loss_output = loss_fn(model_output, batch)
            loss["loss"] += weight * loss_output["loss"]

            # the loss outputs might contain other keys than "loss",
            # we also want to log these, therefore, we add the loss name as prefix to the key
            for key, value in loss_output.items():
                loss[f"{loss_name}_{key}"] = value

        return LossOutput(**loss)
