from diffusers import UNet2DModel, UNet1DModel
import torch

class UNet2D(UNet2DModel):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor): 
        return super().forward(x, time_step).sample
        
class PretrainedUNet2D:
    def __new__(cls, model_id : str, **kwargs,):
        subfolder = kwargs.pop("subfolder", "")
        dummy_model : UNet2DModel = UNet2DModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)
        dummy_model.__class__ = UNet2D
        return dummy_model
    
class UNet1D(UNet1DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x : torch.Tensor, time_step : torch.Tensor): 
        return super().forward(x, time_step).sample