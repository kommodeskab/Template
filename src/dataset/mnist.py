from torch import Tensor
from torchvision.datasets import MNIST
from src.dataset.basedataset import BaseDataset
from src.utils import Data
from torchvision.transforms import ToTensor, Resize, Compose
import torch

class MNISTDataset(BaseDataset):
    def __init__(
        self,
        train: bool,
        img_size: int,
    ):
        self.train = train
        self.transform = Compose([
            Resize((img_size, img_size)),
            ToTensor(),
        ])
        self.mnist = MNIST(root=self.data_path, train=train, download=True)
        
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, index) -> Data:
        X, y = self.mnist[index]
        X : Tensor = self.transform(X)
        X = X.unsqueeze(1)
        y = torch.tensor(y)
            
        return {
            'input': X,
            'label': y
        }