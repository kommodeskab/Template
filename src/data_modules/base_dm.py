import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def split_dataset(
    train_dataset: Dataset, 
    val_dataset: Optional[Dataset], 
    train_val_split: float
    ) -> tuple[Dataset, Dataset]:
    if val_dataset is None:
        return random_split(train_dataset, [train_val_split, 1 - train_val_split])
    else:
        return train_dataset, val_dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        train_val_split: Optional[float] = None,
        **kwargs
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        self.original_dataset = dataset
        self.train_dataset, self.val_dataset = split_dataset(dataset, val_dataset, train_val_split)
        self.kwargs = kwargs
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            **self.kwargs,
            )
        
    def val_dataloader(self):
        kwargs = self.kwargs.copy()
        # remove the shuffle and drop_last for validation dataloader
        for key in ['shuffle', 'drop_last']:
            if key in kwargs:
                kwargs.pop(key)
            
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last=True,
            **kwargs,
            )