import torch
from torch.utils.data import Dataset
from module_name import Sample
import os
from dotenv import load_dotenv

load_dotenv()


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def data_path(self) -> str:
        """
        Returns the `DATA_PATH` variable defined in the `.env` file.
        This is used to define where datasets are stored.
        Usually, `DATA_PATH="/data"`.

        Returns:
            str: The data path.
        """
        return os.getenv("DATA_PATH")

    def sample(self):
        """
        Returns a random sample from the dataset.
        This is useful for testing and debugging.

        Returns:
            Sample: A random sample from the dataset.
        """
        rand_idx = torch.randint(0, len(self), (1,)).item()
        return self.__getitem__(rand_idx)

    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented")

    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError("Get item method not implemented")
