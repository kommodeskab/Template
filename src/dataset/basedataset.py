from torch.utils.data import Dataset
import hashlib
from src import Batch
import os


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()

    @property
    def data_path(self) -> str:
        path = os.environ.get("DATA_PATH")
        if path is None:
            raise ValueError("DATA_PATH environment variable not set")
        return path

    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented")

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError("Get item method not implemented")
