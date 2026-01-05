from torch.utils.data import Dataset
import hashlib
from src import Batch
import os
from dotenv import load_dotenv

load_dotenv()


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()

    @property
    def data_path(self) -> str:
        path = os.getenv("DATA_PATH")
        return path

    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented")

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError("Get item method not implemented")
