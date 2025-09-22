from torch.utils.data import Dataset
import hashlib
from .utils import get_data_path
from ..utils import Data

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()
    
    @property
    def data_path(self):
        return get_data_path()
    
    def __len__(self) -> int:
        raise NotImplementedError("Length method not implemented")

    def __getitem__(self, index : int) -> Data:
        raise NotImplementedError("Get item method not implemented")