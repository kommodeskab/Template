from torch.utils.data import DataLoader, Dataset
from src import Batch


def get_batch_from_dataset(dataset: Dataset, batch_size: int, shuffle: bool = False) -> Batch:
    """
    Returns a single batch of a given size from the dataset.
    This is useful for quickly getting a batch, e.g. for testing purposes.

    Args:
        dataset (Dataset): The dataset to get the batch from.
        batch_size (int): The size of the batch.
        shuffle (bool, optional): Whether to shuffle the dataset before getting the batch. Defaults to False.

    Returns:
        Batch: A batch of data from the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return next(iter(dataloader))
