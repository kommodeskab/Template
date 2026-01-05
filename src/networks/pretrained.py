from torch.nn import Module
from src.utils import model_from_id

class PretrainedModel:
    def __new__(
        cls,
        experiment_id: str,
        model_keyword: str,
    ) -> Module:
        return model_from_id(experiment_id, model_keyword)