from torch.nn import Module
from src.utils import model_from_id


class PretrainedModel:
    def __new__(
        cls,
        project: str,
        id: str,
        model_keyword: str,
        ckpt_filename: str = "last",
    ) -> Module:
        return model_from_id(
            project,
            id,
            model_keyword,
            ckpt_filename,
        )
