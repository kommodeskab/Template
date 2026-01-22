import os
import hydra
from omegaconf import DictConfig
from datetime import datetime
import wandb
import contextlib
import random
import numpy as np
import torch
from hydra.utils import instantiate
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
import tempfile
import shutil
import logging
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temporary_seed(seed: int):
    """
    Context manager for setting a temporary random seed. Sets `random`, `numpy`, `torch` and `torch.cuda` (if available) seeds.
    Code executed inside the context manager will have the specified random seed.
    What is it good for? For reproducing results, e.g. during validation or testing.
    Example usage:
    >>> torch.randn(3) # random tensor
    >>> with temporary_seed(42):
    >>>     torch.randn(3) # tensor with seed 42

    Args:
        seed (int): The temporary random seed to set.
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        yield

    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def get_current_time() -> str:
    """Returns the current time as a string in the format 'ddmmyyHHMMSS'.

    Returns:
        str: The current time as a string.
    """
    now = datetime.now()
    return now.strftime("%d%m%y%H%M%S")


def instantiate_callbacks(callback_cfg: DictConfig | None) -> list[Callback]:
    """
    Function for instantiating callbacks given a `DictConfig`.
    If `callback_cfg` is `None`, an empty list is returned.
    This function is useful for hydra-based configuration of PyTorch Lightning callbacks.

    Args:
        callback_cfg (DictConfig | None): A `DictConfig` containing the callback configurations. If `None`, no callbacks are instantiated.

    Returns:
        list[Callback]: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if callback_cfg is None:
        return callbacks

    for _, callback_params in callback_cfg.items():
        callback = hydra.utils.instantiate(callback_params)
        callbacks.append(callback)

    return callbacks


def get_ckpt_path(
    project: str,
    id: str,
    filename: str = "last",
):
    """
    Returns the path to a specific WandB checkpoint.
    If the checkpoint does not exist locally, it attempts to download it from WandB.
    Example usage:
    >>> ckpt_path = get_ckpt_path("my_project", "12345678", "best")
    >>> ckpt = torch.load(ckpt_path)
    >>> module = DummyModule.load_from_checkpoint(ckpt_path)

    Args:
        project (str): Name of the WandB project.
        id (str): The WandB run ID.
        filename (str, optional): The checkpoint filename. If the checkpoint is remote, then specify the name of the artifact containing the checkpoint. Defaults to "last".

    Returns:
        str: Path to the checkpoint file.
    """

    ckpt_path = f"logs/{project}/{id}/checkpoints/{filename}.ckpt"

    if not os.path.exists(ckpt_path):
        try:
            ckpt_path = download_checkpoint(
                project=project,
                id=id,
                filename=filename,
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download checkpoint with filename '{filename}' for experiment id '{id}' in project '{project}'."
            ) from e

    return ckpt_path


def what_logs_to_delete():
    """
    Prints out the WandB logs that can be safely deleted.
    By "safely deleted", we mean logs that exist locally but not on WandB.
    Here, we assume that if a run ID does not exist on WandB, then it has become obsolete and can be deleted.
    Generally, it is safe to delete most local logs since WandB keeps track of all experiments, including model checkpoints (if they are logged using a `ModelCheckpoint` callback).
    """
    api = wandb.Api()
    project_names = api.projects()
    project_names = [project.name for project in project_names]
    print("It is safe to delete the following folders:")
    for project_name in project_names:
        if not os.path.exists(f"logs/{project_name}"):
            continue

        runs = api.runs(project_name)
        run_ids = [run.id for run in runs]
        local_run_ids = os.listdir(f"logs/{project_name}")
        local_run_ids.sort(reverse=True)

        for local_run_id in local_run_ids:
            if local_run_id not in run_ids:
                # delete the folder
                print(f"logs/{project_name}/{local_run_id}")

    print("Done")


def config_from_id(project: str, id: str) -> dict:
    """
    Returns the config for a specific run.

    Args:
        project (str): The project name
        id (str): The run ID

    Raises:
        ValueError: If the experiment with the given project and ID could not be found.

    Returns:
        dict: The configuration dictionary for the specified run.
    """
    api = wandb.Api()
    name = wandb.api.viewer()["entity"]
    path = f"{name}/{project}/{id}"
    try:
        run = api.run(path)
        logger.info(f"Found experiment {path}.")
        return run.config
    except wandb.errors.CommError:
        pass

    raise ValueError(f"Could not find experiment {path}.")


def model_config_from_id(project: str, id: str, model_keyword: str) -> dict:
    """
    Returns the model config for a specific run.

    Args:
        project (str): The project name.
        id (str): The run ID.
        model_keyword (str): The keyword in the config corresponding to the model.

    Returns:
        dict: The model configuration dictionary for the specified run.
    """
    config = config_from_id(project, id)
    return config["model"][model_keyword]


def module_from_id(
    project: str,
    id: str,
    ckpt_filename: str = "last",
) -> pl.LightningModule:
    """
    Loads a PyTorch Lightning module from a specific WandB run ID and checkpoint.

    Args:
        project (str): The project name
        id (str): The run ID
        ckpt_filename (str, optional): The checkpoint filename. Defaults to "last".

    Returns:
        pl.LightningModule: The loaded PyTorch Lightning module.
    """

    config = config_from_id(project, id)
    model_config = config["model"]
    module: pl.LightningModule = instantiate(model_config)

    ckpt_path = get_ckpt_path(project, id, ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(ckpt["state_dict"])
    logger.info(f"Loaded module from experiment id {id} at checkpoint {ckpt_path}.")

    return module


def model_from_id(
    project: str,
    id: str,
    model_keyword: str,
    ckpt_filename: str = "last",
) -> nn.Module:
    """
    Loads a model from a specific WandB run ID and checkpoint.
    This is NOT a PyTorch Lightning module, but the underlying model inside the module.

    Args:
        project (str): The project name.
        id (str): The run ID.
        model_keyword (str): The keyword in the config corresponding to the model.
        ckpt_filename (str, optional): The checkpoint filename. Defaults to "last".

    Returns:
        nn.Module: The loaded model.
    """
    module = module_from_id(
        project=project,
        id=id,
        ckpt_filename=ckpt_filename,
    )

    model = getattr(module, model_keyword)

    return model


def get_root() -> str:
    """
    Returns:
        str: The root directory of this project based on git.
    """
    return os.popen("git rev-parse --show-toplevel").read().strip()


def get_artifact(
    project: str,
    filename: str,
):
    """Downloads a given artifact from WandB

    Args:
        project (str): The project name
        filename (str): The name of the artifact

    Returns:
        wandb.Artifact: The requested artifact
    """
    api = wandb.Api()
    return api.artifact(f"{project}/{filename}")


def download_checkpoint(
    project: str,
    id: str,
    filename: str,
) -> str:
    """
    Downloads a model checkpoint from WandB and saves it locally.
    Return the path to the downloaded checkpoint.

    Args:
        project (str): The project name
        id (str): The run ID
        filename (str): The name of the artifact

    Returns:
        str: The path to the downloaded checkpoint
    """

    root = get_root()
    artifact = get_artifact(project, filename)
    savedir = f"{root}/logs/{project}/{id}/checkpoints"
    final_path = f"{savedir}/{filename}.ckpt"

    os.makedirs(savedir, exist_ok=True)

    # Use a temporary directory to download first
    # to not overwrite existing files, since it will download as 'model.ckpt' as default
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.download(root=temp_dir)
        temp_file = f"{temp_dir}/model.ckpt"
        assert not os.path.exists(final_path), f"Checkpoint already exists at {final_path}."
        shutil.move(temp_file, final_path)
        logger.info(f"Downloaded checkpoint to {final_path}.")

    return final_path


if __name__ == "__main__":
    what_logs_to_delete()
