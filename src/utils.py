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


@contextlib.contextmanager
def temporary_seed(seed: int):
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
    now = datetime.now()
    return now.strftime("%d%m%y%H%M%S")


def instantiate_callbacks(callback_cfg: DictConfig | None) -> list[Callback]:
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
    ckpt_path = f"logs/{project}/{id}/checkpoints/{filename}.ckpt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    return ckpt_path


def what_logs_to_delete():
    api = wandb.Api()
    project_names = api.projects()
    project_names = [project.name for project in project_names]
    print("It is save to delete the following folders:")
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


def config_from_id(
    project: str,
    id: str
    ) -> dict:
    api = wandb.Api()
    name = wandb.api.viewer()["entity"]
    path = f"{name}/{project}/{id}"
    try:
        run = api.run(path)
        print(f"Found experiment {path}.")
        return run.config
    except wandb.errors.CommError:
        pass

    raise ValueError(f"Could not find experiment {path}.")


def model_config_from_id(
    project: str,
    id: str,
    model_keyword: str
    ) -> dict:
    config = config_from_id(project, id)
    return config["model"][model_keyword]


def model_from_id(
    project: str,
    id: str, 
    model_keyword: str,
    ckpt_filename: str = "last",
    ) -> nn.Module:
    config = config_from_id(id)
    model_config = config["model"]
    module: nn.Module = instantiate(model_config)

    ckpt_path = get_ckpt_path(project, id, ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(ckpt["state_dict"])

    model = getattr(module, model_keyword)
    print(f"Loaded model '{model_keyword}' from experiment id {id} at checkpoint {ckpt_path}.")

    return model


if __name__ == "__main__":
    what_logs_to_delete()
