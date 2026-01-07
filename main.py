from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from src.utils import (
    instantiate_callbacks,
    get_current_time,
    get_ckpt_path,
    model_config_from_id,
)
import pytorch_lightning as pl
import os
import hydra
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Callback
import wandb
import yaml
import logging

os.environ["HYDRA_FULL_ERROR"] = "1"


def update_dict(d: dict | list[dict]) -> None:
    """
    Recursively update the dictionary to replace the model config with the one from the experiment id.
    Why? Because if the the same model is finetuned multiple times, the initialization process will be a mess since it will load all previous configs.
    """
    if isinstance(d, dict):
        if d.get("_target_", None) == "src.networks.PretrainedModel":
            model_keyword = d["model_keyword"]
            experiment_id = d["experiment_id"]
            model_config = model_config_from_id(experiment_id, model_keyword)
            d.clear()
            d.update(model_config)
        for k, v in d.items():
            update_dict(v)
    elif isinstance(d, list):
        for v in d:
            update_dict(v)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    logger = logging.getLogger(__name__)

    pl.seed_everything(cfg.seed)

    id = cfg.continue_from_id

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    update_dict(config)

    logger.info("Config:\n%s", yaml.dump(config, default_flow_style=False, sort_keys=False))

    wandblogger = WandbLogger(
        **cfg.logger,
        project=cfg.project_name,
        name=cfg.task_name,
        id=get_current_time() if not id else str(id),
        config=config,
    )

    if id:
        logger.info(f"Continuing from id: {id}")
        filename = "last" if cfg.ckpt_filename is None else cfg.ckpt_filename
        ckpt_path = get_ckpt_path(id, filename)
        logger.info(f"Using checkpoint path: {ckpt_path}")
    else:
        assert cfg.ckpt_filename is None, "Cannot specify ckpt_filename if not continuing from an experiment id."
        ckpt_path = None

    logger.info("Instantiating callbacks..")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks", None))

    logger.info("Setting up trainer..")
    trainer = Trainer(**cfg.trainer, logger=wandblogger, callbacks=callbacks)

    logger.info("Instantiating model and datamodule..")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.compile:
        logger.info("Compiling model..")
        logger.warning("You cannot compile models on CPU. Make sure you are using a GPU!")
        model = torch.compile(model)

    if cfg.phase == "train":
        logger.info("Beginning training..")
        trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    if cfg.phase == "test":
        if ckpt_path is None:
            logger.warning("No checkpoint path provided for testing. Is this intentional?")
        logger.info("Beginning testing..")
        trainer.test(model, datamodule, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    my_app()
